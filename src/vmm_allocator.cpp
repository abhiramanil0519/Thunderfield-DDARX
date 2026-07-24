#include "kvvmm/vmm_allocator.hpp"

#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace kvvmm {
namespace {

std::size_t alignUp(std::size_t value, std::size_t alignment)
{
    if (alignment == 0) {
        return value;
    }
    const std::size_t remainder = value % alignment;
    return remainder == 0 ? value : value + (alignment - remainder);
}

void validateLogicalPage(const VirtualSequence& sequence, std::size_t logicalPage)
{
    if (logicalPage >= sequence.pageCount()) {
        std::ostringstream out;
        out << "Logical page " << logicalPage << " is outside sequence '" << sequence.name()
            << "' with " << sequence.pageCount() << " reserved pages.";
        throw std::out_of_range(out.str());
    }
}

} // namespace

PhysicalPage::PhysicalPage(
    CUcontext context,
    CUmemGenericAllocationHandle handle,
    std::size_t size,
    std::uint64_t id,
    std::string label)
    : context_(context)
    , handle_(handle)
    , size_(size)
    , id_(id)
    , label_(std::move(label))
{
}

PhysicalPage::~PhysicalPage() noexcept
{
    if (handle_ == 0) {
        return;
    }

    if (cuCtxPushCurrent(context_) == CUDA_SUCCESS) {
        static_cast<void>(cuMemRelease(handle_));
        CUcontext popped = nullptr;
        static_cast<void>(cuCtxPopCurrent(&popped));
    }
}

VirtualSequence::VirtualSequence(
    VmmAllocator* allocator,
    std::string name,
    CUdeviceptr base,
    std::size_t reservedBytes,
    std::size_t pageSize)
    : allocator_(allocator)
    , name_(std::move(name))
    , base_(base)
    , reservedBytes_(reservedBytes)
    , pageSize_(pageSize)
    , pages_(reservedBytes / pageSize)
{
}

VirtualSequence::~VirtualSequence() noexcept
{
    if (allocator_ != nullptr) {
        allocator_->destroySequence(*this);
    }
}

CUdeviceptr VirtualSequence::pageAddress(std::size_t logicalPage) const
{
    std::lock_guard lock(mutex_);
    validateLogicalPage(*this, logicalPage);
    return pageAddressUnlocked(logicalPage);
}

CUdeviceptr VirtualSequence::pageAddressUnlocked(std::size_t logicalPage) const noexcept
{
    return base_ + static_cast<CUdeviceptr>(logicalPage * pageSize_);
}

PhysicalPagePtr VirtualSequence::mappedPage(std::size_t logicalPage) const
{
    std::lock_guard lock(mutex_);
    validateLogicalPage(*this, logicalPage);
    return pages_[logicalPage];
}

std::size_t VirtualSequence::mappedPageCount() const
{
    std::lock_guard lock(mutex_);
    return static_cast<std::size_t>(std::count_if(pages_.begin(), pages_.end(), [](const PhysicalPagePtr& page) {
        return page != nullptr;
    }));
}

VmmAllocator::VmmAllocator(CUdevice device, int deviceOrdinal, CUcontext context, std::size_t requestedPageSize)
    : device_(device)
    , deviceOrdinal_(deviceOrdinal)
    , context_(context)
{
    allocationProp_.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    allocationProp_.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    allocationProp_.location.id = deviceOrdinal_;

    ContextGuard guard(context_);
    KVVMM_CU_CHECK(cuMemGetAllocationGranularity(
        &minimumGranularity_,
        &allocationProp_,
        CU_MEM_ALLOC_GRANULARITY_MINIMUM));
    KVVMM_CU_CHECK(cuMemGetAllocationGranularity(
        &recommendedGranularity_,
        &allocationProp_,
        CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));

    const std::size_t requested = requestedPageSize == 0 ? minimumGranularity_ : requestedPageSize;
    pageSize_ = std::max(minimumGranularity_, alignUp(requested, minimumGranularity_));

    accessDesc_.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    accessDesc_.location.id = deviceOrdinal_;
    accessDesc_.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
}

std::unique_ptr<VirtualSequence> VmmAllocator::reserveSequence(std::string name, std::size_t virtualBytes)
{
    const std::size_t alignedBytes = alignToPage(virtualBytes);
    if (alignedBytes == 0) {
        throw std::invalid_argument("Cannot reserve a zero-byte virtual sequence.");
    }

    CUdeviceptr base = 0;
    {
        ContextGuard guard(context_);
        KVVMM_CU_CHECK(cuMemAddressReserve(&base, alignedBytes, 0, 0, 0));
    }

    reservedVirtualBytes_.fetch_add(alignedBytes);
    return std::unique_ptr<VirtualSequence>(
        new VirtualSequence(this, std::move(name), base, alignedBytes, pageSize_));
}

PhysicalPagePtr VmmAllocator::createPhysicalPage(std::string label)
{
    CUmemGenericAllocationHandle handle{};
    {
        ContextGuard guard(context_);
        KVVMM_CU_CHECK(cuMemCreate(&handle, pageSize_, &allocationProp_, 0));
    }

    const std::uint64_t id = nextPageId_.fetch_add(1);
    createdPhysicalPages_.fetch_add(1);
    return PhysicalPagePtr(new PhysicalPage(context_, handle, pageSize_, id, std::move(label)));
}

void VmmAllocator::mapHandleAt(CUdeviceptr address, CUmemGenericAllocationHandle handle)
{
    KVVMM_CU_CHECK(cuMemMap(address, pageSize_, 0, handle, 0));
    try {
        KVVMM_CU_CHECK(cuMemSetAccess(address, pageSize_, &accessDesc_, 1));
    } catch (...) {
        static_cast<void>(cuMemUnmap(address, pageSize_));
        throw;
    }
}

void VmmAllocator::mapPage(VirtualSequence& sequence, std::size_t logicalPage, const PhysicalPagePtr& page)
{
    if (!page) {
        throw std::invalid_argument("Cannot map a null physical page.");
    }
    if (page->size() != pageSize_) {
        throw std::invalid_argument("Physical page size does not match allocator page size.");
    }

    std::lock_guard lock(sequence.mutex_);
    validateLogicalPage(sequence, logicalPage);
    if (sequence.pages_[logicalPage] != nullptr) {
        std::ostringstream out;
        out << "Sequence '" << sequence.name() << "' logical page " << logicalPage << " is already mapped.";
        throw std::logic_error(out.str());
    }

    ContextGuard guard(context_);
    mapHandleAt(sequence.pageAddressUnlocked(logicalPage), page->handle_);
    sequence.pages_[logicalPage] = page;
    page->mappingCount_.fetch_add(1);
}

void VmmAllocator::unmapPage(VirtualSequence& sequence, std::size_t logicalPage)
{
    std::lock_guard lock(sequence.mutex_);
    validateLogicalPage(sequence, logicalPage);
    if (sequence.pages_[logicalPage] == nullptr) {
        return;
    }

    ContextGuard guard(context_);
    KVVMM_CU_CHECK(cuMemUnmap(sequence.pageAddressUnlocked(logicalPage), pageSize_));
    sequence.pages_[logicalPage]->mappingCount_.fetch_sub(1);
    sequence.pages_[logicalPage].reset();
}

PhysicalPagePtr VmmAllocator::copyOnWrite(
    VirtualSequence& sequence,
    std::size_t logicalPage,
    CUstream stream,
    std::string label)
{
    std::lock_guard lock(sequence.mutex_);
    validateLogicalPage(sequence, logicalPage);

    PhysicalPagePtr oldPage = sequence.pages_[logicalPage];
    if (!oldPage) {
        throw std::logic_error("Cannot copy-on-write an unmapped logical page.");
    }
    if (oldPage->mappingCount() <= 1) {
        return oldPage;
    }

    PhysicalPagePtr freshPage = createPhysicalPage(std::move(label));
    const CUdeviceptr targetAddress = sequence.pageAddressUnlocked(logicalPage);

    ContextGuard guard(context_);

    CUdeviceptr scratchAddress = 0;
    KVVMM_CU_CHECK(cuMemAddressReserve(&scratchAddress, pageSize_, 0, 0, 0));

    bool scratchMapped = false;
    try {
        mapHandleAt(scratchAddress, freshPage->handle_);
        scratchMapped = true;
        KVVMM_CU_CHECK(cuMemcpyDtoDAsync(scratchAddress, targetAddress, pageSize_, stream));
        KVVMM_CU_CHECK(cuStreamSynchronize(stream));

        KVVMM_CU_CHECK(cuMemUnmap(scratchAddress, pageSize_));
        scratchMapped = false;
        KVVMM_CU_CHECK(cuMemAddressFree(scratchAddress, pageSize_));
        scratchAddress = 0;

        KVVMM_CU_CHECK(cuMemUnmap(targetAddress, pageSize_));
        try {
            mapHandleAt(targetAddress, freshPage->handle_);
        } catch (...) {
            mapHandleAt(targetAddress, oldPage->handle_);
            throw;
        }
    } catch (...) {
        if (scratchMapped) {
            static_cast<void>(cuMemUnmap(scratchAddress, pageSize_));
        }
        if (scratchAddress != 0) {
            static_cast<void>(cuMemAddressFree(scratchAddress, pageSize_));
        }
        throw;
    }

    oldPage->mappingCount_.fetch_sub(1);
    freshPage->mappingCount_.fetch_add(1);
    sequence.pages_[logicalPage] = freshPage;
    return freshPage;
}

void VmmAllocator::destroySequence(VirtualSequence& sequence) noexcept
{
    std::lock_guard lock(sequence.mutex_);
    for (std::size_t logicalPage = 0; logicalPage < sequence.pages_.size(); ++logicalPage) {
        unmapPageNoThrow(sequence, logicalPage);
    }

    if (sequence.base_ != 0 && sequence.reservedBytes_ != 0) {
        if (cuCtxPushCurrent(context_) == CUDA_SUCCESS) {
            static_cast<void>(cuMemAddressFree(sequence.base_, sequence.reservedBytes_));
            CUcontext popped = nullptr;
            static_cast<void>(cuCtxPopCurrent(&popped));
        }
        reservedVirtualBytes_.fetch_sub(sequence.reservedBytes_);
    }

    sequence.allocator_ = nullptr;
    sequence.base_ = 0;
    sequence.reservedBytes_ = 0;
}

void VmmAllocator::unmapPageNoThrow(VirtualSequence& sequence, std::size_t logicalPage) noexcept
{
    PhysicalPagePtr& page = sequence.pages_[logicalPage];
    if (!page) {
        return;
    }

    if (cuCtxPushCurrent(context_) == CUDA_SUCCESS) {
        static_cast<void>(cuMemUnmap(sequence.pageAddressUnlocked(logicalPage), pageSize_));
        CUcontext popped = nullptr;
        static_cast<void>(cuCtxPopCurrent(&popped));
    }
    page->mappingCount_.fetch_sub(1);
    page.reset();
}

std::size_t VmmAllocator::alignToPage(std::size_t bytes) const
{
    return alignUp(bytes, pageSize_);
}

std::future<PhysicalPagePtr> BackgroundMapper::mapPageAsync(
    VmmAllocator& allocator,
    VirtualSequence& sequence,
    std::size_t logicalPage,
    std::string label) const
{
    return std::async(
        std::launch::async,
        [&allocator, &sequence, logicalPage, label = std::move(label)]() mutable {
            ContextGuard guard(allocator.context());
            PhysicalPagePtr page = allocator.createPhysicalPage(std::move(label));
            allocator.mapPage(sequence, logicalPage, page);
            return page;
        });
}

} // namespace kvvmm
