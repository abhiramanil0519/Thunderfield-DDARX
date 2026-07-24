#pragma once

#include "kvvmm/cuda_driver.hpp"

#include <atomic>
#include <cstdint>
#include <cuda.h>
#include <cstddef>
#include <future>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace kvvmm {

class VmmAllocator;

class PhysicalPage final {
public:
    ~PhysicalPage() noexcept;

    PhysicalPage(const PhysicalPage&) = delete;
    PhysicalPage& operator=(const PhysicalPage&) = delete;

    [[nodiscard]] std::uint64_t id() const noexcept { return id_; }
    [[nodiscard]] std::size_t size() const noexcept { return size_; }
    [[nodiscard]] const std::string& label() const noexcept { return label_; }
    [[nodiscard]] std::size_t mappingCount() const noexcept { return mappingCount_.load(); }

private:
    friend class VmmAllocator;

    PhysicalPage(
        CUcontext context,
        CUmemGenericAllocationHandle handle,
        std::size_t size,
        std::uint64_t id,
        std::string label);

    CUcontext context_{nullptr};
    CUmemGenericAllocationHandle handle_{};
    std::size_t size_{0};
    std::uint64_t id_{0};
    std::string label_;
    std::atomic_size_t mappingCount_{0};
};

using PhysicalPagePtr = std::shared_ptr<PhysicalPage>;

class VirtualSequence final {
public:
    ~VirtualSequence() noexcept;

    VirtualSequence(const VirtualSequence&) = delete;
    VirtualSequence& operator=(const VirtualSequence&) = delete;

    [[nodiscard]] const std::string& name() const noexcept { return name_; }
    [[nodiscard]] CUdeviceptr base() const noexcept { return base_; }
    [[nodiscard]] std::size_t reservedBytes() const noexcept { return reservedBytes_; }
    [[nodiscard]] std::size_t pageSize() const noexcept { return pageSize_; }
    [[nodiscard]] std::size_t pageCount() const noexcept { return pages_.size(); }
    [[nodiscard]] CUdeviceptr pageAddress(std::size_t logicalPage) const;
    [[nodiscard]] PhysicalPagePtr mappedPage(std::size_t logicalPage) const;
    [[nodiscard]] std::size_t mappedPageCount() const;

private:
    friend class VmmAllocator;

    VirtualSequence(
        VmmAllocator* allocator,
        std::string name,
        CUdeviceptr base,
        std::size_t reservedBytes,
        std::size_t pageSize);

    [[nodiscard]] CUdeviceptr pageAddressUnlocked(std::size_t logicalPage) const noexcept;

    VmmAllocator* allocator_{nullptr};
    std::string name_;
    CUdeviceptr base_{0};
    std::size_t reservedBytes_{0};
    std::size_t pageSize_{0};
    std::vector<PhysicalPagePtr> pages_;
    mutable std::mutex mutex_;
};

class VmmAllocator final {
public:
    VmmAllocator(CUdevice device, int deviceOrdinal, CUcontext context, std::size_t requestedPageSize);
    ~VmmAllocator() = default;

    VmmAllocator(const VmmAllocator&) = delete;
    VmmAllocator& operator=(const VmmAllocator&) = delete;

    [[nodiscard]] std::unique_ptr<VirtualSequence> reserveSequence(std::string name, std::size_t virtualBytes);
    [[nodiscard]] PhysicalPagePtr createPhysicalPage(std::string label);

    void mapPage(VirtualSequence& sequence, std::size_t logicalPage, const PhysicalPagePtr& page);
    void unmapPage(VirtualSequence& sequence, std::size_t logicalPage);
    [[nodiscard]] PhysicalPagePtr copyOnWrite(VirtualSequence& sequence, std::size_t logicalPage, CUstream stream, std::string label);

    [[nodiscard]] CUcontext context() const noexcept { return context_; }
    [[nodiscard]] std::size_t pageSize() const noexcept { return pageSize_; }
    [[nodiscard]] std::size_t minimumGranularity() const noexcept { return minimumGranularity_; }
    [[nodiscard]] std::size_t recommendedGranularity() const noexcept { return recommendedGranularity_; }
    [[nodiscard]] std::uint64_t createdPhysicalPages() const noexcept { return createdPhysicalPages_.load(); }
    [[nodiscard]] std::size_t reservedVirtualBytes() const noexcept { return reservedVirtualBytes_.load(); }

private:
    friend class VirtualSequence;

    void destroySequence(VirtualSequence& sequence) noexcept;
    void mapHandleAt(CUdeviceptr address, CUmemGenericAllocationHandle handle);
    void unmapPageNoThrow(VirtualSequence& sequence, std::size_t logicalPage) noexcept;
    [[nodiscard]] std::size_t alignToPage(std::size_t bytes) const;

    CUdevice device_{0};
    int deviceOrdinal_{0};
    CUcontext context_{nullptr};
    CUmemAllocationProp allocationProp_{};
    CUmemAccessDesc accessDesc_{};
    std::size_t minimumGranularity_{0};
    std::size_t recommendedGranularity_{0};
    std::size_t pageSize_{0};
    std::atomic_uint64_t nextPageId_{1};
    std::atomic_uint64_t createdPhysicalPages_{0};
    std::atomic_size_t reservedVirtualBytes_{0};
};

class BackgroundMapper final {
public:
    [[nodiscard]] std::future<PhysicalPagePtr> mapPageAsync(
        VmmAllocator& allocator,
        VirtualSequence& sequence,
        std::size_t logicalPage,
        std::string label) const;
};

} // namespace kvvmm
