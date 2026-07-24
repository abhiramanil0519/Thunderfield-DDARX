#include "kvvmm/cuda_driver.hpp"
#include "kvvmm/driver_kernels.hpp"
#include "kvvmm/radix_cache.hpp"
#include "kvvmm/vmm_allocator.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstdint>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

struct Options final {
    int device{0};
    std::size_t reserveMiB{256};
    std::size_t pageKiB{0};
    std::size_t prefixBlocks{4};
    bool quiet{false};
};

[[noreturn]] void usage()
{
    std::cout
        << "KV VMM Driver Console\n\n"
        << "Usage:\n"
        << "  kvvmm-demo [options]\n\n"
        << "Options:\n"
        << "  --device N           CUDA device ordinal. Default: 0\n"
        << "  --reserve-mib N      Virtual bytes reserved per sequence. Default: 256\n"
        << "  --page-kib N         Requested physical page size. 0 uses driver minimum granularity.\n"
        << "  --prefix-blocks N    Shared prefix pages to create and reuse. Default: 4\n"
        << "  --quiet              Print only the final verification summary.\n"
        << "  --help               Show this help.\n";
    std::exit(0);
}

std::size_t parseSize(std::string_view text, std::string_view option)
{
    std::size_t consumed = 0;
    const std::string value(text);
    const std::size_t parsed = std::stoull(value, &consumed, 10);
    if (consumed != value.size()) {
        std::ostringstream out;
        out << "Invalid numeric value for " << option << ": '" << value << "'.";
        throw std::invalid_argument(out.str());
    }
    return parsed;
}

Options parseOptions(int argc, char** argv)
{
    Options options;

    for (int i = 1; i < argc; ++i) {
        const std::string_view arg(argv[i]);
        const auto nextValue = [&](std::string_view option) -> std::string_view {
            if (i + 1 >= argc) {
                std::ostringstream out;
                out << "Missing value after " << option << '.';
                throw std::invalid_argument(out.str());
            }
            return argv[++i];
        };

        if (arg == "--help" || arg == "-h") {
            usage();
        } else if (arg == "--device") {
            options.device = static_cast<int>(parseSize(nextValue(arg), arg));
        } else if (arg == "--reserve-mib") {
            options.reserveMiB = parseSize(nextValue(arg), arg);
        } else if (arg == "--page-kib") {
            options.pageKiB = parseSize(nextValue(arg), arg);
        } else if (arg == "--prefix-blocks") {
            options.prefixBlocks = parseSize(nextValue(arg), arg);
        } else if (arg == "--quiet") {
            options.quiet = true;
        } else {
            std::ostringstream out;
            out << "Unknown option: " << arg;
            throw std::invalid_argument(out.str());
        }
    }

    if (options.prefixBlocks == 0) {
        throw std::invalid_argument("--prefix-blocks must be greater than zero.");
    }

    return options;
}

std::uint32_t pageFillValue(std::size_t page)
{
    return 0xa1000000u | static_cast<std::uint32_t>(page & 0x0000ffffu);
}

std::vector<std::int32_t> tokenBlock(std::size_t blockIndex)
{
    constexpr std::size_t tokensPerBlock = 16;
    std::vector<std::int32_t> tokens(tokensPerBlock);
    for (std::size_t i = 0; i < tokens.size(); ++i) {
        tokens[i] = static_cast<std::int32_t>(blockIndex * tokensPerBlock + i + 1);
    }
    return tokens;
}

std::vector<kvvmm::BlockSignature> makePrefixSignatures(std::size_t blocks)
{
    std::vector<kvvmm::BlockSignature> signatures;
    signatures.reserve(blocks);

    for (std::size_t block = 0; block < blocks; ++block) {
        const std::vector<std::int32_t> tokens = tokenBlock(block);
        signatures.push_back(kvvmm::BlockSignature::fromTokens(
            std::span<const std::int32_t>(tokens.data(), tokens.size())));
    }

    return signatures;
}

std::uint32_t readU32(CUcontext context, CUdeviceptr address)
{
    std::uint32_t value = 0;
    kvvmm::ContextGuard guard(context);
    KVVMM_CU_CHECK(cuMemcpyDtoH(&value, address, sizeof(value)));
    return value;
}

void printMapping(const kvvmm::VirtualSequence& sequence, std::size_t pages)
{
    std::cout << "  " << sequence.name() << " virtual base 0x"
              << std::hex << sequence.base() << std::dec << '\n';
    for (std::size_t i = 0; i < pages; ++i) {
        const kvvmm::PhysicalPagePtr page = sequence.mappedPage(i);
        if (!page) {
            std::cout << "    page[" << i << "] -> unmapped\n";
            continue;
        }
        std::cout << "    page[" << i << "] -> physical #" << page->id()
                  << " (" << page->label() << ", mappings=" << page->mappingCount() << ")\n";
    }
}

int runDemo(const Options& options)
{
    kvvmm::DeviceContext device(options.device);
    if (!device.virtualMemoryManagementSupported()) {
        throw std::runtime_error("Selected CUDA device does not report virtual memory management support.");
    }

    const std::size_t requestedPageBytes = options.pageKiB * 1024;
    kvvmm::VmmAllocator allocator(device.device(), device.ordinal(), device.context(), requestedPageBytes);

    const std::size_t minimumPages = options.prefixBlocks + 2;
    const std::size_t requestedReserveBytes = options.reserveMiB * 1024ull * 1024ull;
    const std::size_t reserveBytes = std::max(requestedReserveBytes, minimumPages * allocator.pageSize());

    const auto pageWords64 = allocator.pageSize() / sizeof(std::uint32_t);
    if (pageWords64 > std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error("Selected page size is too large for the demo PTX fill kernel.");
    }
    const auto wordsPerPage = static_cast<std::uint32_t>(pageWords64);

    kvvmm::DriverStream stream(device.context());
    kvvmm::DriverKernelModule kernels(device.context());
    kvvmm::RadixCache<kvvmm::PhysicalPage> prefixCache;

    auto requestA = allocator.reserveSequence("request-A", reserveBytes);
    auto requestB = allocator.reserveSequence("request-B", reserveBytes);

    if (!options.quiet) {
        std::cout << "Device: " << device.name()
                  << " (sm_" << device.computeMajor() << device.computeMinor() << ")\n"
                  << "Total memory: " << kvvmm::formatBytes(device.totalMemoryBytes()) << '\n'
                  << "VMM granularity: min=" << kvvmm::formatBytes(allocator.minimumGranularity())
                  << ", recommended=" << kvvmm::formatBytes(allocator.recommendedGranularity()) << '\n'
                  << "Selected page size: " << kvvmm::formatBytes(allocator.pageSize()) << '\n'
                  << "Virtual reservation per sequence: " << kvvmm::formatBytes(reserveBytes) << "\n\n";
    }

    const std::vector<kvvmm::BlockSignature> prefixSignatures = makePrefixSignatures(options.prefixBlocks);
    std::vector<kvvmm::PhysicalPagePtr> requestAPages;
    requestAPages.reserve(options.prefixBlocks);

    for (std::size_t pageIndex = 0; pageIndex < options.prefixBlocks; ++pageIndex) {
        kvvmm::PhysicalPagePtr page = allocator.createPhysicalPage("prefix-" + std::to_string(pageIndex));
        allocator.mapPage(*requestA, pageIndex, page);
        kernels.fillU32(requestA->pageAddress(pageIndex), pageFillValue(pageIndex), wordsPerPage, stream.get());
        requestAPages.push_back(std::move(page));
    }
    stream.synchronize();

    prefixCache.insertPrefix(
        std::span<const kvvmm::BlockSignature>(prefixSignatures.data(), prefixSignatures.size()),
        std::span<const kvvmm::PhysicalPagePtr>(requestAPages.data(), requestAPages.size()));

    const auto prefixHit = prefixCache.lookupPrefix(
        std::span<const kvvmm::BlockSignature>(prefixSignatures.data(), prefixSignatures.size()));

    if (prefixHit.matchedBlocks() != options.prefixBlocks) {
        throw std::runtime_error("Prefix cache failed to return the prefix that was just inserted.");
    }

    for (std::size_t pageIndex = 0; pageIndex < prefixHit.pages.size(); ++pageIndex) {
        allocator.mapPage(*requestB, pageIndex, prefixHit.pages[pageIndex]);
    }

    kvvmm::BackgroundMapper backgroundMapper;
    auto nextDecodePage = backgroundMapper.mapPageAsync(
        allocator,
        *requestB,
        options.prefixBlocks,
        "request-B-decode-next");

    kernels.patchU32(requestA->pageAddress(0), 1, 0xa110c0deu, stream.get());
    stream.synchronize();

    kvvmm::PhysicalPagePtr requestBDecodePage = nextDecodePage.get();
    kernels.fillU32(requestB->pageAddress(options.prefixBlocks), 0xb2000000u, wordsPerPage, stream.get());
    stream.synchronize();

    const std::size_t cowPageIndex = std::min<std::size_t>(1, options.prefixBlocks - 1);
    kvvmm::PhysicalPagePtr privatePage = allocator.copyOnWrite(
        *requestB,
        cowPageIndex,
        stream.get(),
        "request-B-cow-prefix-" + std::to_string(cowPageIndex));
    kernels.patchU32(requestB->pageAddress(cowPageIndex), 0, 0xbeef1234u, stream.get());
    stream.synchronize();

    const std::uint32_t aValue = readU32(device.context(), requestA->pageAddress(cowPageIndex));
    const std::uint32_t bValue = readU32(device.context(), requestB->pageAddress(cowPageIndex));
    const std::uint32_t bDecodeValue = readU32(device.context(), requestB->pageAddress(options.prefixBlocks));

    const std::uint32_t expectedA = pageFillValue(cowPageIndex);
    if (aValue != expectedA || bValue != 0xbeef1234u || bDecodeValue != 0xb2000000u) {
        std::ostringstream out;
        out << "Verification failed: A=0x" << std::hex << aValue
            << " expected 0x" << expectedA
            << ", B=0x" << bValue
            << " expected 0xbeef1234"
            << ", B_decode=0x" << bDecodeValue
            << " expected 0xb2000000.";
        throw std::runtime_error(out.str());
    }

    if (!options.quiet) {
        std::cout << "Prefix cache: " << prefixHit.matchedBlocks() << '/' << options.prefixBlocks
                  << " blocks reused, radix nodes=" << prefixCache.stats().nodes << "\n\n";

        std::cout << "Mappings after reuse and COW:\n";
        printMapping(*requestA, options.prefixBlocks + 1);
        printMapping(*requestB, options.prefixBlocks + 1);

        const std::size_t logicalMappedBytes =
            (requestA->mappedPageCount() + requestB->mappedPageCount()) * allocator.pageSize();
        const std::size_t committedBytes = allocator.createdPhysicalPages() * allocator.pageSize();

        std::cout << "\nAccounting:\n"
                  << "  total virtual reserved: " << kvvmm::formatBytes(allocator.reservedVirtualBytes()) << '\n'
                  << "  logical mapped bytes:   " << kvvmm::formatBytes(logicalMappedBytes) << '\n'
                  << "  physical pages created: " << allocator.createdPhysicalPages()
                  << " (" << kvvmm::formatBytes(committedBytes) << ")\n"
                  << "  request-B decode page:  physical #" << requestBDecodePage->id() << '\n'
                  << "  request-B COW page:     physical #" << privatePage->id() << "\n\n";
    }

    std::cout << "Verification OK: shared prefix pages were reused, request-B COW stayed private, "
              << "and all GPU writes used flat virtual addresses.\n";

    return 0;
}

} // namespace

int main(int argc, char** argv)
{
    try {
        return runDemo(parseOptions(argc, argv));
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
