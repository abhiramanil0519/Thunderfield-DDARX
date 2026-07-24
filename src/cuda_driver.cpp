#include "kvvmm/cuda_driver.hpp"

#include <array>
#include <cstdint>
#include <sstream>

namespace kvvmm {

std::string describeResult(CUresult result)
{
    const char* name = nullptr;
    const char* description = nullptr;
    cuGetErrorName(result, &name);
    cuGetErrorString(result, &description);

    std::ostringstream out;
    out << (name != nullptr ? name : "CUDA_ERROR_UNKNOWN");
    if (description != nullptr) {
        out << ": " << description;
    }
    return out.str();
}

DriverError::DriverError(CUresult result, const char* expression, const char* file, int line)
    : std::runtime_error([&] {
        std::ostringstream out;
        out << expression << " failed at " << file << ':' << line << " (" << describeResult(result) << ')';
        return out.str();
    }())
    , result_(result)
{
}

void checkDriver(CUresult result, const char* expression, const char* file, int line)
{
    if (result != CUDA_SUCCESS) {
        throw DriverError(result, expression, file, line);
    }
}

ContextGuard::ContextGuard(CUcontext context)
{
    KVVMM_CU_CHECK(cuCtxPushCurrent(context));
    active_ = true;
}

ContextGuard::~ContextGuard() noexcept
{
    if (!active_) {
        return;
    }

    CUcontext popped = nullptr;
    static_cast<void>(cuCtxPopCurrent(&popped));
}

DriverStream::DriverStream(CUcontext context, unsigned int flags)
    : context_(context)
{
    ContextGuard guard(context_);
    KVVMM_CU_CHECK(cuStreamCreate(&stream_, flags));
}

DriverStream::~DriverStream() noexcept
{
    if (stream_ == nullptr) {
        return;
    }

    if (cuCtxPushCurrent(context_) == CUDA_SUCCESS) {
        static_cast<void>(cuStreamDestroy(stream_));
        CUcontext popped = nullptr;
        static_cast<void>(cuCtxPopCurrent(&popped));
    }
}

void DriverStream::synchronize() const
{
    ContextGuard guard(context_);
    KVVMM_CU_CHECK(cuStreamSynchronize(stream_));
}

DeviceContext::DeviceContext(int ordinal)
    : ordinal_(ordinal)
{
    KVVMM_CU_CHECK(cuInit(0));

    int deviceCount = 0;
    KVVMM_CU_CHECK(cuDeviceGetCount(&deviceCount));
    if (deviceCount <= 0) {
        throw std::runtime_error("No CUDA-capable devices are visible to the NVIDIA driver.");
    }
    if (ordinal < 0 || ordinal >= deviceCount) {
        std::ostringstream out;
        out << "Device ordinal " << ordinal << " is out of range; visible device count is " << deviceCount << '.';
        throw std::out_of_range(out.str());
    }

    KVVMM_CU_CHECK(cuDeviceGet(&device_, ordinal_));
    KVVMM_CU_CHECK(cuCtxCreate(&context_, CU_CTX_SCHED_AUTO, device_));
}

DeviceContext::~DeviceContext() noexcept
{
    if (context_ != nullptr) {
        static_cast<void>(cuCtxDestroy(context_));
    }
}

std::string DeviceContext::name() const
{
    std::array<char, 256> buffer{};
    KVVMM_CU_CHECK(cuDeviceGetName(buffer.data(), static_cast<int>(buffer.size()), device_));
    return buffer.data();
}

int DeviceContext::computeMajor() const
{
    int value = 0;
    KVVMM_CU_CHECK(cuDeviceGetAttribute(&value, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device_));
    return value;
}

int DeviceContext::computeMinor() const
{
    int value = 0;
    KVVMM_CU_CHECK(cuDeviceGetAttribute(&value, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device_));
    return value;
}

std::size_t DeviceContext::totalMemoryBytes() const
{
    std::size_t bytes = 0;
    KVVMM_CU_CHECK(cuDeviceTotalMem(&bytes, device_));
    return bytes;
}

bool DeviceContext::virtualMemoryManagementSupported() const
{
#ifdef CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED
    int supported = 0;
    KVVMM_CU_CHECK(cuDeviceGetAttribute(&supported, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, device_));
    return supported != 0;
#else
    return true;
#endif
}

std::string formatBytes(std::size_t bytes)
{
    constexpr std::array<const char*, 5> suffixes = { "B", "KiB", "MiB", "GiB", "TiB" };
    double value = static_cast<double>(bytes);
    std::size_t suffix = 0;

    while (value >= 1024.0 && suffix + 1 < suffixes.size()) {
        value /= 1024.0;
        ++suffix;
    }

    std::ostringstream out;
    if (suffix == 0) {
        out << bytes << ' ' << suffixes[suffix];
    } else {
        out.setf(std::ios::fixed);
        out.precision(value >= 100.0 ? 0 : value >= 10.0 ? 1 : 2);
        out << value << ' ' << suffixes[suffix];
    }
    return out.str();
}

} // namespace kvvmm
