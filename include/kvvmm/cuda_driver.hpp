#pragma once

#include <cuda.h>

#include <cstddef>
#include <stdexcept>
#include <string>

namespace kvvmm {

class DriverError final : public std::runtime_error {
public:
    DriverError(CUresult result, const char* expression, const char* file, int line);

    [[nodiscard]] CUresult result() const noexcept { return result_; }

private:
    CUresult result_;
};

[[nodiscard]] std::string describeResult(CUresult result);

void checkDriver(CUresult result, const char* expression, const char* file, int line);

#define KVVMM_CU_CHECK(expression) \
    ::kvvmm::checkDriver((expression), #expression, __FILE__, __LINE__)

class ContextGuard final {
public:
    explicit ContextGuard(CUcontext context);
    ~ContextGuard() noexcept;

    ContextGuard(const ContextGuard&) = delete;
    ContextGuard& operator=(const ContextGuard&) = delete;

private:
    bool active_{false};
};

class DriverStream final {
public:
    explicit DriverStream(CUcontext context, unsigned int flags = CU_STREAM_NON_BLOCKING);
    ~DriverStream() noexcept;

    DriverStream(const DriverStream&) = delete;
    DriverStream& operator=(const DriverStream&) = delete;

    [[nodiscard]] CUstream get() const noexcept { return stream_; }
    void synchronize() const;

private:
    CUcontext context_{nullptr};
    CUstream stream_{nullptr};
};

class DeviceContext final {
public:
    explicit DeviceContext(int ordinal);
    ~DeviceContext() noexcept;

    DeviceContext(const DeviceContext&) = delete;
    DeviceContext& operator=(const DeviceContext&) = delete;

    [[nodiscard]] CUdevice device() const noexcept { return device_; }
    [[nodiscard]] CUcontext context() const noexcept { return context_; }
    [[nodiscard]] int ordinal() const noexcept { return ordinal_; }
    [[nodiscard]] std::string name() const;
    [[nodiscard]] int computeMajor() const;
    [[nodiscard]] int computeMinor() const;
    [[nodiscard]] std::size_t totalMemoryBytes() const;
    [[nodiscard]] bool virtualMemoryManagementSupported() const;

private:
    int ordinal_{0};
    CUdevice device_{0};
    CUcontext context_{nullptr};
};

[[nodiscard]] std::string formatBytes(std::size_t bytes);

} // namespace kvvmm
