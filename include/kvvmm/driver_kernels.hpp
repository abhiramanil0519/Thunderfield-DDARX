#pragma once

#include "kvvmm/cuda_driver.hpp"

#include <cuda.h>
#include <cstdint>

namespace kvvmm {

class DriverKernelModule final {
public:
    explicit DriverKernelModule(CUcontext context);
    ~DriverKernelModule() noexcept;

    DriverKernelModule(const DriverKernelModule&) = delete;
    DriverKernelModule& operator=(const DriverKernelModule&) = delete;

    void fillU32(CUdeviceptr destination, std::uint32_t value, std::uint32_t words, CUstream stream) const;
    void patchU32(CUdeviceptr destination, std::uint32_t wordIndex, std::uint32_t value, CUstream stream) const;

private:
    CUcontext context_{nullptr};
    CUmodule module_{nullptr};
    CUfunction fillU32_{nullptr};
    CUfunction patchU32_{nullptr};
};

} // namespace kvvmm
