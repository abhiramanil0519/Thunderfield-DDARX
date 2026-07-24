#include "kvvmm/driver_kernels.hpp"

#include <algorithm>

namespace kvvmm {
namespace {

constexpr const char* kPtx = R"ptx(
.version 6.5
.target sm_52
.address_size 64

.visible .entry fill_u32(
    .param .u64 destination,
    .param .u32 value,
    .param .u32 words
)
{
    .reg .pred %p;
    .reg .b32 %r<7>;
    .reg .b64 %rd<4>;

    ld.param.u64 %rd1, [destination];
    ld.param.u32 %r1, [value];
    ld.param.u32 %r2, [words];

    mov.u32 %r3, %tid.x;
    mov.u32 %r4, %ctaid.x;
    mov.u32 %r5, %ntid.x;
    mul.lo.u32 %r6, %r4, %r5;
    add.u32 %r6, %r6, %r3;

    setp.ge.u32 %p, %r6, %r2;
    @%p bra DONE_FILL;

    mul.wide.u32 %rd2, %r6, 4;
    add.u64 %rd3, %rd1, %rd2;
    st.global.u32 [%rd3], %r1;

DONE_FILL:
    ret;
}

.visible .entry patch_u32(
    .param .u64 destination,
    .param .u32 word_index,
    .param .u32 value
)
{
    .reg .b32 %r<3>;
    .reg .b64 %rd<4>;

    ld.param.u64 %rd1, [destination];
    ld.param.u32 %r1, [word_index];
    ld.param.u32 %r2, [value];

    mul.wide.u32 %rd2, %r1, 4;
    add.u64 %rd3, %rd1, %rd2;
    st.global.u32 [%rd3], %r2;
    ret;
}
)ptx";

} // namespace

DriverKernelModule::DriverKernelModule(CUcontext context)
    : context_(context)
{
    ContextGuard guard(context_);
    KVVMM_CU_CHECK(cuModuleLoadDataEx(&module_, kPtx, 0, nullptr, nullptr));
    KVVMM_CU_CHECK(cuModuleGetFunction(&fillU32_, module_, "fill_u32"));
    KVVMM_CU_CHECK(cuModuleGetFunction(&patchU32_, module_, "patch_u32"));
}

DriverKernelModule::~DriverKernelModule() noexcept
{
    if (module_ == nullptr) {
        return;
    }

    if (cuCtxPushCurrent(context_) == CUDA_SUCCESS) {
        static_cast<void>(cuModuleUnload(module_));
        CUcontext popped = nullptr;
        static_cast<void>(cuCtxPopCurrent(&popped));
    }
}

void DriverKernelModule::fillU32(CUdeviceptr destination, std::uint32_t value, std::uint32_t words, CUstream stream) const
{
    if (words == 0) {
        return;
    }

    constexpr unsigned int blockSize = 256;
    const unsigned int gridSize = std::max(1u, (words + blockSize - 1) / blockSize);

    void* args[] = {
        &destination,
        &value,
        &words,
    };

    ContextGuard guard(context_);
    KVVMM_CU_CHECK(cuLaunchKernel(
        fillU32_,
        gridSize,
        1,
        1,
        blockSize,
        1,
        1,
        0,
        stream,
        args,
        nullptr));
}

void DriverKernelModule::patchU32(
    CUdeviceptr destination,
    std::uint32_t wordIndex,
    std::uint32_t value,
    CUstream stream) const
{
    void* args[] = {
        &destination,
        &wordIndex,
        &value,
    };

    ContextGuard guard(context_);
    KVVMM_CU_CHECK(cuLaunchKernel(
        patchU32_,
        1,
        1,
        1,
        1,
        1,
        1,
        0,
        stream,
        args,
        nullptr));
}

} // namespace kvvmm
