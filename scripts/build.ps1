param(
    [string]$Config = "Release",
    [switch]$NoTests
)

$ErrorActionPreference = "Stop"

$repo = Resolve-Path (Join-Path $PSScriptRoot "..")
$build = Join-Path $repo "build"
$tests = if ($NoTests) { "OFF" } else { "ON" }

cmake -S $repo -B $build -DKVVMM_BUILD_TESTS=$tests
cmake --build $build --config $Config

if (-not $NoTests) {
    ctest --test-dir $build -C $Config --output-on-failure
}
