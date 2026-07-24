param(
    [int]$Device = 0,
    [int]$PrefixBlocks = 4,
    [int]$ReserveMiB = 256,
    [int]$PageKiB = 0,
    [string]$Config = "Release"
)

$ErrorActionPreference = "Stop"

$repo = Resolve-Path (Join-Path $PSScriptRoot "..")
$exe = Join-Path $repo "build\$Config\kvvmm-demo.exe"

if (-not (Test-Path $exe)) {
    throw "Demo executable not found at $exe. Run scripts/build.ps1 first."
}

& $exe --device $Device --prefix-blocks $PrefixBlocks --reserve-mib $ReserveMiB --page-kib $PageKiB
