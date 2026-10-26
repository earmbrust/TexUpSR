# TexUpSR quality checks. Invoke from any working directory.
[CmdletBinding()]
param(
    [ValidateSet('All', 'Format', 'Tidy', 'Cppcheck')]
    [string]$Mode = 'All'
)
$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Push-Location $root
try {
    function Run-Tool([string]$name, [string[]]$arguments) {
        $cmd = Get-Command $name -ErrorAction SilentlyContinue
        if (-not $cmd) { throw "Required tool is not on PATH: $name" }
        & $cmd.Source @arguments
        if ($LASTEXITCODE -ne 0) { throw "$name failed (exit $LASTEXITCODE)" }
    }

    if ($Mode -in @('All', 'Format')) {
        $format = Get-Command clang-format -ErrorAction SilentlyContinue
        if (-not $format) { throw 'Required tool is not on PATH: clang-format' }
        $version = (& $format.Source --version | Out-String)
        if ($LASTEXITCODE -ne 0 -or $version -notmatch '\b22\.1\.3\b') {
            throw "clang-format 22.1.3 is required; found: $($version.Trim())"
        }
        $sources = @(git ls-files --cached -- '*.cpp' '*.hpp' '*.h')
        if ($LASTEXITCODE -ne 0 -or $sources.Count -eq 0) { throw 'Could not enumerate tracked C++ files' }
        Write-Host "clang-format 22.1.3: checking $($sources.Count) tracked files"
        Run-Tool 'clang-format' (@('--dry-run', '--Werror', '--style=file') + $sources)
    }

    if ($Mode -in @('All', 'Tidy')) {
        # Dependency-free units only; CUDA/TensorRT/OpenCV units require a full SDK compilation DB.
        $units = @('src/Options.cpp', 'src/Sizing.cpp', 'tests/OptionsTests.cpp', 'tests/SizingTests.cpp')
        $checks = '-*,clang-analyzer-core.*,bugprone-sizeof-expression,bugprone-integer-division'
        Write-Host "clang-tidy: checking $($units.Count) SDK-independent translation units"
        foreach ($unit in $units) {
            Run-Tool 'clang-tidy' @('--quiet', "--checks=$checks", '--warnings-as-errors=*', $unit, '--', '-std=c++17', '-Isrc')
        }
    }

    if ($Mode -in @('All', 'Cppcheck')) {
        Write-Host 'Cppcheck: checking source and tests'
        Run-Tool 'cppcheck' @('--enable=warning,performance,portability', '--std=c++17', '--language=c++', '--error-exitcode=1', '--inline-suppr', '--suppress=missingIncludeSystem', '--suppress=missingInclude', '-Isrc', 'main.cpp', 'src', 'tests')
    }

    Write-Host "Quality checks PASS ($Mode)"
}
finally {
    Pop-Location
}
