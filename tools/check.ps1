# TexUpSR quality checks. Invoke from any working directory.
[CmdletBinding()]
param(
    [ValidateSet('All', 'Format', 'Tidy', 'Cppcheck', 'TidyFull')]
    [string]$Mode = 'All',
    [string]$CompilationDatabase = ''
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

    if ($Mode -in @('All', 'Tidy', 'TidyFull')) {
        $checks = '-*,clang-analyzer-core.*,bugprone-sizeof-expression,bugprone-integer-division'
        if ($Mode -eq 'TidyFull') {
            if (-not $CompilationDatabase) {
                throw 'TidyFull requires -CompilationDatabase pointing to a directory containing compile_commands.json'
            }
            $database = (Resolve-Path -LiteralPath $CompilationDatabase -ErrorAction Stop).Path
            if (Test-Path -LiteralPath $database -PathType Leaf) {
                if ([IO.Path]::GetFileName($database) -ne 'compile_commands.json') {
                    throw 'CompilationDatabase file must be named compile_commands.json'
                }
                $database = Split-Path -Parent $database
            }
            $jsonFile = Join-Path $database 'compile_commands.json'
            if (-not (Test-Path -LiteralPath $jsonFile -PathType Leaf)) {
                throw "Compilation database not found: $jsonFile"
            }
            $entries = @(Get-Content -LiteralPath $jsonFile -Raw | ConvertFrom-Json)
            if ($entries.Count -eq 0) { throw 'Compilation database is empty' }
            $available = New-Object 'System.Collections.Generic.HashSet[string]' ([StringComparer]::OrdinalIgnoreCase)
            foreach ($entry in $entries) {
                if (-not $entry.file -or -not $entry.directory) {
                    throw 'Compilation database entry is missing its file or directory'
                }
                $resolved = [IO.Path]::GetFullPath([IO.Path]::Combine([string]$entry.directory, [string]$entry.file))
                [void]$available.Add($resolved)
            }
            $units = @(git ls-files --cached -- '*.cpp')
            if ($LASTEXITCODE -ne 0 -or $units.Count -eq 0) {
                throw 'Could not enumerate tracked C++ translation units'
            }
            $missing = @($units | Where-Object {
                -not $available.Contains([IO.Path]::GetFullPath((Join-Path $root $_)))
            })
            if ($missing.Count -gt 0) {
                throw "Compilation database lacks $($missing.Count) tracked C++ units: $($missing -join ', ')"
            }
            Write-Host "clang-tidy: checking all $($units.Count) tracked C++ translation units via $jsonFile"
            foreach ($unit in $units) {
                Run-Tool 'clang-tidy' @('--quiet', "--checks=$checks", '--warnings-as-errors=*', '-p', $database, $unit)
            }
        } else {
            # SDK-independent units only; do not silently claim SDK coverage.
            $units = @('src/Options.cpp', 'src/Sizing.cpp', 'src/TensorSizes.cpp',
                       'tests/OptionsTests.cpp', 'tests/SizingTests.cpp', 'tests/TensorSizesTests.cpp')
            Write-Host "clang-tidy: checking $($units.Count) SDK-independent translation units"
            foreach ($unit in $units) {
                Run-Tool 'clang-tidy' @('--quiet', "--checks=$checks", '--warnings-as-errors=*', $unit, '--', '-std=c++17', '-Isrc')
            }
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
