param([switch]$SkipTests)
$ErrorActionPreference = 'Stop'
$cmake = [System.IO.File]::ReadAllText((Resolve-Path 'CMakeLists.txt').Path)
if ($cmake -match 'CopyTestRuntimes\.cmake|copy_if_different|COMMAND\s+\$\{CMAKE_COMMAND\}\s+-E\s+copy') {
    throw 'CMake still contains a runtime DLL deployment rule. Review CMakeLists.txt.'
}
Write-Host 'PASS: no runtime DLL deployment rules found in CMakeLists.txt.'
if (-not $SkipTests) {
    & cmake --build --preset integration
    if ($LASTEXITCODE -ne 0) { throw 'Integration build failed' }
    & ctest --test-dir build-integration -C Release --output-on-failure
    if ($LASTEXITCODE -ne 0) { throw 'CTest failed' }
}
Write-Host 'Validation completed. Existing DLL files in older build directories are not removed or treated as new deployment.'
