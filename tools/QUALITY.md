# TexUpSR quality gate

Use PowerShell from the repository root:

```powershell
pwsh -NoProfile -File tools/check.ps1
```

Optional `-Mode Format`, `-Mode Tidy` and `-Mode Cppcheck` run checks separately. The clang-format version is **22.1.3**; install matching tools before using the all-in-one command. The normal CMake Release build does **not** invoke static analysis.

Enable the tracked Git hook once on each checkout:

```powershell
git config core.hooksPath .githooks
```

The hook verifies the complete working source tree, not only staged changes. Stage the checked contents before committing. Local VS Code PATH settings remain private and are not needed on CI.

## Coverage

- clang-format: all tracked `.cpp`, `.hpp`, and `.h` files.
- Cppcheck: source and tests, treating warning/performance/portability findings as failures.
- Default clang-tidy (`All` or `Tidy`): six SDK-independent translation units, including tensor sizing and its regression tests. This scope stays fast and is enforced by the existing hook and CI quality job.
- Full clang-tidy (`TidyFull`): explicitly opt in with a real compiler-generated `compile_commands.json`. Every tracked `.cpp` must have an entry; otherwise the check **fails**, rather than reporting incomplete results as full coverage.

## Full clang-tidy (local, SDK-backed)

Generate the compilation database with a CMake generator that supports `CMAKE_EXPORT_COMPILE_COMMANDS`, typically Ninja. Visual Studio generators do not export a usable `compile_commands.json`. Use your installed CUDA, TensorRT and OpenCV SDKs and the same C++17 options as the Release build. For example, in an appropriately initialized Visual Studio developer PowerShell environment with Ninja and the SDK paths set:

```powershell
cmake -S . -B build-analysis -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DTensorRT_ROOT="$env:TENSORRT_ROOT" -DCUDAToolkit_ROOT="$env:CUDA_PATH" -DOpenCV_DIR="$env:OpenCV_DIR"
```

Only run this example if `OpenCV_DIR` points to your working OpenCV CMake package and your compiler environment is initialized. Compilation database generation does not build the project. No `compile_commands.json` files should be committed.

Then run:

```powershell
.\tools\check.ps1 -Mode TidyFull -CompilationDatabase .\build-analysis
```

`TidyFull` deliberately fails if the database omits any tracked translation unit or clang-tidy cannot parse its real compile flags/SDK headers. Review findings before modifying checks. Full analysis is **not yet enabled in the automatic hook or CI** until this SDK-backed path is validated locally; the existing fast quality gate continues to run independently of Release builds.

The CI quality job runs before both Windows build matrix entries, including direct invocation of the reusable Windows workflow. CI execution remains unverified while GitHub Actions usage is restricted.
