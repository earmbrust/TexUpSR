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
- clang-tidy: SDK-independent translation units `src/Options.cpp`, `src/Sizing.cpp`, `tests/OptionsTests.cpp`, `tests/SizingTests.cpp`. This limited scope is **intentional**: GPU/OpenCV/TensorRT sources need the correct SDK-backed compilation database, and must not be claimed as clang-tidy-covered yet.

The CI quality job runs before both Windows build matrix entries, including direct invocation of the reusable Windows workflow. Full-SDK clang-tidy coverage should be added as a subsequent, separately validated improvement.
