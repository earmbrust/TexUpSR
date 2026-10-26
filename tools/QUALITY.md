# Quality checks (CMake-native)

Configure the **Ninja** analysis preset once. Use `windows-analysis-local` if your
ignored `CMakeUserPresets.json` defines machine-specific SDK paths:

```text
cmake --preset windows-analysis-local
cmake --build build-analysis --target quality
```

Or use the checked-in `windows-analysis` preset if its environment variables are
configured. The developer terminal must already have access to an x64 MSVC
compiler (on Windows, use a VS Developer PowerShell/Command Prompt). CMake
is responsible for compiler detection; no tool changes the process PATH.

Targets (all available through `cmake --build build-analysis --target <name>`):

- `format-check`: clang-format **22.1.3** checks project C++ sources and headers.
- `tidy`: clang-tidy uses CMake's **actual Ninja compilation database** for every
  project `.cpp` translation unit; checks and warnings-as-errors live in `.clang-tidy`.
- `cppcheck`: warning, portability and performance checks for `main.cpp`, `src`, `tests`.
- `quality`: runs all three checks.

The target list is derived from CMake targets rather than a manually maintained
second source list. Newly added `src/` or `tests/` `.cpp` files missing from the
CMake targets cause configuration to fail. Quality checks are opt-in targets;
normal build, inference, and CTest are unchanged.

The existing `windows-vs2026`/`windows-release` build remains the normal MSBuild
workflow; Ninja is used for analysis because the Visual Studio generator does
not produce `compile_commands.json`. A cross-platform Ninja preset can be added
for each validated platform without rewriting these quality targets.

Enable the optional pre-commit hook with:

```text
git config core.hooksPath .githooks
```

The hook requires the analysis build to be configured first. It checks the
working tree, so stage the checked changes before committing.

CI workflow changes are intentionally **deferred**. The previous Ubuntu quality
job used `tools/check.ps1` without SDKs; converting it to SDK-backed CMake
analysis needs separate CI design and validation. Until then, that workflow's
quality job needs updating before CI is re-enabled.
