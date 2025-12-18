# Code Coverage for SplatRs

This document describes the code coverage instrumentation setup for SplatRs, which measures how much of the codebase is exercised by the test suite.

## Overview

Code coverage helps identify:
- **Test gaps**: Which parts of the code lack test coverage
- **Dead code**: Functions or branches that may be unused
- **Critical paths**: Which code paths are exercised by tests
- **Quality metrics**: Overall test suite effectiveness

SplatRs uses `cargo-llvm-cov`, which integrates with Rust's LLVM-based toolchain to provide accurate coverage metrics.

## Quick Start

### Generate and view HTML coverage report

```bash
make coverage-open
```

This will:
1. Compile the code with coverage instrumentation
2. Run all tests (excluding `#[ignore]` tests)
3. Generate an HTML report
4. Open the report in your default browser

The report shows:
- **Line coverage**: Which lines were executed
- **Region coverage**: Which code regions (e.g., if/else branches) were covered
- **Function coverage**: Which functions were called

## Available Make Targets

| Target | Description |
|--------|-------------|
| `make coverage` | Generate HTML coverage report (saved to `target/coverage/html/`) |
| `make coverage-test` | Generate coverage with test output visible (useful for debugging) |
| `make coverage-open` | Generate coverage and open report in browser |
| `make coverage-clean` | Clean coverage artifacts |

## Manual Usage

If you prefer not to use the Makefile:

```bash
# Generate HTML report
cargo llvm-cov --html --all-features

# View output with test logs
cargo llvm-cov --html --all-features -- --nocapture

# Generate and open in browser
cargo llvm-cov --html --open --all-features

# Clean coverage data
cargo llvm-cov clean --workspace
```

## Configuration

Coverage configuration is defined in `llvm-cov.toml`:

```toml
[report]
# Exclude binary targets, examples, and tests from coverage
exclude = [
    "src/bin/*",
    "examples/*",
    "tests/*",
]

[html]
# HTML report output directory
output-dir = "target/coverage/html"

[json]
# JSON report for CI integration (future)
output-file = "target/coverage/coverage.json"
```

### Why exclude tests and binaries?

- **Tests (`tests/*`)**: Test code doesn't need coverage - we measure coverage *of* the library code *by* the tests
- **Examples (`examples/*`)**: Example code is for demonstration, not production
- **Binaries (`src/bin/*`)**: Binary targets are thin wrappers around library code; we measure the library instead

This keeps coverage metrics focused on the actual library implementation in `src/`.

## Understanding Coverage Metrics

The coverage report provides three key metrics:

### 1. Line Coverage
**What it measures**: Percentage of executable lines that were run during tests.

```rust
fn example(x: i32) -> i32 {
    if x > 0 {        // ✓ Covered
        return x * 2; // ✓ Covered if test calls with x > 0
    }
    0                 // ✗ Uncovered if no test calls with x <= 0
}
```

### 2. Region Coverage
**What it measures**: Percentage of code regions (branches) that were executed.

Even if all lines are covered, region coverage can be lower if not all branches are tested:

```rust
fn example(x: i32) -> &'static str {
    if x > 0 { "positive" } else { "non-positive" }
    // ✓ 100% line coverage if either branch runs
    // ✓ 100% region coverage only if BOTH branches run
}
```

### 3. Function Coverage
**What it measures**: Percentage of functions that were called at least once.

```rust
pub fn used_function() { }     // ✓ Covered if called
pub fn unused_function() { }   // ✗ Uncovered if never called
```

## Interpreting Results

### Good Coverage Targets

- **Line coverage**: Aim for 70-80%+ for critical modules
- **Function coverage**: 80%+ for public APIs
- **Region coverage**: Don't obsess over 100% - some error paths may be hard to test

### What to prioritize

1. **Core algorithms** (`src/render/`, `src/optim/`): High coverage critical
2. **GPU code**: Harder to test, but gradient checks help
3. **I/O code** (`src/io/`): Important for robustness
4. **Utilities**: Lower priority for 100% coverage

### Ignored tests

Some tests are marked `#[ignore]` because they:
- Require large datasets not in the repo
- Take a long time to run (30+ seconds)
- Are for visual/manual validation

These tests **are not included** in coverage by default. To include them:

```bash
cargo llvm-cov --html --all-features -- --ignored
```

## Troubleshooting

### Coverage data shows 0% or is missing

**Cause**: Tests didn't compile or run.

**Fix**: Run tests first to verify they pass:
```bash
cargo test --all-features
```

### "Warning: N functions have mismatched data"

**Cause**: LLVM coverage data has minor inconsistencies (usually harmless).

**Fix**: This warning is usually safe to ignore. If coverage looks wrong, try:
```bash
make coverage-clean
make coverage
```

### Coverage report doesn't open

**Cause**: `--open` flag failed to find browser.

**Fix**: Manually open the report:
```bash
open target/coverage/html/index.html    # macOS
xdg-open target/coverage/html/index.html # Linux
```

### Tests fail during coverage run

**Cause**: Test compilation errors or test failures.

**Fix**:
1. Run tests normally to see the full error: `cargo test`
2. Fix the failing tests
3. Re-run coverage: `make coverage`

### GPU tests cause issues

**Cause**: GPU tests may fail in some environments (CI, headless servers).

**Fix**: GPU tests are mostly `#[ignore]`d, so they won't affect coverage by default.

## Future Enhancements

### CI Integration
We plan to add coverage tracking to CI:
- Generate coverage on every PR
- Track coverage trends over time
- Fail PRs that significantly decrease coverage

### Coverage Badges
Add a coverage badge to README.md showing current coverage percentage.

### Differential Coverage
Show coverage only for changed lines in PRs.

## Baseline Coverage (Dec 2024)

Initial coverage report generated with:
- 21 tests passing
- 13 tests ignored (require datasets or are slow)
- Key test runs:
  - `test_m8_multiview_train_smoke_tandt`: 33.13s
  - `test_m9_densify_smoke_tandt`: 11.97s

Full baseline report: `target/coverage/html/index.html`

## Additional Resources

- [cargo-llvm-cov Documentation](https://github.com/taiki-e/cargo-llvm-cov)
- [LLVM Coverage Mapping](https://llvm.org/docs/CoverageMappingFormat.html)
- [Rust Testing Best Practices](https://doc.rust-lang.org/book/ch11-00-testing.html)
