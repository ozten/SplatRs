 - Missing the “educational” unit tests promised in AGENTS.md for core math: Camera::world_to_camera, Camera::project, Gaussian::covariance_matrix,
    Gaussian2D::inverse_covariance, and SH evaluation lack targeted unit tests that check invariants/identities, not just “does it run” (src/core/
    camera.rs, src/core/gaussian.rs, src/render/*).
  - Many tests are integration/E2E or GPU comparisons rather than unit tests of math primitives; this makes regressions harder to localize and weakens
    the “teach through tests” goal (tests/* broadly).
  - Gradient checks are present (good), but they are clustered around end‑to‑end render paths; there are few micro‑gradient checks on single functions
    (e.g., perspective_jacobian, SH basis) that would explain the math and be resilient to unrelated pipeline changes (src/core/math.rs, src/render/
    full_diff.rs).
  - Assertions often compare rendered images or expect exact CPU/GPU parity without tolerances or explanatory checks of intermediate invariants (e.g.,
    orthogonality, symmetry, determinant), which are more stable and educational (tests/gpu_*, tests/m3_*, tests/m4_*).
  - Several tests write outputs for manual inspection but don’t assert on metadata/structure (e.g., PLY header correctness, image dimensions, pixel
    statistics), missing easy automated checks that still teach (tests/m1_colmap_load.rs, tests/m3_render_spheres.rs, tests/m4_render_gaussians.rs).

  Quality rating

  - Unit test quality: 5/10.
    You have good coverage of “it runs and looks plausible,” and some solid gradient checks, but the unit tests don’t yet demonstrate the math in
    isolation or teach the derivations the way your AGENTS.md principles call for.

  How to do them better (prose, concrete patterns)

  - Write “math invariant” tests that teach the property you’re relying on.
    Example: for Gaussian::covariance_matrix, assert symmetry, positive‑definiteness (all eigenvalues > 0), and that rotating a diagonal covariance
    produces off‑diagonal terms. This reinforces why factorization is used.
  - For camera projection, test a tiny canonical scene with known values and derive the answer in the test.
    Example: camera at identity with fx=fy=100, cx=50, cy=60, point (1,2,4) should project to (75,110). Include comments showing the arithmetic so the
    test doubles as documentation.
  - Add “micro gradient checks” for isolated functions.
    Example: finite‑difference perspective_jacobian against numerical gradients for random points with fixed fx/fy. This makes it easier to trust the
    Jacobian without running the whole renderer.
  - Prefer structured assertions over image diffs for unit tests.
    Example: for a Gaussian splat, assert that the peak is at the mean, that values fall off radially, and that the integral over a small window is
    within a tolerance. This is far more robust than pixel‑exact golden images.
  - Create deterministic “toy scenes” that are fully in‑memory and tiny.
    Example: one Gaussian, one camera, 8×8 image. That makes expected results easy to reason about and keeps tests fast and precise.
  - Separate educational tests from end‑to‑end tests.
    Put unit tests in src/... modules with explicit derivations; keep E2E in tests/ with #[ignore]. This aligns with your “tests as documentation”
    priority.
