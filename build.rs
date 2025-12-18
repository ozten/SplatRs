fn main() {
    // Only run tauri-build when building the viewer binary
    #[cfg(feature = "viewer")]
    tauri_build::build()
}
