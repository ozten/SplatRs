use sugar_rs::viewer::{commands::*, state::AppState};

fn main() {
    // Initialize GPU renderer and state
    let state = AppState::new().expect("Failed to initialize GPU");

    tauri::Builder::default()
        .manage(state)
        .invoke_handler(tauri::generate_handler![
            log_to_stdout,
            list_models,
            load_model,
            render_frame,
            get_camera_by_id,
            set_disable_sh,
            get_disable_sh
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
