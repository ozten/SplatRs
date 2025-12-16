//! OBJ format export for meshes.
//!
//! OBJ is a simple text-based mesh format widely supported by 3D tools.
//! We'll use it to export meshes extracted from SuGaR (M14).

use crate::io::LoadError;
use std::path::Path;

/// Mesh data structure (to be defined in sugar::mesh module).
///
/// For now, just a placeholder type.
pub struct Mesh {
    // TODO: Move to sugar::mesh module when implementing M14
}

/// Save a mesh to OBJ format.
///
/// OBJ format is simple:
/// ```text
/// v x y z           # Vertex positions
/// vn nx ny nz       # Vertex normals (optional)
/// f i j k           # Faces (triangles)
/// ```
pub fn save_obj(mesh: &Mesh, path: &Path) -> Result<(), LoadError> {
    // TODO: Implement for M14
    unimplemented!("See M14 - OBJ mesh export")
}
