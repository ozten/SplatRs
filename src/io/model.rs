//! Binary model format for saving and loading trained Gaussian Splatting models.
//!
//! File format: `.gs` (Gaussian Splatting Model)
//!
//! Layout:
//! ```text
//! Header (256 bytes):
//!   - Magic: "GSPLAT\0\0" (8 bytes)
//!   - Version: u32 (4 bytes)
//!   - Num Gaussians: u64 (8 bytes)
//!   - SH degree: u32 (4 bytes)
//!   - Scene bounds min: 3 × f32 (12 bytes)
//!   - Scene bounds max: 3 × f32 (12 bytes)
//!   - Training iterations: u64 (8 bytes)
//!   - Training PSNR: f32 (4 bytes)
//!   - Compression: u32 (4 bytes) - 0=none, 1=lz4
//!   - Training width: u32 (4 bytes) - training image width
//!   - Training height: u32 (4 bytes) - training image height
//!   - Training downsample factor: f32 (4 bytes) - 0.25 = 25% resolution
//!   - Reserved: 176 bytes (for future use)
//!
//! Gaussian Data (per Gaussian):
//!   - Position: 3 × f32 (12 bytes)
//!   - Scale (log): 3 × f32 (12 bytes)
//!   - Rotation (quaternion w,x,y,z): 4 × f32 (16 bytes)
//!   - Opacity (logit): f32 (4 bytes)
//!   - SH coefficients: 16 × 3 × f32 (192 bytes)
//!   Total per Gaussian: 236 bytes
//! ```
//!
//! Example file sizes:
//! - 10K Gaussians: ~2.4 MB uncompressed
//! - 100K Gaussians: ~24 MB uncompressed
//! - With LZ4 compression: typically 5-10x smaller

use crate::core::{Gaussian, GaussianCloud};
use nalgebra::{UnitQuaternion, Vector3};
use std::io::{Read, Write};
use std::path::Path;

const MAGIC: &[u8; 8] = b"GSPLAT\0\0";
const VERSION: u32 = 2;  // Version 2: added dataset_path field
const HEADER_SIZE: usize = 256;

/// Compression method
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Compression {
    None = 0,
    #[cfg(feature = "lz4")]
    Lz4 = 1,
}

/// Metadata about the trained model
#[derive(Debug, Clone)]
pub struct ModelMetadata {
    /// Number of Gaussians
    pub num_gaussians: u64,

    /// SH degree (0-3)
    pub sh_degree: u32,

    /// Scene bounding box (min corner)
    pub bounds_min: Vector3<f32>,

    /// Scene bounding box (max corner)
    pub bounds_max: Vector3<f32>,

    /// Number of training iterations completed
    pub training_iterations: u64,

    /// Final training PSNR (dB)
    pub training_psnr: f32,

    /// Compression method used
    pub compression: Compression,

    /// Training image width (after downsampling)
    pub training_width: u32,

    /// Training image height (after downsampling)
    pub training_height: u32,

    /// Downsample factor applied during training (e.g., 0.25 = 25% of original)
    pub training_downsample_factor: f32,

    /// Dataset path used for training (for COLMAP camera access)
    pub dataset_path: String,
}

impl Default for ModelMetadata {
    fn default() -> Self {
        Self {
            num_gaussians: 0,
            sh_degree: 3,
            bounds_min: Vector3::new(-1.0, -1.0, -1.0),
            bounds_max: Vector3::new(1.0, 1.0, 1.0),
            training_iterations: 0,
            training_psnr: 0.0,
            compression: Compression::None,
            training_width: 0,
            training_height: 0,
            training_downsample_factor: 1.0,
            dataset_path: String::new(),
        }
    }
}

/// Error type for model I/O
#[derive(Debug)]
pub enum ModelError {
    Io(std::io::Error),
    InvalidMagic,
    UnsupportedVersion(u32),
    UnsupportedCompression(u32),
    InvalidData(String),
}

impl From<std::io::Error> for ModelError {
    fn from(e: std::io::Error) -> Self {
        ModelError::Io(e)
    }
}

impl std::fmt::Display for ModelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelError::Io(e) => write!(f, "I/O error: {}", e),
            ModelError::InvalidMagic => write!(f, "Invalid file magic (not a .gs file)"),
            ModelError::UnsupportedVersion(v) => write!(f, "Unsupported version: {}", v),
            ModelError::UnsupportedCompression(c) => write!(f, "Unsupported compression: {}", c),
            ModelError::InvalidData(msg) => write!(f, "Invalid data: {}", msg),
        }
    }
}

impl std::error::Error for ModelError {}

/// Save a GaussianCloud to a .gs file
pub fn save_model<P: AsRef<Path>>(
    path: P,
    cloud: &GaussianCloud,
    metadata: &ModelMetadata,
) -> Result<(), ModelError> {
    let file = std::fs::File::create(path)?;
    let mut writer = std::io::BufWriter::new(file);

    // Write header
    write_header(&mut writer, cloud, metadata)?;

    // Write dataset path (variable length string)
    let dataset_path_bytes = metadata.dataset_path.as_bytes();
    let path_len = dataset_path_bytes.len() as u32;
    writer.write_all(&path_len.to_le_bytes())?;
    writer.write_all(dataset_path_bytes)?;

    // Write Gaussian data
    write_gaussians(&mut writer, &cloud.gaussians, metadata.compression)?;

    Ok(())
}

/// Load a GaussianCloud from a .gs file
pub fn load_model<P: AsRef<Path>>(path: P) -> Result<(GaussianCloud, ModelMetadata), ModelError> {
    let file = std::fs::File::open(path)?;
    let mut reader = std::io::BufReader::new(file);

    // Read header
    let mut metadata = read_header(&mut reader)?;

    // Read dataset path (variable length string) - only for version 2+
    if metadata.num_gaussians > 0 {  // Use this as a proxy for valid file
        // Try to read dataset path length
        let mut len_buf = [0u8; 4];
        if reader.read_exact(&mut len_buf).is_ok() {
            let path_len = u32::from_le_bytes(len_buf) as usize;
            // Sanity check - path shouldn't be more than 4KB
            if path_len < 4096 {
                let mut path_buf = vec![0u8; path_len];
                if reader.read_exact(&mut path_buf).is_ok() {
                    if let Ok(dataset_path) = String::from_utf8(path_buf) {
                        metadata.dataset_path = dataset_path;
                    }
                }
            }
        }
    }

    // Read Gaussian data
    let gaussians = read_gaussians(&mut reader, metadata.num_gaussians as usize, metadata.compression)?;

    let cloud = GaussianCloud::from_gaussians(gaussians);
    Ok((cloud, metadata))
}

/// Write header to file
fn write_header<W: Write>(
    writer: &mut W,
    cloud: &GaussianCloud,
    metadata: &ModelMetadata,
) -> Result<(), ModelError> {
    let mut header = vec![0u8; HEADER_SIZE];
    let mut offset = 0;

    // Magic
    header[offset..offset + 8].copy_from_slice(MAGIC);
    offset += 8;

    // Version
    header[offset..offset + 4].copy_from_slice(&VERSION.to_le_bytes());
    offset += 4;

    // Num Gaussians
    let num_gaussians = cloud.len() as u64;
    header[offset..offset + 8].copy_from_slice(&num_gaussians.to_le_bytes());
    offset += 8;

    // SH degree
    header[offset..offset + 4].copy_from_slice(&metadata.sh_degree.to_le_bytes());
    offset += 4;

    // Scene bounds min
    header[offset..offset + 4].copy_from_slice(&metadata.bounds_min.x.to_le_bytes());
    offset += 4;
    header[offset..offset + 4].copy_from_slice(&metadata.bounds_min.y.to_le_bytes());
    offset += 4;
    header[offset..offset + 4].copy_from_slice(&metadata.bounds_min.z.to_le_bytes());
    offset += 4;

    // Scene bounds max
    header[offset..offset + 4].copy_from_slice(&metadata.bounds_max.x.to_le_bytes());
    offset += 4;
    header[offset..offset + 4].copy_from_slice(&metadata.bounds_max.y.to_le_bytes());
    offset += 4;
    header[offset..offset + 4].copy_from_slice(&metadata.bounds_max.z.to_le_bytes());
    offset += 4;

    // Training iterations
    header[offset..offset + 8].copy_from_slice(&metadata.training_iterations.to_le_bytes());
    offset += 8;

    // Training PSNR
    header[offset..offset + 4].copy_from_slice(&metadata.training_psnr.to_le_bytes());
    offset += 4;

    // Compression
    header[offset..offset + 4].copy_from_slice(&(metadata.compression as u32).to_le_bytes());
    offset += 4;

    // Training width
    header[offset..offset + 4].copy_from_slice(&metadata.training_width.to_le_bytes());
    offset += 4;

    // Training height
    header[offset..offset + 4].copy_from_slice(&metadata.training_height.to_le_bytes());
    offset += 4;

    // Training downsample factor
    header[offset..offset + 4].copy_from_slice(&metadata.training_downsample_factor.to_le_bytes());
    offset += 4;

    // Reserved (176 bytes) - already zeroed
    // offset is now at 80 (68 + 12)

    writer.write_all(&header)?;
    Ok(())
}

/// Read header from file
fn read_header<R: Read>(reader: &mut R) -> Result<ModelMetadata, ModelError> {
    let mut header = vec![0u8; HEADER_SIZE];
    reader.read_exact(&mut header)?;

    let mut offset = 0;

    // Check magic
    if &header[offset..offset + 8] != MAGIC {
        return Err(ModelError::InvalidMagic);
    }
    offset += 8;

    // Version
    let version = u32::from_le_bytes(header[offset..offset + 4].try_into().unwrap());
    if version != VERSION {
        return Err(ModelError::UnsupportedVersion(version));
    }
    offset += 4;

    // Num Gaussians
    let num_gaussians = u64::from_le_bytes(header[offset..offset + 8].try_into().unwrap());
    offset += 8;

    // SH degree
    let sh_degree = u32::from_le_bytes(header[offset..offset + 4].try_into().unwrap());
    offset += 4;

    // Scene bounds min
    let min_x = f32::from_le_bytes(header[offset..offset + 4].try_into().unwrap());
    offset += 4;
    let min_y = f32::from_le_bytes(header[offset..offset + 4].try_into().unwrap());
    offset += 4;
    let min_z = f32::from_le_bytes(header[offset..offset + 4].try_into().unwrap());
    offset += 4;
    let bounds_min = Vector3::new(min_x, min_y, min_z);

    // Scene bounds max
    let max_x = f32::from_le_bytes(header[offset..offset + 4].try_into().unwrap());
    offset += 4;
    let max_y = f32::from_le_bytes(header[offset..offset + 4].try_into().unwrap());
    offset += 4;
    let max_z = f32::from_le_bytes(header[offset..offset + 4].try_into().unwrap());
    offset += 4;
    let bounds_max = Vector3::new(max_x, max_y, max_z);

    // Training iterations
    let training_iterations = u64::from_le_bytes(header[offset..offset + 8].try_into().unwrap());
    offset += 8;

    // Training PSNR
    let training_psnr = f32::from_le_bytes(header[offset..offset + 4].try_into().unwrap());
    offset += 4;

    // Compression
    let compression_u32 = u32::from_le_bytes(header[offset..offset + 4].try_into().unwrap());
    let compression = match compression_u32 {
        0 => Compression::None,
        #[cfg(feature = "lz4")]
        1 => Compression::Lz4,
        #[cfg(not(feature = "lz4"))]
        1 => return Err(ModelError::UnsupportedCompression(compression_u32)),
        _ => return Err(ModelError::UnsupportedCompression(compression_u32)),
    };
    offset += 4;

    // Training resolution (backward compatible - default to 0 if not present)
    let training_width = if offset + 4 <= header.len() {
        u32::from_le_bytes(header[offset..offset + 4].try_into().unwrap())
    } else {
        0
    };
    offset += 4;

    let training_height = if offset + 4 <= header.len() {
        u32::from_le_bytes(header[offset..offset + 4].try_into().unwrap())
    } else {
        0
    };
    offset += 4;

    let training_downsample_factor = if offset + 4 <= header.len() {
        f32::from_le_bytes(header[offset..offset + 4].try_into().unwrap())
    } else {
        1.0
    };

    Ok(ModelMetadata {
        num_gaussians,
        sh_degree,
        bounds_min,
        bounds_max,
        training_iterations,
        training_psnr,
        compression,
        training_width,
        training_height,
        training_downsample_factor,
        dataset_path: String::new(),  // Will be populated in load_model for v2+
    })
}

/// Write Gaussians to file
fn write_gaussians<W: Write>(
    writer: &mut W,
    gaussians: &[Gaussian],
    compression: Compression,
) -> Result<(), ModelError> {
    // Serialize all Gaussians to binary
    let mut data = Vec::new();
    for g in gaussians {
        write_gaussian(&mut data, g)?;
    }

    // Compress if requested
    match compression {
        Compression::None => {
            writer.write_all(&data)?;
        }
        #[cfg(feature = "lz4")]
        Compression::Lz4 => {
            // Write uncompressed size first (for decompression buffer allocation)
            writer.write_all(&(data.len() as u64).to_le_bytes())?;

            // Compress and write
            let compressed = lz4_flex::compress_prepend_size(&data);
            writer.write_all(&compressed)?;
        }
    }

    Ok(())
}

/// Read Gaussians from file
fn read_gaussians<R: Read>(
    reader: &mut R,
    num_gaussians: usize,
    compression: Compression,
) -> Result<Vec<Gaussian>, ModelError> {
    // Read data (decompressing if needed)
    let data = match compression {
        Compression::None => {
            // Calculate expected size
            let gaussian_size = 236; // position(12) + scale(12) + rotation(16) + opacity(4) + sh(192)
            let expected_size = num_gaussians * gaussian_size;

            let mut data = vec![0u8; expected_size];
            reader.read_exact(&mut data)?;
            data
        }
        #[cfg(feature = "lz4")]
        Compression::Lz4 => {
            // Read uncompressed size
            let mut size_bytes = [0u8; 8];
            reader.read_exact(&mut size_bytes)?;
            let _uncompressed_size = u64::from_le_bytes(size_bytes);

            // Read compressed data
            let mut compressed = Vec::new();
            reader.read_to_end(&mut compressed)?;

            // Decompress
            lz4_flex::decompress_size_prepended(&compressed)
                .map_err(|e| ModelError::InvalidData(format!("LZ4 decompression failed: {}", e)))?
        }
    };

    // Deserialize Gaussians
    let mut gaussians = Vec::with_capacity(num_gaussians);
    let mut offset = 0;

    for _ in 0..num_gaussians {
        let g = read_gaussian(&data[offset..])?;
        gaussians.push(g);
        offset += 236; // Size of one Gaussian
    }

    Ok(gaussians)
}

/// Write a single Gaussian to a byte buffer
fn write_gaussian<W: Write>(writer: &mut W, g: &Gaussian) -> Result<(), ModelError> {
    // Position (12 bytes)
    writer.write_all(&g.position.x.to_le_bytes())?;
    writer.write_all(&g.position.y.to_le_bytes())?;
    writer.write_all(&g.position.z.to_le_bytes())?;

    // Scale (12 bytes)
    writer.write_all(&g.scale.x.to_le_bytes())?;
    writer.write_all(&g.scale.y.to_le_bytes())?;
    writer.write_all(&g.scale.z.to_le_bytes())?;

    // Rotation (16 bytes) - quaternion as (w, x, y, z)
    let q = g.rotation.quaternion();
    writer.write_all(&q.w.to_le_bytes())?;
    writer.write_all(&q.i.to_le_bytes())?;
    writer.write_all(&q.j.to_le_bytes())?;
    writer.write_all(&q.k.to_le_bytes())?;

    // Opacity (4 bytes)
    writer.write_all(&g.opacity.to_le_bytes())?;

    // SH coefficients (192 bytes) - 16 × 3 floats
    for sh in &g.sh_coeffs {
        writer.write_all(&sh[0].to_le_bytes())?;
        writer.write_all(&sh[1].to_le_bytes())?;
        writer.write_all(&sh[2].to_le_bytes())?;
    }

    Ok(())
}

/// Read a single Gaussian from a byte buffer
fn read_gaussian(data: &[u8]) -> Result<Gaussian, ModelError> {
    if data.len() < 236 {
        return Err(ModelError::InvalidData("Insufficient data for Gaussian".to_string()));
    }

    let mut offset = 0;

    // Position
    let px = f32::from_le_bytes(data[offset..offset + 4].try_into().unwrap());
    offset += 4;
    let py = f32::from_le_bytes(data[offset..offset + 4].try_into().unwrap());
    offset += 4;
    let pz = f32::from_le_bytes(data[offset..offset + 4].try_into().unwrap());
    offset += 4;
    let position = Vector3::new(px, py, pz);

    // Scale
    let sx = f32::from_le_bytes(data[offset..offset + 4].try_into().unwrap());
    offset += 4;
    let sy = f32::from_le_bytes(data[offset..offset + 4].try_into().unwrap());
    offset += 4;
    let sz = f32::from_le_bytes(data[offset..offset + 4].try_into().unwrap());
    offset += 4;
    let scale = Vector3::new(sx, sy, sz);

    // Rotation (quaternion)
    let qw = f32::from_le_bytes(data[offset..offset + 4].try_into().unwrap());
    offset += 4;
    let qx = f32::from_le_bytes(data[offset..offset + 4].try_into().unwrap());
    offset += 4;
    let qy = f32::from_le_bytes(data[offset..offset + 4].try_into().unwrap());
    offset += 4;
    let qz = f32::from_le_bytes(data[offset..offset + 4].try_into().unwrap());
    offset += 4;

    // Normalize quaternion (important!)
    let rotation = UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(qw, qx, qy, qz));

    // Opacity
    let opacity = f32::from_le_bytes(data[offset..offset + 4].try_into().unwrap());
    offset += 4;

    // SH coefficients
    let mut sh_coeffs = [[0.0f32; 3]; 16];
    for sh in &mut sh_coeffs {
        sh[0] = f32::from_le_bytes(data[offset..offset + 4].try_into().unwrap());
        offset += 4;
        sh[1] = f32::from_le_bytes(data[offset..offset + 4].try_into().unwrap());
        offset += 4;
        sh[2] = f32::from_le_bytes(data[offset..offset + 4].try_into().unwrap());
        offset += 4;
    }

    Ok(Gaussian::new(position, scale, rotation, opacity, sh_coeffs))
}

/// Compute scene bounding box from Gaussians
pub fn compute_bounds(gaussians: &[Gaussian]) -> (Vector3<f32>, Vector3<f32>) {
    if gaussians.is_empty() {
        return (Vector3::zeros(), Vector3::zeros());
    }

    let mut min = gaussians[0].position;
    let mut max = gaussians[0].position;

    for g in gaussians {
        min.x = min.x.min(g.position.x);
        min.y = min.y.min(g.position.y);
        min.z = min.z.min(g.position.z);

        max.x = max.x.max(g.position.x);
        max.y = max.y.max(g.position.y);
        max.z = max.z.max(g.position.z);
    }

    (min, max)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_save_load_roundtrip() {
        use tempfile::NamedTempFile;

        // Create a simple cloud
        let mut cloud = GaussianCloud::new();
        for i in 0..10 {
            let g = Gaussian::new(
                Vector3::new(i as f32, 0.0, 0.0),
                Vector3::new(0.0, 0.0, 0.0), // log scale = exp(0) = 1
                UnitQuaternion::identity(),
                0.0, // logit opacity
                [[0.5; 3]; 16],
            );
            cloud.push(g);
        }

        // Compute metadata
        let (bounds_min, bounds_max) = compute_bounds(&cloud.gaussians);
        let metadata = ModelMetadata {
            num_gaussians: cloud.len() as u64,
            sh_degree: 3,
            bounds_min,
            bounds_max,
            training_iterations: 1000,
            training_psnr: 25.5,
            compression: Compression::None,
            training_width: 256,
            training_height: 256,
            training_downsample_factor: 0.25,
            dataset_path: String::new(),
        };

        // Save and load
        let temp_file = NamedTempFile::new().unwrap();
        save_model(temp_file.path(), &cloud, &metadata).unwrap();

        let (loaded_cloud, loaded_metadata) = load_model(temp_file.path()).unwrap();

        // Verify metadata
        assert_eq!(loaded_metadata.num_gaussians, 10);
        assert_eq!(loaded_metadata.sh_degree, 3);
        assert_eq!(loaded_metadata.training_iterations, 1000);
        assert_relative_eq!(loaded_metadata.training_psnr, 25.5, epsilon = 1e-6);
        assert_eq!(loaded_metadata.training_width, 256);
        assert_eq!(loaded_metadata.training_height, 256);
        assert_relative_eq!(loaded_metadata.training_downsample_factor, 0.25, epsilon = 1e-6);

        // Verify Gaussians
        assert_eq!(loaded_cloud.len(), 10);
        for i in 0..10 {
            let orig = &cloud.gaussians[i];
            let loaded = &loaded_cloud.gaussians[i];

            assert_relative_eq!(orig.position.x, loaded.position.x, epsilon = 1e-6);
            assert_relative_eq!(orig.position.y, loaded.position.y, epsilon = 1e-6);
            assert_relative_eq!(orig.position.z, loaded.position.z, epsilon = 1e-6);

            assert_relative_eq!(orig.opacity, loaded.opacity, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_bounds_computation() {
        let cloud = GaussianCloud::from_gaussians(vec![
            Gaussian::new(
                Vector3::new(0.0, 0.0, 0.0),
                Vector3::zeros(),
                UnitQuaternion::identity(),
                0.0,
                [[0.0; 3]; 16],
            ),
            Gaussian::new(
                Vector3::new(10.0, 5.0, -3.0),
                Vector3::zeros(),
                UnitQuaternion::identity(),
                0.0,
                [[0.0; 3]; 16],
            ),
        ]);

        let (min, max) = compute_bounds(&cloud.gaussians);

        assert_relative_eq!(min.x, 0.0, epsilon = 1e-6);
        assert_relative_eq!(min.y, 0.0, epsilon = 1e-6);
        assert_relative_eq!(min.z, -3.0, epsilon = 1e-6);

        assert_relative_eq!(max.x, 10.0, epsilon = 1e-6);
        assert_relative_eq!(max.y, 5.0, epsilon = 1e-6);
        assert_relative_eq!(max.z, 0.0, epsilon = 1e-6);
    }
}
