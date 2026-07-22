# Training Metadata JSONL Schema

**Issue**: hq-2zi
**Status**: Design Specification
**Version**: 1.0
**Date**: 2026-01-21

## Overview

This document specifies the JSONL (JSON Lines) format for tracking training metadata alongside PLY checkpoint files. This replaces the monolithic .gs format with a separation of concerns:
- **PLY files**: Gaussian parameters (standard, portable)
- **JSONL file**: Training metadata (append-only log)

## Goals

1. **Separation of Concerns**: Gaussian data (PLY) separate from training metadata (JSONL)
2. **Append-Only Log**: Track full training history without rewriting files
3. **Easy Analysis**: Simple to parse, plot training curves, find best checkpoint
4. **Backward Compatible**: Include all fields from current .gs format
5. **Human Readable**: JSON text format for easy inspection and debugging

## File Structure

### Naming Convention

```
training_output/
├── checkpoint_0000.ply          # Initial point cloud
├── checkpoint_0100.ply          # After 100 iterations
├── checkpoint_0200.ply          # After 200 iterations
├── ...
├── checkpoint_1000.ply          # Final model
└── training_history.jsonl       # Metadata for all checkpoints
```

### JSONL Format

- **Extension**: `.jsonl` (JSON Lines)
- **Encoding**: UTF-8
- **Line Format**: One JSON object per line, newline-separated
- **Ordering**: Chronological (append-only)
- **Size**: ~200-500 bytes per checkpoint (negligible compared to PLY files)

## Schema Definition

### Core Fields

Each line in `training_history.jsonl` is a JSON object with these fields:

```json
{
  "checkpoint_file": "checkpoint_1000.ply",
  "timestamp": "2026-01-21T02:30:45.123Z",
  "iteration": 1000,
  "num_gaussians": 100000,
  "sh_degree": 3,
  "training_psnr": 28.5,
  "training_loss": 0.0042,
  "scene_bounds": {
    "min": [-5.2, -3.1, -2.0],
    "max": [5.2, 3.1, 2.0]
  },
  "training_config": {
    "image_width": 1920,
    "image_height": 1080,
    "downsample_factor": 0.25,
    "num_training_images": 100
  },
  "dataset_info": {
    "dataset_path": "/path/to/dataset",
    "dataset_type": "colmap"
  },
  "optimization_state": {
    "learning_rate": 0.00016,
    "position_lr": 0.00016,
    "scale_lr": 0.005,
    "rotation_lr": 0.001,
    "opacity_lr": 0.05,
    "sh_lr": 0.0025
  },
  "performance": {
    "training_time_seconds": 125.5,
    "iteration_time_ms": 125.0,
    "memory_usage_mb": 4096
  }
}
```

### Field Specifications

#### Required Fields

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `checkpoint_file` | string | Filename of corresponding PLY checkpoint | `"checkpoint_1000.ply"` |
| `timestamp` | string | ISO 8601 timestamp when checkpoint was saved | `"2026-01-21T02:30:45.123Z"` |
| `iteration` | integer | Training iteration number | `1000` |
| `num_gaussians` | integer | Number of Gaussians in this checkpoint | `100000` |
| `sh_degree` | integer | Spherical harmonics degree (0-3) | `3` |
| `training_psnr` | float | Peak Signal-to-Noise Ratio in dB | `28.5` |
| `training_loss` | float | Training loss value | `0.0042` |

#### Scene Bounds

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `scene_bounds.min` | [float, float, float] | Minimum (x, y, z) of bounding box | `[-5.2, -3.1, -2.0]` |
| `scene_bounds.max` | [float, float, float] | Maximum (x, y, z) of bounding box | `[5.2, 3.1, 2.0]` |

#### Training Configuration

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `training_config.image_width` | integer | Original training image width | `1920` |
| `training_config.image_height` | integer | Original training image height | `1080` |
| `training_config.downsample_factor` | float | Downsample factor applied | `0.25` (= 25% of original) |
| `training_config.num_training_images` | integer | Number of images in training set | `100` |

**Actual Training Resolution**:
```
actual_width = image_width * downsample_factor
actual_height = image_height * downsample_factor
```

#### Dataset Information

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `dataset_info.dataset_path` | string | Path to dataset directory | `"/path/to/dataset"` |
| `dataset_info.dataset_type` | string | Type of dataset | `"colmap"`, `"nerfstudio"`, etc. |

#### Optimization State (Optional)

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `optimization_state.learning_rate` | float | Global learning rate | `0.00016` |
| `optimization_state.position_lr` | float | Position learning rate | `0.00016` |
| `optimization_state.scale_lr` | float | Scale learning rate | `0.005` |
| `optimization_state.rotation_lr` | float | Rotation learning rate | `0.001` |
| `optimization_state.opacity_lr` | float | Opacity learning rate | `0.05` |
| `optimization_state.sh_lr` | float | SH coefficient learning rate | `0.0025` |

#### Performance Metrics (Optional)

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `performance.training_time_seconds` | float | Total training time up to this checkpoint | `125.5` |
| `performance.iteration_time_ms` | float | Average time per iteration | `125.0` |
| `performance.memory_usage_mb` | integer | GPU/RAM memory usage | `4096` |

## Example JSONL File

```jsonl
{"checkpoint_file":"checkpoint_0000.ply","timestamp":"2026-01-21T02:00:00.000Z","iteration":0,"num_gaussians":10000,"sh_degree":0,"training_psnr":15.2,"training_loss":0.15,"scene_bounds":{"min":[-5.2,-3.1,-2.0],"max":[5.2,3.1,2.0]},"training_config":{"image_width":1920,"image_height":1080,"downsample_factor":0.25,"num_training_images":100},"dataset_info":{"dataset_path":"/data/scene1","dataset_type":"colmap"},"performance":{"training_time_seconds":0.0,"iteration_time_ms":0.0,"memory_usage_mb":2048}}
{"checkpoint_file":"checkpoint_0100.ply","timestamp":"2026-01-21T02:05:12.500Z","iteration":100,"num_gaussians":15000,"sh_degree":1,"training_psnr":22.3,"training_loss":0.08,"scene_bounds":{"min":[-5.2,-3.1,-2.0],"max":[5.2,3.1,2.0]},"training_config":{"image_width":1920,"image_height":1080,"downsample_factor":0.25,"num_training_images":100},"dataset_info":{"dataset_path":"/data/scene1","dataset_type":"colmap"},"optimization_state":{"learning_rate":0.00016,"position_lr":0.00016,"scale_lr":0.005,"rotation_lr":0.001,"opacity_lr":0.05,"sh_lr":0.0025},"performance":{"training_time_seconds":12.5,"iteration_time_ms":125.0,"memory_usage_mb":3072}}
{"checkpoint_file":"checkpoint_0200.ply","timestamp":"2026-01-21T02:10:25.000Z","iteration":200,"num_gaussians":20000,"sh_degree":2,"training_psnr":25.1,"training_loss":0.05,"scene_bounds":{"min":[-5.2,-3.1,-2.0],"max":[5.2,3.1,2.0]},"training_config":{"image_width":1920,"image_height":1080,"downsample_factor":0.25,"num_training_images":100},"dataset_info":{"dataset_path":"/data/scene1","dataset_type":"colmap"},"optimization_state":{"learning_rate":0.00016,"position_lr":0.00016,"scale_lr":0.005,"rotation_lr":0.001,"opacity_lr":0.05,"sh_lr":0.0025},"performance":{"training_time_seconds":25.0,"iteration_time_ms":125.0,"memory_usage_mb":3584}}
{"checkpoint_file":"checkpoint_1000.ply","timestamp":"2026-01-21T02:30:45.123Z","iteration":1000,"num_gaussians":100000,"sh_degree":3,"training_psnr":28.5,"training_loss":0.0042,"scene_bounds":{"min":[-5.2,-3.1,-2.0],"max":[5.2,3.1,2.0]},"training_config":{"image_width":1920,"image_height":1080,"downsample_factor":0.25,"num_training_images":100},"dataset_info":{"dataset_path":"/data/scene1","dataset_type":"colmap"},"optimization_state":{"learning_rate":0.00016,"position_lr":0.00016,"scale_lr":0.005,"rotation_lr":0.001,"opacity_lr":0.05,"sh_lr":0.0025},"performance":{"training_time_seconds":125.5,"iteration_time_ms":125.0,"memory_usage_mb":4096}}
```

## Implementation Notes

### Writing Metadata

```rust
use std::fs::OpenOptions;
use std::io::Write;
use serde::Serialize;

#[derive(Serialize)]
struct CheckpointMetadata {
    checkpoint_file: String,
    timestamp: String,
    iteration: u64,
    num_gaussians: usize,
    sh_degree: u32,
    training_psnr: f32,
    training_loss: f32,
    scene_bounds: SceneBounds,
    training_config: TrainingConfig,
    dataset_info: DatasetInfo,
    #[serde(skip_serializing_if = "Option::is_none")]
    optimization_state: Option<OptimizationState>,
    #[serde(skip_serializing_if = "Option::is_none")]
    performance: Option<Performance>,
}

fn append_checkpoint_metadata(
    jsonl_path: &Path,
    metadata: &CheckpointMetadata,
) -> Result<(), std::io::Error> {
    // Open file in append mode (create if doesn't exist)
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(jsonl_path)?;

    // Serialize to JSON (single line, no pretty-printing)
    let json = serde_json::to_string(metadata)?;

    // Write line + newline
    writeln!(file, "{}", json)?;

    Ok(())
}
```

### Reading Metadata

```rust
use std::fs::File;
use std::io::{BufRead, BufReader};
use serde::Deserialize;

#[derive(Deserialize)]
struct CheckpointMetadata {
    // ... same fields as Serialize version
}

fn read_training_history(
    jsonl_path: &Path,
) -> Result<Vec<CheckpointMetadata>, std::io::Error> {
    let file = File::open(jsonl_path)?;
    let reader = BufReader::new(file);

    let mut history = Vec::new();

    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue; // Skip empty lines
        }

        let metadata: CheckpointMetadata = serde_json::from_str(&line)?;
        history.push(metadata);
    }

    Ok(history)
}

fn get_latest_checkpoint(
    jsonl_path: &Path,
) -> Result<Option<CheckpointMetadata>, std::io::Error> {
    let history = read_training_history(jsonl_path)?;
    Ok(history.into_iter().last())
}

fn find_best_checkpoint(
    jsonl_path: &Path,
) -> Result<Option<CheckpointMetadata>, std::io::Error> {
    let history = read_training_history(jsonl_path)?;
    Ok(history.into_iter().max_by(|a, b| {
        a.training_psnr.partial_cmp(&b.training_psnr).unwrap()
    }))
}

fn filter_by_iteration_range(
    jsonl_path: &Path,
    min_iter: u64,
    max_iter: u64,
) -> Result<Vec<CheckpointMetadata>, std::io::Error> {
    let history = read_training_history(jsonl_path)?;
    Ok(history
        .into_iter()
        .filter(|m| m.iteration >= min_iter && m.iteration <= max_iter)
        .collect())
}
```

### Plotting Training Curves

```python
import json
import matplotlib.pyplot as plt

def plot_training_curves(jsonl_path):
    iterations = []
    psnr_values = []
    loss_values = []
    num_gaussians = []

    with open(jsonl_path, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            iterations.append(data['iteration'])
            psnr_values.append(data['training_psnr'])
            loss_values.append(data['training_loss'])
            num_gaussians.append(data['num_gaussians'])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(iterations, psnr_values)
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('PSNR (dB)')
    axes[0].set_title('Training Quality')
    axes[0].grid(True)

    axes[1].plot(iterations, loss_values)
    axes[1].set_xlabel('Iteration')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Training Loss')
    axes[1].grid(True)
    axes[1].set_yscale('log')

    axes[2].plot(iterations, num_gaussians)
    axes[2].set_xlabel('Iteration')
    axes[2].set_ylabel('Number of Gaussians')
    axes[2].set_title('Model Complexity')
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=150)
    plt.show()
```

## Comparison with .gs Format

### Advantages over .gs

| Aspect | .gs Format | JSONL + PLY |
|--------|-----------|-------------|
| **Gaussian data** | Proprietary binary | Standard PLY (industry compatible) |
| **Metadata** | Binary header | Human-readable JSON |
| **Training history** | Single snapshot | Full append-only log |
| **Checkpoint overhead** | Rewrite entire file | Append one line (~500 bytes) |
| **Tool compatibility** | SplatRs only | Any PLY viewer |
| **Debugging** | Hex editor required | Text editor for metadata |
| **Analysis** | Custom parser | Standard JSON tools |
| **File size** | Smaller (with LZ4) | Larger (no compression) |

### Migration from .gs

To convert existing .gs files to PLY + JSONL:

```rust
fn migrate_gs_to_ply(gs_path: &Path, output_dir: &Path) -> Result<(), Box<dyn std::error::Error>> {
    // Load .gs file
    let (cloud, gs_metadata) = load_model(gs_path)?;

    // Generate output paths
    let ply_path = output_dir.join(format!(
        "checkpoint_{:04}.ply",
        gs_metadata.training_iterations
    ));
    let jsonl_path = output_dir.join("training_history.jsonl");

    // Save PLY
    save_ply(&cloud, &ply_path, gs_metadata.sh_degree)?;

    // Create JSONL metadata
    let metadata = CheckpointMetadata {
        checkpoint_file: ply_path.file_name().unwrap().to_string_lossy().to_string(),
        timestamp: chrono::Utc::now().to_rfc3339(),
        iteration: gs_metadata.training_iterations,
        num_gaussians: cloud.len(),
        sh_degree: gs_metadata.sh_degree,
        training_psnr: gs_metadata.training_psnr,
        training_loss: 0.0, // Not stored in .gs
        scene_bounds: SceneBounds {
            min: gs_metadata.bounds_min.into(),
            max: gs_metadata.bounds_max.into(),
        },
        training_config: TrainingConfig {
            image_width: gs_metadata.training_width as usize,
            image_height: gs_metadata.training_height as usize,
            downsample_factor: gs_metadata.training_downsample_factor,
            num_training_images: 0, // Not stored in .gs
        },
        dataset_info: DatasetInfo {
            dataset_path: gs_metadata.dataset_path.clone(),
            dataset_type: "colmap".to_string(),
        },
        optimization_state: None,
        performance: None,
    };

    // Append to JSONL
    append_checkpoint_metadata(&jsonl_path, &metadata)?;

    println!("Migrated {} -> {}", gs_path.display(), ply_path.display());
    Ok(())
}
```

## Best Practices

1. **Checkpoint Naming**:
   - Use zero-padded iteration numbers: `checkpoint_0000.ply`, `checkpoint_0100.ply`
   - Makes files sort correctly in filesystem
   - Easy to find specific iterations

2. **Append Atomicity**:
   - Write metadata AFTER successfully saving PLY file
   - Use file locking if multiple processes might append
   - Consider temp file + rename for atomicity

3. **File Rotation**:
   - Keep all checkpoints during training
   - After training, optionally delete intermediate checkpoints
   - Always keep training_history.jsonl (it's tiny)

4. **Error Handling**:
   - If JSONL append fails, log warning but don't fail training
   - Metadata is recoverable from PLY files if needed
   - Consider adding checksum field for validation

5. **Extensibility**:
   - Unknown fields in JSON are ignored by deserializer
   - Easy to add new fields without breaking compatibility
   - Use `#[serde(skip_serializing_if = "Option::is_none")]` for optional fields

## Validation Checklist

- [ ] Each line is valid JSON (no trailing commas, proper escaping)
- [ ] checkpoint_file exists and matches iteration number
- [ ] Timestamps are monotonically increasing (or equal)
- [ ] Iterations are monotonically increasing
- [ ] File ends with newline
- [ ] No blank lines (except possibly at end)
- [ ] All required fields present in each record
- [ ] scene_bounds min < max for all axes
- [ ] downsample_factor in range (0, 1]
- [ ] sh_degree in range [0, 3]

## Version History

- **v1.0** (2026-01-21): Initial specification
  - Append-only JSONL format
  - All fields from .gs format
  - Optional optimization and performance fields
  - Migration guide from .gs
