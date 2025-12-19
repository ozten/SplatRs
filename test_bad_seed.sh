#!/bin/bash
# Quick 500-iteration test for bad seed troubleshooting

CONFIG_NAME=$1
SEED=${2:-42}

if [ -z "$CONFIG_NAME" ]; then
    echo "Usage: $0 <config_name> [seed]"
    echo "Example: $0 baseline 42"
    exit 1
fi

OUT_DIR="runs/debug_${CONFIG_NAME}_seed${SEED}"

echo "=== Testing: $CONFIG_NAME (seed=$SEED) ==="

case $CONFIG_NAME in
    baseline)
        cargo run --release --features gpu --bin sugar-train -- \
            --preset micro --iters 500 --max-images 40 --seed $SEED \
            --dataset-root datasets/tandt_db/tandt/train --gpu \
            --out-dir "$OUT_DIR"
        ;;

    dense_aggressive)
        # More aggressive densification
        cargo run --release --features gpu --bin sugar-train -- \
            --preset micro --iters 500 --max-images 40 --seed $SEED \
            --densify-grad-threshold 0.0001 \
            --dataset-root datasets/tandt_db/tandt/train --gpu \
            --out-dir "$OUT_DIR"
        ;;

    more_gaussians)
        # Start with more Gaussians
        cargo run --release --features gpu --bin sugar-train -- \
            --preset micro --iters 500 --max-images 40 --seed $SEED \
            --max-gaussians 25000 \
            --dataset-root datasets/tandt_db/tandt/train --gpu \
            --out-dir "$OUT_DIR"
        ;;

    higher_lr)
        # Higher learning rates
        cargo run --release --features gpu --bin sugar-train -- \
            --preset micro --iters 500 --max-images 40 --seed $SEED \
            --lr 0.004 --lr-position 0.00032 \
            --dataset-root datasets/tandt_db/tandt/train --gpu \
            --out-dir "$OUT_DIR"
        ;;

    densify_100)
        # Densify at iter 100 instead of 500
        cargo run --release --features gpu --bin sugar-train -- \
            --preset micro --iters 500 --max-images 40 --seed $SEED \
            --densify-interval 100 \
            --dataset-root datasets/tandt_db/tandt/train --gpu \
            --out-dir "$OUT_DIR"
        ;;

    *)
        echo "Unknown config: $CONFIG_NAME"
        echo "Available: baseline, dense_aggressive, more_gaussians, higher_lr, densify_100"
        exit 1
        ;;
esac

# Report results
if [ -f "$OUT_DIR/m8_test_view_rendered_0500.png" ]; then
    SIZE=$(ls -lh "$OUT_DIR/m8_test_view_rendered_0500.png" | awk '{print $5}')
    echo "=== Result: $CONFIG_NAME ==="
    echo "File size: $SIZE"

    # Check if it's a solid color (< 10KB is bad)
    SIZE_BYTES=$(stat -f%z "$OUT_DIR/m8_test_view_rendered_0500.png" 2>/dev/null || stat -c%s "$OUT_DIR/m8_test_view_rendered_0500.png")
    if [ $SIZE_BYTES -lt 10000 ]; then
        echo "Status: ❌ BROKEN (solid color)"
    else
        echo "Status: ✅ Potentially working"
    fi
fi
