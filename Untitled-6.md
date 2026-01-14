cargo run --bin sugar-train -- --multiview --max-images 5 --max-gaussians 2000 --downsample 0.125 --iters 50 --val-interval 10 --max-test-views 1 --log-interval 1 --loss l2  --scene datasets/tandt_db/tandt/train/sparse/0

--densify-interval 25 --densify-max-gaussians 8000 --densify-grad-threshold 0.1 --prune-opacity-threshold 0.01 --split-sigma-threshold 0.05

Then step up quality gradually (more iters, then --loss l1-dssim, then --learn-opacity, then --learn-position).

 cargo run --release --features gpu --bin sugar-train -- \
    --preset m7 \
    --dataset-root datasets/tandt_db/tandt/train \
    --gpu \
    --iters 1000


  cargo run --bin sugar-train -- \
    --preset m8 \
    --dataset-root datasets/tandt_db/tandt/train \
    --out-dir output/

  cargo run --bin sugar-render -- \
    --model output/model.gs \
    --camera-id 5 \
    --dataset-root datasets/tandt_db/tandt/train \
    --out render.png


for i in {0..10}
do
   # Pad the number to 4 digits for the camera file (e.g., 0002)
   CAM_ID=$(printf "%04d" $i)
   
   ./target/debug/sugar-render \
     --model "test_output/model.gs" \
     --camera-json "test_output/train_2000_20251216/colmap_train_sequence/cameras/cam_${CAM_ID}.json" \
     --dataset-root "datasets/tandt_db/tandt/train" \
     --out "test_output/cam${i}_tandt_train.png"
done

for i in {0..10}
do
   # Pad the number to 4 digits for the camera file (e.g., 0002)
   CAM_ID=$(printf "%04d" $i)
   
   ./target/debug/sugar-render \
     --model "runs/20260101_1640_micro/model.gs" \
     --camera-id "${i}" \
     --dataset-root "datasets/tandt_db/tandt/train" \
     --out "test_output/cam${i}_tandt_train.png"
done



SUGAR_GPU_TIMING=1  time cargo run --release --bin sugar-train --features gpu -- \
       --preset micro --dataset-root datasets/tandt_db/tandt/train --gpu