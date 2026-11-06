# Continuing the work

Open items, roughly in the order they would pay off, each with a place to start.

## Deployment

1. Time the encoder on the Orin. Only a theoretical budget exists so far
   ([deployment.md](deployment.md#compute-budget-on-the-jetson)). Run
   `benchmarks/pointnet2_jetson_bench.py` on the device, ideally with the driver and SLAM running,
   and record the JetPack and PyTorch versions with the numbers.
2. Run the live node on the robot. `ros2_inference/live_inference_node.py` has only been exercised
   through its shared per-scan code on recorded bags.
3. Publish the observation. Nothing publishes the 256-d latent for a policy yet. A small node that
   wraps `deploy.ScanEncoder` in a `PointCloud2` subscriber and publishes one message per scan
   (a `std_msgs/Float32MultiArray`, for example) would give the downstream policy its interface.

## Real data

4. Fix the input order for SalsaNext and the CNN. Their paths in
   `ros2_inference/_shared_inference.py` still receive scans in Ouster order, since the fix only
   covered PointNet++. The order conversion in `sensor.py` and the before/after method in
   `real_eval/before_after.py` show how it was done there.
5. Redo the ring-noise fine-tune. `outputs/pointnet2_ringnoise_finetuned` was trained while the
   measured dropout was still applied upside down. Retrain with `train.py --init-from ...
   --augment-noise` ([training-and-evaluation.md](training-and-evaluation.md#fine-tuning-with-measured-real-sensor-noise)).
6. Get a quantitative real-world score. Real data has no labels, but the objects in the real
   recordings were placed at measured positions. Combined with the robot's odometry, those
   positions would give per-object ground truth for at least part of each bag.
7. Regenerate the demo's bag3 player and video, which still show the run from before the
   input-order fix. `python demo/build_demo.py --only bag_player:bag3_pointnet2` rebuilds the
   player. For the video, point the `media` entry in `demo/sources.json` at a new
   `inference.mp4` from `run_inference_on_bag.py` and run `--only media:bag3_pointnet2`.

## Training and data

8. Train on the missing classes. Human, car and bus have no examples, so the class heads have
   never learned them, and the only way to get examples is new Gazebo scenarios that contain
   those models.
9. Hold validation out at the scenario level. Validation and test currently share 19 scenarios
   ([data.md](data.md#the-split)); a retrain could give validation scenarios of its own.
10. Fix a random seed and save the training history. No seed is set, and per-epoch losses are
    only printed. `train_utils.fit()` already passes every epoch to `log_fn`, so writing a CSV
    there takes a few lines.
11. Add tunnels and domain randomization. Every scenario is the same tunnel, so the benchmark says
    nothing about other environments. More reconstructed tunnels, and systematic randomization of
    sensor noise and materials, would help narrow the sim-to-real gap.

## Infrastructure

12. Build and test the Docker images. `docker-compose.yml` and `docker/pipeline.Dockerfile` are
    written but have not been built yet. The pipeline image's import check still imports `cv2`,
    which no code uses any more.
13. Replace the stand-in tunnel scan (`noslil.ply`) in the demo once a cleaner capture exists.

## Conventions to keep

- Simulated and real results are reported separately, and nothing on real data is called an
  accuracy.
- Every number in the demo comes from a script in this repo that writes JSON
  (`kevin_pipeline/fair_comparison.py`, `evaluate.py`, `real_eval/`). Nobody fills tables by hand.
- New behaviour gets a test in `tests/` that runs on synthetic data, so the suite keeps passing on
  a clone without data or GPU.
