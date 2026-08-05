const DATA = {
 "config": {
  "steps": 40000,
  "batch_size": 256,
  "lr": 0.0002,
  "warmup": 500,
  "weight_decay": 0.0001,
  "ema_decay": 0.9999,
  "base_channels": 64,
  "grad_clip": 1.0,
  "num_workers": 4,
  "log_every": 50,
  "sample_every": 1000,
  "ckpt_every": 1000,
  "out": "runs/base",
  "device": "mps",
  "seed": 0,
  "n_params": 9283587,
  "steps_per_epoch": 195
 },
 "metrics": [
  {
   "step": 50,
   "epoch": 0.2564102564102564,
   "loss": 1.1972598910331727,
   "lr": 2.04e-05,
   "sec_per_step": 1.123485655784607,
   "wall_h": 0.015603967441452874
  }
 ],
 "samples": [],
 "sanity": {
  "hardware": {
   "batch_size": 256,
   "steps_run": 195,
   "steps_per_epoch": 195,
   "sec_per_step": 0.9816696093632624,
   "epoch_minutes": 3.190426230430603,
   "device": "mps",
   "oom": false
  },
  "overfit_16": {
   "final_loss": 0.011569969356060028,
   "recon_mae": 0.03317539393901825,
   "steps": 500,
   "seconds": 41.081897258758545
  },
  "solver_invariance": {
   "mae_uint8": 1.8108114004135132,
   "max_uint8": 12.0
  }
 },
 "evals": []
};
