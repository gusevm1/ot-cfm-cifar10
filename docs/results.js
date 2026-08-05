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
  },
  {
   "step": 100,
   "epoch": 0.5128205128205128,
   "loss": 1.0877812838554382,
   "lr": 4.0400000000000006e-05,
   "sec_per_step": 0.9222443389892578,
   "wall_h": null
  },
  {
   "step": 150,
   "epoch": 0.7692307692307693,
   "loss": 0.8959847259521484,
   "lr": 6.04e-05,
   "sec_per_step": 0.9968048810958863,
   "wall_h": null
  },
  {
   "step": 200,
   "epoch": 1.0256410256410255,
   "loss": 0.651819988489151,
   "lr": 8.04e-05,
   "sec_per_step": 1.1201866579055786,
   "wall_h": null
  },
  {
   "step": 250,
   "epoch": 1.2820512820512822,
   "loss": 0.43125206530094146,
   "lr": 0.0001004,
   "sec_per_step": 1.2048922204971313,
   "wall_h": null
  },
  {
   "step": 300,
   "epoch": 1.5384615384615385,
   "loss": 0.29028089314699174,
   "lr": 0.0001204,
   "sec_per_step": 1.2526025199890136,
   "wall_h": null
  },
  {
   "step": 350,
   "epoch": 1.794871794871795,
   "loss": 0.23643940240144729,
   "lr": 0.0001404,
   "sec_per_step": 1.2708909797668457,
   "wall_h": null
  },
  {
   "step": 400,
   "epoch": 2.051282051282051,
   "loss": 0.21800722032785416,
   "lr": 0.00016040000000000002,
   "sec_per_step": 1.268093342781067,
   "wall_h": null
  },
  {
   "step": 450,
   "epoch": 2.3076923076923075,
   "loss": 0.20746463000774384,
   "lr": 0.00018040000000000002,
   "sec_per_step": 1.2457327032089234,
   "wall_h": null
  },
  {
   "step": 500,
   "epoch": 2.5641025641025643,
   "loss": 0.20465752124786377,
   "lr": 0.0002,
   "sec_per_step": 1.2453542852401733,
   "wall_h": null
  },
  {
   "step": 550,
   "epoch": 2.8205128205128207,
   "loss": 0.19709290117025374,
   "lr": 0.0001999992092940719,
   "sec_per_step": 1.2689015579223633,
   "wall_h": null
  },
  {
   "step": 600,
   "epoch": 3.076923076923077,
   "loss": 0.19369253277778625,
   "lr": 0.00019999683718879195,
   "sec_per_step": 1.283491759300232,
   "wall_h": null
  },
  {
   "step": 650,
   "epoch": 3.3333333333333335,
   "loss": 0.19080267131328582,
   "lr": 0.00019999288372167287,
   "sec_per_step": 1.2849283409118653,
   "wall_h": null
  },
  {
   "step": 700,
   "epoch": 3.58974358974359,
   "loss": 0.18578296899795532,
   "lr": 0.00019998734895523525,
   "sec_per_step": 1.2826671600341797,
   "wall_h": null
  },
  {
   "step": 750,
   "epoch": 3.8461538461538463,
   "loss": 0.18654108107089995,
   "lr": 0.00019998023297700658,
   "sec_per_step": 1.2225511980056762,
   "wall_h": null
  },
  {
   "step": 800,
   "epoch": 4.102564102564102,
   "loss": 0.18278607338666916,
   "lr": 0.00019997153589951973,
   "sec_per_step": 1.2194983577728271,
   "wall_h": null
  },
  {
   "step": 850,
   "epoch": 4.358974358974359,
   "loss": 0.18278518468141555,
   "lr": 0.00019996125786031138,
   "sec_per_step": 1.2161363410949706,
   "wall_h": null
  },
  {
   "step": 900,
   "epoch": 4.615384615384615,
   "loss": 0.18217229574918747,
   "lr": 0.00019994939902191964,
   "sec_per_step": 1.2054786014556884,
   "wall_h": null
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
 "evals": [],
 "fid": [],
 "solver": null,
 "compare": [],
 "real_reference": null
};
