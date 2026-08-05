// Renders docs/index.html's inline script against empty / current / fully-populated data.
// The page has to degrade gracefully while a run is still in flight. Run: node test_page.js
const fs = require("fs");

const html = fs.readFileSync("docs/index.html", "utf8");
const js = html.match(/<script>([\s\S]*?)<\/script>/)[1];
const els = {};
global.document = {
  getElementById: (id) => (els[id] ||= { set innerHTML(v) { this._h = v; }, get innerHTML() { return this._h; } }),
};

const cases = {
  empty: { config: null, metrics: [], samples: [], sanity: null, evals: [] },
  full: {
    config: { n_params: 9283587, device: "mps", batch_size: 256, ema_decay: 0.9999, steps: 40000,
              base_channels: 64, weight_decay: 1e-4, lr: 2e-4, warmup: 500, steps_per_epoch: 195 },
    metrics: [{ step: 50, loss: 1.19, lr: 2e-5, sec_per_step: 1.1 },
              { step: 40000, loss: 0.31, lr: 0, sec_per_step: 0.95 }],
    samples: [{ step: 1000, src: "samples/0001000.png" }],
    sanity: { overfit_16: { final_loss: 0.0116, recon_mae: 0.0332 },
              solver_invariance: { mae_uint8: 1.81, max_uint8: 12 },
              hardware: { steps_run: 195, sec_per_step: 0.98, epoch_minutes: 3.19, device: "mps" } },
    evals: [{ step: 40000, method: "dopri5", n: 10000, fid: 31.2, fid_ref: "cifar10-train",
              sample_seconds: 1400, solver_steps: null, grid: "samples/grid_dopri5.png" },
            { step: 40000, method: "euler", solver_steps: 20, n: 10000, sample_seconds: 280 }],
  },
};
if (fs.existsSync("docs/results.js")) {
  cases.current = JSON.parse(fs.readFileSync("docs/results.js", "utf8").replace(/^const DATA = /, "").replace(/;\s*$/, ""));
}

for (const [name, DATA] of Object.entries(cases)) {
  Object.keys(els).forEach((k) => delete els[k]);
  try {
    new Function("DATA", js)(DATA);
    console.log(`  ${name}: ok`);
  } catch (e) {
    console.log(`  ${name}: FAIL ${e.message}`);
    process.exitCode = 1;
  }
}
