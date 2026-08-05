// Renders docs/index.html's inline script against empty / current / fully-populated data, then
// exercises the interactive controls. The page has to degrade gracefully while a run is in flight
// and must not throw when a section's data is missing. Run: node test_page.js
//
// The DOM stub is deliberately minimal — enough to execute every branch, not a browser. It catches
// bad field access and broken maths, not layout.
const fs = require("fs");

const html = fs.readFileSync("docs/index.html", "utf8");
const js = html.match(/<script>([\s\S]*?)<\/script>/)[1];

function makeEl(tag = "div") {
  const attrs = {};
  return {
    tagName: tag, className: "", innerHTML: "", textContent: "", children: [], value: 0,
    append(...kids) { this.children.push(...kids); },
    setAttribute(k, v) { attrs[k] = String(v); },
    getAttribute(k) { return attrs[k] ?? null; },
    querySelector() { return makeEl(); },
    addEventListener(ev, fn) { this["on" + ev] = fn; },
  };
}

function run(name, DATA) {
  const els = {};
  global.document = {
    getElementById: (id) => (els[id] ||= makeEl()),
    createElement: (tag) => makeEl(tag),
  };
  global.Image = function () { return { set src(v) {} }; };
  global.setInterval = () => 1;
  global.clearInterval = () => {};
  try {
    new Function("DATA", js)(DATA);
    // Exercise the controls that only fire on user input.
    els.tsmooth?.onchange?.({ target: { checked: false } });
    els.tlog?.onchange?.({ target: { checked: true } });
    console.log(`  ${name}: ok`);
  } catch (e) {
    console.log(`  ${name}: FAIL ${e.message}\n${e.stack.split("\n")[1] || ""}`);
    process.exitCode = 1;
  }
}

const full = {
  config: { n_params: 9283587, device: "mps", batch_size: 256, ema_decay: 0.9999, steps: 40000,
            base_channels: 64, weight_decay: 1e-4, lr: 2e-4, warmup: 500, steps_per_epoch: 195,
            log_every: 50 },
  metrics: [{ step: 50, loss: 1.19, lr: 2e-5, sec_per_step: 1.1 },
            { step: 20000, loss: 0.42, lr: 1e-4, sec_per_step: 1.0 },
            { step: 40000, loss: 0.31, lr: 0, sec_per_step: 0.95 }],
  samples: [{ step: 1000, src: "samples/0001000.png", delta: null },
            { step: 2000, src: "samples/0002000.png", delta: 18.4 },
            { step: 40000, src: "samples/0040000.png", delta: 1.2 }],
  fid: [{ step: 2000, fid: 92.1, n: 2000, method: "euler", solver_steps: 20, seconds: 240 },
        { step: 40000, fid: 34.7, n: 2000, method: "euler", solver_steps: 20, seconds: 240 }],
  compare: [{ step: 1000, n: 16, src: "compare/0001000_nn.png", nn_mean: 21.4, nn_min: 15.2, nn_max: 27.9 },
            { step: 40000, n: 16, src: "compare/0040000_nn.png", nn_mean: 14.8, nn_min: 9.1, nn_max: 19.3 }],
  real_reference: "compare/real_reference.png",
  solver: { step: 40000, n: 64, entries: [
    { method: "dopri5", steps: null, nfe: 122, seconds: 8.1, mae_vs_dopri5: 0, src: "solver/dopri5.png" },
    { method: "euler", steps: 1, nfe: 1, seconds: 0.1, mae_vs_dopri5: 41.2, src: "solver/euler1.png" },
    { method: "euler", steps: 50, nfe: 50, seconds: 3.4, mae_vs_dopri5: 0.9, src: "solver/euler50.png" }] },
  sanity: { overfit_16: { final_loss: 0.0116, recon_mae: 0.0332 },
            solver_invariance: { mae_uint8: 1.81, max_uint8: 12 },
            hardware: { steps_run: 195, sec_per_step: 0.98, epoch_minutes: 3.19, device: "mps" } },
  evals: [{ step: 40000, method: "dopri5", n: 10000, fid: 31.2, fid_ref: "cifar10-train",
            sample_seconds: 1400, solver_steps: null, grid: "samples/grid_dopri5.png" },
          { step: 40000, method: "euler", solver_steps: 20, n: 10000, sample_seconds: 280 }],
};

run("empty", { config: null, metrics: [], samples: [], sanity: null, evals: [], fid: [], solver: null,
               compare: [], real_reference: null });
// One log point, no checkpoint yet — what the page looks like in the first minutes of a run.
run("just started", { ...full, metrics: full.metrics.slice(0, 1), samples: [], fid: [], solver: null,
                      evals: [], compare: [], real_reference: null, sanity: full.sanity });
run("mid run", { ...full, solver: null, evals: [] });
// A single comparison point: the NN chart has one point and must not divide by zero.
run("one compare point", { ...full, compare: full.compare.slice(0, 1), solver: null, evals: [] });
run("full", full);
if (fs.existsSync("docs/results.js")) {
  run("current", JSON.parse(fs.readFileSync("docs/results.js", "utf8")
    .replace(/^const DATA = /, "").replace(/;\s*$/, "")));
}
