"""Collect run artifacts into docs/ for the results page. Re-run any time; it is idempotent."""

import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

DOCS = Path("docs")


def main(run="runs/base", sanity="runs/sanity"):
    run, sanity = Path(run), Path(sanity)
    (DOCS / "samples").mkdir(parents=True, exist_ok=True)

    data = {"config": None, "metrics": [], "samples": [], "sanity": None, "evals": [],
            "fid": [], "solver": None, "compare": [], "real_reference": None}

    if (run / "config.json").exists():
        data["config"] = json.loads((run / "config.json").read_text())

    for key, path in (("metrics", run / "metrics.jsonl"), ("fid", run / "fid.jsonl"),
                      ("compare", run / "compare" / "compare.jsonl")):
        if path.exists():
            data[key] = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]

    # Every grid: the page scrubs through them. Each is a ~90 KB 256x256 PNG.
    # `delta` is the mean |Δ| in uint8 against the previous grid — same fixed noise batch,
    # so it measures how much the samples are still moving, i.e. convergence speed.
    prev = None
    for g in sorted((run / "samples").glob("*.png")):
        shutil.copy(g, DOCS / "samples" / g.name)
        cur = np.asarray(Image.open(g).convert("RGB"), dtype=np.float32)
        delta = None if prev is None else float(np.abs(cur - prev).mean())
        prev = cur
        data["samples"].append({"step": int(g.stem), "src": f"samples/{g.name}", "delta": delta})

    for name in ("overfit16.png", "solver_invariance.png"):
        if (sanity / name).exists():
            shutil.copy(sanity / name, DOCS / "samples" / name)
    if (sanity / "sanity.json").exists():
        data["sanity"] = json.loads((sanity / "sanity.json").read_text())

    for r in sorted((run / "eval").glob("result_*.json")):
        e = json.loads(r.read_text())
        g = run / "eval" / f"grid_{e['method']}.png"
        if g.exists():
            shutil.copy(g, DOCS / "samples" / g.name)
            e["grid"] = f"samples/{g.name}"
        data["evals"].append(e)

    if data["compare"]:
        (DOCS / "compare").mkdir(exist_ok=True)
        for c in data["compare"]:
            shutil.copy(run / "compare" / Path(c["src"]).name, DOCS / "compare" / Path(c["src"]).name)
    ref = run / "compare" / "real_reference.png"
    if ref.exists():
        shutil.copy(ref, DOCS / "compare" / ref.name)
        data["real_reference"] = "compare/real_reference.png"

    if (run / "solver" / "solver.json").exists():
        data["solver"] = json.loads((run / "solver" / "solver.json").read_text())
        (DOCS / "solver").mkdir(exist_ok=True)
        for e in data["solver"]["entries"]:
            shutil.copy(run / "solver" / Path(e["src"]).name, DOCS / "solver" / Path(e["src"]).name)

    (DOCS / "results.js").write_text("const DATA = " + json.dumps(data, indent=1) + ";\n")
    print(f"docs/results.js  <-  {len(data['metrics'])} log points, {len(data['samples'])} grids, "
          f"{len(data['fid'])} FID points, {len(data['compare'])} NN comparisons, "
          f"{len(data['evals'])} evals, "
          f"{len(data['solver']['entries']) if data['solver'] else 0} solver budgets")


if __name__ == "__main__":
    import sys
    main(*sys.argv[1:])
