"""Collect run artifacts into docs/ for the results page. Re-run any time; it is idempotent."""

import json
import shutil
from pathlib import Path

DOCS = Path("docs")


def main(run="runs/base", sanity="runs/sanity"):
    run, sanity = Path(run), Path(sanity)
    (DOCS / "samples").mkdir(parents=True, exist_ok=True)

    data = {"config": None, "metrics": [], "samples": [], "sanity": None, "evals": []}

    if (run / "config.json").exists():
        data["config"] = json.loads((run / "config.json").read_text())

    if (run / "metrics.jsonl").exists():
        for line in (run / "metrics.jsonl").read_text().splitlines():
            if line.strip():
                data["metrics"].append(json.loads(line))

    # Keep ~8 evenly spaced grids so the page stays small.
    grids = sorted((run / "samples").glob("*.png"))
    if grids:
        keep = grids if len(grids) <= 8 else [grids[round(i * (len(grids) - 1) / 7)] for i in range(8)]
        for g in dict.fromkeys(keep):
            shutil.copy(g, DOCS / "samples" / g.name)
            data["samples"].append({"step": int(g.stem), "src": f"samples/{g.name}"})

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

    (DOCS / "results.js").write_text("const DATA = " + json.dumps(data, indent=1) + ";\n")
    print(f"docs/results.js  <-  {len(data['metrics'])} log points, "
          f"{len(data['samples'])} grids, {len(data['evals'])} evals")


if __name__ == "__main__":
    import sys
    main(*sys.argv[1:])
