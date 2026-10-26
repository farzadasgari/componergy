# scripts/run_review.py
"""
Review: calibration of the p-values (B3), is the effect compound-specific (B1), portfolio robustness (B2).

    python -m scripts.run_review --part {b1,b2,b3,all} [--quick] [--n-jobs N]

Adds files under figures/review/ only (--quick: figures/review/quick/). Nothing the published figures or
robustness tables are built from is touched; a SHA-256 freeze check before and after proves it.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--part", choices=["b1", "b2", "b3", "all"], default="all")
    p.add_argument("--quick", action="store_true", help="cut every draw count by 10x (smoke test)")
    p.add_argument("--n-jobs", type=int, default=3, help="joblib workers for the null simulations")
    return p.parse_args()


def _write_json(path, obj) -> None:
    tmp = str(path) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)
    os.replace(tmp, path)


def _git(*args):
    from componergy.paths import ROOT
    r = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
    return r.stdout.strip()


def main() -> int:
    args = parse_args()
    t0 = time.time()

    import numpy as np
    import pandas as pd

    from componergy import paths
    from componergy.figures.stats_log import write_figure_log
    from componergy.netcdf_io import atomic_to_csv
    from componergy.review import analyses as A
    from componergy.review import data as D
    from componergy.review import freeze, nullsim, report

    root_out = paths.REVIEW_DIR
    out = root_out / "quick" if args.quick else root_out
    for sub in ("b1", "b2", "b3"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    before_path = root_out / "freeze_before.json"
    if not before_path.exists():
        freeze.write_freeze(before_path, freeze.hash_tree(paths.ROOT))
    before = json.loads(before_path.read_text())["files"]

    cfg = A.Cfg(n_jobs=args.n_jobs)
    if args.quick:
        cfg = cfg.quick()
    print(f"config: {cfg}")

    meta_path = out / "review.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    dropped = meta.get("Bootstrap draws dropped", {})

    print("loading inputs (opens the indices file; ~2 min)...")
    ctx = D.load_all(root_out / "statewide_drivers.csv")
    parts = ["b3", "b1", "b2"] if args.part == "all" else [args.part]

    choice_path = out / "b3" / "b3_choice.json"
    if choice_path.exists():
        choice = json.loads(choice_path.read_text())
    else:
        choice = {"source": "default: b3 not run, p_norm_shift for every family",
                  "families": {f: {"chosen": "p_norm_shift"} for f in ("generation", "demand", "imbalance")}}

    b1_tab = b2_tab = None
    if "b3" in parts:
        print("B3d effective events...")
        eff, eps = A.effective_events(ctx)
        atomic_to_csv(eff, out / "b3" / "b3_effective_events.csv", index=False)
        atomic_to_csv(eps, out / "b3" / "b3_episodes.csv", index=False)

        print(f"B3c null simulations (N1 + N2, {cfg.n_draws} draws, {cfg.n_perm_sim} perms)...")
        drv = ctx["drivers"]
        z = drv[["sti", "sapei3", "scdhi3"]].values.T
        windows = {}
        for wname, names in (("generation", ["Hydro share", "Fossil share", "Hydro GWh", "Fossil GWh"]),
                             ("demand", ["Sonoran Desert", "Mojave Desert", "Imbalance"])):
            idx = ctx["windows"][wname]["index"]
            outs = {o["name"]: o for o in ctx["outcomes"]}
            windows[wname] = {"pos": drv.index.get_indexer(idx), "index_ns": nullsim.to_ns(idx),
                              "Y": np.column_stack([outs[n]["y"].values for n in names]), "names": names,
                              "trends": [outs[n]["trend"] for n in names]}
        sim = nullsim.run_nullsim(z, nullsim.to_ns(drv.index), windows, z.T, n_draws_n2=cfg.n_draws,
                                  n_perm=cfg.n_perm_sim, n_jobs=cfg.n_jobs, seed=cfg.seed,
                                  min_cond=20 if args.quick else 200, max_total=200 if args.quick else 2000)
        rej = sim["rejection"]
        # families of the series: imbalance has its own family although it shares the demand window
        rej["family"] = rej["series"].map(lambda s: "imbalance" if s == "Imbalance" else
                                          "demand" if s in ("Sonoran Desert", "Mojave Desert") else "generation")
        atomic_to_csv(rej, out / "b3" / "b3_nullsim_rejection.csv", index=False)
        atomic_to_csv(sim["diagnostics"], out / "b3" / "b3_nullsim_diagnostics.csv", index=False)
        report.nullsim_figure(rej, out / "b3" / "b3_nullsim_rejection.jpg")
        choice = A.choose_inference(rej, A.p_floors(ctx), cfg.n_perm)
        choice["n_draws"] = sim["n_draws"]
        choice["conditional_pool"] = sim["conditional"]
        choice["var_order_bic"] = sim["var_order"]
        choice["quick"] = bool(args.quick)
        _write_json(choice_path, choice)
        print("chosen p-types:", {f: v["chosen"] for f, v in choice["families"].items()})

        print("B3a alternative inference on the headline series...")
        cmp_, info = A.b3_inference_comparison(ctx, cfg, choice)
        atomic_to_csv(cmp_, out / "b3" / "b3_inference_comparison.csv", index=False)
        dropped["B3a"] = info["boot_draws_dropped"]
        meta["B3a reproduction"] = {"max_abs_diff_vs_event_effects": info["max_abs_diff_vs_event_effects"]}

    if "b1" in parts:
        print("B1a full event table...")
        b1_tab, info = A.b1_event_table(ctx, cfg, choice)
        atomic_to_csv(b1_tab, out / "b1" / "b1_event_table.csv", index=False)
        dropped["B1a"] = info["boot_draws_dropped"]
        print("B1b intensity-conditioned compound effect...")
        inten, drv_sum = A.b1_intensity(ctx, cfg, choice)
        atomic_to_csv(inten, out / "b1" / "b1_intensity_conditioned.csv", index=False)
        atomic_to_csv(drv_sum, out / "b1" / "b1_driver_summary.csv", index=False)
        print("B1c continuous interaction models...")
        coef, comp, marg = A.b1_continuous(ctx, cfg, choice)
        atomic_to_csv(coef, out / "b1" / "b1_continuous_coefficients.csv", index=False)
        atomic_to_csv(comp, out / "b1" / "b1_model_comparison.csv", index=False)
        atomic_to_csv(marg, out / "b1" / "b1_marginal_heat.csv", index=False)
        atomic_to_csv(A.b1_decision(ctx, coef, comp, inten), out / "b1" / "b1_decision.csv", index=False)
        dropped["B1c"] = int(coef["n_boot_dropped"].max())

    if "b2" in parts:
        print("B2 specifications S0-S4...")
        b2_tab, info = A.b2_specifications(ctx, cfg, choice)
        atomic_to_csv(b2_tab, out / "b2" / "b2_specifications.csv", index=False)
        dropped["B2"] = info["boot_draws_dropped"]
        print("B2 leave-one-episode-out and jackknife...")
        _, loeo, jack = A.b2_leave_one_episode_out(ctx, cfg)
        atomic_to_csv(loeo, out / "b2" / "b2_leave_one_episode_out.csv", index=False)
        atomic_to_csv(jack, out / "b2" / "b2_jackknife_year.csv", index=False)
        print("B2 substitution accounting...")
        subst, decomp = A.b2_substitution(ctx, cfg, b2_tab)
        atomic_to_csv(subst, out / "b2" / "b2_substitution.csv", index=False)
        atomic_to_csv(decomp, out / "b2" / "b2_share_decomposition.csv", index=False)
        atomic_to_csv(A.b2_decision(b2_tab, loeo), out / "b2" / "b2_decision.csv", index=False)
        dropped["B2 substitution"] = int(subst["n_boot_dropped"].max())

    if args.part in ("all", "b3"):
        print("add-ons (A5, provenance)...")
        _write_json(out / "addons.json", A.addons(ctx, paths.ROOT))

    if b1_tab is not None or b2_tab is not None:
        rep = A.reproduction_checks(ctx, paths.FIGURES_DIR, b1_tab, b2_tab)
        old = pd.DataFrame(meta.get("Reproduction", []))
        merged = pd.concat([old[~old.get("table", pd.Series(dtype=str)).isin(rep["table"])] if len(old) else old,
                            rep], ignore_index=True)
        meta["Reproduction"] = json.loads(merged.to_json(orient="records"))

    print("freeze check...")
    after = freeze.hash_tree(paths.ROOT)
    freeze.write_freeze(root_out / "freeze_after.json", after)
    d = freeze.diff_hashes(before, after)
    meta["Freeze check"] = {"files hashed": len(after), "changed files": len(d["changed"]),
                            "added files": len(d["added"]), "removed files": len(d["removed"]),
                            "changed list": d["changed"] or "none", "added list": d["added"] or "none",
                            "removed list": d["removed"] or "none",
                            "digest before": freeze.digest(before)[:16], "digest after": freeze.digest(after)[:16]}
    meta["Bootstrap draws dropped"] = dropped
    import joblib, scipy, xarray
    meta["Run"] = {
        "quick": bool(args.quick), "parts": parts, "config": str(cfg),
        "runtime_seconds": round(time.time() - t0, 1),
        "python": platform.python_version(),
        "packages": f"numpy {np.__version__}, pandas {pd.__version__}, scipy {scipy.__version__}, "
                    f"xarray {xarray.__version__}, joblib {joblib.__version__}",
        "git commit": _git("rev-parse", "HEAD"),
        "git dirty (uncommitted changes in tree)": bool(_git("status", "--porcelain")),
        "rank-deficient bootstrap draws dropped": dropped,
        "command": " ".join(sys.argv),
    }
    write_figure_log(out / "review", meta)
    report.build_report(out, meta)
    print(f"wrote {out}; changed published files: {len(d['changed'])}")
    return 0 if not d["changed"] else 1


if __name__ == "__main__":
    sys.exit(main())
