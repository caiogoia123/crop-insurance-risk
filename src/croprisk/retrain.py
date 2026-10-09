"""Monthly retrain with a champion/challenger gate (runs on the VM).

1. refresh data (PSR only if the CKAN files changed; NASA POWER tail; ONI);
2. rebuild the modeling table;
3. holdout H = most recent matured safra; challenger = trained on safras <= H-1;
4. champion = the bundle in use, scored on the same holdout;
5. promote only if the challenger is not worse (AUC and Brier within tolerance).

Every run appends one JSON line (metrics, PSI, decision) to logs/retrain.jsonl.
Exit code 10 = promoted (the host script then restarts the API and checks
/health, rolling back CURRENT if it fails); 0 = champion kept; other = failure,
in which case nothing was changed.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import traceback
from datetime import UTC, datetime

import pandas as pd

from croprisk import config, dataset, metrics
from croprisk.data import enso, geo, power, psr
from croprisk.features import build as fbuild
from croprisk.serving.bundle import Bundle, current_dir
from croprisk.train import build_tables, fit_and_evaluate, promote, write_bundle

log = logging.getLogger("croprisk.retrain")

AUC_TOL = 0.002  # challenger may be at most this much below the champion
BRIER_TOL = 0.001
KEEP_BUNDLES = 3
PROMOTED = 10
LOG_PATH = config.ROOT / "logs" / "retrain.jsonl"


def refresh_data() -> dict:
    out = {}
    out["psr_files"] = [p.name for p in psr.download_all()]
    munis = geo.municipality_cells(geo.download_municipalities())
    try:
        out["power"] = power.update_cells(munis[["cell_id", "cell_lat", "cell_lon"]], workers=2)
    except Exception as e:  # stale climate is acceptable for the pre-season model
        log.warning("POWER update failed, using cached series: %s", e)
        out["power"] = f"failed: {e}"
    enso.download(force=True)
    return out


def score_champion(champion: Bundle, hold: pd.DataFrame) -> dict:
    X = hold[champion.meta["features"]]
    p = champion.booster.predict(X)
    return metrics.binary_metrics(hold["y"], p)


def cleanup_bundles(models_dir, keep: int = KEEP_BUNDLES) -> None:
    current = (models_dir / "CURRENT").read_text().strip()
    dirs = sorted([d for d in models_dir.iterdir() if d.is_dir()], key=lambda d: d.stat().st_mtime)
    for d in dirs[:-keep]:
        if d.name != current:
            for f in sorted(d.rglob("*"), reverse=True):
                f.unlink() if f.is_file() else f.rmdir()
            d.rmdir()


def run() -> int:
    record: dict = {"started_at": datetime.now(UTC).isoformat(timespec="seconds")}
    record["data"] = refresh_data()
    dataset.main()
    fbuild.main()
    df = pd.read_parquet(fbuild.FEATURES_PATH)
    holdout = int(df["safra_year"].max())
    champion = Bundle.load(current_dir(config.MODELS))
    record["holdout_safra"] = holdout
    record["champion"] = {
        "version": champion.meta["version"],
        "train_safras": champion.meta["train_safras"],
    }
    if champion.meta["train_safras"][1] >= holdout:
        # cannot happen under the protocol unless data shrank; never compare on seen data
        record["decision"] = "skip: champion trained on the holdout safra"
        return finish(record, 0)

    os.environ.setdefault("CROPRISK_TUNING", json.dumps(champion.meta["tuning"]))
    res = fit_and_evaluate(df, holdout)
    champ_met = score_champion(champion, df[df["safra_year"] == holdout])
    chall_met = res["metrics"]
    record["champion"]["holdout"] = champ_met
    record["challenger"] = {
        "holdout": chall_met,
        "top_feature_psi": dict(list(res["feature_psi"].items())[:10]),
    }
    not_worse = (
        chall_met["auc"] >= champ_met["auc"] - AUC_TOL
        and chall_met["brier"] <= champ_met["brier"] + BRIER_TOL
    )
    same = res["model"].booster.model_to_string() == champion.booster.model_to_string()
    if same:
        record["decision"] = "keep: challenger identical to champion"
        return finish(record, 0)
    if not not_worse:
        record["decision"] = "keep: challenger worse on holdout"
        return finish(record, 0)
    policies = pd.read_parquet(dataset.POLICIES_PATH)
    out = write_bundle(res, build_tables(policies), holdout)
    promote(out)
    cleanup_bundles(config.MODELS)
    record["decision"] = f"promoted {out.name}"
    return finish(record, PROMOTED)


def finish(record: dict, code: int) -> int:
    record["finished_at"] = datetime.now(UTC).isoformat(timespec="seconds")
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")
    log.info("retrain decision: %s", record.get("decision"))
    return code


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    try:
        code = run()
    except Exception:
        finish({"decision": "failed", "error": traceback.format_exc()[-2000:]}, 1)
        raise
    sys.exit(code)


if __name__ == "__main__":
    main()
