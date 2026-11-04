"""Background-shortcut audit runner, a thin layer over `abvid.audit`.

    python experiments/background_shortcut/run.py table   # data and folds only; fits nothing
"""
import argparse
from pathlib import Path

from abvid.audit import background_shortcut as bs
from abvid.audit import run_record
from abvid.paths import resolve

CONFIG = Path(__file__).with_name("config.json")


def table(config, allow_dirty):
    ids, _ = bs.load_features(resolve(config["features"]["path"]), config["features"]["sha256"])
    rows = bs.clip_table(ids)
    folds = [fold for seed in config["seeds"]
             for fold in bs.leave_one_site_out(rows, budget=config["budget_per_class"], seed=seed)]
    run = run_record.start("background_shortcut", config_path=CONFIG,
                           features=config["features"], allow_dirty=allow_dirty)
    run_record.write_once(run, "site_summary.json", bs.site_summary(rows))
    run_record.write_once(run, "folds.json", [
        {key: [ids[i] for i in value] if isinstance(value, list) else value
         for key, value in fold.items()} for fold in folds])
    print(run)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("stage", choices=["table"])
    parser.add_argument("--allow-dirty", action="store_true",
                        help="run from uncommitted code; the dirty paths are recorded")
    args = parser.parse_args()
    table(bs.load_config(CONFIG), args.allow_dirty)


if __name__ == "__main__":
    main()
