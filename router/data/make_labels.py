"""Turn graded answers into cheapest-correct-tier labels and train/val/calib/test splits.

label = 0 (LITE) if lite_correct else 1 (MID) if mid_correct else 2 (FULL).
Queries no tier answers correctly get label 2 (spend the battery anyway).
Splits are stratified by label with the seed from config.
Output: labeled.csv = queries + graded columns + label, split.
"""
import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from shared.config import load_config, repo_path  # noqa: E402
from shared.tiers import Tier  # noqa: E402

log = logging.getLogger("make_labels")
SPLIT_ORDER = ["train", "val", "calib", "test"]


def cheapest_correct_tier(lite_correct: int, mid_correct: int, full_correct: int) -> int:
    if lite_correct:
        return int(Tier.LITE)
    if mid_correct:
        return int(Tier.MID)
    return int(Tier.FULL)


def _stratify_or_none(labels: pd.Series, n_test: int):
    # Stratification needs at least 2 members per class and room for every class in each part.
    counts = labels.value_counts()
    if counts.min() < 2 or n_test < len(counts) or len(labels) - n_test < len(counts):
        log.warning("too few examples to stratify a split of %d; falling back to a random split", len(labels))
        return None
    return labels


def assign_splits(df: pd.DataFrame, shares: dict[str, float], seed: int) -> pd.Series:
    """Peel off test, then calib, then val from what is left; the remainder is train."""
    split = pd.Series("train", index=df.index)
    rest = df
    for name in ["test", "calib", "val"]:
        n = round(shares[name] * len(df))
        part, rest = train_test_split(rest, test_size=len(rest) - n, train_size=n, random_state=seed,
                                      stratify=_stratify_or_none(rest.label, len(rest) - n))
        split[part.index] = name
    return split


def main():
    cfg = load_config()
    data_dir = repo_path(cfg["router"]["data_dir"])
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--queries", type=Path, default=data_dir / "queries.csv")
    parser.add_argument("--graded", type=Path, default=data_dir / "graded.csv")
    parser.add_argument("--out", type=Path, default=data_dir / "labeled.csv")
    parser.add_argument("--seed", type=int, default=cfg["seed"])
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    df = pd.read_csv(args.queries).merge(pd.read_csv(args.graded), on="query_id", how="inner")
    df["label"] = [cheapest_correct_tier(*r) for r in df[["lite_correct", "mid_correct", "full_correct"]].values]
    df["split"] = assign_splits(df, cfg["router"]["splits"], args.seed)
    df.to_csv(args.out, index=False)

    print(f"\nWrote {len(df)} labeled queries to {args.out}")
    print("Label distribution per split (0=LITE, 1=MID, 2=FULL):")
    print(pd.crosstab(df.split, df.label, margins=True).reindex(SPLIT_ORDER + ["All"]).to_string())
    print("\nLabel distribution per category:")
    print(pd.crosstab(df.category, df.label).to_string())
    print(f"\nQueries no tier answered correctly (labelled FULL): "
          f"{int((df[['lite_correct', 'mid_correct', 'full_correct']].sum(axis=1) == 0).sum())}")


if __name__ == "__main__":
    main()
