"""
Reproduce the raw-data (Section 6) statistics of the paper

  "Does Diversity Actually Improve Long-term Retention? An Empirical
   Investigation with Failure Analysis of RL-based Diversity Control"

directly from the shipped derived table `diversity_retention_results.csv`.
No raw KuaiRand download is needed for this script; the table already contains
one row per user-date session (session length, diversity metrics, return gap).

Usage:
    python reproduce.py [path/to/diversity_retention_results.csv]
"""

import sys
import warnings
import numpy as np
import pandas as pd
from scipy import stats

# A user whose sessions have constant diversity or constant return_gap yields an
# undefined (NaN) within-user correlation; such users are dropped below, so the
# accompanying warning is expected and silenced for clean output.
warnings.filterwarnings("ignore", category=stats.ConstantInputWarning)

CSV = sys.argv[1] if len(sys.argv) > 1 else "diversity_retention_results.csv"
DIV_METRICS = ["tag_unique_ratio", "video_type_entropy", "music_type_entropy"]


def sig(p):
    return "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))


def partial_corr(x, y, z):
    """Pearson correlation between x and y, controlling for z (residual method)."""
    def residual(a, b):
        slope = np.cov(a, b)[0, 1] / np.var(b)
        return a - slope * b
    return stats.pearsonr(residual(x, z), residual(y, z))


def main():
    df = pd.read_csv(CSV)
    print(f"Loaded {len(df):,} sessions from {CSV}\n")

    # ------------------------------------------------------------------ 6.2
    print("=" * 64)
    print("Section 6.2  Aggregate correlations (diversity vs. return_gap)")
    print("=" * 64)
    gap = df["return_gap"].values.astype(float)
    for m in DIV_METRICS:
        col = df[m].values.astype(float)
        pr, pp = stats.pearsonr(col, gap)
        sr, sp = stats.spearmanr(col, gap)
        print(f"  {m:<20}  Pearson r={pr:+.4f} {sig(pp):>3}   "
              f"Spearman r={sr:+.4f} {sig(sp):>3}")

    # partial correlation controlling for session length + confounder structure
    tag = df["tag_unique_ratio"].values.astype(float)
    slen = df["session_len"].values.astype(float)
    pr, pp = partial_corr(tag, gap, slen)
    print("\n  Confound check (tag_unique_ratio, controlling for session_len):")
    print(f"    corr(session_len, tag_unique_ratio) = {stats.pearsonr(slen, tag)[0]:+.4f}")
    print(f"    corr(session_len, return_gap)       = {stats.pearsonr(slen, gap)[0]:+.4f}")
    print(f"    partial r(tag, gap | session_len)   = {pr:+.4f} {sig(pp)}")

    # ------------------------------------------------------------------ 6.3
    print("\n" + "=" * 64)
    print("Section 6.3  Quartile analysis (tag_unique_ratio)")
    print("=" * 64)
    q = df.copy()
    q["quartile"] = pd.qcut(q["tag_unique_ratio"], q=4, duplicates="drop")
    print(f"  {'bin':<28} {'n':>8}  {'mean_gap':>8}  {'return<=1d':>10}")
    print("  " + "-" * 58)
    for b, grp in q.groupby("quartile", observed=True):
        n = len(grp)
        mean_g = grp["return_gap"].mean()
        ret = (grp["return_gap"] <= 1).mean() * 100
        print(f"  {str(b):<28} {n:>8,}  {mean_g:>8.3f}  {ret:>9.1f}%")

    # ------------------------------------------------------------------ 6.4
    print("\n" + "=" * 64)
    print("Section 6.4  Within-user analysis (min 5 sessions/user)")
    print("=" * 64)
    corrs = []
    for _, grp in df.groupby("user_id"):
        if len(grp) < 5:
            continue
        r, _ = stats.spearmanr(grp["tag_unique_ratio"], grp["return_gap"])
        if not np.isnan(r):
            corrs.append(r)
    corrs = np.array(corrs)
    t, p = stats.ttest_1samp(corrs, 0)
    print(f"  users analyzed              : {len(corrs):,}")
    print(f"  mean within-user r          : {corrs.mean():+.4f} {sig(p)}")
    print(f"  users with r > 0 (slower)   : {(corrs > 0).mean() * 100:.1f}%")
    print(f"  users with r < 0 (faster)   : {(corrs < 0).mean() * 100:.1f}%")
    print(f"  users with r = 0            : {(corrs == 0).mean() * 100:.1f}%")
    print(f"  one-sample t-test (mu=0)    : t={t:.3f}  p={p:.3e}  {sig(p)}")


if __name__ == "__main__":
    main()
