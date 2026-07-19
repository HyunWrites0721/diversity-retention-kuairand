"""
Explore the diversity-retention signal in the raw KuaiRand data.

For each user session (user x date):
  - compute the diversity of the recommended items in the session
    (tag-based categorical diversity)
  - compute the return gap (days until the user's next session)
  - analyze the Pearson/Spearman correlation between the two

Input data (KuaiRand-Pure):
  - log_standard_4_08_to_4_21_pure.csv
  - log_standard_4_22_to_5_08_pure.csv
  - log_random_4_22_to_5_08_pure.csv
  - video_features_basic_pure.csv (tag, video_type, music_type)

Note: DATA_DIR below points at a local copy of KuaiRand-Pure; edit it to match
your machine before running.

Usage:
    python diversity_retention_signal.py
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy import stats

DATA_DIR = '/home/data/KuaiSim/dataset/kuairand/KuaiRand-Pure/data'
OUT_DIR  = '/home/data/LTV/analysis'
os.makedirs(OUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Data loading
# ---------------------------------------------------------------------------

def load_logs():
    print("[1] Loading logs ...")
    logs = []
    for fname in [
        'log_standard_4_08_to_4_21_pure.csv',
        'log_standard_4_22_to_5_08_pure.csv',
        'log_random_4_22_to_5_08_pure.csv',
    ]:
        df = pd.read_csv(os.path.join(DATA_DIR, fname),
                         usecols=['user_id','video_id','date'])
        logs.append(df)
        print(f"  {fname}: {len(df):,} rows")

    log = pd.concat(logs, ignore_index=True)
    log = log.drop_duplicates()
    print(f"  total (dedup): {len(log):,} rows  |  users: {log['user_id'].nunique():,}  |  dates: {log['date'].nunique()}")
    return log


def load_video_features():
    print("[2] Loading video features ...")
    vf = pd.read_csv(
        os.path.join(DATA_DIR, 'video_features_basic_pure.csv'),
        usecols=['video_id', 'tag', 'video_type', 'music_type'],
    )
    # tag: comma-separated string -> frozenset
    def parse_tags(s):
        if pd.isna(s):
            return frozenset()
        return frozenset(str(s).split(','))

    vf['tag_set'] = vf['tag'].apply(parse_tags)
    print(f"  {len(vf):,} videos  |  tag nunique: {vf['tag'].nunique()}")
    return vf[['video_id', 'tag_set', 'video_type', 'music_type']]


# ---------------------------------------------------------------------------
# 2. Diversity computation (per session)
# ---------------------------------------------------------------------------

def session_diversity(video_ids, vid2tags, vid2type, vid2music):
    """
    Compute the diversity of a single session (list of items) three ways.

    Returns dict:
      tag_unique_ratio   : number of unique tags / total tags in the session
      video_type_entropy : entropy of the video_type distribution (normalized)
      music_type_entropy : entropy of the music_type distribution (normalized)
    """
    n = len(video_ids)
    if n == 0:
        return {'tag_unique_ratio': 0.0, 'video_type_entropy': 0.0, 'music_type_entropy': 0.0}

    # tag unique ratio: fraction of unique tags among all tags
    all_tags = []
    for vid in video_ids:
        all_tags.extend(vid2tags.get(vid, set()))
    tag_unique_ratio = len(set(all_tags)) / max(len(all_tags), 1)

    # video_type entropy
    types = [vid2type.get(vid, 'UNKNOWN') for vid in video_ids]
    type_counts = pd.Series(types).value_counts(normalize=True)
    vt_entropy = float(stats.entropy(type_counts)) if len(type_counts) > 1 else 0.0

    # music_type entropy
    musics = [vid2music.get(vid, -1) for vid in video_ids]
    music_counts = pd.Series(musics).value_counts(normalize=True)
    mt_entropy = float(stats.entropy(music_counts)) if len(music_counts) > 1 else 0.0

    return {
        'tag_unique_ratio':   tag_unique_ratio,
        'video_type_entropy': vt_entropy,
        'music_type_entropy': mt_entropy,
    }


def compute_session_diversity(log, vf):
    print("[3] Computing session diversity ...")

    vid2tags  = dict(zip(vf['video_id'], vf['tag_set']))
    vid2type  = dict(zip(vf['video_id'], vf['video_type']))
    vid2music = dict(zip(vf['video_id'], vf['music_type']))

    sessions = log.groupby(['user_id', 'date'])['video_id'].apply(list).reset_index()
    sessions.columns = ['user_id', 'date', 'video_ids']
    sessions['session_len'] = sessions['video_ids'].apply(len)

    # keep only sessions with at least 3 items
    sessions = sessions[sessions['session_len'] >= 3].copy()
    print(f"  sessions (len>=3): {len(sessions):,}  |  users: {sessions['user_id'].nunique():,}")

    div_records = sessions['video_ids'].apply(
        lambda vids: session_diversity(vids, vid2tags, vid2type, vid2music)
    )
    div_df = pd.DataFrame(div_records.tolist())
    sessions = pd.concat([sessions.reset_index(drop=True), div_df], axis=1)
    return sessions


# ---------------------------------------------------------------------------
# 3. Retention (return_gap) computation
# ---------------------------------------------------------------------------

def compute_return_gap(sessions):
    print("[4] Computing return_gap ...")

    # convert `date` to an actual datetime
    sessions = sessions.copy()
    sessions['date_dt'] = pd.to_datetime(sessions['date'].astype(str), format='%Y%m%d')

    # sort per user, then compute the next session date
    sessions = sessions.sort_values(['user_id', 'date_dt'])
    sessions['next_date'] = sessions.groupby('user_id')['date_dt'].shift(-1)
    sessions['return_gap'] = (sessions['next_date'] - sessions['date_dt']).dt.days

    # drop the last session per user (next_date = NaN)
    sessions = sessions[sessions['return_gap'].notna()].copy()
    sessions['return_gap'] = sessions['return_gap'].astype(int)

    # inspect the return_gap distribution
    print(f"  valid sessions: {len(sessions):,}")
    print(f"  return_gap distribution: mean={sessions['return_gap'].mean():.2f}  "
          f"median={sessions['return_gap'].median():.1f}  "
          f"std={sessions['return_gap'].std():.2f}")
    gap_dist = sessions['return_gap'].value_counts().sort_index().head(10)
    print(f"  gap frequency (top 10):\n{gap_dist.to_string()}")
    return sessions


# ---------------------------------------------------------------------------
# 4. Correlation analysis
# ---------------------------------------------------------------------------

def correlation_analysis(sessions):
    print("\n" + "=" * 60)
    print("Diversity-retention correlation analysis")
    print("=" * 60)

    div_metrics = ['tag_unique_ratio', 'video_type_entropy', 'music_type_entropy']
    target = 'return_gap'

    results = {}
    for metric in div_metrics:
        col = sessions[metric].values
        gap = sessions[target].values

        pearson_r,  pearson_p  = stats.pearsonr(col, gap)
        spearman_r, spearman_p = stats.spearmanr(col, gap)

        results[metric] = {
            'pearson_r':  pearson_r,
            'pearson_p':  pearson_p,
            'spearman_r': spearman_r,
            'spearman_p': spearman_p,
        }

        sig_p = '***' if pearson_p < 0.001 else ('**' if pearson_p < 0.01 else ('*' if pearson_p < 0.05 else 'ns'))
        sig_s = '***' if spearman_p < 0.001 else ('**' if spearman_p < 0.01 else ('*' if spearman_p < 0.05 else 'ns'))

        print(f"\n  [{metric}]")
        print(f"    diversity range : {col.min():.4f} ~ {col.max():.4f}  mean={col.mean():.4f}")
        print(f"    Pearson  r={pearson_r:+.4f}  p={pearson_p:.2e}  {sig_p}")
        print(f"    Spearman r={spearman_r:+.4f}  p={spearman_p:.2e}  {sig_s}")

    # session-length correction: partial correlation controlling for session_len
    print(f"\n  [partial correlation controlling for session_len - tag_unique_ratio]")
    from scipy.stats import pearsonr
    def partial_corr(x, y, z):
        """Partial correlation between x and y, controlling for z."""
        def residual(a, b):
            slope = np.cov(a, b)[0, 1] / np.var(b)
            return a - slope * b
        rx = residual(x, z)
        ry = residual(y, z)
        return pearsonr(rx, ry)

    col = sessions['tag_unique_ratio'].values.astype(float)
    gap = sessions['return_gap'].values.astype(float)
    slen = sessions['session_len'].values.astype(float)
    pr, pp = partial_corr(col, gap, slen)
    sig = '***' if pp < 0.001 else ('**' if pp < 0.01 else ('*' if pp < 0.05 else 'ns'))
    print(f"    partial Pearson r={pr:+.4f}  p={pp:.2e}  {sig}")

    return results


# ---------------------------------------------------------------------------
# 5. Quartile comparison of return_gap
# ---------------------------------------------------------------------------

def quantile_analysis(sessions):
    print("\n" + "=" * 60)
    print("Return_gap by diversity quartile (tag_unique_ratio)")
    print("=" * 60)

    sessions = sessions.copy()
    sessions['div_quartile'] = pd.qcut(
        sessions['tag_unique_ratio'], q=4,
        duplicates='drop',
    )

    print(f"\n  {'quartile':<12} {'n':>7}  {'mean_gap':>9}  {'median_gap':>11}  {'return<2d':>10}")
    print("  " + "-" * 55)
    for q, grp in sessions.groupby('div_quartile'):
        n       = len(grp)
        mean_g  = grp['return_gap'].mean()
        med_g   = grp['return_gap'].median()
        ret2    = (grp['return_gap'] <= 1).mean() * 100
        print(f"  {str(q):<12} {n:>7,}  {mean_g:>9.3f}  {med_g:>11.1f}  {ret2:>9.1f}%")

    # Q1 vs Q4 t-test
    q1 = sessions[sessions['div_quartile'] == 'Q1\n(low)']['return_gap']
    q4 = sessions[sessions['div_quartile'] == 'Q4\n(high)']['return_gap']
    t, p = stats.ttest_ind(q1, q4)
    print(f"\n  Q1 vs Q4 t-test: t={t:.3f}  p={p:.4e}")
    if p < 0.05:
        direction = "lower (more diverse -> faster return)" if q4.mean() < q1.mean() else "higher (more diverse -> slower return)"
        print(f"  -> significant difference. Q4 mean_gap is {direction} than Q1")
    else:
        print(f"  -> no significant difference (p>=0.05)")


# ---------------------------------------------------------------------------
# 6. Within-user analysis (within-user correlation)
# ---------------------------------------------------------------------------

def within_user_analysis(sessions, min_sessions=5):
    print("\n" + "=" * 60)
    print(f"Within-user correlation (per-user diversity-gap relationship, min_sessions={min_sessions})")
    print("=" * 60)
    print("  (removes between-user fixed effects to measure the pure effect of diversity)")

    user_corrs = []
    for uid, grp in sessions.groupby('user_id'):
        if len(grp) < min_sessions:
            continue
        r, p = stats.spearmanr(grp['tag_unique_ratio'], grp['return_gap'])
        if not np.isnan(r):
            user_corrs.append(r)

    user_corrs = np.array(user_corrs)
    print(f"\n  users analyzed : {len(user_corrs):,}")
    print(f"  within-user r  : mean={user_corrs.mean():+.4f}  median={np.median(user_corrs):+.4f}  std={user_corrs.std():.4f}")

    # fraction of users with negative correlation (diversity up -> gap down = faster return)
    neg_frac = (user_corrs < 0).mean()
    pos_frac = (user_corrs > 0).mean()
    print(f"  r<0 (diversity linked to faster return): {neg_frac*100:.1f}%")
    print(f"  r>0 (diversity linked to slower return): {pos_frac*100:.1f}%")

    # is the group mean significantly different from zero?
    t, p = stats.ttest_1samp(user_corrs, 0)
    sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns'))
    print(f"  one-sample t-test (mu=0): t={t:.3f}  p={p:.4e}  {sig}")
    if p < 0.05:
        direction = "diversity up -> faster return" if user_corrs.mean() < 0 else "diversity up -> slower return"
        print(f"  -> significant: {direction}")
    else:
        print(f"  -> not significant: no within-user diversity-retention relationship")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    log = load_logs()
    vf  = load_video_features()
    sessions = compute_session_diversity(log, vf)
    sessions = compute_return_gap(sessions)
    correlation_analysis(sessions)
    quantile_analysis(sessions)
    within_user_analysis(sessions, min_sessions=5)

    # save results
    out_path = os.path.join(OUT_DIR, 'diversity_retention_results.csv')
    sessions[['user_id','date','session_len',
              'tag_unique_ratio','video_type_entropy','music_type_entropy',
              'return_gap']].to_csv(out_path, index=False)
    print(f"\n[done] per-session results saved -> {out_path}")


if __name__ == '__main__':
    main()
