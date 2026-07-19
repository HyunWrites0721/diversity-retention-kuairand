"""
KuaiRand 원본 데이터에서 diversity-retention 신호 탐색.

각 유저 세션(user × date)에 대해:
  - 세션 내 추천 아이템들의 diversity 계산 (tag 기반 categorical diversity)
  - 다음 세션까지의 복귀 간격(return_gap) 계산
  - 두 변수의 Pearson/Spearman 상관관계 분석

사용 데이터:
  - log_standard_4_08_to_4_21_pure.csv
  - log_standard_4_22_to_5_08_pure.csv
  - log_random_4_22_to_5_08_pure.csv
  - video_features_basic_pure.csv (tag, video_type, music_type)

Usage:
    /opt/conda/envs/ltv/bin/python /home/data/LTV/analysis/diversity_retention_signal.py
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
# 1. 데이터 로드
# ---------------------------------------------------------------------------

def load_logs():
    print("[1] 로그 로드 중 ...")
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
    print(f"  전체 (dedup): {len(log):,} rows  |  유저: {log['user_id'].nunique():,}  |  날짜: {log['date'].nunique()}")
    return log


def load_video_features():
    print("[2] video feature 로드 중 ...")
    vf = pd.read_csv(
        os.path.join(DATA_DIR, 'video_features_basic_pure.csv'),
        usecols=['video_id', 'tag', 'video_type', 'music_type'],
    )
    # tag: 쉼표로 구분된 문자열 → frozenset
    def parse_tags(s):
        if pd.isna(s):
            return frozenset()
        return frozenset(str(s).split(','))

    vf['tag_set'] = vf['tag'].apply(parse_tags)
    print(f"  video {len(vf):,}개  |  tag nunique: {vf['tag'].nunique()}")
    return vf[['video_id', 'tag_set', 'video_type', 'music_type']]


# ---------------------------------------------------------------------------
# 2. Diversity 계산 (세션 단위)
# ---------------------------------------------------------------------------

def session_diversity(video_ids, vid2tags, vid2type, vid2music):
    """
    한 세션(아이템 목록)의 diversity를 3가지 방식으로 계산.

    Returns dict:
      tag_unique_ratio   : 세션 내 고유 태그 수 / 세션 길이
      video_type_entropy : video_type 분포 엔트로피 (정규화)
      music_type_entropy : music_type 분포 엔트로피 (정규화)
    """
    n = len(video_ids)
    if n == 0:
        return {'tag_unique_ratio': 0.0, 'video_type_entropy': 0.0, 'music_type_entropy': 0.0}

    # tag unique ratio: 전체 태그 중 고유 태그 비율
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
    print("[3] 세션 diversity 계산 중 ...")

    vid2tags  = dict(zip(vf['video_id'], vf['tag_set']))
    vid2type  = dict(zip(vf['video_id'], vf['video_type']))
    vid2music = dict(zip(vf['video_id'], vf['music_type']))

    sessions = log.groupby(['user_id', 'date'])['video_id'].apply(list).reset_index()
    sessions.columns = ['user_id', 'date', 'video_ids']
    sessions['session_len'] = sessions['video_ids'].apply(len)

    # 최소 3개 이상의 아이템이 있는 세션만 분석
    sessions = sessions[sessions['session_len'] >= 3].copy()
    print(f"  세션 (len≥3): {len(sessions):,}개  |  유저: {sessions['user_id'].nunique():,}")

    div_records = sessions['video_ids'].apply(
        lambda vids: session_diversity(vids, vid2tags, vid2type, vid2music)
    )
    div_df = pd.DataFrame(div_records.tolist())
    sessions = pd.concat([sessions.reset_index(drop=True), div_df], axis=1)
    return sessions


# ---------------------------------------------------------------------------
# 3. Retention (return_gap) 계산
# ---------------------------------------------------------------------------

def compute_return_gap(sessions):
    print("[4] return_gap 계산 중 ...")

    # date를 실제 날짜로 변환
    sessions = sessions.copy()
    sessions['date_dt'] = pd.to_datetime(sessions['date'].astype(str), format='%Y%m%d')

    # 유저별로 정렬 후 다음 세션 날짜 계산
    sessions = sessions.sort_values(['user_id', 'date_dt'])
    sessions['next_date'] = sessions.groupby('user_id')['date_dt'].shift(-1)
    sessions['return_gap'] = (sessions['next_date'] - sessions['date_dt']).dt.days

    # 마지막 세션(next_date=NaN) 제외
    sessions = sessions[sessions['return_gap'].notna()].copy()
    sessions['return_gap'] = sessions['return_gap'].astype(int)

    # return_gap 분포 확인
    print(f"  유효 세션: {len(sessions):,}개")
    print(f"  return_gap 분포: mean={sessions['return_gap'].mean():.2f}  "
          f"median={sessions['return_gap'].median():.1f}  "
          f"std={sessions['return_gap'].std():.2f}")
    gap_dist = sessions['return_gap'].value_counts().sort_index().head(10)
    print(f"  gap 빈도 (상위 10):\n{gap_dist.to_string()}")
    return sessions


# ---------------------------------------------------------------------------
# 4. 상관관계 분석
# ---------------------------------------------------------------------------

def correlation_analysis(sessions):
    print("\n" + "=" * 60)
    print("Diversity-Retention 상관관계 분석")
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
        print(f"    diversity 범위  : {col.min():.4f} ~ {col.max():.4f}  mean={col.mean():.4f}")
        print(f"    Pearson  r={pearson_r:+.4f}  p={pearson_p:.2e}  {sig_p}")
        print(f"    Spearman r={spearman_r:+.4f}  p={spearman_p:.2e}  {sig_s}")

    # 세션 길이 보정: session_len을 통제한 편상관
    print(f"\n  [session_len 편상관 보정 — tag_unique_ratio]")
    from scipy.stats import pearsonr
    def partial_corr(x, y, z):
        """z를 통제한 x-y 편상관"""
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
    print(f"    편Pearson r={pr:+.4f}  p={pp:.2e}  {sig}")

    return results


# ---------------------------------------------------------------------------
# 5. 분위별 return_gap 비교
# ---------------------------------------------------------------------------

def quantile_analysis(sessions):
    print("\n" + "=" * 60)
    print("Diversity 분위별 return_gap 비교 (tag_unique_ratio)")
    print("=" * 60)

    sessions = sessions.copy()
    sessions['div_quartile'] = pd.qcut(
        sessions['tag_unique_ratio'], q=4,
        duplicates='drop',
    )

    print(f"\n  {'분위':<12} {'n':>7}  {'mean_gap':>9}  {'median_gap':>11}  {'return<2d':>10}")
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
        direction = "낮음 (다양할수록 빨리 복귀)" if q4.mean() < q1.mean() else "높음 (다양할수록 늦게 복귀)"
        print(f"  → 유의미한 차이 있음. Q4 mean_gap이 Q1보다 {direction}")
    else:
        print(f"  → 유의미한 차이 없음 (p≥0.05)")


# ---------------------------------------------------------------------------
# 6. 사용자별 분석 (within-user correlation)
# ---------------------------------------------------------------------------

def within_user_analysis(sessions, min_sessions=5):
    print("\n" + "=" * 60)
    print(f"Within-user 상관관계 (유저별 개인 내 diversity-gap 관계, min_sessions={min_sessions})")
    print("=" * 60)
    print("  (유저 간 고정효과를 제거하여 다양성의 순수 효과 측정)")

    user_corrs = []
    for uid, grp in sessions.groupby('user_id'):
        if len(grp) < min_sessions:
            continue
        r, p = stats.spearmanr(grp['tag_unique_ratio'], grp['return_gap'])
        if not np.isnan(r):
            user_corrs.append(r)

    user_corrs = np.array(user_corrs)
    print(f"\n  분석 유저 수 : {len(user_corrs):,}")
    print(f"  개인내 상관  : mean={user_corrs.mean():+.4f}  median={np.median(user_corrs):+.4f}  std={user_corrs.std():.4f}")

    # 상관이 음수(다양성↑→gap↓=빨리 복귀)인 유저 비율
    neg_frac = (user_corrs < 0).mean()
    pos_frac = (user_corrs > 0).mean()
    print(f"  r<0 (다양성이 빠른 복귀와 연관): {neg_frac*100:.1f}%")
    print(f"  r>0 (다양성이 늦은 복귀와 연관): {pos_frac*100:.1f}%")

    # 집단 평균이 0과 유의하게 다른지
    t, p = stats.ttest_1samp(user_corrs, 0)
    sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns'))
    print(f"  one-sample t-test (μ=0): t={t:.3f}  p={p:.4e}  {sig}")
    if p < 0.05:
        direction = "다양성↑ → 빠른 복귀" if user_corrs.mean() < 0 else "다양성↑ → 늦은 복귀"
        print(f"  → 유의미: {direction}")
    else:
        print(f"  → 유의미하지 않음: 개인 내에서도 diversity-retention 관계 없음")


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

    # 결과 저장
    out_path = os.path.join(OUT_DIR, 'diversity_retention_results.csv')
    sessions[['user_id','date','session_len',
              'tag_unique_ratio','video_type_entropy','music_type_entropy',
              'return_gap']].to_csv(out_path, index=False)
    print(f"\n[done] 세션별 결과 저장 → {out_path}")


if __name__ == '__main__':
    main()
