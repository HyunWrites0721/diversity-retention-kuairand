# Does Diversity Actually Improve Long-term Retention? — Raw-data Analysis

Reproducibility repository for the **Section 6 (raw-data) analysis** of the paper:

> *Does Diversity Actually Improve Long-term Retention? An Empirical Investigation
> with Failure Analysis of RL-based Diversity Control.* FRAME'26 (co-located with
> ACM RecSys 2026).

It contains everything needed to reproduce the paper's claim that, in KuaiRand,
higher session diversity is **not** associated with better long-term retention —
including the session-length confound that reverses the naive correlation and the
within-user analysis.

## Contents

| File | Description |
|------|-------------|
| `diversity_retention_results.csv` | Derived table, one row per user–date session (session length, three diversity metrics, return gap). 231,829 sessions. **CC BY-SA 4.0** (see `ATTRIBUTION.md`). |
| `reproduce.py` | Recomputes all Section 6 statistics **from the CSV alone** — no raw download needed. |
| `diversity_retention_signal.py` | The original derivation script that builds the CSV from raw KuaiRand-Pure logs (documents provenance; see note below). |
| `ATTRIBUTION.md` | KuaiRand attribution and CC BY-SA 4.0 notice for the derived data. |
| `LICENSE` | MIT license for the code in this repository. |

## Quick reproduction

```bash
pip install numpy pandas scipy
python reproduce.py
```

Expected output (matches the paper):

```
Section 6.2  Aggregate correlations
  tag_unique_ratio      Pearson r=+0.0883 ***   Spearman r=+0.1737 ***
  video_type_entropy    Pearson r=-0.0262 ***   Spearman r=-0.0884 ***
  music_type_entropy    Pearson r=-0.0406 ***   Spearman r=-0.0760 ***
  Confound check (controlling for session_len):
    corr(session_len, tag_unique_ratio) = -0.5997
    corr(session_len, return_gap)       = -0.1747
    partial r(tag, gap | session_len)   = -0.0209 ***

Section 6.3  Quartile analysis
  (0.0823, 0.625]   60,456   mean_gap 2.169   return<=1d 76.4%
  (0.625, 0.75]     55,872   mean_gap 2.811   return<=1d 62.7%
  (0.75, 1.0]      115,501   mean_gap 2.828   return<=1d 57.4%

Section 6.4  Within-user analysis (min 5 sessions/user)
  users analyzed            : 20,018
  mean within-user r        : +0.1360 ***
  users with r > 0 (slower) : 66.3%
  users with r < 0 (faster) : 32.4%
  one-sample t-test (mu=0)  : t=53.149  p~0  ***
```

## Regenerating the CSV from raw KuaiRand-Pure (optional)

`diversity_retention_signal.py` documents how `diversity_retention_results.csv`
was produced. It reads the raw KuaiRand-Pure logs
(`log_standard_*.csv`, `log_random_*.csv`, `video_features_basic_pure.csv`), so to
run it you must (1) download KuaiRand-Pure from <https://kuairand.com/> and
(2) edit the `DATA_DIR` path near the top of the script to point at your local
copy. The shipped CSV lets you skip this step for the analysis in `reproduce.py`.

## Data source and license

The derived table is computed from **KuaiRand-Pure** (Gao et al., CIKM 2022),
distributed under **CC BY-SA 4.0**. Under the ShareAlike term, the derived table
here is released under the same license, with attribution — see `ATTRIBUTION.md`.
The code in this repository is released under the MIT license (`LICENSE`).

## Citation

```bibtex
@inproceedings{gao2022kuairand,
  title     = {KuaiRand: An Unbiased Sequential Recommendation Dataset with Randomly Exposed Videos},
  author    = {Gao, Chongming and Li, Shijun and Zhang, Yuan and Chen, Jiawei and Li, Biao and Lei, Wenqiang and Jiang, Peng and He, Xiangnan},
  booktitle = {CIKM},
  year      = {2022}
}
```
