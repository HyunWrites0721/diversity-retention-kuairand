# Does Diversity Actually Improve Long-Term Retention? — Raw-data Analysis

Reproducibility repository for the **Section 6 (raw-data) analysis** of the paper:

> *Does Diversity Actually Improve Long-Term Retention? An Empirical Investigation
> with Failure Analysis of RL-Based Diversity Control.* FRAME'26: Methodology First -
> Rethinking Research Assessment in RecSys Workshop (co-located with ACM RecSys 2026),
> September 28, 2026, Minneapolis, Minnesota, USA.
>
> The camera-ready version of the paper is included as [`paper.pdf`](paper.pdf).

It reproduces the paper's Section 6 finding that, in KuaiRand, the diversity–retention
association is **near-null once user activity is controlled**. Naive within-user
correlations look strong and even change sign across diversity metrics (tag vs.
video-/music-type entropy), but that structure is a **session-length confound**:
adjusting for it collapses all three metrics to a weak, uniform faster-return signal
(*r* between −0.02 and −0.035). It includes the confound check and the raw-vs-length-adjusted
within-user analysis.

## Contents

| File | Description |
|------|-------------|
| `paper.pdf` | Camera-ready version of the paper (FRAME'26). **CC BY 4.0**, copyright by its authors. |
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
Section 6.2  Aggregate correlations (diversity vs. return_gap)
  metric                 Pearson   Spearman  partial|len
  tag_unique_ratio       +0.0883    +0.1737    -0.0209 ***
  video_type_entropy     -0.0262    -0.0884    -0.0143 ***
  music_type_entropy     -0.0406    -0.0760    -0.0171 ***
  Confounder structure (why raw signs disagree):
    corr(session_len, tag_unique_ratio)   = -0.5997
    corr(session_len, video_type_entropy) = +0.0698
    corr(session_len, music_type_entropy) = +0.1369
    corr(session_len, return_gap)         = -0.1747
  -> raw signs disagree, but all three partials (ctrl len) are weakly negative.

Section 6.3  Binned analysis (tag_unique_ratio)
  tag_unique_ratio is heavily tied at its maximum, so a quartile split
  collapses into three unequal bins; the actual cut points are reported.
  return<=1d = next-day return rate (fraction with return_gap <= 1).

  bin                                 n  mean_gap  return<=1d
  (0.0823, 0.625]                60,456     2.169       76.4%
  (0.625, 0.75]                  55,872     2.811       62.7%
  (0.75, 1.0]                   115,501     2.828       57.4%

Section 6.4  Within-user analysis: raw vs length-adjusted (min 5 sessions)
  metric                 users    raw_r    adj_r   adj_t
  tag_unique_ratio      20,018  +0.1360  -0.0353   -12.1
  video_type_entropy    11,536  -0.1031  -0.0214    -7.5
  music_type_entropy    19,585  -0.0638  -0.0211    -7.6
  -> Raw signs disagree (a session-length confound); once length is
     controlled, all three collapse to a weak, uniform faster-return signal.
```

## Regenerating the CSV from raw KuaiRand-Pure (optional)

`diversity_retention_signal.py` documents how `diversity_retention_results.csv`
was produced. It reads the raw KuaiRand-Pure logs
(`log_standard_*.csv`, `log_random_*.csv`, `video_features_basic_pure.csv`), so to
run it you must (1) download KuaiRand-Pure from <https://kuairand.com/> and
(2) point the script at your local copy of its `data/` directory via the
`KUAIRAND_DIR` environment variable, e.g.:

```bash
KUAIRAND_DIR=/path/to/KuaiRand-Pure/data python diversity_retention_signal.py
```

The shipped CSV lets you skip this step for the analysis in `reproduce.py`.

## Data source and license

The derived table is computed from **KuaiRand-Pure** (Gao et al., CIKM 2022),
distributed under **CC BY-SA 4.0**. Under the ShareAlike term, the derived table
here is released under the same license, with attribution — see `ATTRIBUTION.md`.
The code in this repository is released under the MIT license (`LICENSE`).
The paper (`paper.pdf`) is © 2026 its authors and released under CC BY 4.0.

## Citation

```bibtex
@inproceedings{gao2022kuairand,
  title     = {KuaiRand: An Unbiased Sequential Recommendation Dataset with Randomly Exposed Videos},
  author    = {Gao, Chongming and Li, Shijun and Zhang, Yuan and Chen, Jiawei and Li, Biao and Lei, Wenqiang and Jiang, Peng and He, Xiangnan},
  booktitle = {CIKM},
  year      = {2022}
}
```
