# Data Attribution and License

The per-session table `diversity_retention_results.csv` in this directory is a
**derived dataset** computed from **KuaiRand-Pure**:

> Gao, C., Li, S., Zhang, Y., Chen, J., Li, B., Lei, W., Jiang, P., & He, X. (2022).
> KuaiRand: An Unbiased Sequential Recommendation Dataset with Randomly Exposed
> Videos. *CIKM 2022*. <https://kuairand.com/>

KuaiRand is distributed under the **Creative Commons Attribution-ShareAlike 4.0
International (CC BY-SA 4.0)** license
(<https://creativecommons.org/licenses/by-sa/4.0/>).

In accordance with the ShareAlike term of that license, this derived table is
likewise released under **CC BY-SA 4.0**, with attribution to the original KuaiRand
authors as cited above. Each row corresponds to one user–date session and contains
only aggregate session-level statistics — diversity metrics (`tag_unique_ratio`,
`video_type_entropy`, `music_type_entropy`), session length, and return gap. No
personal data is introduced beyond what KuaiRand-Pure already exposes.

The analysis code in this directory (`diversity_retention_signal.py`) is original
work and is not a derivative of the dataset; it is covered by the repository's own
code license, independently of the CC BY-SA 4.0 data license above.
