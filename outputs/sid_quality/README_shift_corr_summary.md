# Local Shift-invariant Correlation Distance 结果

设置：两个 heartbeat 比较时，允许其中一个在 `-6..+6` 个采样点内小范围平移；每个平移都算 correlation distance，取最小值作为 pattern distance。

| 实验 | active | perplexity | intra shift-corr | inter shift-corr | inter/intra ratio | intra P50 | intra P90 | intra best shift | inter best shift |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| V2 原始 no balancing | 780 | 703.59 | 0.4583 | 0.6459 | 1.4093 | 0.4162 | 0.8786 | 3.04 | 3.59 |
| H clean | 730 | 652.17 | 0.3088 | 0.6144 | 1.9897 | 0.2445 | 0.6533 | 1.31 | 3.50 |

## 结论

H clean 在允许小范围时间错位后仍然明显更好：

- intra shift-corr distance：`0.4583 -> 0.3088`，同一个 word 内部 pattern 更像。
- inter/intra ratio：`1.4093 -> 1.9897`，不同 word 之间更分得开。
- intra P90：`0.8786 -> 0.6533`，说明最混杂的一批同 word pair 也改善了。
- intra best shift mean：`3.04 -> 1.31`，说明 H 同一 word 内部不仅形状更像，时间错位也更小。

可以汇报为：

> 我们进一步使用 local shift-invariant correlation distance，不强制 R-peak 对齐，只允许 heartbeat 在小范围内自动寻找最佳对齐。结果显示，H clean 的同 word 内部 pattern distance 明显更低，inter/intra 分离比接近 2，说明 clean 后的 SID word 更接近“同一种 ECG 形态”，而不是只靠时间点对齐。
