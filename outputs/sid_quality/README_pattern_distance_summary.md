# R-peak 对齐后的 Pattern Distance 总结

## 做了什么

这次重新评估了两组 codebook：

- V2 原始 no balancing
- V2-clean H

每组都算两套距离：

- `raw_zscore`：每个 heartbeat token 做 z-score 标准化，然后直接逐点比较。
- `rpeak_aligned_zscore`：每个 heartbeat token 做 z-score 标准化，再把最大绝对峰值对齐到中心位置，然后比较。

这样做的目的，是避免 ECG heartbeat 因为时间上错开几个采样点，就被误判成完全不同的 pattern。

## 结果

| 实验 | 距离模式 | intra RMSE | inter RMSE | inter/intra RMSE | intra corr dist | inter corr dist | corr 分离比 | silhouette |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| V2 原始 | raw z-score | 1.1644 | 1.3471 | 1.1569 | 0.7416 | 0.9429 | 1.2714 | -0.2521 |
| V2 原始 | R-peak aligned | 1.1163 | 1.2831 | 1.1495 | 0.7261 | 0.9013 | 1.2414 | -0.3598 |
| H clean | raw z-score | 0.8652 | 1.3432 | 1.5525 | 0.4421 | 0.9402 | 2.1269 | -0.2145 |
| H clean | R-peak aligned | 0.9696 | 1.2651 | 1.3048 | 0.5880 | 0.8882 | 1.5105 | -0.3123 |

## 人话解释

R-peak 对齐后，核心结论没有变：

**H 仍然明显优于原始 V2 codebook。**

在 R-peak aligned 比较下：

- V2 原始 intra RMSE：`1.1163`
- H intra RMSE：`0.9696`
- V2 原始 corr 分离比：`1.2414`
- H corr 分离比：`1.5105`

这说明 H 不是只在“逐点对齐”的情况下更好。即使允许 heartbeat 在时间上有一些错位，H 的同一个 word 内部仍然更像，不同 word 之间仍然更分得开。

## 一个需要注意的现象

H 在 `raw z-score` 下的结果反而比 `R-peak aligned` 更好：

- H raw intra corr dist：`0.4421`
- H aligned intra corr dist：`0.5880`

这可能说明：H 本身已经把峰位置比较稳定的 heartbeat 聚在一起了。此时再用 `argmax(abs)` 强行对齐，可能会引入额外误差，原因包括：

- 最大绝对值点不一定是真正的 R 峰；
- 不同导联存在极性差异；
- 残余边缘伪影可能比真实 QRS 峰更大；
- 平移后的边缘 padding 会改变波形。

所以汇报时建议这样讲：

> ECG 是时序信号，不能只看逐点 RMSE。我们补充了 R-peak 对齐后的 morphology pattern distance。结果显示，H 在 R-peak aligned 条件下仍然具有更低 intra distance 和更高 inter/intra separation，说明 clean 后的 SID word 语义更干净。但 naive R-peak alignment 也有局限，下一步更适合加入局部 shift-invariant distance 或 DTW。

## 下一步更推荐的指标

不要只依赖硬性的 R-peak 对齐。更适合 ECG pattern 的做法是局部平移不变距离：

```text
对于两个 heartbeat x 和 y：
尝试 -6 到 +6 个采样点的小范围平移；
每个平移都计算 correlation distance；
取最小 distance 作为这对 heartbeat 的 pattern distance。
```

这样更符合目标：

**只要 pattern 像，小范围时间错位不应该被惩罚太重。**

