# 数字人面部行为驱动模型评测说明

为保证不同队伍结果可比，评测时涉及的以下内容必须统一：

- 评测输入输出格式
- 样本顺序
- 邻居集合 `A(n)` 的定义
- 指标公式与聚合方式

参赛队可以自行实现评测代码，但不得自行修改上述评测协议。


##  评测输入格式

参赛队提交给评测脚本的核心文件为：

- `prediction_emotion.npy`

其形状必须为：

```text
[N, K, T, 25]
```

其中：

- `N`：评测样本数
- `K`：每个样本生成的候选序列数，默认 `10`
- `T`：序列长度，默认 `750`
- `25`：面部行为特征维度

25 维顺序固定为：

- 前 15 维：AU
- 中间 2 维：VA
- 后 8 维：EXP

参赛队内部可以使用任意中间表示，例如 3DMM、blendshape、视频帧、隐变量等，但最终评测前必须转换为上述统一格式。


## 样本顺序定义

评测脚本中的样本顺序严格由 `person_specific_val.csv` 确定。

该文件读取规则与原项目 `ReactionDatasetTest` 一致：

1. 读取 `person_specific_val.csv`
2. 使用其中第 2 列作为 `speaker_path`
3. 使用其中第 3 列作为 `listener_path`
4. 正向样本顺序为：
   - `speaker_path -> listener_path`
5. 再交换说话者/听者方向，追加一遍：
   - `listener_path -> speaker_path`
6. 最终评测样本总数为原始行数的 `2` 倍

因此，`prediction_emotion.npy` 的第 1 维 `N` 必须与该展开后的官方顺序完全一致，不能自行排序，不能自行删样本。


## 邻居集合 A(n) 定义

对每个样本 `n`，其邻居集合 `A(n)` 由官方邻居矩阵唯一确定：

```text
A(n) = { j | M[n, j] = 1 }
```

其中：

- `M` 为 `person_specific_masked_neighbour_emotion_val.npy`
- `M` 的第 `n` 行对应第 `n` 个评测样本
- `M` 的第 `j` 列对应第 `j` 个参考真实序列

注意：

- 参赛队**不需要**自行构建 `A(n)`
- 参赛队**不得**自行替换、扩展或缩减邻居集合
- 邻居矩阵的行列顺序与 `person_specific_val.csv` 展开后的样本顺序一一对应


## 本地开放的指标

本评测脚本开放以下本地指标：

- `FRCorr`
- `FRCorr*`
- `FRdist`
- `FRDiv`
- `FRDvs`
- `FRVar`
- `FRSyn`

说明：

- 在当前开放的 person-specific 评测协议下，脚本中的 `FRCorr*` 与 `FRCorr` 使用同一套 `A(n)`，因此数值相同
- `FRRea` 依赖额外的渲染与 FID 流程，不在本公开脚本中开放，建议由组委会统一复评


## 公式说明

### FRCorr / FRCorr*

对每个样本 `n`、每个候选序列 `k`：

```text
score(n, k) = max_{j in A(n)} CCC(pred[n, k], gt[j])
```

聚合方式：

```text
FRCorr = FRCorr* = (1 / N) * sum_n sum_k score(n, k)
```

数值方向：

- 越大越好


###  FRdist

对每个样本 `n`、每个候选序列 `k`：

```text
dist(n, k) = min_{j in A(n)} d(pred[n, k], gt[j])
```

其中加权 DTW 距离为：

```text
d(x, y)
= (1 / 15) * DTW(x[:, 0:15], y[:, 0:15])
+ 1 * DTW(x[:, 15:17], y[:, 15:17])
+ (1 / 8) * DTW(x[:, 17:25], y[:, 17:25])
```

聚合方式：

```text
FRdist = (1 / N) * sum_n sum_k dist(n, k)
```

数值方向：

- 越小越好


### FRDiv

对同一样本内部的 `K` 条候选序列计算离散程度。

```text
FRDiv_n = sum_{a,b} ||z(n,a) - z(n,b)||^2 / (K * (K - 1) * D')
FRDiv = (1 / N) * sum_n FRDiv_n
```

其中 `z(n, k)` 表示将 `[T, 25]` 展平后的向量。

数值方向：

- 一般越大表示多样性越强


### FRDvs

固定同一个候选编号 `k`，比较不同样本之间的差异：

```text
FRDvs = sum_k sum_{n,m} ||z(n,k) - z(m,k)||^2 / (N * (N - 1) * K * D')
```

数值方向：

- 一般越大表示跨样本差异越强


### FRVar

对时间维计算方差并整体平均：

```text
FRVar = mean_{n,k,d}(Var_t(pred[n,k,t,d]))
```

数值方向：

- 越大表示时序变化越强


###  FRSyn

对每对 `(pred[n,k], speaker[n])` 在 `[-(2*fps-1), ..., +(2*fps-1)]` 的 lag 范围内计算时滞互相关，取峰值对应 lag 的绝对值并平均：

```text
FRSyn = mean(|offset(n, k)|)
```

数值方向：

- 越小越好

