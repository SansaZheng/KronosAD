# 基于 Kronos 的金融时间序列异常检测

这是在 [官方 Kronos](https://github.com/shiyu-coder/Kronos) 上新增的**预测误差异常检测扩展**，
不是原论文已有的检测器，也不是 TranAD 的两阶段重构模型。支持 Kronos-mini 和本地离线权重。
基础代码来自上游提交 `67b630e67f6a18c9e9be918d9b4337c960db1e9a`，保留原 MIT 许可证。

## 1. 定义与实现

在单个资产、单一采样频率的 OHLCVA 序列上，对时刻 t：

```text
历史窗口 W_t = x[t-L:t]          # 不包含目标 x[t]
预测值 pred[t] = Kronos(W_t)     # pred_len=1，多次采样后取平均
scale[t,i] = std(W_t[:,i], ddof=0) + epsilon[i]
score[t,i] = ((x[t,i] - pred[t,i]) / scale[t,i]) ** 2
dimension_label[t,i] = int(score[t,i] >= threshold[i])
label[t] = OR_i dimension_label[t,i]
anomaly_score[t] = max_i(score[t,i] / threshold[i])
```

因此 `anomaly_score >= 1` 等价于该时间戳任一维度触发异常。
异常含义是“偏离模型基于历史所预测的市场动态”，不等价于操纵、欺诈或数据错误。
维度标签表示偏离发生在哪个字段，不能据此推断因果根源。

- **参考/预热段**：CSV 前 L 行，仅提供初始历史。本扩展不训练模型，使用已有预训练权重。
- **校准段**：行号 `[L, calibration_end)`，逐点预测后拟合每维 POT。
- **测试段**：行号 `[calibration_end, test_end)`，使用冻结阈值。
- 边界是从 0 开始的行位置，右端不包含。参考段和测试段不要求等长。
- 测试开始可以使用此前校准段作为历史。后续时刻会使用此前已到达的测试观测；
  这符合在线预测，但历史异常可能污染后续上下文。不会用预测替换真实历史。
- 标准差只使用当前时刻之前的历史；实际目标的残差**不截断**。
  Kronos 自身仍会对模型输入做标准化和截断。
- `model.eval()` 和 `tokenizer.eval()` 关闭推理时 dropout，默认随机种子 42。
  不同硬件/依赖版本的逐位一致性不保证。正式实验建议指定权重提交版本。

### 缺失维度与输入约束

必须包含 `timestamps,open,high,low,close`；`volume,amount` 可选。
所有实际存在的行情字段都必须有限、非缺失；时间戳必须严格递增且不重复。
额外的标签列会被忽略，绝不会用于预测或选阈值。

缺少 volume 时为模型补 0；缺少 amount 时沿用上游方法，用 volume 乘以 OHLC 均价估算。
这些补出来的字段**不参与评分或诊断**。若 amount 有真实观测而 volume 缺失，保留真实 amount。
若某列虽已存在、但其实是你预先推算的，请用 `--features` 将其排除。
单个 CSV 只放一种资产、一个时间粒度；不要混合多个资产。
程序校验时间顺序，但不自动推断交易日历、停牌、复权或数据源单位。
采样频率与这些数据语义应在校准和测试期间一致。

### 固定 POT 阈值

对每维校准分数，用初始高分位数 u 选出超额值 `score > u`，
将 `score-u` 拟合为广义帕累托分布 GPD（位置固定为 0）。若校准分数数量为 n、
超额数量为 k，阈值为：

```text
D = u + GPD.inverse_survival(q_i * n / k)
```

这是**固定 POT**，不是在线动态更新的 SPOT。POT 的依据参见
[Siffer 等，2017](https://www.kdd.org/kdd2017/papers/view/anomaly-detection-in-streams-with-extreme-value-theory)。

- 默认初始分位数 `0.90`、最少 `20` 个非退化超额；这只是可运行默认配置，需要验证。
- 默认 `--risk 0.01 --risk-mode family`，将名义总体风险均分为每维 `q_i=0.01/m`。
  这是 Bonferroni 分配；仅在边际尾部概率估计有效时才有相应解释，不能保证真实误报率。
- `--risk-mode per-feature` 表示每维都使用该 risk，此时“任一维异常”的总体误报可能更高。
- 初始分位数越高，需要越长校准数据。分数恒定、超额不足或拟合失败时明确报错，
  **不会悄悄改用普通分位数**。需要扩大主要正常的校准段或重新选择初始分位数。
- 只拟合分布上尾的超额值。没有照搬 TranAD 的 coefficient、low quantile 或实验 F1。
- 校准数据应主要正常，且与后续部署分布相近。程序不使用标签自动剔除污染数据。
  强自相关、市场漂移或异常污染可能使尾部分布估计失效。

## 2. 安装

建议在独立 Python 3.10+ 环境中，从仓库根目录执行：

```bash
python -m pip install -r requirements-anomaly.txt
```

新增加的运行依赖只有 SciPy。原仓库依赖包括 PyTorch、pandas、Hugging Face Hub 等。

## 3. 用自带数据跑通

仓库自带 `tests/data/regression_input.csv`，包含 2500 行 OHLCVA 数据。
下面使用前 64 行预热，接着 436 行校准，再检测 20 行。
这段行情没有提供异常真值，**仅用于功能演示，不能计算可信的检测 F1**。

```bash
python -m anomaly \
  --csv tests/data/regression_input.csv \
  --lookback 64 \
  --calibration-end 500 \
  --test-end 520 \
  --device cpu \
  --sample-count 1 \
  --output anomaly_results/mini_demo \
  --plot
```

默认下载 `NeoQuasar/Kronos-mini` 和 `NeoQuasar/Kronos-Tokenizer-2k`。
默认 CPU 单线程，优先保证小规模演示容易运行。
输入窗口不得超过 `--max-context`（mini 默认 2048），不接受静默截断。

## 4. 离线权重：无需连接 Hugging Face

将此前下载好的四个文件放置为：

```text
weights/
├── Kronos-mini/
│   ├── config.json
│   └── model.safetensors
└── Kronos-Tokenizer-2k/
    ├── config.json
    └── model.safetensors
```

在仓库根目录执行（相对路径以当前工作目录为准）：

```bash
HF_HUB_OFFLINE=1 python -m anomaly \
  --csv tests/data/regression_input.csv \
  --model ./weights/Kronos-mini \
  --tokenizer ./weights/Kronos-Tokenizer-2k \
  --local-files-only \
  --lookback 64 \
  --calibration-end 500 \
  --test-end 520 \
  --output anomaly_results/mini_offline \
  --plot
```

该方式不需要关闭证书校验。权重不会被提交到 Git。

## 5. 自己的数据与参数

最小 CSV 示例：

```csv
timestamps,open,high,low,close,volume,amount
2025-01-02 09:35:00,10.0,10.2,9.9,10.1,1000,10100
2025-01-02 09:40:00,10.1,10.3,10.0,10.2,1200,12240
```

实际数据需要足够多行，满足历史窗口、POT 校准尾部和测试段的需求。
例如用前 2000 行作为参考和校准，后续全部作为测试：

```bash
python -m anomaly \
  --csv /absolute/path/market.csv \
  --lookback 128 \
  --calibration-end 2000 \
  --features open high low close volume \
  --epsilon volume=1 \
  --sample-count 5 \
  --initial-quantile 0.95 \
  --risk 0.01 \
  --risk-mode family \
  --device cuda:0 \
  --output anomaly_results/my_market
```

`epsilon` 默认每维 `1e-5`；它是加到历史标准差上的正数，具有原特征的单位。
不同单位、极低波动或零方差序列需要在参考数据上设定合理的每维数值。
`--sample-count` 增大可减小采样噪声，同时增加计算成本；校准和测试使用同样参数。
默认 `--temperature 0.6 --top-p 0.9`。
使用 Hub 权重时可通过 `--model-revision` 和 `--tokenizer-revision` 指定提交 SHA。
切换 small/base 时需同时指定匹配 Tokenizer，并设置 `--max-context 512`。

## 6. 输出

程序要求输出目录尚不存在，以免覆盖历史结果。

| 文件 | 内容 |
| --- | --- |
| `calibration_scores.csv` | 校准段真实值、预测值、标准化尺度、分数、阈值和诊断结果 |
| `test_anomalies.csv` | 测试段同类信息及最终时间戳标签 |
| `thresholds.json` | 每维 GPD 参数、峰数、风险、阈值、运行参数、分段和汇总 |
| `anomalies.png` | 指定 `--plot` 时生成的收盘价预测与异常标记 |

关键列：`row` 为原 CSV 中的 0 起始行位置；`timestamp` 为目标时间；
`observed_close`、`predicted_close`、`scale_close`、`score_close` 为维度级信息；
`threshold_close`、`is_anomaly_close` 为阈值与维度标签。
`is_anomaly` 为逻辑或标签，`anomaly_dimensions` 列出触发的维度。
图中红叉表示**任意维度**触发，不一定是 close 异常。

`test_alarm_fraction` 只是报警比例；`calibration_alarm_fraction_in_sample` 是用于拟合
同一阈值的校准段内诊断比例；二者都不是准确率、真实误报率或 F1。
正式评估需要独立真值与预先固定的点级/事件级评估协议，不应根据测试标签调阈值。

## 7. 验证与代码结构

```bash
python -m pip install pytest
python -m pytest tests/test_anomaly.py -q
```

可选真实模型集成测试：

```bash
HF_HUB_OFFLINE=1 KRONOS_MINI_WEIGHTS_DIR="$PWD/weights" \
  python -m pytest tests/test_anomaly.py tests/test_anomaly_mini.py -q
```

离线测试使用可控预测器，不下载权重。覆盖历史窗口与未来隔离、真实极端值不截断、
缺失可选维度、零方差、错误输入、POT 超额值拟合、指数尾的阈值恢复、
冻结阈值、等号边界与维度逻辑或。真实 mini 的端到端验证另见
[`anomaly_validation.md`](anomaly_validation.md)。
上游 `tests/test_kronos_regression.py` 是 small 权重的回归测试，需要另行下载，
不属于这里的轻量离线测试。

```text
anomaly/scoring.py    历史窗口、模型预测、标准化平方残差
anomaly/pot.py        固定 POT 校准、维度标签和时间戳逻辑或
anomaly/__main__.py   CSV 命令行入口与结果保存
tests/test_anomaly.py 离线正确性测试
```
