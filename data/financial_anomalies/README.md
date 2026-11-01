# KronosAD 金融交易异常数据

检索与下载日期：**2026-11-02（Asia/Singapore）**。原始文件和已转换的数据都已实际下载，
不是只提供链接或下载器。数据目录约 31 MB，单个文件最大约 26.4 MB，无需 Git LFS。

## 已包含的数据

| 数据包 | 原始来源 | 可直接运行的序列 | 行数 | 标记为正的点 | 标签性质 |
| --- | --- | ---: | ---: | ---: | --- |
| `meme2026` | USENIX Security 2026 / Zenodo 17830944 | 39 条连续小时级片段 | 80,204 | 323 | 作者算法识别的泵拉砸盘区间，非人工逐点真值 |
| `sapienza_binance` | ICCCN 2020 事件名单 + Binance 原始 K 线 | 12 个案例的连续 5 分钟行情 | 17,029 | 84 | 公告时间扩展的弱标签 |
| `recent_binance` | Binance BTCUSDT、ETHUSDT | 2 条连续 5 分钟行情 | 1,728 | 不适用 | 无标签，`label=-1` |
| 合计 | | **53 条** | **98,961** | | |

近期行情覆盖 **2026-10-29 00:00 至 2026-10-31 23:55 UTC**。
检索时 2026-11-01 的完整 UTC 日归档和 2026-10 月归档尚未发布（返回 404），
因此没有把未取得的最新日期写入数据集。

所有下载文件的 URL、字节数和 SHA256 见 [`raw_manifest.json`](raw_manifest.json)。
完整目录与各序列的路径、单位、分段边界、标签语义见 [`catalog.csv`](catalog.csv)。
检索过但未纳入的候选数据及原因见 [`SEARCH_NOTES.md`](SEARCH_NOTES.md)。

## 哪些标签可以用于什么评估

### 1. Meme 币：作者算法区间标签

来源：[Zenodo 17830944](https://doi.org/10.5281/zenodo.17830944)，
对应论文 *A Midsummer Meme's Dream: Investigating Market Manipulations in the Meme Coin Ecosystem*，
作者 Alberto Maria Mongardini、Alessandro Mei，USENIX Security 2026。

- **完整原始研究包**已保存为 `raw/meme2026/meme_coin-main.zip`，26,446,048 字节，
  MD5 `d6676f869dec06a624f2528c20dfcee7`，与 Zenodo 发布者校验值一致。
- 包中包含日级、小时级 OHLCV、洗盘/LPI/rug-pull 相关数据与研究代码。
  本次只将符合 Kronos 输入要求的**小时级行情**转换为直接运行的文件；其他材料仍在原始包内。
- 原始小时包有 **584 个 CSV、331,865 行**，其中 **176 个为空文件**。
  不少非空序列有长时间缺口，因此只保留每段至少 600 行、严格连续的小时片段，
  共 39 段，来自 33 个来源文件。没有补造空缺 K 线。
- 91 条作者泵拉砸盘事件保留于 `raw/meme2026/pump_and_dump_events.csv`。
  对保留的行情段，将 `[pump_start, dump_end]` 内的 bar-open 时间标记为 1，其余为 0。
  原事件表实际覆盖 2024-08 至 2025-03，文件名的 “after_mid_january” 不作为过滤依据。
- 作者标签由 RSI、涨幅、成交额和后续下跌等规则回溯识别；它们不是监管认定或独立人工真值。
  在这套标签上计算的 F1 衡量的是与作者规则的一致性，不能直接解释成真实操纵识别准确率。
- 原始文件列 `highest/lowest/closed` 分别转换为 `high/low/close`。
- 来源为 GeckoTerminal OHLCV；按照该接口默认 USD 计价以及论文中美元成交额的解释，
  原 `volume` 作为报价成交额放入 `amount`，不将其冒充代币数量 `volume`。
  原包没有保存 API 请求的 currency 参数；若需要验证绝对单位，应进一步向作者核对，
  或仅使用 OHLC 做稳健性对照。原始数值在原包中完整保留。
- 原始文件名已将地址转为小写，包括 Solana。这里仅用这些**来源别名**关联标签；
  不能把文件名中的小写 Solana 别名当作链上规范地址。原事件表保留作者原地址。
- 高回报代币选择以及连续性筛选都带来选择偏差，不代表整个市场。

逐文件的空数据、缺口筛选与保留情况见 [`meme_quality_report.csv`](meme_quality_report.csv)。

### 2. 历史 Binance 泵拉砸盘：公告事件弱标签

事件来源：[SystemsLab-Sapienza/pump-and-dump-dataset](https://github.com/SystemsLab-Sapienza/pump-and-dump-dataset)，
固定版本 `d71250d4cb055dde2d415c8cba38a0dcd6eb6e16`，MIT。
论文：*Pump and Dumps in the Bitcoin Era: Real Time Detection of Cryptocurrency Market Manipulations*，ICCCN 2020，
DOI [10.1109/ICCCN49398.2020.9209660](https://doi.org/10.1109/ICCCN49398.2020.9209660)。

- 完整原始名单共 **1,110 行事件**，包含 Binance、Yobit、Cryptopia、Bittrex、KuCoin；
  原始数据以实际下载后的行数为准。
- 本次为可运行 starter benchmark：从 Binance 事件按年份（2018–2021）分别选最早、中间、
  最晚的去重事件，共 **12 个**。完整名单没有被这 12 个案例替代。
- 所有配对必须是论文指定的 **SYM/BTC**，不是同名代币的 USDT 交易对，也不是其他交易所。
- 从 Binance Vision 下载事件日之前 3 个 UTC 日、事件日及后 1 日，共 5 日的真实 5 分钟 K 线。
  12 个案例中，PPT 源行情存在时间缺口，只保留了最长的合格连续片段，未填补缺失数据。
- 原始 `quote asset volume` 映射为 `amount`；`volume` 保持原始基础资产数量。
- `label_event_onset=1`：该 K 线区间包含公告时刻。
- `label=1`：该 K 线与 `[公告时刻-2分钟, 公告时刻+30分钟)` 有交集。
  该区间是本项目**明确指定的弱标注规则**，不是来源论文人工标出的泵开始/结束边界。
  来源提到部分活动可比公告提前约 120 秒；公告后的 30 分钟为本项目约定，未在测试结果上调优。
- `label=0` 仅表示不在已知事件窗口中，不能证明该时刻没有其他异常。
- 上游附带的手工标注文件是工程特征序列，不能还原出精确 OHLC，因而没有把其标签
  冒充本次 K 线的人工逐点真值。
- 年代早于论文所述 Kronos 预训练截止期，不能宣称此包是严格时间外的零样本评估。

完整选择记录见 [`sapienza_selected_events.csv`](sapienza_selected_events.csv)。

### 3. BTC / ETH 近期行情：完全无标签

`prepared/recent_binance/` 各包含 864 根真实 5 分钟 K 线，覆盖三个完整 UTC 日。
`label=-1` 表示未知，不会把未知点标记为正常。
可用于近期数据上的运行、漂移分析或人工检查，不能据此直接计算异常检测 F1。

## 报价单位换算与原始数据保留

KronosPredictor 在输入标准化时加上固定 `1e-5`，直接输入很小的币价会让该常数压过真实波动。
因此 prepared 数据采用**可逆的报价单位换算**，而非修改收益率或生成行情：

- `sapienza_binance`：BTC 报价乘 `1e8`，转换为 satoshi；OHLC 和 `amount` 同乘该倍数，
  基础资产数量 `volume` 不变。
- `meme2026`：每段只根据最初 64 根历史 K 线的 open 中位数选择十进制倍数，
  使参考价格约为 100–1000；OHLC 与 `amount` 同乘该倍数，之后整个序列保持该单位。
  单位选择不读取未来价格或标签。
- `recent_binance`：保留 USDT 报价，倍数为 1。

每段的 `quote_multiplier` 和 `quote_unit` 均记录于 `catalog.csv`；
**原价格/成交额 = prepared 数值 / quote_multiplier**。
来源文件未修改。原始包、转换数据的许可不因换单位而改变。

## 运行

在仓库根目录安装原有异常检测依赖：

```bash
python -m pip install -r requirements-anomaly.txt
```

推荐通过目录工具运行，自动读取该段的参考边界与各维度 epsilon，避免给小币价套用不合适的尺度：

```bash
# 权重已放在 weights/ 中时，完全离线运行
HF_HUB_OFFLINE=1 python scripts/data/run_catalog_series.py \
  --series-id AMBBTC_2019-01-01_s00 \
  --model ./weights/Kronos-mini \
  --tokenizer ./weights/Kronos-Tokenizer-2k \
  --local-files-only \
  --output anomaly_results/amb_case \
  --plot
```

有 323 个算法正标记点的 meme 包中，可选择例如：

```bash
HF_HUB_OFFLINE=1 python scripts/data/run_catalog_series.py \
  --series-id bsc_0xfebe8c1ed424dbf688551d4e2267e7a53698f0aa_s00 \
  --model ./weights/Kronos-mini \
  --tokenizer ./weights/Kronos-Tokenizer-2k \
  --local-files-only \
  --output anomaly_results/meme_vinu
```

所有 prepared 文件都是 `pandas.read_csv()` 可直接读取的 `.csv.gz`，列包含：

```text
timestamps,open,high,low,close,[volume],[amount],label,[label_event_onset]
```

这里的可运行表示输入格式和参考长度合格，不保证所有序列都有足够非退化 POT 超额。
若拟合失败，应记录失败并审查参考数据，不应通过测试标签反复选择阈值。

标签列不会输入模型。没有提供维度级真值，**不要伪造每个 OHLCVA 维度的异常标签**。
参考划分在数据准备阶段用已公开事件注释预先指定，避开已知正标记并保留 24 bar 缓冲；
这不是在线无标签自动寻找正常区间。校准段仍不保证绝对正常。
模型预测、标准化和 POT 拟合均不读取标签。不要根据测试标签选择阈值或报告最优 F1。

## 校验与离线重建

所有 66 个 Binance ZIP 均已与发布者提供的 `.CHECKSUM` 文件进行 SHA256 核对。
目录校验包括：源文件/派生文件哈希、有限值、正价格、OHLC 关系、非负成交量/额、
时间唯一且连续、至少 600 行、已知正标记不进入建议校准段。

本次 **27 项自动测试通过**，三个数据包各有一个真实 Kronos-mini 离线运行验证。
校验、运行参数和报警计数见 [`validation_report.json`](validation_report.json)；
这些计数不是准确率、真实检出数量或 F1。

```bash
python scripts/data/validate_financial_data.py

# 原始文件都在仓库中，此命令无需网络即可重建 prepared 数据
python scripts/data/prepare_financial_data.py --offline

python -m pip install pytest
python -m pytest tests/test_financial_data.py tests/test_anomaly.py -q
```

`prepare_financial_data.py` 不执行来源研究包内的 notebook 或脚本，只读取数据。
重建命令使用固定事件选择与固定日期，不会自动将未来日期或新事件混入当前实验。

## 许可与引用

见 [`DATA_LICENSES.md`](DATA_LICENSES.md)。**仓库代码的 MIT 许可证不覆盖第三方数据**。
Meme 原始/派生数据为 CC BY 4.0；Binance 原始及派生行情为 CC BY-NC-SA 4.0 并附带
Binance Vision Dataset Terms，仅用于其允许的非商业研究场景；Sapienza 原始事件名单为 MIT。
