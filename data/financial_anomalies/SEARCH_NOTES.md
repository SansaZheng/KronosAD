# 检索记录与选择依据

核验日期：2026-11-02。这里的“符合 Kronos”要求至少有真实 OHLC 和有序时间戳，
可选基础成交量与报价成交额；带标签时还必须说明标签的来源、时间粒度和可信程度。
不能把无关金融表格或统计特征补成虚假的 K 线。

| 候选来源 | 核验结果 | 本次处理 |
| --- | --- | --- |
| [USENIX Security 2026 meme coin artifact](https://doi.org/10.5281/zenodo.17830944) | 开放，CC BY 4.0，26.4 MB，含 OHLCV 与作者算法事件区间 | 完整下载原包；转换严格连续的小时行情 |
| [ICCCN 2020 Sapienza P&D](https://github.com/SystemsLab-Sapienza/pump-and-dump-dataset) | MIT，事件公告名单；手工标签附着的是工程特征而非原始 OHLC | 下载完整事件表；用相同交易所/交易对的真实行情构建 12 个弱标签案例 |
| [Binance Public Data](https://github.com/binance/binance-public-data) | 官方公开 OHLCVA，SHA256 校验；2026-08-26 条款为 CC BY-NC-SA 4.0 + 附加条款 | 下载 60 个历史日归档及 6 个近期日归档，保留非商业许可 |
| [2026 Unified P&D benchmark](https://zenodo.org/records/21377192) | 描述符合 OHLCV+标签，但文件访问为 Restricted | 仅记录来源，未绕过访问限制，也未声称已下载 |
| [SIGMOD 2023 target coin prediction](https://github.com/Bayi-Hu/Pump-and-Dump-Detection-on-Cryptocurrency) | 主要为 Telegram 事件/序列特征、目标币种预测；未发现明确仓库再分发许可证 | 不把特征序列当 K 线，也未复制到本公共仓库 |
| [Mendeley trade/order dataset](https://data.mendeley.com/datasets/fsn6fn7ht8/1) | CC BY 4.0，订单与成交日志；介绍未给出与本任务对应的逐时点 OHLC 标签 | 作为需要进一步聚合和核验的候选，未混入可直接运行包 |
| [JFDS high-frequency markets](https://doi.org/10.5281/zenodo.11048480) | 192.3 MB，LOB 与交易操纵模拟，表示方式不同 | 留作订单簿方向候选，不把盘口/中间价拼成伪 OHLC |
| [MagSeven anomaly dataset](https://appliedaihub.org/datasite/magseven/overview/) | 完整版付费，公开部分为派生的相对特征；标签是统计规则 | 未购买，未把工程特征当真实 K 线 |
| [NAB](https://github.com/numenta/NAB) / Yahoo S5 | 通用异常基准多为单变量流量；股票 ticker 的社交讨论量也不是 OHLC | 不通过复制单变量制造四个价格字段 |

本次数据不是“已具备独立逐点真值的全市场金融异常标准答案”。两类带标记数据分别是
作者算法区间标签和公告事件弱标签；近期行情完全无标签。
未发现可以直接使用的 OHLCVA **维度级**独立真值，因此没有制造维度标签。
