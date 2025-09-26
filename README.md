# Nasdaq-100-Top-N-
Nasdaq 100 成分股 Top-N 策略回测
Question:

Nasdaq 100 指数（NDX）在过去近 30 年表现优秀，其中包含 100 只成分股，以科技股为
主。
问题是：如果我们只选择市值最大的 N 只股票（如 Top 10、Top 30、Top 50 等），是否能
获得比指数本身更高的回报？最优的 N 应该是多少？




Method:

样本区间：2007-12-31 至 2025-06-27
基准：QQQ（纳斯达克 100 ETF，总回报价格）
调仓频率：每年一次(可根据修改参数调整成每月每日)
权重方式：
Value-weight（市值加权）
Equal-weight（等权重）
扫描范围：N = 1 -100



Back Test Result:

1. Value-weight 策略
最优 N = 7, CAGR = 18.7%, 总回报率 ≈ 1900%显著优于基准 QQQ（CAGR 14.9%，总回
报 ≈ 1033%）
(From logs) N= 7 STRAT_TR=1908.19% STRAT_CAGR=18.71% QQQ_TR=1033.37% 
QQQ_CAGR=14.90% [2007-12-31 ~ 2025-06-27]


3. Equal-weight 策略
最优 N = 8, CAGR = 19.6%, 总回报率 ≈ 2200%, 进一步超过基准 QQQ 与 value-weight 策
略
(From logs) N= 8 STRAT_TR=2194.95% STRAT_CAGR=19.62% QQQ_TR=1033.37% 
QQQ_CAGR=14.90% [2007-12-31 ~ 2025-06-27]



Conclusion:

NDX 整体表现很好，但并非最优选择：从 2007 年以来，直接持有 NDX 的 CAGR 约
14.9%，而通过 Top-N 策略可以显著提高到 ~19%。
Top 7–8 股组合最优：无论市值加权还是等权重，最佳 N 都在 7–8 之间，说明过度分散会
稀释回报，而集中持有头部股票能捕捉到科技巨头的长期成长。
Equal-weight 略胜一筹：Top-8 等权重组合在 CAGR 和总回报率上都优于市值加权，说明
避免过度集中在少数超级巨头（如 AAPL、MSFT）能带来更佳风险回报。
