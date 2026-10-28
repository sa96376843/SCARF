# SCARF

SCARF: Stability-Constrained Active Relearning with Reliability Feedback for Temporal Malware Detection

本仓库提供论文对应的实验代码、配置、历史结果，以及 2026-10-27 动态反馈补充实验的逐月预测和查询记录。原始固定门控方法与动态扩展分别保留，便于按相同协议核对结果。

## 内容

| 目录 | 内容 |
|---|---|
| `SCARF_TDSC/code` | 原始 SCARF-v3、基线、组件消融、扩展比较和分析脚本 |
| `SCARF_TDSC/results*` | 历史逐月结果、协议、标签查询及效率记录 |
| `SCARF_TDSC/tables` | 既有结果表及其中间 CSV |
| `experiments_2026-10-27/code_snapshot` | 实际运行的动态扩展代码快照与冻结配置 |
| `experiments_2026-10-27/runs` | 全部 27 组的逐月指标、预测、查询、伪标签、反馈及状态 |
| `experiments_2026-10-27/analysis` | 三种子汇总、配对区间、触发统计与核验记录 |
| `diagnostics` | 阈值判决变化、Brier 类别贡献和历史逐月差异 |

原始数据通过 [DATA.md](DATA.md) 中的来源获取。本仓库不包含原始 Parquet、可执行恶意软件、模型检查点、Python 环境或论文内部评审材料。代码和实验数值按源文件原样保留，逐文件来源及校验值见 `publication_manifest.json` 和 `SHA256SUMS.json`。历史绘图脚本保留原实验版本的样式。

## 实验协议与结果

数据为 BODMAS 时间戳 Parquet 镜像，134,428 条记录。2019 年 8 月用于初始训练，9 月用于校准，2019 年 10 月至 2020 年 9 月逐月前向评估。动态扩展包括九种策略、三个随机种子（13、42、87），每月总标注预算为 100。原始结果与新环境重跑分别报告。

补充实验已完成 27 组、324 条月度评估。配置文件的 `CODE_READY_REAL_DATA_NOT_RUN` 是运行前冻结状态标签，保留以维持原配置哈希；实际完成状态见各运行目录的 `status.json`。

| 策略 | F1 (%) | 实际 FPR (%) | TPR@1%FPR (%) |
|---|---:|---:|---:|
| EnsembleMargin | 98.402 | 2.354 | 98.871 |
| FrozenGate | 98.343 | 2.392 | 98.562 |
| DynamicGate | 98.328 | 2.426 | 98.531 |
| FrozenThreshold | 98.449 | 2.014 | 98.531 |

以上为三种子月度宏平均；完整九策略结果见 `experiments_2026-10-27/analysis/comparison.csv`。动态门控只在种子 13 的三个后续月份影响查询，本轮没有显示相对冻结门控的检测收益。冻结阈值保留相同的概率分数，降低实际误报率，同时使实际召回率从 99.197% 降至 99.069%。TPR@1%FPR 是 ROC 插值指标，与实际阈值召回率分开解释。F1 的平均提升不代表配对区间排除了零。

## 环境与复现

补充实验使用 Python 3.12.14；具体依赖版本记录在 `experiments_2026-10-27/code_snapshot/requirements.txt` 和各运行的 `run_manifest.json` 中。

```bash
python -m venv .venv
# 激活该环境后安装依赖
python -m pip install -r experiments_2026-10-27/code_snapshot/requirements.txt

# 机制/时序测试采用合成 fixture，不是论文实验结果
python -m unittest discover -s experiments_2026-10-27/code_snapshot/tests -v

# 从已发布记录重新汇总；输出到新目录
python experiments_2026-10-27/code_snapshot/summarize.py --runs experiments_2026-10-27/runs --out analysis_new

# 获取并校验数据后，运行全部九策略 × 三种子
python experiments_2026-10-27/code_snapshot/run_dynamic.py --data data/bodmas-timestamps.parquet --out runs_new --cache cache_new

# 从原始数据和保存的预测复核月度指标、标签预算及反馈时序
python experiments_2026-10-27/verify_results.py
```

新运行应使用新输出目录，保留已发布的完整结果。固定的标注预算包含每月 20 个均匀审计对象。月份 t 的预测先冻结，返回的反馈用于后续查询和模型更新，不用于修改当月预测。

历史实验可在另一个环境安装 `SCARF_TDSC/requirements.txt` 后执行：

```bash
python SCARF_TDSC/code/run_experiments.py --data data/bodmas-timestamps.parquet --out historical_baselines_new --cache cache_historical
python SCARF_TDSC/code/run_scarf_v3.py --data data/bodmas-timestamps.parquet --out historical_scarf_new --cache cache_historical
python SCARF_TDSC/code/run_ablations_v3.py --data data/bodmas-timestamps.parquet --out historical_ablation_new --cache cache_historical
python SCARF_TDSC/code/run_extended_comparisons.py --suite comparisons --data data/bodmas-timestamps.parquet --out historical_extended_new --cache cache_historical
```

`run_extended_comparisons.py` 还支持 `--suite sensitivity` 和 `--suite efficiency`。历史环境与补充实验环境的部分数值存在小幅差异，具体原因尚未完全定位，见 `historical_reproduction_differences.csv`。不能将重新运行、配置一致或门控触发等同于性能结论已经独立复现。
