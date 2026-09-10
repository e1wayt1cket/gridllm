# make_training_summary.py
"""Consolidate every training run and evaluation on disk into summary tables.

Writes CSV plus a readable Markdown digest. The CSVs are the machine-readable
record (they include the columns the digest shortens away); the Markdown is
what gets pasted into a handoff note or a paper draft.

Both consumer-accounting generations are reported separately. Results written
before 2026-09-09 use the older convention, where storage discharge offset the
consumer bill; that flipped the sign of consumer surplus on three of the four
scenarios, so the two must not be averaged together.

Usage:
  python make_training_summary.py [--out-dir results] [--quiet]
"""

import os
import argparse

import pandas as pd

import data_aggregator as da

# Runs shorter than this are smoke tests or pilots rather than results.
_REAL_RUN_MIN_EPISODES = 30

_CONVENTION_TITLES = {
    "p_dis_excluded": "现行口径（2026-09-09 起）：储能放电不抵扣消费者账单",
    "p_dis_offset_legacy": "旧口径（2026-09-09 前）：储能放电抵消购电，CS/CP 符号相反",
    "unknown": "无消费者账目",
}

_SCENARIO_ORDER = ["baseline", "high_re", "peak_load", "congestion"]


def _fmt_table(df: pd.DataFrame, columns: dict, decimals: int = 1) -> str:
    """Render a frame as a Markdown table with named, formatted columns."""
    header = "| " + " | ".join(columns.values()) + " |"
    rule = "|" + "|".join("---" for _ in columns) + "|"
    lines = [header, rule]
    for row in df.to_dict("records"):
        cells = []
        for key in columns:
            v = row.get(key)
            if v is None or (isinstance(v, float) and pd.isna(v)):
                cells.append("—")
            elif isinstance(v, float):
                cells.append(f"{v:,.{decimals}f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def build_digest(runs: pd.DataFrame, evals: pd.DataFrame) -> str:
    """The Markdown digest as one string."""
    out = ["# 训练与评估汇总", ""]

    real = runs[runs["episodes"].fillna(0) >= _REAL_RUN_MIN_EPISODES]
    out += [
        f"- 训练 run 总数：**{len(runs)}**（其中正式 run {len(real)} 个，"
        f"其余为 smoke / pilot）",
        f"- 评估结果文件：**{evals['policy_label'].nunique()}** 个，"
        f"fleet 评估点 {len(evals)} 个",
        "",
    ]

    if len(real):
        by_algo = real.groupby("algo").size().to_dict()
        out += [
            "- 算法分布：" + "、".join(f"{k} {v}"
                                       for k, v in sorted(by_algo.items())),
            "",
        ]

    # ---- per-run table ----
    out += ["## 一、训练运行", ""]
    cols = {"date": "时间", "algo": "算法", "seed": "种子",
            "episodes": "集数", "best_episode": "best 集",
            "final_mean_reward": "final reward", "critic_max": "critic 峰值",
            "critic_trend": "损失趋势"}
    shown = real[list(cols)].copy()
    shown["critic_trend"] = shown["critic_trend"].map(
        {"falling": "下降", "rising": "上升", "unknown": "—"})
    # A bounded, falling loss is the healthy signature; anything else is worth
    # looking at before trusting the run's numbers.
    diverging = shown[shown["critic_max"].fillna(0) > 10]
    out += [_fmt_table(shown, cols), ""]
    if len(diverging):
        out += [f"> ⚠ {len(diverging)} 个 run 的 critic loss 峰值超过 10，"
                f"需检查是否发散。", ""]
    else:
        out += [f"> 全部 {len(shown)} 个正式 run 的 critic loss 有界"
                f"（峰值 ≤ {shown['critic_max'].max():.2f}），未见发散。", ""]
    out += [
        "> 损失趋势一列多为「上升」属预期：critic 的 TD 目标在除以 "
        "`reward_std()` 归一化，策略变好时该尺度随之变化，因此**损失上升不等于发散**。"
        "判据是峰值是否有界（上表 critic 峰值），而非趋势方向。",
        "",
    ]

    # ---- evaluation, one table per convention ----
    out += ["## 二、评估结果（按场景，fleet 均值）", ""]
    for conv in da.CONSUMER_CONVENTIONS:
        sub = evals[evals["cs_convention"] == conv]
        if not len(sub):
            continue
        out += [f"### {_CONVENTION_TITLES[conv]}", ""]
        grouped = sub.groupby("scenario")[
            ["profit_delta", "genuine_welfare_delta", "cs_delta",
             "market_power_arb", "market_power_power"]] \
            .mean().reset_index()
        grouped["n"] = sub.groupby("scenario").size().values
        grouped["_o"] = grouped["scenario"].map(
            {s: i for i, s in enumerate(_SCENARIO_ORDER)})
        grouped = grouped.sort_values("_o", na_position="last")
        cols = {"scenario": "场景", "n": "点数",
                "profit_delta": "利润增量", "genuine_welfare_delta": "真实福利增量",
                "cs_delta": "CS 增量", "market_power_arb": "套利",
                "market_power_power": "市场力"}
        out += [_fmt_table(grouped, cols), ""]

    out += [
        "口径说明：`cs_delta` 与 `cp_delta` 在两代口径之间符号可能相反，"
        "因此分开汇总、不合并平均。`profit_delta` 与 `genuine_welfare_delta` "
        "不受口径影响。",
        "",
    ]
    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=str, default="results",
                        help="Directory for the CSV and Markdown outputs")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    runs = da.run_summary()
    evals = da.experiment_table()          # both conventions, annotated

    runs_path = os.path.join(args.out_dir, "training_run_summary.csv")
    eval_path = os.path.join(args.out_dir, "training_eval_summary.csv")
    md_path = os.path.join(args.out_dir, "training_summary.md")
    runs.to_csv(runs_path, index=False)
    evals.to_csv(eval_path, index=False)

    digest = build_digest(runs, evals)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(digest)

    if not args.quiet:
        print(digest)
        print()
    print(f"wrote {runs_path}")
    print(f"wrote {eval_path}")
    print(f"wrote {md_path}")
    if da.warnings():
        print(f"\n{len(da.warnings())} discovery warning(s):")
        for w in da.warnings():
            print(f"  - {w}")


if __name__ == "__main__":
    main()
