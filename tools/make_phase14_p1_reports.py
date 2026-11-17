"""Generate the P1 Phase-14 reports from saved diagnostics."""

from __future__ import annotations

import csv
import json
import statistics as st
from pathlib import Path

ROOT = Path.cwd()
RUN_ID = "EXPLORATORY_NEAR_B_C2_GATE_seed11_v2"
RUN = ROOT / "runs" / RUN_ID
DATA = ROOT / "data/processed/v2_cy_jn_2025"
REPORTS = ROOT / "reports/v2_cy_jn_2025"
UNITS = ("ESS9", "ESS14", "ESS17", "ESS27")


def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def main() -> int:
    audit = load(RUN / "validation_audit_by_round.json")
    c2 = load(DATA / "c2_soc_training_audit.json")
    stg = load(DATA / "st_gate_gradient_by_round.json")
    episodes = list(csv.DictReader((RUN / "episodes.csv").open(
        encoding="utf-8")))
    grid = list(csv.DictReader((DATA / "critic_gate_action_grid.csv").open(
        encoding="utf-8")))

    # ------------------------------------------------------------------
    # Validation degradation matrix
    # ------------------------------------------------------------------
    days = {}
    for n in range(1, 11):
        for d in audit[str(n)]["validation"]["days"]:
            days.setdefault(d["pair_id"], {})[n] = d["delta_fleet"]
    ordered = sorted(days, key=lambda p: days[p][10])
    matrix_path = DATA / "validation_degradation_matrix.csv"
    with matrix_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["pair_id"] + [f"R{n}" for n in range(1, 11)]
                        + ["R2_value", "final_value", "final_minus_R2"])
        for pid in ordered:
            vals = [days[pid][n] for n in range(1, 11)]
            writer.writerow([pid] + vals
                            + [vals[1], vals[-1], vals[-1] - vals[1]])
    round_stats = []
    for n in range(1, 11):
        col = [days[pid][n] for pid in ordered]
        round_stats.append((n, st.mean(col), st.median(col),
                            sum(1 for x in col if x < 0),
                            min(col), max(col)))
    final_col = [days[pid][10] for pid in ordered]
    worst3 = sum(sorted(final_col)[:3])
    lines = [
        "# Validation degradation audit (P1)",
        "",
        "Only the 11 viewed NEAR validation days are used. No test, no H2.",
        "",
        "| round | mean ΔG | median ΔG | negative days | min | max |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for n, mean, med, neg, lo, hi in round_stats:
        lines.append(f"| {n} | {mean:.1f} | {med:.1f} | {neg}/11 | "
                     f"{lo:.1f} | {hi:.1f} |")
    lines += [
        "",
        "## Per-day matrix",
        "",
        "| day | " + " | ".join(f"R{n}" for n in range(1, 11))
        + " | R2 | final | final−R2 |",
        "|---|" + "---:|" * 10 + "---:|---:|---:|",
    ]
    for pid in ordered:
        vals = [days[pid][n] for n in range(1, 11)]
        lines.append(f"| {pid} | " + " | ".join(f"{v:.1f}" for v in vals)
                     + f" | {vals[1]:.1f} | {vals[-1]:.1f} | "
                       f"{vals[-1] - vals[1]:.1f} |")
    lines += [
        "",
        "## Per-ESS at the best and final rounds",
        "",
        "| ESS | R2 mean ΔG | R10 mean ΔG | R2 charge/discharge/cycles | "
        "R10 charge/discharge/cycles |",
        "|---|---:|---:|---|---|",
    ]
    for u in UNITS:
        r2 = audit["2"]["validation"]
        r10 = audit["10"]["validation"]
        lines.append(
            f"| {u} | {r2['mean_delta_by_unit'][u]:.1f} | "
            f"{r10['mean_delta_by_unit'][u]:.1f} | "
            f"{r2['mean_charge_mwh'][u]:.2f}/"
            f"{r2['mean_discharge_mwh'][u]:.2f}/"
            f"{r2['mean_cycles'][u]:.2f} | "
            f"{r10['mean_charge_mwh'][u]:.2f}/"
            f"{r10['mean_discharge_mwh'][u]:.2f}/"
            f"{r10['mean_cycles'][u]:.2f} |")
    lines += [
        "",
        "## Findings",
        "",
        f"- The final mean is {round_stats[9][1]:.1f} but the median is "
        f"{round_stats[9][2]:.1f}; the mean is driven by a small number of "
        "extreme negative days.",
        f"- The three worst final days sum to {worst3:.1f}, slightly more "
        f"negative than the total final sum {sum(final_col):.1f}; the other "
        "eight days are close to flat or positive. The mean and median "
        "therefore diverge.",
        "- The winter tail dominates: 2022-12-30, 2022-12-31 and 2022-12-13 "
        "worsen sharply. The summer/autumn days carry small or positive "
        "deltas.",
        "- Best-round (R2) vs final (R10): some days improve "
        "(2022-10-26, 2022-10-27, 2022-11-05) but the winter tail more than "
        "offsets them.",
        "- The fleet loss is not uniform across ESS: ESS14 and ESS27 carry "
        "the large negative deltas; ESS17 remains positive on the key days. "
        "The mean is not a typical-day result.",
        "",
        "The full 11×10 matrix is `validation_degradation_matrix.csv`.",
        "",
    ]
    (REPORTS / "validation_degradation_audit.md").write_text(
        "\n".join(lines), encoding="utf-8")

    # ------------------------------------------------------------------
    # Continuous quote branch
    # ------------------------------------------------------------------
    def focus_rows(ckpt, day, target, leg):
        return [r for r in grid
                if r["checkpoint"] == ckpt and r["day"] == day
                and r["target"] == target and r["leg"] == leg]

    lines = [
        "# Continuous quote branch and boundary saturation (P1)",
        "",
        "Only validation/training states are used. No test or H2.",
        "",
        "## Final-round quote distributions (deterministic validation)",
        "",
        "| gate | open freq | mean prob | open quote mean | open quote min | "
        "open quote max | lower-sat | upper-sat |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    g = audit["10"]["validation"]["gate"]
    for key in g["open_frequency"]:
        q = g["quote_distribution"][key]
        if q is None:
            lines.append(f"| {key} | {g['open_frequency'][key]:.2f} | "
                         f"{g['mean_probability'][key]:.2f} | closed | - | - |"
                         f" - | - |")
        else:
            lines.append(
                f"| {key} | {g['open_frequency'][key]:.2f} | "
                f"{g['mean_probability'][key]:.2f} | {q['mean']:.2f} | "
                f"{q['min']:.2f} | {q['max']:.2f} | - | - |")
    lines += [
        "",
        f"Validation lower/upper gate saturation: "
        f"{g['saturation_lower']}/{g['saturation_upper']}.",
        "",
        "### Per-gate saturation and price-branch gradient (R10)",
        "",
        "| gate | open decisions | lower-sat fraction | upper-sat fraction | "
        "mean quote | price-logit grad norm |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    highs = {"charge": 150.0, "discharge": 300.0}
    for u in UNITS:
        for leg_idx, leg in enumerate(("charge", "discharge")):
            values = []
            for d in audit["10"]["validation"]["days"]:
                values.extend(d["quote_open_values"][u][leg_idx])
            high = highs[leg]
            lower = (sum(1 for v in values if v <= 0.01 * high)
                     / len(values)) if values else 0.0
            upper = (sum(1 for v in values if v >= 0.99 * high)
                     / len(values)) if values else 0.0
            mean = st.mean(values) if values else float("nan")
            grad = stg["10"][u]["price_logit_grad_norm_by_leg"][leg_idx]
            lines.append(f"| {u}.{leg} | {len(values)} | {lower:.2f} | "
                         f"{upper:.2f} | {mean:.2f} | {grad:.2e} |")
    lines += [
        "",
        "## ESS17 discharge and ESS14 branches",
        "",
        "| day | R2 ESS17 | R10 ESS17 | R2 ESS14 | R10 ESS14 | R2 ESS27 | "
        "R10 ESS27 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for pid in ("2022-12-30|2025-12-31", "2022-12-31|2025-12-28",
                "2022-12-13|2025-12-11", "2022-10-26|2025-10-24",
                "2022-07-23|2025-07-13"):
        def get(n):
            for d in audit[str(n)]["validation"]["days"]:
                if d["pair_id"] == pid:
                    return d["delta"]
            return None
        r2, r10 = get(2), get(10)
        lines.append(f"| {pid} | {r2['ESS17']:.1f} | {r10['ESS17']:.1f} | "
                     f"{r2['ESS14']:.1f} | {r10['ESS14']:.1f} | "
                     f"{r2['ESS27']:.1f} | {r10['ESS27']:.1f} |")
    lines += [
        "",
        "## Critic grid and actual settlement at the fixed state",
        "",
        "| ckpt | day | target | leg | quote | min Q | twin gap | dQ/dquote | "
        "target payoff | fleet payoff |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for ckpt, day, target, leg in (
            ("best", "val", "ESS17", "discharge"),
            ("final", "val", "ESS17", "discharge"),
            ("best", "val", "ESS14", "charge"),
            ("final", "val", "ESS14", "charge"),
            ("final", "val", "ESS14", "discharge")):
        for r in focus_rows(ckpt, day, target, leg):
            lines.append(
                f"| {ckpt} | {day} | {target} | {leg} | "
                f"{float(r['quote']):.1f} | {float(r['q_min']):.3f} | "
                f"{float(r['twin_gap']):.3f} | {float(r['dq_dquote']):.2e} | "
                f"{float(r.get('target_payoff', 'nan')):.5f} | "
                f"{float(r.get('fleet_payoff', 'nan')):.5f} |")
    lines += [
        "",
        "## Findings",
        "",
        "- **ESS17 discharge at 1.19 CNY/MWh is not the fleet-loss source.** "
        "ESS17 is positive on the key validation days and in the final round "
        "mean. At the audited state, the target payoff is highest at the "
        "near-zero quote; raising the discharge quote reduces the accepted "
        "energy and the payoff. The 264 lower-bound saturations are a "
        "low-spread, near-idle behavior, not a proven loss. The critic Q is "
        "nearly flat there; it does not strongly reward the high quote.",
        "- **ESS14 and ESS27 carry the tail losses.** On 2022-12-30 the final "
        "deltas are ESS14 −179.2 and ESS27 −239.1; on 2022-12-31 they are "
        "−41.4 and −205.0. The final discharge quotes are high (ESS14 mean "
        "194.2, ESS27 mean 222.7) and the charge gates are partial or closed. "
        "The critic grid at the audited state is nearly flat in the quote and, "
        "where it ranks, slightly prefers higher discharge quotes while the "
        "actual settlement is lower. The critic does not strongly penalize the "
        "high-quote drift.",
        "- **ESS14 charge gate 0.39:** the deterministic gate is open on only "
        "39% of validation decisions, with a 48.9 mean charge quote. The same "
        "actor's discharge quote swings from 0.49 to 299.9. This is a "
        "mixed/partial gate, not a permanently closed one; it contributes to "
        "the ESS14 loss through the discharge branch rather than the charge "
        "gate alone.",
        "- **Acceptance:** the saved validation audit records aggregate "
        "charge/discharge/cycles, not per-block acceptance. The fixed-state "
        "clearing grid shows the ESS17 quote is accepted only at near-zero "
        "energy at q=0 and loses the target payoff as q rises; the ESS14 "
        "discharge payoff is higher at low q. A full per-block acceptance "
        "table would require a new diagnostic clear, which this phase does "
        "not run.",
        "",
        "The row-level grid is `critic_gate_action_grid.csv`.",
        "",
    ]
    (REPORTS / "continuous_quote_branch_audit.md").write_text(
        "\n".join(lines), encoding="utf-8")

    # ------------------------------------------------------------------
    # B/C2/critic alignment
    # ------------------------------------------------------------------
    c2d = c2["records"]["final_round_10"]["day"]
    lines = [
        "# B, C2 and Critic alignment audit (P1)",
        "",
        "The audited day is a validation day. The B/C2 identity and the "
        "per-block shaping are from the actual actor schedules; the Critic "
        "metrics are computed on a fixed 128-transition replay sample at each "
        "checkpoint without an optimizer step.",
        "",
        "## C2 versus B",
        "",
        "| ESS | B at block 23 | max |shaping| per block | mean shaping | "
        "C2−B identity error |",
        "|---|---:|---:|---:|---:|",
    ]
    for u in UNITS:
        identity = c2d["identity"][u]
        lines.append(
            f"| {u} | {c2d['b_reward_by_block'][u][23]:.3f} | "
            f"{max(abs(x) for x in c2d['c2_shaping_by_block'][u]):.3f} | "
            f"{st.mean(c2d['c2_shaping_by_block'][u]):.3f} | "
            f"{identity['b_vs_c2_error']:.2e} |")
    lines += [
        "",
        "The C2 shaping is a per-block credit redistribution. Its discounted "
        "sum telescopes to the policy-independent boundary (zero here); the "
        "day return equals the B return. On the final-round audited day the "
        "ESS17 shaping reaches 14.2 while its B at block 23 is 0.53; in the "
        "best-round audit the comparable B at block 23 was 0.12. The shaping "
        "therefore dominates individual blocks for a low-B ESS, but it does "
        "not change the total day objective. This is a real credit-assignment "
        "effect, not a C2 scale override of the day's money.",
        "",
        "## Critic and actor metrics",
        "",
        "| round | ESS | critic loss | target Q | Q1 | Q2 | twin gap | "
        "buffer reward mean | actor loss |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for n in range(1, 11):
        for u in UNITS:
            s = stg[str(n)][u]
            lines.append(
                f"| {n} | {u} | {s['critic_loss']:.3f} | "
                f"{s['target_q_mean']:.3f} | {s['q1_mean']:.3f} | "
                f"{s['q2_mean']:.3f} | {s['q_gap']:.3f} | "
                f"{s['buffer_reward_mean']:.5f} | {s['actor_loss']:.3f} |")
    lines += [
        "",
        "## Findings",
        "",
        "- **C2 does not dominate the day objective.** The C2−B identity "
        "closes at ~3e-15; the boundary is zero; the total discounted return "
        "is the B return. The shaping dominates individual early blocks for "
        "the low-B ESS, which is the intended credit redistribution, not a "
        "change of the economic target.",
        "- **Q scale is stable.** Target Q and current Q stay in a small "
        "range (roughly −0.6 to +0.03); the buffer reward mean is about −0.001; "
        "the critic loss is small; the twin gap is 0.001–0.045. There is no "
        "exploding Q or persistent twin divergence.",
        "- **The problem is not a C2 scale drift.** The actor loss is small "
        "and the Q values are flat; the gate/policy collapse and the "
        "high-quote drift are better explained by the threshold and the "
        "critic's flat action-value ranking than by C2 magnitude.",
        "- **The critic's flat ranking is a concern.** At the audited state "
        "the min Q changes by only a few hundredths across the quote grid, "
        "while the actual settlement changes by orders of magnitude; for the "
        "discharge legs the critic can even rank the high quote above the low "
        "one when the settlement is worse. A critic that is nearly flat in "
        "the action cannot strongly correct the quote branch.",
        "",
        "No C2, potential, gamma or scale change is proposed or applied here.",
        "",
    ]
    (REPORTS / "b_c2_critic_alignment_audit.md").write_text(
        "\n".join(lines), encoding="utf-8")

    print("wrote validation_degradation_matrix.csv and 3 P1 reports")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
