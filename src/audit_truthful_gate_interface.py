# audit_truthful_gate_interface.py
"""Contract checks for the gate policy adapter. Pure torch; no env/training."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

import truthful_gate_actor_adapter as ga

OUT = "data/processed/v2_cy_jn_2025/pilot/gate_interface_test_results.json"


def run() -> dict:
    checks = {}
    closed = ga.forward_policy(torch.tensor([[-4.0, -4.0, 0.0, 0.0]]))
    checks["closed_is_exact_truth"] = closed["quotes"].tolist() == [[0.0, 0.0]]
    opened = ga.forward_policy(torch.tensor([[3.0, 3.0, 0.0, 0.0]]))
    checks["open_is_legal"] = bool(
        (opened["quotes"] >= 0).all()
        and opened["quotes"][0, 0] <= ga.LEG_HIGHS[0]
        and opened["quotes"][0, 1] <= ga.LEG_HIGHS[1])
    checks["gates_are_independent"] = bool(
        ga.forward_policy(torch.tensor([[-1.0, 1.0, 0.0, 0.0]]))["quotes"]
        .tolist() == [[0.0, 150.0]])
    checks["evaluation_is_deterministic"] = torch.equal(
        ga.target_quotes(torch.tensor([[1.0, -1.0, 0.3, -0.3]])),
        ga.target_quotes(torch.tensor([[1.0, -1.0, 0.3, -0.3]])))
    a = ga.forward_policy(torch.zeros(8, 4), train=True,
                          generator=torch.Generator().manual_seed(5))
    b = ga.forward_policy(torch.zeros(8, 4), train=True,
                          generator=torch.Generator().manual_seed(5))
    checks["training_sampling_reproducible"] = torch.equal(
        a["hard_gates"], b["hard_gates"])
    out = torch.tensor([[-10.0, -10.0, 0.0, 0.0]], requires_grad=True)
    res = ga.forward_policy(out, train=True,
                            generator=torch.Generator().manual_seed(0))
    checks["closed_sample_is_hard_zero"] = res["hard_gates"].tolist() == [
        [0.0, 0.0]]
    res["quotes"].sum().backward()
    checks["st_gate_gradient_nonzero"] = bool(out.grad[0, 0].item() != 0.0)
    checks["closed_price_gradient_zero"] = bool(out.grad[0, 2].item() == 0.0)
    open_out = torch.tensor([[2.0, 2.0, 0.0, 0.0]], requires_grad=True)
    ga.forward_policy(open_out)["quotes"].sum().backward()
    checks["open_price_gradient_nonzero"] = bool(
        open_out.grad[0, 2].item() > 0.0)
    checks["critic_action_is_two_quotes"] = ga.critic_action(
        torch.tensor([[1.0, 0.0, 0.0, 0.0]])).shape == (1, 2)
    checks["target_is_evaluation_action"] = torch.equal(
        ga.target_quotes(torch.tensor([[1.0, -1.0, 0.2, 0.2]])),
        ga.forward_policy(torch.tensor([[1.0, -1.0, 0.2, 0.2]]))["quotes"])
    modes = [int(m) for m in ga.forward_policy(
        torch.tensor([[1.0, 1.0, 0.0, 0.0], [-1.0, 1.0, 0.0, 0.0],
                      [1.0, -1.0, 0.0, 0.0], [-1.0, -1.0, 0.0, 0.0]]))["mode"]]
    checks["per_unit_modes_cover_four"] = sorted(modes) == [0, 1, 2, 3]
    checks["fleet_mode_range"] = ga.fleet_mode([3, 3, 3, 3]) == 255
    monitor = ga.GateMonitor(
        probs=a["probs"][:1], sampled=a["hard_gates"][:1],
        executed=a["quotes"][:1], price_logits=a["price_logits"][:1],
        price_quotes=a["price_quotes"][:1], mode=a["mode"][:1])
    row = monitor.row("ESS9")
    checks["monitor_fields_present"] = all(k in row for k in (
        "gate_prob_charge", "gate_sampled_charge", "gate_prob_discharge",
        "gate_sampled_discharge", "price_logit_charge",
        "price_logit_discharge", "executed_charge", "executed_discharge",
        "near_upper_charge", "near_lower_charge", "mode"))
    warm = ga.forward_policy(torch.zeros(1, 4), temperature=5.0)["probs"]
    cold = ga.forward_policy(torch.zeros(1, 4), temperature=1.0)["probs"]
    checks["temperature_moves_probability_toward_half"] = bool(
        abs(float(warm[0, 0]) - 0.5) <= abs(float(cold[0, 0]) - 0.5) + 1e-12)
    return {"checks": checks, "all_pass": all(checks.values())}


def main(argv=None) -> int:
    payload = run()
    path = Path.cwd() / OUT
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"wrote {path}")
    return 0 if payload["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
