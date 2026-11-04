# Truthful-gate policy interface audit (V2-CY-JN-2025)

Adapter: `src/truthful_gate_actor_adapter.py`; contract checks:
`gate_interface_test_results.json` (15/15 pass); tests:
`tests/test_truthful_gate_actor_adapter.py`. It is not wired into
`rl_spec.py`, `rl_bidding.py` or `rl_td3.py`; no training.

## The policy vector and the market action

Per ESS the policy outputs **four** numbers:

```
[gate_logit_charge, gate_logit_discharge, price_logit_charge, price_logit_discharge]
```

Forward action (evaluation):

```
g_k = 1[gate_logit_k > 0]
quote_k = 0                              if g_k = 0
        = h_k * sigmoid(price_logit_k)   if g_k = 1
```

The executed market action is **two quotes** per ESS; the gate selects the
declaration mode and does not represent a physical charge/discharge switch.

## Contract

| interface | definition |
|---|---|
| Actor output | 4 logits per ESS (2 gate + 2 price) |
| Forward quote | hard gate; closed → exact 0; open → \(h\sigma(z)\) in [0,h] |
| Critic input | the **executed 2 quotes** per ESS, not the gate probability |
| Replay buffer | the executed 2 quotes (for the critic target), plus the 4 policy logits, gate probability and sampled gate for the estimator and monitoring |
| Target actor | non-sampling hard gate → two quotes |
| Exploration | Bernoulli(\(\sigma(\text{gate logit}/T)\)) during training, with a seeded generator; deterministic hard gate in evaluation |
| Actor loss | straight-through through the gate; exact sigmoid derivative through the open price branch |

Why the critic sees the quote, not the gate: the market acts on the quote. If
the critic saw the continuous probability while the environment executed a
different hard action, its Q would be trained on an action never taken. The
adapter's `critic_action` is the evaluation/target action, so the critic's
action dimension stays 2 per ESS and its semantics match the market.

## Straight-through is biased

The gate's hard threshold has no derivative. The adapter's surrogate is
`prob * (1 - prob)`, applied only when `train=True`; evaluation has no gate
gradient. **This is a biased estimator and must not be described as an
unbiased deterministic policy gradient.** A Bernoulli/REINFORCE gate remains
the unbiased option (higher variance); Gumbel-Softmax is another biased
surrogate. The choice is not made here.

## Verified properties (`gate_interface_test_results.json`)

- closed gate → exactly `[0,0]` at any finite logit; open gate legal and
  non-negative;
- gates independent per leg; per-unit mode covers 0..3; fleet mode base-4
  0..255;
- evaluation and target actions deterministic;
- training sampling reproducible with a seeded `torch.Generator`;
- closed gate: gate-logit gradient non-zero via the surrogate, price-logit
  gradient zero; open gate: price gradient non-zero;
- critic action shape `(n,2)`;
- monitor row carries gate probability, sampled gate, price logit/quote,
  executed quote, near-lower/upper flags and the joint mode;
- temperature moves the gate probability toward one half.

## Monitoring fields

Per ESS and per gate: probability, sampled value, open frequency (aggregated by
the caller), price logit, price quote, executed quote, boundary proximity, and
the joint mode. Fleet-level: the count of the 256 possible base-4 modes actually
seen, which catches a gate that is permanently open or closed and a unit that
collapses to one behaviour.

## Not done

- No trainer integration, no entropy term, no final estimator choice.
- No empirical test that the gate improves returns; that needs training on
  train/validation and a new holdout.
