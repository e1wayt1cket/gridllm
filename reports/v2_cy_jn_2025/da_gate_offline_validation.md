# D+A offline and end-to-end validation (P0)

Executed by `src/audit_da_gate_offline.py`; all 64 checks passed. Data:
`da_gate_offline_validation.json`.

## Mapping

| check | result |
|---|---|
| monotone on 400,001 logits in [−100,100] | pass |
| bounded in [0,1], quote in [0,h] | pass |
| finite outputs and derivatives | pass |
| no negative quote, no quote clipping | pass |
| gate closed → exactly 0 | pass |
| gate open → legal and covers the interval | pass |
| analytic vs central difference | pass; ≤1e-8, ≤1e-5 at the joint |
| high-region minimum gradient | 9.66 CNY/logit vs sigmoid 0.897 (10.77×) |
| logit for 258 / 299 | 1.837 / 2.575 |

## Gate

| episode | temperature |
|---:|---:|
| 0 | 4.000 |
| 26 | 3.264 |
| 53 | 2.500 |
| 79 | 1.764 |
| 106 | 1.000 |
| 159 | 1.000 |

ST gradients are finite at logits −4, 0, 4, 8 and at τ=1 and τ=4. At a
saturated logit 4 the τ=4 gradient is 5.13 against 1.84 at τ=1; at logit 8 it
is 2.74 against 0.035. At the midpoint τ=4 is 6.53 against 26.10, the expected
temperature trade. Deterministic evaluation uses the sign of the logit, and
the Bernoulli frequency over 8,000 gate draws at p=0.5 is 0.4996.

## Sampler

A 48-transition buffer (two days) was sampled with 16 non-empty joint cells,
cell counts 1–8, median 3. The per-transition probability range was
0.0171–0.0293 against a uniform 0.0208; the uniform-ratio range was
0.71–1.22; the effective sample size was 46.85 (97.6% of the buffer). The
sampled rows match their buffer indices exactly; the probabilities sum to one
and respect the cap; the non-empty cells and ESS are recorded. The final
pilot buffer is larger (see the D2 report).

## End-to-end

One complete day with forced exploration (epsilon=1):

| check | result |
|---|---|
| clears | 24/24 |
| all OPTIMAL | yes |
| voltage / line violation cells | 0 / 0 |
| terminal SOC max error | 0.0 |
| B delta identity | pass |
| B paid only at block 23 | pass |
| C2 discounted identity | pass for all units |
| potential starts at zero | pass |
| executed quotes legal | pass |
| stored gates binary | pass |
| explored legs | 75 |
| replay length | 24 |
| v3 save/restore | pass; open-gate action round-trips |
| v2 checkpoint refused | pass |
| sampler/DA metadata saved | pass |

## Boundary

The validation writes only a temporary checkpoint, which is deleted after the
round trip. It does not read H2 or the legacy test. The pilot run is separate.
