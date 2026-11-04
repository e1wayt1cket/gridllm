# Winter loss block reconstruction (P1)

Frozen R2 best and R10 final checkpoints on the three validation days. The B1 block payoffs are the terminal-neutral rolling-truthful arm. Every block uses the deterministic policy; no SOC forcing.

| ckpt | day | ESS | worst blocks (block: B1 diff) | day ΔG | charge/discharge MWh |
|---:|---|---|---|---:|---|
| 2 | 2022-12-13|2025-12-11 | ESS14 | 17: -35.4; 22: -26.9; 23: -22.5 | 4.3 | 1.20/1.01 |
| 2 | 2022-12-13|2025-12-11 | ESS27 | 12: -88.2; 3: -70.1; 16: -42.3 | -29.7 | 1.00/0.84 |
| 2 | 2022-12-30|2025-12-31 | ESS14 | 23: -30.8; 22: -17.0; 13: -9.7 | 11.8 | 1.10/0.93 |
| 2 | 2022-12-30|2025-12-31 | ESS27 | 23: -24.3; 11: -17.0; 8: -14.0 | 4.2 | 1.06/0.90 |
| 2 | 2022-12-31|2025-12-28 | ESS14 | 23: -33.2; 16: -22.3; 22: -19.3 | -3.2 | 1.40/1.19 |
| 2 | 2022-12-31|2025-12-28 | ESS27 | 0: -50.6; 1: -23.0; 23: -14.9 | 0.6 | 0.96/0.81 |
| 10 | 2022-12-13|2025-12-11 | ESS14 | 3: -71.5; 17: -71.5; 16: -66.3 | -56.7 | 0.70/0.59 |
| 10 | 2022-12-13|2025-12-11 | ESS27 | 12: -88.2; 3: -80.4; 16: -42.3 | -115.9 | 0.60/0.50 |
| 10 | 2022-12-30|2025-12-31 | ESS14 | 10: -111.1; 14: -43.9; 11: -41.9 | -179.2 | 0.70/0.59 |
| 10 | 2022-12-30|2025-12-31 | ESS27 | 10: -128.1; 14: -52.3; 11: -49.0 | -239.1 | 0.70/0.59 |
| 10 | 2022-12-31|2025-12-28 | ESS14 | 17: -98.1; 0: -47.6; 1: -35.1 | -41.4 | 0.14/0.12 |
| 10 | 2022-12-31|2025-12-28 | ESS27 | 17: -108.1; 0: -53.4; 1: -39.5 | -205.0 | 0.66/0.56 |

## Findings

- The losses concentrate in the high-quote discharge blocks: the policy's per-block payoff is below B1 exactly where ESS14/ESS27 discharge at high quotes, and the SOC/charge path then has to recover.
- The day B reward is paid only at block 23; the per-block `diff_vs_b1` is the cash contribution, and its sum is the day ΔG. The B reward at 23 is ΔG / fixed_scale.
- The C2 shaping is recorded per block. It redistributes credit; it does not change the day ΔG.
- The winter tail loss is not an ESS17 boundary effect: ESS17 is not in this table; the two loss agents are ESS14 and ESS27.

Full per-block data: `winter_block_reconstruction.csv`.

## R10 mechanism detail

| day | ESS | block | discharge quote | p_dis MWh | SOC | B1 payoff | policy payoff | diff |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 2022-12-30 | ESS14 | 10 | 299.37 | 2.9e−8 | 0.900 | 111.11 | 6e−6 | −111.11 |
| 2022-12-30 | ESS27 | 10 | 282.13 | 3.3e−8 | 0.900 | 128.14 | 8e−6 | −128.14 |
| 2022-12-13 | ESS14 | 3 | 299.66 | 1.0e−8 | 0.500 | 71.55 | −2e−6 | −71.55 |
| 2022-12-13 | ESS14 | 17 | 160.58 | 8.1e−8 | 0.900 | 71.46 | 1.5e−5 | −71.46 |
| 2022-12-13 | ESS27 | 12 | 274.19 | 2.6e−9 | 0.673 | 88.21 | −1e−6 | −88.21 |
| 2022-12-31 | ESS14 | 17 | 272.50 | 5.6e−9 | 0.440 | 98.11 | 7e−7 | −98.11 |
| 2022-12-31 | ESS27 | 17 | 258.43 | 4.8e−8 | 0.880 | 108.13 | 1.3e−5 | −108.13 |

The effective discharge offer includes the 100 CNY/MWh cycle cost. The high
quote plus cycle cost is at or near the block LMP (block means 287–425
CNY/MWh), and the policy's dispatch collapses to numerical zero while B1
(effective offer 100) captures the 71–128 CNY block payoff. The high quote is
the direct cause of the lost block; the winter price is the state that makes
it material. The battery is often left at SOC 0.90, so the loss is also a
forgone discharge opportunity, not an over-charge.

The R2 best checkpoint has the same pattern but smaller losses and lower
quotes (e.g. 12-30 block 11/22/23: −30.8/−17.0/−24.3 at ESS14/27); the R10
checkpoint raises the quote and the loss. The C2 shaping on the loss blocks is
−8.7 to −3.7 CNY, recorded per block; the day B reward is at block 23 and does
not change the cash interpretation of these per-block differences.
