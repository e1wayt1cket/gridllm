# Critic gate/action value audit (P0)

Fixed state: block 10 of one training and one validation day. Other agents keep the checkpoint's deterministic quotes. The quote grid varies only the target leg; the other target leg is 0. The critic values are offline; the settlement columns are actual window clearings. They are not a full policy counterfactual.

| ckpt | day | target | leg | gate | quote | Q1 | Q2 | min Q | twin gap | dQ/dquote | gate proxy grad | target payoff | fleet payoff | accepted |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| best | train | ESS14 | charge | closed | 0.0 | -0.718 | -0.725 | -0.725 | 0.007 | 1.132e-03 | 1.080e-11 | 0.00 | 0.00 | True |
| best | train | ESS14 | charge | open_0 | 0.0 | -0.718 | -0.725 | -0.725 | 0.007 | 1.132e-03 | 1.080e-11 | 0.00 | 0.00 | True |
| best | train | ESS14 | charge | open_25 | 37.5 | -0.675 | -0.728 | -0.728 | 0.052 | -2.199e-04 | 6.805e-04 | 0.00 | 0.00 | True |
| best | train | ESS14 | charge | open_50 | 75.0 | -0.652 | -0.703 | -0.703 | 0.051 | 5.535e-04 | -6.259e-04 | 0.00 | 0.00 | True |
| best | train | ESS14 | charge | open_75 | 112.5 | -0.695 | -0.684 | -0.695 | 0.011 | -2.077e-04 | -1.118e-03 | 0.00 | 0.00 | True |
| best | train | ESS14 | charge | open_100 | 150.0 | -0.684 | -0.712 | -0.712 | 0.029 | 1.295e-04 | -5.714e-03 | 0.00 | 0.00 | True |
| best | train | ESS14 | discharge | closed | 0.0 | -0.718 | -0.725 | -0.725 | 0.007 | -1.629e-03 | -1.338e-10 | 0.00 | 0.00 | True |
| best | train | ESS14 | discharge | open_0 | 0.0 | -0.718 | -0.725 | -0.725 | 0.007 | -1.629e-03 | -1.338e-10 | 0.00 | 0.00 | True |
| best | train | ESS14 | discharge | open_25 | 75.0 | -0.666 | -0.733 | -0.733 | 0.067 | 9.154e-04 | -1.622e-02 | 0.00 | 0.00 | True |
| best | train | ESS14 | discharge | open_50 | 150.0 | -0.675 | -0.724 | -0.724 | 0.049 | -6.127e-04 | -3.245e-02 | -0.00 | 0.00 | True |
| best | train | ESS14 | discharge | open_75 | 225.0 | -0.764 | -0.750 | -0.764 | 0.014 | -9.462e-04 | -4.867e-02 | -0.00 | 0.00 | True |
| best | train | ESS14 | discharge | open_100 | 300.0 | -0.835 | -0.865 | -0.865 | 0.030 | -2.407e-03 | -6.489e-02 | -0.00 | 0.00 | True |
| best | train | ESS17 | discharge | closed | 0.0 | -0.575 | -0.538 | -0.575 | 0.037 | -4.879e-04 | -6.778e-12 | 0.00 | 0.00 | True |
| best | train | ESS17 | discharge | open_0 | 0.0 | -0.575 | -0.538 | -0.575 | 0.037 | -4.879e-04 | -6.778e-12 | 0.00 | 0.00 | True |
| best | train | ESS17 | discharge | open_25 | 75.0 | -0.545 | -0.448 | -0.545 | 0.097 | -7.694e-05 | 7.195e-03 | 0.00 | 0.00 | True |
| best | train | ESS17 | discharge | open_50 | 150.0 | -0.525 | -0.446 | -0.525 | 0.078 | 4.616e-05 | -1.178e-02 | -0.00 | 0.00 | True |
| best | train | ESS17 | discharge | open_75 | 225.0 | -0.536 | -0.474 | -0.536 | 0.062 | -1.953e-04 | -5.163e-02 | -0.00 | 0.00 | True |
| best | train | ESS17 | discharge | open_100 | 300.0 | -0.602 | -0.536 | -0.602 | 0.066 | -1.048e-03 | -5.672e-02 | -0.00 | 0.00 | True |
| best | val | ESS14 | charge | closed | 0.0 | -0.687 | -0.677 | -0.687 | 0.010 | 6.199e-04 | 9.963e-12 | 0.00 | 0.00 | True |
| best | val | ESS14 | charge | open_0 | 0.0 | -0.687 | -0.677 | -0.687 | 0.010 | 6.199e-04 | 9.963e-12 | 0.00 | 0.00 | True |
| best | val | ESS14 | charge | open_25 | 37.5 | -0.639 | -0.689 | -0.689 | 0.049 | -1.270e-04 | 5.110e-04 | 0.00 | 0.00 | True |
| best | val | ESS14 | charge | open_50 | 75.0 | -0.620 | -0.660 | -0.660 | 0.039 | 6.715e-04 | 6.675e-04 | 0.00 | 0.00 | True |
| best | val | ESS14 | charge | open_75 | 112.5 | -0.677 | -0.650 | -0.677 | 0.027 | 2.907e-04 | -1.654e-03 | 0.00 | 0.00 | True |
| best | val | ESS14 | charge | open_100 | 150.0 | -0.674 | -0.672 | -0.674 | 0.002 | 1.376e-03 | -6.307e-03 | 0.00 | 0.00 | True |
| best | val | ESS14 | discharge | closed | 0.0 | -0.687 | -0.677 | -0.687 | 0.010 | -6.866e-04 | -3.267e-10 | 0.00 | 0.00 | True |
| best | val | ESS14 | discharge | open_0 | 0.0 | -0.687 | -0.677 | -0.687 | 0.010 | -6.866e-04 | -3.267e-10 | 0.00 | 0.00 | True |
| best | val | ESS14 | discharge | open_25 | 75.0 | -0.635 | -0.701 | -0.701 | 0.066 | 1.391e-05 | -3.963e-02 | 0.00 | 0.00 | True |
| best | val | ESS14 | discharge | open_50 | 150.0 | -0.667 | -0.704 | -0.704 | 0.037 | -4.863e-04 | -7.925e-02 | -0.00 | 0.00 | True |
| best | val | ESS14 | discharge | open_75 | 225.0 | -0.744 | -0.735 | -0.744 | 0.010 | -1.017e-03 | -1.189e-01 | -0.00 | 0.00 | True |
| best | val | ESS14 | discharge | open_100 | 300.0 | -0.817 | -0.864 | -0.864 | 0.047 | -2.251e-03 | -1.585e-01 | -0.00 | 0.00 | True |
| best | val | ESS17 | discharge | closed | 0.0 | -0.552 | -0.509 | -0.552 | 0.043 | -2.254e-04 | 3.351e-12 | 0.00 | 0.00 | True |
| best | val | ESS17 | discharge | open_0 | 0.0 | -0.552 | -0.509 | -0.552 | 0.043 | -2.254e-04 | 3.351e-12 | 0.00 | 0.00 | True |
| best | val | ESS17 | discharge | open_25 | 75.0 | -0.524 | -0.405 | -0.524 | 0.119 | 9.895e-05 | 4.794e-03 | 0.00 | 0.00 | True |
| best | val | ESS17 | discharge | open_50 | 150.0 | -0.508 | -0.421 | -0.508 | 0.087 | -1.755e-04 | -1.851e-02 | -0.00 | 0.00 | True |
| best | val | ESS17 | discharge | open_75 | 225.0 | -0.532 | -0.462 | -0.532 | 0.071 | -7.758e-04 | -5.292e-02 | -0.00 | 0.00 | True |
| best | val | ESS17 | discharge | open_100 | 300.0 | -0.599 | -0.522 | -0.599 | 0.077 | -1.105e-03 | -7.960e-02 | -0.00 | 0.00 | True |
| final | train | ESS14 | charge | closed | 0.0 | -0.199 | -0.202 | -0.202 | 0.004 | 2.099e-05 | -1.661e-13 | 0.00 | 0.00 | True |
| final | train | ESS14 | charge | open_0 | 0.0 | -0.199 | -0.202 | -0.202 | 0.004 | 2.099e-05 | -1.661e-13 | 0.00 | 0.00 | True |
| final | train | ESS14 | charge | open_25 | 37.5 | -0.197 | -0.202 | -0.202 | 0.004 | 1.901e-05 | -2.015e-05 | 0.00 | 0.00 | True |
| final | train | ESS14 | charge | open_50 | 75.0 | -0.197 | -0.201 | -0.201 | 0.004 | 2.109e-05 | -4.029e-05 | 0.00 | 0.00 | True |
| final | train | ESS14 | charge | open_75 | 112.5 | -0.196 | -0.200 | -0.200 | 0.004 | 1.891e-05 | -6.044e-05 | 0.00 | 0.00 | True |
| final | train | ESS14 | charge | open_100 | 150.0 | -0.196 | -0.200 | -0.200 | 0.003 | 1.130e-05 | -8.058e-05 | 0.00 | 0.00 | True |
| final | train | ESS14 | discharge | closed | 0.0 | -0.199 | -0.202 | -0.202 | 0.004 | 2.129e-05 | 2.125e-14 | 0.00 | 0.00 | True |
| final | train | ESS14 | discharge | open_0 | 0.0 | -0.199 | -0.202 | -0.202 | 0.004 | 2.129e-05 | 2.125e-14 | 0.00 | 0.00 | True |
| final | train | ESS14 | discharge | open_25 | 75.0 | -0.198 | -0.200 | -0.200 | 0.002 | 3.535e-05 | 4.285e-06 | 0.00 | 0.00 | True |
| final | train | ESS14 | discharge | open_50 | 150.0 | -0.198 | -0.199 | -0.199 | 0.001 | 1.792e-05 | 5.666e-06 | -0.00 | 0.00 | True |
| final | train | ESS14 | discharge | open_75 | 225.0 | -0.196 | -0.197 | -0.197 | 0.001 | 1.544e-05 | 6.232e-06 | -0.00 | 0.00 | True |
| final | train | ESS14 | discharge | open_100 | 300.0 | -0.194 | -0.196 | -0.196 | 0.002 | 1.746e-05 | 1.286e-05 | -0.00 | 0.00 | True |
| final | train | ESS17 | discharge | closed | 0.0 | -0.234 | -0.230 | -0.234 | 0.005 | 1.211e-05 | 6.044e-13 | 0.00 | 0.00 | True |
| final | train | ESS17 | discharge | open_0 | 0.0 | -0.234 | -0.230 | -0.234 | 0.005 | 1.211e-05 | 6.044e-13 | 0.00 | 0.00 | True |
| final | train | ESS17 | discharge | open_25 | 75.0 | -0.233 | -0.226 | -0.233 | 0.007 | 1.462e-05 | 7.548e-05 | 0.00 | 0.00 | True |
| final | train | ESS17 | discharge | open_50 | 150.0 | -0.232 | -0.223 | -0.232 | 0.008 | 2.274e-05 | 2.797e-04 | -0.00 | -0.00 | True |
| final | train | ESS17 | discharge | open_75 | 225.0 | -0.233 | -0.220 | -0.233 | 0.012 | -1.609e-05 | -3.376e-04 | -0.00 | -0.00 | True |
| final | train | ESS17 | discharge | open_100 | 300.0 | -0.234 | -0.217 | -0.234 | 0.017 | -1.609e-05 | -4.155e-04 | -0.00 | -0.00 | True |
| final | val | ESS14 | charge | closed | 0.0 | -0.198 | -0.202 | -0.202 | 0.004 | 2.099e-05 | -2.599e-13 | 0.00 | 0.00 | True |
| final | val | ESS14 | charge | open_0 | 0.0 | -0.198 | -0.202 | -0.202 | 0.004 | 2.099e-05 | -2.599e-13 | 0.00 | 0.00 | True |
| final | val | ESS14 | charge | open_25 | 37.5 | -0.197 | -0.201 | -0.201 | 0.004 | 2.850e-05 | -3.152e-05 | 0.00 | 0.00 | True |
| final | val | ESS14 | charge | open_50 | 75.0 | -0.196 | -0.200 | -0.200 | 0.004 | 2.327e-05 | -6.304e-05 | 0.00 | 0.00 | True |
| final | val | ESS14 | charge | open_75 | 112.5 | -0.196 | -0.199 | -0.199 | 0.004 | 1.892e-05 | -9.456e-05 | 0.00 | 0.00 | True |
| final | val | ESS14 | charge | open_100 | 150.0 | -0.196 | -0.199 | -0.199 | 0.003 | 1.152e-05 | -1.261e-04 | 0.00 | 0.00 | True |
| final | val | ESS14 | discharge | closed | 0.0 | -0.198 | -0.202 | -0.202 | 0.004 | 2.127e-05 | 6.688e-15 | 0.00 | 0.00 | True |
| final | val | ESS14 | discharge | open_0 | 0.0 | -0.198 | -0.202 | -0.202 | 0.004 | 2.127e-05 | 6.688e-15 | 0.00 | 0.00 | True |
| final | val | ESS14 | discharge | open_25 | 75.0 | -0.198 | -0.200 | -0.200 | 0.003 | 3.532e-05 | 6.339e-06 | 0.00 | 0.00 | True |
| final | val | ESS14 | discharge | open_50 | 150.0 | -0.197 | -0.199 | -0.199 | 0.001 | 1.690e-05 | 5.046e-06 | -0.00 | 0.00 | True |
| final | val | ESS14 | discharge | open_75 | 225.0 | -0.196 | -0.197 | -0.197 | 0.001 | 1.544e-05 | 9.299e-06 | -0.00 | 0.00 | True |
| final | val | ESS14 | discharge | open_100 | 300.0 | -0.194 | -0.196 | -0.196 | 0.002 | 1.956e-05 | 2.426e-05 | -0.00 | 0.00 | True |
| final | val | ESS17 | discharge | closed | 0.0 | -0.236 | -0.231 | -0.236 | 0.004 | 1.212e-05 | 7.179e-13 | 0.00 | 0.00 | True |
| final | val | ESS17 | discharge | open_0 | 0.0 | -0.236 | -0.231 | -0.236 | 0.004 | 1.212e-05 | 7.179e-13 | 0.00 | 0.00 | True |
| final | val | ESS17 | discharge | open_25 | 75.0 | -0.234 | -0.228 | -0.234 | 0.007 | 1.415e-05 | 8.309e-05 | 0.00 | 0.00 | True |
| final | val | ESS17 | discharge | open_50 | 150.0 | -0.233 | -0.225 | -0.233 | 0.008 | 2.244e-05 | 2.923e-04 | -0.00 | -0.00 | True |
| final | val | ESS17 | discharge | open_75 | 225.0 | -0.234 | -0.222 | -0.234 | 0.012 | -1.592e-05 | -3.703e-04 | -0.00 | -0.00 | True |
| final | val | ESS17 | discharge | open_100 | 300.0 | -0.235 | -0.219 | -0.235 | 0.016 | -1.609e-05 | -4.937e-04 | -0.00 | -0.00 | True |

## How to read it

- A rising `min Q` with quote while the actual target/fleet payoff falls is evidence that the critic prefers a quote the market does not reward.
- A near-zero `dQ/dquote` means the critic is locally flat in the quote; a large positive/negative value means it actively ranks the quote.
- The gate proxy gradient is the straight-through surrogate at the target actor's gate logit. It is not a true hard-gate derivative.
- The settlement is a one-block rolling-window clearing at the fixed state; it is the correct economic comparison for this state, not a claim about a full-day policy.

Full grid for all four ESS: `critic_gate_action_grid.csv`.

## Findings

- **The critic is nearly flat in the quote.** At the final-checkpoint validation state the ESS17 discharge min Q moves from −0.2355 at q=0 to −0.2348 at q=300, while the actual target payoff falls from 1.8e−4 to 3.0e−6 and the fleet payoff falls from 1.7e−4 to −9e−6. The ESS14 discharge min Q moves from −0.2020 to −0.1962 while the target payoff falls from 2.0e−4 to 1.5e−4. The critic weakly prefers the high quote exactly where the settlement is worse.
- **ESS14 charge is the exception:** both the min Q and the actual settlement slightly favour the higher charge quote at the audited state (−0.2020→−0.1988; 1.98e−4→2.08e−4). The critic is not uniformly wrong.
- **No strong systematic valuation of refusal/high/low.** The Q range across the grid is only a few hundredths; `dQ/dquote` is 1e−5–1e−3; the twin gap is 0.001–0.12. The critic does not strongly value a low/refusal quote either. The flatness is itself the problem: the actor receives little action-value signal against the high-quote drift.
- **The gate closed/open duplicate at q=0:** the critic sees the same quote and gives the same Q; the gate-logit proxy gradient is near zero for a closed gate. The gate is not a separately valued action in the critic.
- **Scope:** one block on one training and one validation day, best and final checkpoints. This is an offline value check, not a full policy counterfactual. The settlement columns are actual window clearings at that state; they are the correct economic comparison for the state, not a day-level claim.
