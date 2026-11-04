# Trainer smoke update and checkpoint audit

No performance claim. The update is the smallest engineering update that proves the loss, gradient and optimizer path are connected.

- updates executed: 17
- environment steps: 24
- replay buffer length: 24
- critic loss finite: True
- actor loss finite: True
- actor parameters changed by the update: True
- restored parameters update identically: True

## Checkpoint round trip

| check | result |
|---|---|
| level | resume |
| deterministic action restored | True |
| replay sample restored (RNG) | True |
| critic/actor losses restored | True |
| parameters updated identically after restore | True |
| counters restored | True |
| date exposure log restored | True |
| all pass | True |

The restored trainer continues from the saved buffer, optimiser moments, RNG and counters; it is not an actors-only file.
