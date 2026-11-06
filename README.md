# AV2 Trajectory Prediction

Vehicle trajectory forecasting on Argoverse 2.

## Current progress

- Constant Velocity baseline completed.
- History-only Transformer training and evaluation completed.
- Checkpoint saving, resuming and loading verified.
- Initial failure-case analysis completed.
- Map conditioning and Conditional Diffusion are planned.

## Experiment setting

- Training: 5,000 scenarios
- Validation: 100 fixed scenarios
- History: 50 frames
- Future: 60 frames
- Sampling interval: 0.1 s
- One predicted trajectory per scenario (K=1)
- Miss: final displacement error greater than 2 m

## Results

| Model | ADE (m) | FDE (m) | MR |
|---|---:|---:|---:|
| Constant Velocity | 5.406 | 14.111 | 92% |
| History Transformer | 3.779 | 9.482 | 95% |

Transformer checkpoint selected by validation FDE at epoch 138.
These are vehicle-only subset results, not full AV2 benchmark results.