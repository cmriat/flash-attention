# Flash-Attention Merge Validation

Date: 2026-04-26

This note records the local validation status for the merged
`/home/jovyan/dev/gemma4/flash-attention` tree against the FA3 package already
installed in the `attention-core` pixi environment.

## Baselines

- Installed package:
  - `flash-attn-3 == 2026.3.12`
  - build tag: `*cmriat_main_403feae*`
- Local merged tree:
  - repo: `/home/jovyan/dev/gemma4/flash-attention`
  - branch: `main`
  - base commit during validation: `e7abacd`

## Benchmark Method

- Benchmark harness: `attention-core` existing benchmark configs
- Environment: `pixi run -m /home/jovyan/dev/gemma4/attention-core -e dev`
- Local FA override:
  - `PYTHONPATH=/home/jovyan/dev/gemma4/flash-attention/hopper`
- Note:
  - the standard varlen regression was reproduced both through the benchmark
    harness and through a direct `flash_attn_varlen_func` forward+backward smoke
    test
  - the fix was validated first on direct `D=96/192` varlen backward, then on
    the original `example_varlen_core.py` benchmark config

## Results

### 1. Dense FA3

Source config:

- `/home/jovyan/dev/gemma4/attention-core/benchmark/configs/example_fixlen_core.py`

Shape:

- `B=2, S=2048, H=32, D=128, bf16`

`fa3` rows:

| Variant | fwd_ms | fwd_bwd_ms | Delta fwd | Delta fwd_bwd |
| --- | ---: | ---: | ---: | ---: |
| installed `403feae` | `0.2091` | `0.9806` | `-` | `-` |
| local merged | `0.2100` | `0.9296` | `+0.41%` | `-5.20%` |

Conclusion:

- Dense FA3 is effectively unchanged on forward.
- Forward + backward is slightly faster on the local merged tree.

### 2. Standard Varlen FA3

Source config:

- `/home/jovyan/dev/gemma4/attention-core/benchmark/configs/example_varlen_core.py`

Shape:

- `B=1, S=4096, H=128, D=192, rope_dim=64, bf16`

Installed package full-config row:

| Variant | fwd_ms | fwd_bwd_ms | Delta fwd | Delta fwd_bwd |
| --- | ---: | ---: | ---: | ---: |
| installed `403feae` | `2.2998` | `11.7608` | `-` | `-` |
| local merged (fixed) | `2.2990` | `11.3059` | `-0.03%` | `-3.87%` |

Root cause and fix:

- The regression was not in the attention-core wrapper.
- Direct local Hopper repro showed:
  - forward for `D=96/192` finished normally
  - backward for `D=96/192` stalled on the merged tree
- The working baseline at `403feae` used explicit SM90 backward dispatch bodies
  for `hdim96` and `hdim192`.
- The merged tree had moved those paths onto the unified `run_mha_bwd_<...>`
  launcher; restoring the old explicit dispatch bodies for `hdim96/192`
  removed the stall immediately.

Validation after fix:

- Direct smoke:
  - `D=96, S=512`: `1.2690s`
  - `D=96, S=2048`: `1.1616s`
  - `D=192, S=2048`: `0.0175s`
- Benchmark:
  - `example_varlen_core.py` now completes normally
  - `varlen_fa3`: `fwd_ms=2.2990`, `fwd_bwd_ms=11.3059`

Conclusion:

- Standard varlen FA3 benchmark is no longer blocked.
- After restoring the `hdim96/192` SM90 backward dispatch, local merged
  performance is back in family with the installed `403feae` package.

### 3. Sink Paths

Source config:

- `/home/jovyan/dev/gemma4/attention-core/benchmark/configs/example_varlen_sink_native_compare.py`

Shape:

- `B=3, S=16384, H=128, D=128, normal-split varlen, bf16`

Timing loop for this comparison:

- `warmup=1`, `iters=3`

Rows:

| Core | Installed fwd_ms | Local fwd_ms | Delta fwd | Installed fwd_bwd_ms | Local fwd_bwd_ms | Delta fwd_bwd |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `varlen_fa3_with_sink_attentioncore` | `6.0646` | `5.9906` | `-1.22%` | `21.1955` | `20.7688` | `-2.01%` |
| `varlen_fa3_native_sink` | `4.3463` | `4.3132` | `-0.76%` | `20.7572` | `19.0433` | `-8.26%` |

Conclusion:

- Sink paths do not show a regression in this benchmark.
- Both the attention-core sink wrapper and native sink kernel are slightly
  faster on the local merged tree.

## Precision Context

The local merged tree already matches the previous arbitrary-mask precision
baseline on the Gemma single-kernel path, and sink parity is restored.

Known remaining issue:

- `attention-core` 8-GPU Gemma AMFA strict fp32-reference end-to-end metrics
  still fail on parameter gradients.
- Current probing indicates that this is a contract problem between:
  - distributed bf16 production-style execution
  - single-card strict fp32 reference
- It is not yet closed as a final pass/fail decision.

## Overall Status

- Dense FA3: acceptable
- Standard varlen FA3: acceptable after restoring `hdim96/192` backward dispatch
- Sink paths: acceptable
- Arbitrary-mask precision path: restored to expected baseline
- Remaining open item:
  - `attention-core` 8-GPU Gemma AMFA strict fp32-reference end-to-end metrics
