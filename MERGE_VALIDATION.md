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
- Timing mode:
  - reuse the existing benchmark configs and input shapes
  - disable `profile` and `dump_memory`
  - reason: `fwd_ms` / `fwd_bwd_ms` are measured before profiler trace merge, so
    profiling only adds long post-processing time and can hide kernel timing
- Local FA override:
  - `PYTHONPATH=/home/jovyan/dev/gemma4/attention-core:/home/jovyan/dev/gemma4/flash-attention/hopper`
  - `ATTENTION_CORE_FA3_HOPPER_DIR=/home/jovyan/dev/gemma4/flash-attention/hopper`

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

| Variant | fwd_ms | fwd_bwd_ms |
| --- | ---: | ---: |
| installed `403feae` | `2.2998` | `11.7608` |

Isolated `varlen_fa3` only, same shape, reduced timing loop (`warmup=2`, `iters=5`):

| Variant | status |
| --- | --- |
| installed `403feae` | `fwd_ms=2.1509`, `fwd_bwd_ms=11.2430` |
| local merged | `timeout after 80s wall clock` |

Conclusion:

- Local merged `varlen_fa3` has a performance or liveness regression on the
  standard varlen benchmark path.
- This is not a profiler artifact. The timeout was reproduced with profiler and
  memory dump disabled.

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
- Sink paths: acceptable
- Arbitrary-mask precision path: restored to expected baseline
- Standard varlen FA3 benchmark path: blocked by a local merged regression /
  hang and must be fixed before treating the merge as performance-clean
