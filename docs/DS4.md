# DS4

## Project Overview

DwarfStar 4 (ds4) is a self-contained C inference engine specifically for DeepSeek V4 Flash. It is not a generic GGUF runner. The codebase is pure C99 (no C++) with Objective-C only where Metal requires it. GPU backends are Metal (primary, macOS) and CUDA (Linux). The CPU backend exists only for correctness/debug reference.

The design philosophy is "single-model integration focused local AI": always target the best current open weights model that is practically fast on high-end personal machines (96-128GB+ RAM Macs, DGX Spark). The specific model may change over time, but the constraint stays narrow. The project uses asymmetric 2/8-bit quantization (only routed MoE experts are quantized to 2-bit; shared experts, projections, and routing stay at full precision).

## Build Commands

```sh
make                       # macOS Metal (default)
make cuda-spark            # Linux CUDA, DGX Spark / GB10
make cuda-generic          # Linux CUDA, other local GPUs
make cuda CUDA_ARCH=sm_N   # Linux CUDA, explicit arch
make cpu                   # CPU-only diagnostics build (no GPU)
make clean
```

Build produces four binaries: `ds4` (CLI), `ds4-server` (HTTP API server), `ds4-bench` (speed benchmarking), `ds4-eval` (capability evaluation).

## Testing

```sh
make test                          # build and run all tests (ds4_test)
./ds4_test --server                # API parsing, chat rendering, tool-call parsing, KV bookkeeping
./ds4_test --logprob-vectors       # tokenizer/attention/logits regression against official vectors
./ds4_test --long-context          # long-context fact-recall regression
./ds4_test --tool-call-quality     # DSML tool-call emission (needs model)
./ds4_test --metal-kernels         # isolated Metal kernel numeric checks
make cuda-regression               # CUDA-specific regression (Linux only)
```

Override default model path with `DS4_TEST_MODEL=/path/to/model.gguf`. The test source is `tests/ds4_test.c` which `#include`s `ds4_server.c` to test server internals.

## Architecture

The codebase is intentionally flat — a handful of large C files, not a framework:

- **`ds4.c` + `ds4.h`**: Core engine. Model loading (mmap-backed GGUF), tokenizer, CPU reference inference, Metal/CUDA graph scheduling, session management, disk-cache payload serialization. `ds4_engine` is the loaded model; `ds4_session` is one mutable inference timeline with live KV cache and logits.
- **`ds4_cli.c`**: Interactive REPL (linenoise-based) and one-shot prompt mode. Manages multi-turn chat transcript and live KV checkpoint.
- **`ds4_server.c`**: OpenAI/Anthropic/Responses-compatible HTTP API. Worker queue, SSE streaming, DSML tool-call mapping and exact replay, disk KV cache policy. Endpoints: `/v1/chat/completions`, `/v1/responses`, `/v1/messages`, `/v1/completions`, `/v1/models`.
- **`ds4_metal.m`**: Objective-C Metal runtime and kernel dispatch wrappers.
- **`ds4_cuda.cu`**: CUDA kernel implementations.
- **`ds4_gpu.h`**: Shared GPU backend interface between Metal and CUDA.
- **`ds4_bench.c`**: Speed benchmarking at context frontiers (incremental prefill + generation probes).
- **`ds4_eval.c`**: Capability regression suite (92 embedded GPQA/SuperGPQA/AIME/COMPSEC items).

Supporting libraries (vendored): `rax.c/h` (radix tree, used by server for tool-call replay maps), `linenoise.c/h` (line editing for CLI).

### Key Subsystems

- **`metal/*.metal`**: Metal compute kernels (attention, RoPE, MoE routing, quantized matmul, norms, etc.).
- **`gguf-tools/`**: Offline GGUF generation, imatrix collection, quantization, and quality scoring against official DeepSeek continuations.
- **`dir-steering/`**: Directional steering vector tools and examples.
- **`speed-bench/`**: Benchmark CSV data and `plot_speed.py` for graph generation.
- **`tests/test-vectors/`**: Official continuation vectors for regression checks.

### Session / KV Cache Model

A `ds4_session` owns the live KV cache. Callers provide full token prefixes and `ds4_session_sync()` determines whether to reuse, extend, or rebuild graph state. The server maintains one in-memory session at a time; the disk KV cache (`--kv-disk-dir`) lets different sessions survive switches and restarts. Cache keys are SHA1 of rendered byte prefixes.

### Tool Call Flow

The server renders OpenAI/Anthropic tool schemas into DeepSeek's DSML format. Generated DSML tool calls are parsed back into API tool-call objects. Exact DSML replay (keyed by tool ID in a radix tree) preserves KV prefix alignment across turns. Canonicalization is the fallback when exact replay data is missing.

## Code Conventions

- C99 only. No C++.
- Public API boundary is `ds4.h` — CLI/server code must not depend on tensor internals.
- Model loading is mmap-backed; do not eagerly copy the full GGUF.
- Correctness before speed. Do not introduce faster paths with unexplained drift.
- CPU backend is reference/debug only. Avoid large CPU inference on macOS (kernel VM bug can crash the system).
- No permanent semantic variants behind flags; diagnostic switches are fine.
- The `DS4_NO_GPU` preprocessor define builds the CPU-only variant.

## Quality / Speed Regression

Before submitting changes to inference backends, verify both correctness and speed:

```sh
# Correctness
make test

# Speed (before/after CSV comparison)
./ds4-bench -m ds4flash.gguf --prompt-file speed-bench/promessi_sposi.txt \
  --ctx-start 2048 --ctx-max 65536 --step-incr 2048 --gen-tokens 128 \
  --csv /tmp/ds4-speed.csv

# Quantization quality (compare NLL scores)
make -C gguf-tools quality-score
gguf-tools/quality-testing/score_official MODEL.gguf \
  gguf-tools/quality-testing/data/manifest.tsv /tmp/scores.tsv 4096
```
