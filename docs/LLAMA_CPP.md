# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

IMPORTANT: Ensure you've thoroughly reviewed the [AGENTS.md](AGENTS.md) file before beginning any work.

## Build Commands

CMake is the only build system (the Makefile just prints a deprecation error).

```bash
# Configure and build (CPU-only, Release)
cmake -B build
cmake --build build --config Release -j$(nproc)

# With CUDA
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j$(nproc)

# Debug build
cmake -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j$(nproc)

# Binaries go to build/bin/
```

Key CMake options: `GGML_CUDA`, `GGML_METAL`, `GGML_VULKAN`, `GGML_SYCL`, `GGML_RPC`, `LLAMA_BUILD_TESTS`, `LLAMA_BUILD_SERVER`, `LLAMA_BUILD_EXAMPLES`.

## Running Tests

```bash
# Run all ctest tests
cd build && ctest --output-on-failure

# Run a single test by name
cd build && ctest -R test-chat --output-on-failure

# Run full CI locally (CPU-only)
mkdir tmp && bash ./ci/run.sh ./tmp/results ./tmp/mnt

# Server tests (Python, pytest-based)
cd tools/server/tests && pip install -r requirements.txt && ./tests.sh

# Run a single server test
cd tools/server/tests && pytest -v -x unit/test_basic.py

# Slow server tests (tool calls, etc.)
SLOW_TESTS=1 ./tests.sh
```

## Architecture Overview

### Layer Structure

- **ggml** (`ggml/`): Low-level tensor computation library. Defines operators, memory management, and hardware backend interfaces. This is a separate library with its own public API in `ggml/include/`.
  - Backend implementations live in `ggml/src/ggml-cuda/`, `ggml/src/ggml-cpu/`, `ggml/src/ggml-metal/`, etc.
  - Each backend implements the same operator interface; `test-backend-ops` verifies cross-backend consistency.
- **llama** (`src/`, `include/`): Model loading, tokenization, inference, sampling, and KV cache management. Public C API in `include/llama.h`, C++ smart-pointer wrappers in `include/llama-cpp.h`.
- **common** (`common/`): Shared utilities used by tools and examples — argument parsing (`arg.h`), chat templating (`chat.h`), grammar/PEG parsing, JSON schema conversion, sampling wrappers, HTTP helpers, download/HF-cache support.
- **tools** (`tools/`): Production-quality programs — `server`, `cli`, `quantize`, `perplexity`, `llama-bench`, `imatrix`, `gguf-split`, etc.
- **examples** (`examples/`): Simpler demonstration programs and model conversion scripts.

### Key Source Files in `src/`

The llama library is split by concern: `llama-model.*` (model loading/architecture), `llama-context.*` (inference context), `llama-vocab.*` (tokenizer), `llama-sampler.*` (sampling chains), `llama-grammar.*` (constrained decoding), `llama-kv-cache.*` (KV cache implementations), `llama-arch.*` (architecture enum/tensor name mappings), `llama-graph.*` (computation graph building), `llama-batch.*` (batch management).

### Server Architecture

`tools/server/` implements an OpenAI-compatible HTTP server:
- `server-context.*`: Holds the `llama_context` and manages inference slots.
- `server-slot` (in server-context): Abstraction over a single parallel inference sequence.
- `server-routes` (in server-http): Maps HTTP endpoints to server_context operations.
- Web UI is built separately and optionally embedded (controlled by `LLAMA_BUILD_UI`).
- Server tests are Python/pytest in `tools/server/tests/`.

### Matrix Multiplication Convention

ggml uses unconventional matmul: `C = ggml_mul_mat(ctx, A, B)` computes C = B * A^T (not A * B).

## Coding Conventions

- C/C++, no external dependencies beyond vendored single-header libs
- `snake_case` everywhere; naming optimizes for longest common prefix (`number_big` not `big_number`)
- Enum values: `UPPER_CASE`, prefixed with enum name (e.g., `LLAMA_VOCAB_TYPE_BPE`)
- Pattern: `<class>_<method>` where method is `<action>_<noun>` (e.g., `llama_model_init`, `llama_sampler_get_seed`)
- Use `init`/`free` for constructor/destructor naming
- 4-space indentation, brackets on the same line, `void * ptr`, `int & a`
- Use sized integer types (`int32_t`) in public API
- Avoid templates and fancy STL; prefer basic `for` loops
- Filenames: C/C++ lowercase with dashes, Python lowercase with underscores
- New ggml operators must have corresponding `test-backend-ops` test cases
- When modifying server: refer to `tools/server/README-dev.md` for scope and architecture

## Important Resources

- [Build docs](docs/build.md)
- [How to add a new model](docs/development/HOWTO-add-model.md)
- [Server dev docs](tools/server/README-dev.md)
- [PEG parser docs](docs/development/parsing.md)
- [PR template](.github/pull_request_template.md)
- [CONTRIBUTING.md](CONTRIBUTING.md)
