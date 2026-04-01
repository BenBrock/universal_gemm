# Universal GEMM

`universal_gemm` is a prototype workspace for experimenting with DTensor-based
distributed GEMM variants, with a particular focus on one-sided / stationary
algorithms backed by NVSHMEM.

At a high level, the repo contains:

- core helpers for DTensor tile reasoning and execution
- implementations of Stationary-B and Stationary-C execution plans
- example entry points for running distributed matmuls
- a benchmark suite for comparing plain DTensor matmul against the stationary
  methods

## Core Files

- `dtensor_utils.py`: DTensor/NVSHMEM helpers, tile access, execution helpers
- `stationary_b_plan.py`: Stationary-B plan construction and execution
- `stationary_c_plan.py`: Stationary-C plan construction and execution
- `dtensor_scratch.py`: NVSHMEM-backed scratch allocation
- `tile_bounds.py`: tile geometry helpers
- `accumulate_kernels.py`: low-level accumulation helpers
- `dtensor_profile.py`: lightweight profiling/reporting utilities

## Main Entry Points

- `dtensor_example.py`: direct distributed matmul driver
- `benchmark/run_matmul_experiments.py`: benchmark suite runner
- `benchmark/matmul_bench_worker.py`: single-case worker launched under `torchrun`
- `benchmark/render_benchmark_report.py`: render HTML reports from stored results

## Environment Notes

The stationary methods assume:

- CUDA GPUs
- NCCL process groups
- NVSHMEM availability
- DTensor local tensors backed by symmetric memory

For direct stationary execution, set:

```bash
export TORCH_DTENSOR_USE_NVSHMEM=1
```

The benchmark suite sets this automatically for stationary methods and leaves it
unset for plain DTensor runs.

## Running the Main Example

From the repo root:

```bash
torchrun --nproc_per_node=8 ./dtensor_example.py \
  --stationary-method stationary_c \
  --a-partition row \
  --b-partition row \
  --c-partition row \
  --replication 1 \
  --m 1024 \
  --n 1024 \
  --k 1024
```

Supported stationary methods are:

- `stationary_b`
- `stationary_c`
- `auto`

## Running the Benchmark Suite

Quick smoke run:

```bash
python benchmark/run_matmul_experiments.py \
  --nproc-per-node 2 \
  --methods dtensor,stationary_c,stationary_b \
  --replication-factors 1 \
  --partition-triples "row,row,row" \
  --gpt3-m-values "" \
  --square-sizes 128 \
  --odd-large-values 256 \
  --odd-small-value 64 \
  --warmup-iters 0 \
  --measure-iters 1 \
  --max-cases 3
```

Render an HTML report from stored results:

```bash
python benchmark/render_benchmark_report.py \
  --results-json benchmark/results/<suite>/results.json
```

See `benchmark/README_matmul_suite.md` for more benchmark-specific detail.

## Repository State

This repo currently mixes:

- tracked source files
- local benchmark outputs
- local job-generation outputs
- scratch scripts and experiment artifacts

The cleanup goal is to keep the useful local data available while making the
tracked repo easier to understand and safer to work in.
