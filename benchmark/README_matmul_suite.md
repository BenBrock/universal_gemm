# DTensor / Universal GEMM Benchmark Suite

This suite has two phases:

- Phase 1: run distributed experiments and store machine-readable results
- Phase 2: render an interactive HTML report from stored results

## Phase 1: Run Experiments

Run the full default suite (GPT-3, square, and odd-dimension experiments):

```bash
python universal_gemm/benchmark/run_matmul_experiments.py \
  --nproc-per-node 4 \
  --replication-factors 1,2 \
  --partition-triples "row,row,row;row,column,row;block,block,block" \
  --theoretical-tflops 312
```

Defaults include:

- Methods: `dtensor,stationary_c,stationary_b`
- GPT-3 expansion: `m in {1024,2048,4096,8192}`, `k=12288`, `n=49152`
- GPT-3 shrinking: same `m`, with `k=49152`, `n=12288`
- Square sizes: `1024,2048,4096,8192`
- Odd-dimension large values: `4096,8192,16384,32768` with the other dims fixed at `1024`

Output artifacts are written to:

- `results.json` (full metadata + all measurements)
- `results.csv` (flat tabular export)
- `runs/<case>/torchrun.log` (per-case execution logs)

Quick smoke run:

```bash
python universal_gemm/benchmark/run_matmul_experiments.py \
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

## Phase 2: Render HTML Report

Generate an interactive report from results:

```bash
python universal_gemm/benchmark/render_benchmark_report.py \
  --results-json /path/to/results.json \
  --output-html /path/to/report.html
```

Optional initial filters:

```bash
python universal_gemm/benchmark/render_benchmark_report.py \
  --results-json /path/to/results.json \
  --methods dtensor,stationary_c \
  --groups gpt3_expansion,gpt3_shrinking,square
```

The report includes:

- TFLOPs (median) curves for each method/partition/replication series
- Dashed theoretical-max TFLOPs line when `--theoretical-tflops` is provided
- Interactive method toggles and experiment section toggles
