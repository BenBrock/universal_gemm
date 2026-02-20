#!/usr/bin/env python3
import argparse
import csv
import dataclasses
import datetime as dt
import json
import os
import pathlib
import socket
import subprocess
import sys
from dataclasses import dataclass

import torch


THIS_DIR = pathlib.Path(__file__).resolve().parent
WORKER = THIS_DIR / "matmul_bench_worker.py"


@dataclass(frozen=True)
class ShapeCase:
    experiment_group: str
    experiment_label: str
    varied_dim: str
    varied_value: int
    m: int
    n: int
    k: int


@dataclass(frozen=True)
class MethodCase:
    method: str
    a_partition: str
    b_partition: str
    c_partition: str
    replication_factor: int


def _parse_csv_list(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def _parse_int_list(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def _parse_partition_triples(value: str) -> list[tuple[str, str, str]]:
    triples: list[tuple[str, str, str]] = []
    for item in value.split(";"):
        item = item.strip()
        if not item:
            continue
        parts = [p.strip() for p in item.split(",")]
        if len(parts) != 3:
            raise ValueError(
                "Partition triples must be formatted as "
                "'a,b,c;a,b,c;...' with each value in {row,column,block}"
            )
        a, b, c = parts
        for x in (a, b, c):
            if x not in {"row", "column", "block"}:
                raise ValueError(f"Invalid partition token {x!r}")
        triples.append((a, b, c))
    return triples


def _default_output_dir() -> pathlib.Path:
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    return THIS_DIR / "results" / f"matmul_suite_{stamp}"


def _build_shape_suite(
    *,
    gpt3_m_values: list[int],
    square_sizes: list[int],
    odd_large_values: list[int],
    odd_small_value: int,
) -> list[ShapeCase]:
    suite: list[ShapeCase] = []

    # GPT-3 MLP expansion: [m x 12k] @ [12k x 48k]
    for m in gpt3_m_values:
        suite.append(
            ShapeCase(
                experiment_group="gpt3_expansion",
                experiment_label="GPT-3 expansion (k=12288, n=49152)",
                varied_dim="m",
                varied_value=m,
                m=m,
                k=12 * 1024,
                n=48 * 1024,
            )
        )

    # GPT-3 MLP shrinking: [m x 48k] @ [48k x 12k]
    for m in gpt3_m_values:
        suite.append(
            ShapeCase(
                experiment_group="gpt3_shrinking",
                experiment_label="GPT-3 shrinking (k=49152, n=12288)",
                varied_dim="m",
                varied_value=m,
                m=m,
                k=48 * 1024,
                n=12 * 1024,
            )
        )

    for s in square_sizes:
        suite.append(
            ShapeCase(
                experiment_group="square",
                experiment_label="Square matrices",
                varied_dim="size",
                varied_value=s,
                m=s,
                n=s,
                k=s,
            )
        )

    for big in odd_large_values:
        suite.append(
            ShapeCase(
                experiment_group="odd_m_large",
                experiment_label=f"Odd-dimension: large m (n=k={odd_small_value})",
                varied_dim="m",
                varied_value=big,
                m=big,
                n=odd_small_value,
                k=odd_small_value,
            )
        )
        suite.append(
            ShapeCase(
                experiment_group="odd_n_large",
                experiment_label=f"Odd-dimension: large n (m=k={odd_small_value})",
                varied_dim="n",
                varied_value=big,
                m=odd_small_value,
                n=big,
                k=odd_small_value,
            )
        )
        suite.append(
            ShapeCase(
                experiment_group="odd_k_large",
                experiment_label=f"Odd-dimension: large k (m=n={odd_small_value})",
                varied_dim="k",
                varied_value=big,
                m=odd_small_value,
                n=odd_small_value,
                k=big,
            )
        )
    return suite


def _run_case(
    *,
    python_exe: str,
    nproc_per_node: int,
    master_port: int,
    shape_case: ShapeCase,
    method_case: MethodCase,
    warmup_iters: int,
    measure_iters: int,
    seed: int,
    theoretical_tflops: float | None,
    output_dir: pathlib.Path,
    run_index: int,
) -> dict:
    case_key = (
        f"{shape_case.experiment_group}"
        f"_m{shape_case.m}_n{shape_case.n}_k{shape_case.k}"
        f"_{method_case.method}"
        f"_A{method_case.a_partition}B{method_case.b_partition}C{method_case.c_partition}"
        f"_rep{method_case.replication_factor}"
    )
    case_dir = output_dir / "runs" / case_key
    case_dir.mkdir(parents=True, exist_ok=True)
    result_file = case_dir / "result.json"
    log_file = case_dir / "torchrun.log"

    cmd = [
        python_exe,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--rdzv-endpoint",
        f"localhost:{master_port}",
        "--nproc_per_node",
        str(nproc_per_node),
        str(WORKER),
        "--method",
        method_case.method,
        "--m",
        str(shape_case.m),
        "--n",
        str(shape_case.n),
        "--k",
        str(shape_case.k),
        "--a-partition",
        method_case.a_partition,
        "--b-partition",
        method_case.b_partition,
        "--c-partition",
        method_case.c_partition,
        "--replication-factor",
        str(method_case.replication_factor),
        "--warmup-iters",
        str(warmup_iters),
        "--measure-iters",
        str(measure_iters),
        "--seed",
        str(seed + run_index),
        "--result-file",
        str(result_file),
    ]
    if theoretical_tflops is not None:
        cmd.extend(["--theoretical-tflops", str(theoretical_tflops)])

    env = os.environ.copy()
    env.pop("TORCH_DTENSOR_USE_NVSHMEM", None)
    if method_case.method in {"stationary_c", "stationary_b"}:
        env["TORCH_DTENSOR_USE_NVSHMEM"] = "1"

    proc = subprocess.run(
        cmd,
        cwd=str(THIS_DIR.parent),
        env=env,
        text=True,
        capture_output=True,
    )
    with open(log_file, "w", encoding="utf-8") as f:
        if proc.stdout:
            f.write(proc.stdout)
        if proc.stderr:
            if proc.stdout:
                f.write("\n")
            f.write(proc.stderr)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Benchmark case failed (exit={proc.returncode}): {' '.join(cmd)}\n"
            f"See log: {log_file}"
        )
    if not result_file.exists():
        raise RuntimeError(f"Worker completed without result file: {result_file}")

    with open(result_file, "r", encoding="utf-8") as f:
        worker_result = json.load(f)

    merged = {
        "experiment_group": shape_case.experiment_group,
        "experiment_label": shape_case.experiment_label,
        "varied_dim": shape_case.varied_dim,
        "varied_value": shape_case.varied_value,
        "case_key": case_key,
        **dataclasses.asdict(shape_case),
        **dataclasses.asdict(method_case),
        **worker_result,
    }
    return merged


def _write_results(output_dir: pathlib.Path, payload: dict) -> None:
    json_path = output_dir / "results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=False)

    csv_path = output_dir / "results.csv"
    rows = payload["results"]
    if not rows:
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["no_results"])
        return

    columns = sorted(rows[0].keys())
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 1: run DTensor/Universal GEMM experiment suite")
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--nproc-per-node", type=int, default=torch.cuda.device_count())
    parser.add_argument("--master-port-base", type=int, default=29600)
    parser.add_argument("--warmup-iters", type=int, default=3)
    parser.add_argument("--measure-iters", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260218)
    parser.add_argument("--methods", type=str, default="dtensor,stationary_c,stationary_b")
    parser.add_argument("--replication-factors", type=str, default="1,2")
    parser.add_argument(
        "--partition-triples",
        type=str,
        default="row,row,row;row,column,row;block,block,block",
        help="Semicolon-separated 'a,b,c' triples",
    )
    parser.add_argument("--gpt3-m-values", type=str, default="1024,2048,4096,8192")
    parser.add_argument("--square-sizes", type=str, default="1024,2048,4096,8192")
    parser.add_argument("--odd-large-values", type=str, default="4096,8192,16384,32768")
    parser.add_argument("--odd-small-value", type=int, default=1024)
    parser.add_argument("--max-cases", type=int, default=0, help="0 means run all")
    parser.add_argument("--theoretical-tflops", type=float, default=None)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--output-dir", type=str, default=str(_default_output_dir()))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.nproc_per_node <= 0:
        raise ValueError("--nproc-per-node must be positive")

    methods = _parse_csv_list(args.methods)
    for method in methods:
        if method not in {"dtensor", "stationary_c", "stationary_b"}:
            raise ValueError(f"Unsupported method {method!r}")

    replication_factors = _parse_int_list(args.replication_factors)
    partition_triples = _parse_partition_triples(args.partition_triples)
    shape_suite = _build_shape_suite(
        gpt3_m_values=_parse_int_list(args.gpt3_m_values),
        square_sizes=_parse_int_list(args.square_sizes),
        odd_large_values=_parse_int_list(args.odd_large_values),
        odd_small_value=args.odd_small_value,
    )

    method_cases: list[MethodCase] = []
    for method in methods:
        for rep in replication_factors:
            if args.nproc_per_node % rep != 0:
                continue
            for a_p, b_p, c_p in partition_triples:
                method_cases.append(
                    MethodCase(
                        method=method,
                        a_partition=a_p,
                        b_partition=b_p,
                        c_partition=c_p,
                        replication_factor=rep,
                    )
                )
    if not method_cases:
        raise RuntimeError("No runnable method/partition/replication combinations generated.")

    all_cases: list[tuple[ShapeCase, MethodCase]] = [
        (shape_case, method_case)
        for shape_case in shape_suite
        for method_case in method_cases
    ]
    if args.max_cases > 0:
        all_cases = all_cases[: args.max_cases]

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    failures: list[dict] = []

    for idx, (shape_case, method_case) in enumerate(all_cases):
        port = args.master_port_base + (idx % 1000)
        print(
            f"[{idx + 1}/{len(all_cases)}] "
            f"{shape_case.experiment_group} "
            f"(m={shape_case.m}, n={shape_case.n}, k={shape_case.k}) "
            f"method={method_case.method} "
            f"parts={method_case.a_partition}/{method_case.b_partition}/{method_case.c_partition} "
            f"rep={method_case.replication_factor}",
            flush=True,
        )
        try:
            row = _run_case(
                python_exe=args.python,
                nproc_per_node=args.nproc_per_node,
                master_port=port,
                shape_case=shape_case,
                method_case=method_case,
                warmup_iters=args.warmup_iters,
                measure_iters=args.measure_iters,
                seed=args.seed,
                theoretical_tflops=args.theoretical_tflops,
                output_dir=output_dir,
                run_index=idx,
            )
            results.append(row)
        except Exception as exc:
            failure = {
                "index": idx,
                "shape_case": dataclasses.asdict(shape_case),
                "method_case": dataclasses.asdict(method_case),
                "error": str(exc),
            }
            failures.append(failure)
            print(f"  FAILED: {exc}", flush=True)
            if not args.continue_on_error:
                break

    payload = {
        "created_at": dt.datetime.now().isoformat(),
        "host": socket.gethostname(),
        "nproc_per_node": args.nproc_per_node,
        "warmup_iters": args.warmup_iters,
        "measure_iters": args.measure_iters,
        "theoretical_tflops": args.theoretical_tflops,
        "results": results,
        "failures": failures,
    }
    _write_results(output_dir, payload)

    print(
        f"Wrote {len(results)} successful results and {len(failures)} failures to {output_dir}",
        flush=True,
    )
    if failures and not args.continue_on_error:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
