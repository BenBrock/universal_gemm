comm_issue: float = 0.0
comm_sync: float = 0.0
compute: float = 0.0
total_runtime: float = 0.0


def add_comm_issue(duration_s: float) -> None:
    global comm_issue
    comm_issue += float(duration_s)


def add_comm_sync(duration_s: float) -> None:
    global comm_sync
    comm_sync += float(duration_s)


def add_compute(duration_s: float) -> None:
    global compute
    compute += float(duration_s)


def reset_stats() -> None:
    global comm_issue, comm_sync, compute, total_runtime
    comm_issue = 0.0
    comm_sync = 0.0
    compute = 0.0
    total_runtime = 0.0


def add_total_runtime(duration_s: float) -> None:
    global total_runtime
    total_runtime += float(duration_s)


def set_total_runtime(duration_s: float) -> None:
    global total_runtime
    total_runtime = float(duration_s)


def print_stats(runtime_total_s: float | None = None) -> None:
    total = comm_issue + comm_sync + compute
    print(f"comm_issue: {comm_issue}, comm_sync: {comm_sync}, compute: {compute}")
    if total > 0:
        print(
            f"comm_issue: {100 * (comm_issue / total)}%, "
            f"comm_sync: {100 * (comm_sync / total)}%, "
            f"compute: {100 * (compute / total)}"
        )

    benchmark_total = total_runtime if runtime_total_s is None else float(runtime_total_s)
    if benchmark_total > 0:
        residual = benchmark_total - total
        print(
            f"total_runtime: {benchmark_total}, accounted: {total}, "
            f"unaccounted: {residual}"
        )
        print(
            f"accounted: {100 * (total / benchmark_total)}%, "
            f"unaccounted: {100 * (residual / benchmark_total)}%"
        )
