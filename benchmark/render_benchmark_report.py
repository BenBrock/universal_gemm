#!/usr/bin/env python3
import argparse
import collections
import json
import pathlib
from typing import Any


def _default_output_html(results_path: pathlib.Path) -> pathlib.Path:
    return results_path.with_name("report.html")


def _load_results(path: pathlib.Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if "results" not in payload:
        raise ValueError(f"Expected 'results' in {path}")
    return payload


def _group_for_plot(
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for row in rows:
        exp_key = row["experiment_group"]
        exp = grouped.setdefault(
            exp_key,
            {
                "experiment_group": exp_key,
                "experiment_label": row["experiment_label"],
                "varied_dim": row["varied_dim"],
                "series": collections.defaultdict(list),
                "theoretical_tflops": row.get("theoretical_tflops"),
            },
        )
        series_key = (
            f"{row['method']} | "
            f"A={row['a_partition']} B={row['b_partition']} C={row['c_partition']} | "
            f"rep={row['replication_factor']}"
        )
        exp["series"][series_key].append(row)
        if exp["theoretical_tflops"] is None and row.get("theoretical_tflops") is not None:
            exp["theoretical_tflops"] = row["theoretical_tflops"]

    # Sort each trace by x-axis varied value.
    for exp in grouped.values():
        for series_key in list(exp["series"].keys()):
            exp["series"][series_key] = sorted(
                exp["series"][series_key], key=lambda r: r["varied_value"]
            )
        exp["series"] = dict(exp["series"])
    return grouped


def _infer_method_from_series_name(name: str) -> str:
    return name.split("|", 1)[0].strip()


def _build_plot_payload(
    grouped: dict[str, dict[str, Any]],
    *,
    methods_filter: set[str] | None,
    groups_filter: set[str] | None,
) -> list[dict[str, Any]]:
    plots: list[dict[str, Any]] = []
    for exp_key in sorted(grouped.keys()):
        if groups_filter and exp_key not in groups_filter:
            continue
        exp = grouped[exp_key]
        traces = []
        x_values_all: list[int] = []
        for series_name, rows in exp["series"].items():
            method = _infer_method_from_series_name(series_name)
            if methods_filter and method not in methods_filter:
                continue
            xs = [int(r["varied_value"]) for r in rows]
            ys = [float(r["median_tflops"]) for r in rows]
            x_values_all.extend(xs)
            traces.append(
                {
                    "name": series_name,
                    "x": xs,
                    "y": ys,
                    "mode": "lines+markers",
                    "type": "scatter",
                    "meta": {"method": method, "kind": "series"},
                }
            )
        if not traces:
            continue

        theo = exp.get("theoretical_tflops")
        if theo is not None and x_values_all:
            x_sorted = sorted(set(x_values_all))
            traces.append(
                {
                    "name": f"Theoretical max ({theo:.2f} TFLOPs)",
                    "x": x_sorted,
                    "y": [float(theo)] * len(x_sorted),
                    "mode": "lines",
                    "type": "scatter",
                    "line": {"dash": "dash", "width": 2},
                    "meta": {"method": "theoretical", "kind": "theoretical"},
                }
            )

        plots.append(
            {
                "id": exp_key,
                "title": exp["experiment_label"],
                "x_title": f"Varied {exp['varied_dim']}",
                "y_title": "Achieved TFLOPs (median)",
                "traces": traces,
            }
        )
    return plots


def _render_html(plots: list[dict[str, Any]], output_path: pathlib.Path, methods: list[str]) -> None:
    payload_json = json.dumps(plots, indent=2)
    method_controls = "\n".join(
        [
            f'<label><input type="checkbox" class="method-toggle" value="{m}" checked> {m}</label>'
            for m in methods
        ]
    )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>DTensor / Universal GEMM Benchmark Report</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    body {{
      font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
      margin: 0;
      padding: 24px;
      background: linear-gradient(180deg, #f8fafc, #eef2ff);
      color: #0f172a;
    }}
    h1 {{ margin-top: 0; }}
    .controls {{
      position: sticky;
      top: 0;
      background: rgba(255,255,255,0.92);
      border: 1px solid #cbd5e1;
      border-radius: 12px;
      padding: 12px;
      margin-bottom: 18px;
      backdrop-filter: blur(6px);
    }}
    .controls-row {{
      display: flex;
      gap: 16px;
      flex-wrap: wrap;
      margin-bottom: 8px;
    }}
    .controls label {{ margin-right: 8px; }}
    .plot-card {{
      background: white;
      border: 1px solid #cbd5e1;
      border-radius: 12px;
      padding: 10px 14px 16px 14px;
      margin-bottom: 18px;
      box-shadow: 0 4px 16px rgba(2, 6, 23, 0.06);
    }}
    .plot {{
      width: 100%;
      min-height: 420px;
    }}
  </style>
</head>
<body>
  <h1>DTensor / Universal GEMM Benchmark Report</h1>
  <div class="controls">
    <div class="controls-row">
      <strong>Methods:</strong>
      {method_controls}
    </div>
    <div class="controls-row" id="experiment-toggles"></div>
  </div>
  <div id="plots-root"></div>
  <script>
    const plots = {payload_json};
    const root = document.getElementById("plots-root");
    const experimentToggles = document.getElementById("experiment-toggles");
    const plotHandles = [];

    for (const p of plots) {{
      const card = document.createElement("div");
      card.className = "plot-card";
      card.dataset.expId = p.id;

      const title = document.createElement("h3");
      title.textContent = p.title;
      card.appendChild(title);

      const plotDiv = document.createElement("div");
      plotDiv.className = "plot";
      card.appendChild(plotDiv);
      root.appendChild(card);

      Plotly.newPlot(
        plotDiv,
        p.traces,
        {{
          margin: {{t: 20, r: 24, b: 60, l: 70}},
          xaxis: {{title: p.x_title, tickformat: ",d"}},
          yaxis: {{title: p.y_title}},
          legend: {{orientation: "h"}}
        }},
        {{responsive: true}}
      );

      const expToggle = document.createElement("label");
      expToggle.innerHTML = `<input type="checkbox" class="exp-toggle" value="${{p.id}}" checked> ${{p.title}}`;
      experimentToggles.appendChild(expToggle);

      plotHandles.push({{id: p.id, card, plotDiv, traces: p.traces}});
    }}

    function selectedMethods() {{
      return new Set(
        Array.from(document.querySelectorAll(".method-toggle"))
          .filter(x => x.checked)
          .map(x => x.value)
      );
    }}

    function applyMethodFilter() {{
      const enabledMethods = selectedMethods();
      for (const handle of plotHandles) {{
        const vis = handle.traces.map((tr) => {{
          const meta = tr.meta || {{}};
          if (meta.kind === "theoretical") {{
            return true;
          }}
          return enabledMethods.has(meta.method);
        }});
        Plotly.restyle(handle.plotDiv, "visible", vis);
      }}
    }}

    function applyExperimentFilter() {{
      const enabledExps = new Set(
        Array.from(document.querySelectorAll(".exp-toggle"))
          .filter(x => x.checked)
          .map(x => x.value)
      );
      for (const handle of plotHandles) {{
        handle.card.style.display = enabledExps.has(handle.id) ? "block" : "none";
      }}
    }}

    document.querySelectorAll(".method-toggle").forEach((el) => {{
      el.addEventListener("change", applyMethodFilter);
    }});
    document.addEventListener("change", (ev) => {{
      if (ev.target && ev.target.classList.contains("exp-toggle")) {{
        applyExperimentFilter();
      }}
    }});
  </script>
</body>
</html>
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 2: render interactive HTML benchmark report")
    parser.add_argument("--results-json", type=str, required=True)
    parser.add_argument("--output-html", type=str, default=None)
    parser.add_argument(
        "--methods",
        type=str,
        default="dtensor,stationary_c,stationary_b",
        help="Comma-separated method filter for initial report rendering",
    )
    parser.add_argument(
        "--groups",
        type=str,
        default="",
        help="Optional comma-separated experiment group filter "
        "(e.g. gpt3_expansion,square,odd_m_large)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    results_path = pathlib.Path(args.results_json)
    output_html = pathlib.Path(args.output_html) if args.output_html else _default_output_html(results_path)

    payload = _load_results(results_path)
    rows = payload["results"]

    methods_filter = {x.strip() for x in args.methods.split(",") if x.strip()}
    groups_filter = {x.strip() for x in args.groups.split(",") if x.strip()} if args.groups else None

    grouped = _group_for_plot(rows)
    plots = _build_plot_payload(
        grouped,
        methods_filter=methods_filter if methods_filter else None,
        groups_filter=groups_filter,
    )
    _render_html(plots, output_html, sorted(methods_filter))
    print(f"Wrote report: {output_html}")


if __name__ == "__main__":
    main()
