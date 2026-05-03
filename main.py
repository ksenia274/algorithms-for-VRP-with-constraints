"""
Usage:
    python main.py run        --config <path> [--seed N] [--time N] [--instance S] [--output P]
    python main.py benchmark  --config <path> [--seed N] [--time N] [--instance S] [--output P]
    python main.py visualize  <path>
    python main.py compare    <bench1> <bench2> ... [--output P]
"""
from __future__ import annotations

import argparse
from pathlib import Path


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="main.py",
        description="VRP with fairness constraints — CLI dispatcher",
    )
    sub = p.add_subparsers(dest="command", required=True)

    # ── run ──────────────────────────────────────────────────────────────
    run_p = sub.add_parser("run", help="Single solver run from YAML config")
    run_p.add_argument("--config", type=Path, required=True)
    run_p.add_argument("--seed", type=int, default=None)
    run_p.add_argument("--time", "--time-limit", dest="time_limit", type=int, default=None)
    run_p.add_argument("--instance", type=str, default=None)
    run_p.add_argument("--output", type=Path, default=None)

    # ── benchmark ────────────────────────────────────────────────────────
    bench_p = sub.add_parser("benchmark", help="Batch run from BenchmarkConfig YAML")
    bench_p.add_argument("--config", type=Path, required=True)
    bench_p.add_argument("--seed", type=int, default=None)
    bench_p.add_argument("--time", "--time-limit", dest="time_limit", type=int, default=None)
    bench_p.add_argument("--instance", type=str, default=None)
    bench_p.add_argument("--output", type=Path, default=None)

    # ── visualize ────────────────────────────────────────────────────────
    vis_p = sub.add_parser("visualize", help="Generate plots for a run or benchmark directory")
    vis_p.add_argument("path", type=Path)

    # ── compare ──────────────────────────────────────────────────────────
    cmp_p = sub.add_parser("compare", help="Compare metrics across multiple benchmarks")
    cmp_p.add_argument("paths", type=Path, nargs="+")
    cmp_p.add_argument("--output", type=Path, default=None)

    # ── spb-map ───────────────────────────────────────────────────────────
    spb_p = sub.add_parser("spb-map", help="Solve SPb instance and draw Folium map")
    spb_p.add_argument("--load-json", type=Path, default=None)
    spb_p.add_argument("--algorithm", default="hgs_rebalance")
    spb_p.add_argument("--time", type=int, default=30)
    spb_p.add_argument("--vehicles", type=int, default=10)
    spb_p.add_argument("--capacity", type=int, default=None)
    spb_p.add_argument("--seed", type=int, default=42)
    spb_p.add_argument("--output-map", type=Path, default=Path("map_results.html"))
    spb_p.add_argument("--save-run", action="store_true")

    return p


def main() -> None:
    args = _parser().parse_args()

    if args.command == "run":
        from runtime.cli.run_command import cmd_run
        cmd_run(
            args.config,
            seed=args.seed,
            time_limit=args.time_limit,
            instance=args.instance,
            output=args.output,
        )

    elif args.command == "benchmark":
        from runtime.cli.benchmark_command import cmd_benchmark
        cmd_benchmark(
            args.config,
            seed=args.seed,
            time_limit=args.time_limit,
            instance=args.instance,
            output=args.output,
        )

    elif args.command == "visualize":
        from runtime.cli.visualize_command import cmd_visualize
        cmd_visualize(args.path)

    elif args.command == "compare":
        from runtime.cli.compare_command import cmd_compare
        cmd_compare(args.paths, output=args.output)

    elif args.command == "spb-map":
        from scripts.run_spb_map import cmd_spb_map
        cmd_spb_map(
            load_json=args.load_json,
            algorithm=args.algorithm,
            time_limit=args.time,
            vehicles=args.vehicles,
            capacity=args.capacity,
            seed=args.seed,
            output_map=args.output_map,
            save_run=args.save_run,
        )


if __name__ == "__main__":
    main()
