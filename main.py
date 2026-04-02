"""
Режимы:
    python main.py                           # простой HGS на одном инстансе
    python main.py --fairness                # HGS + fairness rebalancing
    python main.py --alns                    # ALNS solver
    python main.py --alns --fairness         # ALNS solver c fairness
    python main.py benchmark                 # прогон на всех solomon инстансах и вывод в CSV
    python main.py visualise                 # построить графики из results CSV
    python main.py visualise --csv results/fairness_benchmark.csv
"""

import argparse


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command")

    bench = subparsers.add_parser("benchmark", help="Run full benchmark")
    bench.add_argument("--dataset", choices=["solomon", "yandex"], default="solomon")
    bench.add_argument("--yandex-path", default="vrp_problems")
    bench.add_argument("--time", type=int, default=30)
    bench.add_argument("--capacity", type=int, default=200)
    bench.add_argument("--vehicles", type=int, default=25)
    bench.add_argument("--max-cost-increase", type=float, default=8.0)
    bench.add_argument("--rebalance-iters", type=int, default=5000)
    bench.add_argument("--seed", type=int, default=42)
    bench.add_argument("--output", default="results")
    bench.add_argument("--alns", action="store_true", help="Use ALNS solver instead of HGS")
    bench.add_argument("--fairness", action="store_true", help="Enable fairness objective")
    bench.add_argument("--alns-iterations", type=int, default=25000)
    bench.add_argument("--fairness-weight", type=float, default=100.0)

    vis = subparsers.add_parser("visualise", help="Build charts from benchmark CSV")
    vis.add_argument("--csv", default="results/fairness_benchmark.csv")
    vis.add_argument("--output", default="visualization/output")

    parser.add_argument("--instance", default="R101")
    parser.add_argument("--time", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--capacity", type=int, default=200)
    parser.add_argument("--vehicles", type=int, default=25)
    parser.add_argument("--fairness", action="store_true")
    parser.add_argument("--max-cost-increase", type=float, default=5.0)
    parser.add_argument("--rebalance-iters", type=int, default=3000)
    parser.add_argument("--alns", action="store_true")
    parser.add_argument("--alns-iterations", type=int, default=25000)
    parser.add_argument("--fairness-weight", type=float, default=100.0)

    args = parser.parse_args()

    if args.command == "benchmark":
        from scripts.run_benchmark import run_benchmark
        run_benchmark(args)
    elif args.command == "visualise":
        from scripts.run_visualise import run_visualise
        run_visualise(args)
    elif args.alns:
        from scripts.run_alns import run_alns
        run_alns(args)
    elif args.fairness:
        from scripts.run_hgs import run_fairness
        run_fairness(args)
    else:
        from scripts.run_hgs import run_simple
        run_simple(args)


if __name__ == "__main__":
    main()