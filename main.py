"""
Режимы:
    python main.py --algorithm hgs_simple               # простой HGS на одном инстансе
    python main.py --algorithm hgs_rebalance            # HGS + fairness rebalancing
    python main.py --algorithm hgs_rs                   # HGS + rectangle splitting
    python main.py --algorithm hgs_penalty              # HGS + итеративная штрафная матрица расстояний
    python main.py --algorithm hgs_adaptive             # HGS + адаптивные веса fairness прямо в ILS (форк PyVRP)
    python main.py --algorithm alns                     # ALNS solver с fairness
    python main.py benchmark                            # прогон на всех solomon инстансах и вывод в CSV
    python main.py visualise                            # построить графики из results CSV
    python main.py visualise --csv results/fairness_benchmark.csv
"""

import argparse
from enum import Enum


class Algorithm(Enum):
    HGS_SIMPLE = 'hgs_simple'
    HGS_REBALANCE = 'hgs_rebalance'
    HGS_RECTANGLE_SPLITTING = 'hgs_rs'
    HGS_PENALTY = 'hgs_penalty'
    HGS_ADAPTIVE = 'hgs_adaptive'
    ALNS = 'alns'

    def __str__(self):
        return self.value


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command")

    bench = subparsers.add_parser("benchmark", help="Run full benchmark")
    bench.add_argument("--dataset", choices=["solomon", "yandex"], default="solomon")
    bench.add_argument("--yandex-path", default="vrp_problems")
    bench.add_argument("--algorithm", type=Algorithm, choices=list(Algorithm), default=Algorithm.HGS_REBALANCE)
    bench.add_argument("--time", type=int, default=30)
    bench.add_argument("--capacity", type=int, default=200)
    bench.add_argument("--vehicles", type=int, default=25)
    bench.add_argument("--max-cost-increase", type=float, default=8.0)
    bench.add_argument("--rebalance-iters", type=int, default=5000)
    bench.add_argument("--alns-iterations", type=int, default=25000)
    bench.add_argument("--fairness-weight", type=float, default=100.0)
    bench.add_argument("--fair-restarts", type=int, default=5)
    bench.add_argument("--seed", type=int, default=42)
    bench.add_argument("--output", default="results")
    bench.add_argument("--route-balance", type=float, default=500.0)
    bench.add_argument("--decay", type=float, default=0.9999)
    bench.add_argument("--strategy", choices=["linear", "fairness_signal"], default="linear")
    bench.add_argument("--min-weight", type=float, default=0.0)
    bench.add_argument("--max-weight", type=float, default=1e9)
    bench.add_argument("--update-every", type=int, default=1)

    vis = subparsers.add_parser("visualise", help="Build charts from benchmark CSV")
    vis.add_argument("--csv", default="results/fairness_benchmark.csv")
    vis.add_argument("--output", default="visualization/output")

    parser.add_argument("--instance", default="R101")
    parser.add_argument("--algorithm", type=Algorithm, choices=list(Algorithm))
    parser.add_argument("--time", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--capacity", type=int, default=200)
    parser.add_argument("--vehicles", type=int, default=25)
    parser.add_argument("--max-cost-increase", type=float, default=5.0)
    parser.add_argument("--rebalance-iters", type=int, default=3000)
    parser.add_argument("--alns-iterations", type=int, default=25000)
    parser.add_argument("--fairness-weight", type=float, default=100.0)
    parser.add_argument("--fair-restarts", type=int, default=5)
    parser.add_argument("--route-balance", type=float, default=500.0)
    parser.add_argument("--decay", type=float, default=0.9999)
    parser.add_argument("--strategy", choices=["linear", "fairness_signal"], default="linear")
    parser.add_argument("--trace", action="store_true",
                        help="Save per-iteration adaptive trace CSV to results/")
    parser.add_argument("--trace-dir", default="results",
                        help="Directory for trace CSV (default: results)")
    parser.add_argument("--target-cv", type=float, default=0.2,
                        help="FS: CV target (default: 0.2)")
    parser.add_argument("--hold-band", type=float, default=0.05,
                        help="FS: band around target_cv (default: 0.05)")
    parser.add_argument("--boost-multiplier", type=float, default=1.05,
                        help="FS: weight boost factor (default: 1.05)")
    parser.add_argument("--decay-multiplier", type=float, default=0.995,
                        help="FS: weight decay factor (default: 0.995)")
    parser.add_argument("--min-weight", type=float, default=0.0,
                        help="Min adaptive weight (default: 0.0)")
    parser.add_argument("--max-weight", type=float, default=1e9,
                        help="Max adaptive weight (default: 1e9)")

    args = parser.parse_args()

    if args.command == "benchmark":
        from scripts.run_benchmark import run_benchmark
        run_benchmark(args)
    elif args.command == "visualise":
        from scripts.run_visualise import run_visualise
        run_visualise(args)
    elif args.algorithm == Algorithm.ALNS:
        from scripts.run_alns import run_alns
        run_alns(args)
    elif args.algorithm == Algorithm.HGS_REBALANCE:
        from scripts.run_hgs import run_fairness_rebalance
        run_fairness_rebalance(args)
    elif args.algorithm == Algorithm.HGS_RECTANGLE_SPLITTING:
        from scripts.run_hgs_rs import run_hgs_rs
        run_hgs_rs(args)
    elif args.algorithm == Algorithm.HGS_PENALTY:
        from scripts.run_hgs_penalty import run_hgs_penalty
        run_hgs_penalty(args)
    elif args.algorithm == Algorithm.HGS_ADAPTIVE:
        from scripts.run_hgs_adaptive import run_hgs_adaptive
        run_hgs_adaptive(args)
    elif args.algorithm == Algorithm.HGS_SIMPLE:
        from scripts.run_hgs import run_simple
        run_simple(args)
    else:
        raise Exception("--algorithm is required")


if __name__ == "__main__":
    main()
