def run_visualise(args):
    from visualization.fairness_charts import plot_all
    plot_all(csv_path=args.csv, output_dir=args.output)