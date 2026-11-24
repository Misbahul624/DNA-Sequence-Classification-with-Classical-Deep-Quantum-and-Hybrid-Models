"""DNA sequence classification — 12 classical / deep / quantum / hybrid models.

    python run.py                          # BOTH tasks, all 12 models -> results/REPORT.md
    python run.py --task family            # gene-family task only
    python run.py --task species           # human vs chimpanzee vs dog only
    python run.py --models 1,2,6,10        # a subset
    python run.py -v                       # also per-batch progress inside each epoch
    python run.py -q                       # quiet: one line per model
    python run.py --full                   # longer training
"""
# Keep this file free of top-level imports: XGBoost trains in a spawned child process that
# re-imports this script, and that child must never load torch (macOS libomp conflict).

if __name__ == "__main__":
    from dna.pipeline import main

    main()
