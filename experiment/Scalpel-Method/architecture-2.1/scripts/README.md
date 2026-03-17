# Scripts

1. Train the router (multi-label):
   python scripts/router_train.py --config ../config.example.json

2. Split the base LLM into experts:
   python scripts/split_llm.py --config ../config.example.json

3. Benchmark modular expert loading + inference:
   python scripts/bench_modular_infer.py --config ../config.example.json --categories Science,History

4. Compare benchmark runs:
   python scripts/compare_benchmarks.py --inputs ../artifacts/benchmarks --print_table

5. Benchmark fragmentation / data loss (expert vs full/base model):
   python scripts/bench_fragmentation.py --config ../config.example.json --categories Science,History

Outputs go to `artifacts/` by default (see config).
