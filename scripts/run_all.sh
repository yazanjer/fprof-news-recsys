#!/bin/bash
# Full benchmark: 2 datasets x 3 scorers x 10 seeds, lambda grid, both list constructions,
# CPLEX check, significance tests. Set SAVE_MODELS=1 to keep the trained weights.
set -e
cd "$(dirname "$0")/.."
export FPROF_DATA=${FPROF_DATA:-./data}
for D in MIND EBNeRD; do
  ( for M in MF H-NCF EH-NCF; do
      FPROF_OUT=results/$D FPROF_SAVE_MODELS=${SAVE_MODELS:+models} python -u run.py --datasets $D --models $M
    done ) &
done
wait
python -c "import aggregate,json; [json.dump(aggregate.summarise(f'results/{d}'), open(f'results/{d}.json','w')) for d in ('MIND','EBNeRD')]"
echo "Summaries written to results/MIND.json and results/EBNeRD.json"
echo "Tables and figures: python make_results.py results <thesis-directory>"
