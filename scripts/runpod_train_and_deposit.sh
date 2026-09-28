#!/bin/bash
# RunPod job used for the published models: trains all 60 models with weight saving,
# writes the summaries, packages weights and results, and creates a Zenodo draft.
# Expects: repository checked out in the current directory; ZENODO_TOKEN in the environment.
cd "$(dirname "$0")/.."
pip install -q "pandas<3" pyarrow scikit-learn scipy docplex cplex requests 2>&1 | tail -2
python -c "import tensorflow as tf; print('TF', tf.__version__, 'GPUS', tf.config.list_physical_devices('GPU'))"
export FPROF_DATA=/workspace/data
for D in MIND EBNeRD; do
  ( for m in MF H-NCF EH-NCF; do
      FPROF_OUT=/workspace/results/$D FPROF_SAVE_MODELS=/workspace/models python -u run.py --datasets $D --models $m 2>&1 \
        | grep -v -E '^ *[0-9]+/[0-9]+|BUNDLE' >> /workspace/log_$D.txt
    done; touch /workspace/done_$D ) &
done
wait
python -c "import aggregate,json; [json.dump(aggregate.summarise(f'/workspace/results/{d}'), open(f'/workspace/results/{d}.json','w')) for d in ('MIND','EBNeRD')]"
grep -h " done " /workspace/log_*.txt | wc -l
cd /workspace
tar czf fprof_models_MIND.tar.gz -C models MIND
tar czf fprof_models_EBNeRD.tar.gz -C models EBNeRD
tar czf fprof_results.tar.gz results
ls -la /workspace/*.tar.gz
