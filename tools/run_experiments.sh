#!/bin/bash

# Number of batches and experiments
TOTAL_BATCHES=10
EXPERIMENTS_PER_BATCH=8

for batch in $(seq 1 $TOTAL_BATCHES); do
    echo "=== Starting Batch $batch ==="
    
    for experiment in $(seq 1 $EXPERIMENTS_PER_BATCH); do
        # Generate a random seed
        SEED=$RANDOM
        echo "Running Experiment $experiment in Batch $batch with seed=$SEED"
        run_training --config-name=double_ticker seed=$SEED &
    done

    # Wait for all experiments in the current batch to finish
    wait
    echo "=== Completed Batch $batch ==="
    echo
done

echo "All batches and experiments have been completed."
