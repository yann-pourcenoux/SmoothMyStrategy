#!/bin/bash
# Example usage:
# ./run_sweep.sh <sweep_id> <number_of_agents>
# Example:
# ./run_sweep.sh "sweep/abc123" 5

# Number of sweeping agents
NUM_SWEEP_AGENTS="$2"
    
for experiment in $(seq 1 $NUM_SWEEP_AGENTS); do
        echo "Running Sweeping agent $experiment"
        # Start of Selection
        wandb agent "$1" &
    done

    # Wait for all sweeping agents to finish
    wait
    echo "=== Completed Sweeping agent $experiment ==="
    echo
done

echo "All sweeping agents have been completed."
