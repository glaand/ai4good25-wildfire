import wandb
import statistics
from collections import defaultdict

# --- Configuration ---
# Your W&B Entity (username or team name)
ENTITY = "bluefrog-university-of-zurich" 
# Your W&B Project Name
PROJECT = "ai4good25-wildfire-src"
# List of the specific sweep IDs you want to analyze
SWEEP_IDS = ["serlb54o", "amgs0fsy", "hqgpmiai"] 
# The metrics you want to analyze
METRICS_TO_ANALYZE = ["test_AP", "test_UCE"] 

# --- W&B API Initialization ---
# Assuming you have run 'wandb login' or set the WANDB_API_KEY environment variable
api = wandb.Api()

print(f"--- Analyzing Metrics from Project: {ENTITY}/{PROJECT} ---")

# --- Function to Analyze a Single Sweep ---
def analyze_sweep(sweep_id):
    """Fetches runs for a sweep, extracts metric values, and calculates stats."""
    full_sweep_path = f"{ENTITY}/{PROJECT}/{sweep_id}"
    
    try:
        # Fetch the sweep object
        sweep = api.sweep(full_sweep_path)
        print(f"\n## 📊 Sweep ID: {sweep_id} ({sweep.name})")

        runs = sweep.runs
        
        # Check if any runs are available
        if not runs:
            print("No runs found for this sweep.")
            return

        # Dictionary to store all collected metric values (e.g., {"test_AP": [0.8, 0.7, ...]})
        metric_values = defaultdict(list)

        # 1. Collect all metric values from run summaries
        for run in runs:
            # Check if the run finished and has a summary
            if run.state == "finished" and run.summary:
                for metric in METRICS_TO_ANALYZE:
                    # Check if the metric exists in the final summary
                    if metric in run.summary:
                        metric_values[metric].append(run.summary[metric])
        
        # 2. Calculate and print statistics
        for metric in METRICS_TO_ANALYZE:
            values = metric_values.get(metric, [])
            num_runs = len(values)

            if num_runs > 0:
                # Calculate Mean
                mean_val = statistics.mean(values)
                
                # Calculate Standard Deviation (Sample Standard Deviation)
                # We check for more than 1 value to prevent a statistics.StatisticsError
                std_dev_val = statistics.stdev(values) if num_runs > 1 else 0.0

                print(f"--- {metric} ---")
                print(f"  Runs Analyzed: **{num_runs}**")
                print(f"  Mean: **{mean_val:.4f}**")
                print(f"  Standard Deviation (Std Dev): **{std_dev_val:.4f}**")
            else:
                print(f"--- {metric} ---")
                print(f"  No valid final summary data found for {metric} in any run.")

    except wandb.CommError as e:
        print(f"Error accessing sweep {sweep_id}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred for sweep {sweep_id}: {e}")

# --- Main Execution ---
for sweep_id in SWEEP_IDS:
    analyze_sweep(sweep_id)

print("\n--- Script finished ---")
