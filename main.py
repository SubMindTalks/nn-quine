import os
import json
import subprocess
from pathlib import Path
from tqdm import tqdm
import sys

# Constants
VANILLA_QUINE_SCRIPT = "mains/vanilla_quine.py"
REAL_QUINE_SCRIPT = "mains/real_quine.py"
CONFIGS_DIR = "configs"
RESULTS_DIR = "results"

# Ensure results directory exists
Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)

# Function to train a quine
def train_quine(config_path):
    config_name = os.path.basename(config_path)
    model_type = "real" if "real_quine" in config_path else "vanilla"
    script = VANILLA_QUINE_SCRIPT if model_type == "vanilla" else REAL_QUINE_SCRIPT
    result_file = os.path.join(RESULTS_DIR, f"{model_type}_quine_{config_name.replace('.json', '.txt')}")

    print(f"Training {model_type.capitalize()} Quine with config {config_name}...")

    python_executable = sys.executable

    try:
        result = subprocess.run(
            [python_executable, script, config_path, "--eval"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        with open(result_file, "w") as f:
            f.write(result.stdout)
        if result.returncode == 0:
            print(f"Completed {model_type.capitalize()} Quine with config {config_name}")
        else:
            print(f"Error training {model_type.capitalize()} Quine with config {config_name}")
            print(result.stderr)
    except Exception as e:
        print(f"Exception occurred: {e}")


# Function to summarize results
def summarize_results():
    print("\nSummarizing results...\n")
    for result_file in Path(RESULTS_DIR).glob("*.txt"):
        if "vanilla_quine" in result_file.stem:
            model_type = "Vanilla Quine"
        elif "real_quine" in result_file.stem:
            model_type = "Real Quine"
        else:
            model_type = "Unknown"

        config_name = result_file.stem.replace("vanilla_quine_", "").replace("real_quine_", "")
        print(f"Results for {model_type} - {config_name}:")
        with open(result_file, "r") as f:
            lines = f.readlines()
            eval_results = [line.strip() for line in lines if "Param #" in line or "Loss" in line]
            for line in eval_results:
                print(line)
        print("-" * 80)


# Main function
def main():
    configs = Path(CONFIGS_DIR).glob("**/*.json")

    for config_path in tqdm(list(configs), desc="Training Quines"):
        train_quine(str(config_path))

    summarize_results()


if __name__ == "__main__":
    main()
