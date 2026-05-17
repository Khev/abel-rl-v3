import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import sys

def plot_results():
    # Find the latest results directory in gemini/cov
    results_dir = "gemini/cov"
    if not os.path.exists(results_dir):
        print(f"Directory not found: {results_dir}")
        return

    # Get all subdirectories
    subdirs = [os.path.join(results_dir, d) for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d))]
    if not subdirs:
        print("No results found.")
        return

    # Sort by modification time to get the latest
    latest_dir = max(subdirs, key=os.path.getmtime)
    print(f"Plotting results from: {latest_dir}")

    csv_path = os.path.join(latest_dir, "accuracies.csv")
    if not os.path.exists(csv_path):
        print(f"CSV not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)

    plt.figure(figsize=(10, 6))
    plt.plot(df['timesteps'], df['train_greedy_acc'], label='Train Greedy', marker='o')
    plt.plot(df['timesteps'], df['test_greedy_acc'], label='Test Greedy', marker='x')
    plt.plot(df['timesteps'], df['test_beam_acc'], label='Test Beam', marker='s')
    
    plt.xlabel('Timesteps')
    plt.ylabel('Accuracy')
    plt.title('CoV Agent Performance on Open Equations')
    plt.legend()
    plt.grid(True)
    
    output_path = os.path.join("gemini", "cov_results.png")
    plt.savefig(output_path)
    print(f"Plot saved to: {output_path}")

if __name__ == "__main__":
    plot_results()
