import os
import json
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]

DATABASE_DIR = "Database/"  # Acknowledged and manually transferred long-term data storage from OUTPUT_DIR
PLOTS_DIR = "Plots/"


def visualize_data(data, plots_dir):
    # Determine maximum dimension length
    max_len = max(len(d) for d in data) if data else 0

    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    # Prepare data by dimension
    data_by_dimension = []
    for dim_idx in range(max_len):
        values = [snap[dim_idx] for snap in data if dim_idx < len(snap)]
        data_by_dimension.append(values)

    # Overlay Plot: Vertical Scatter (Index vs Value)
    plt.figure(figsize=(12, 6))

    for dim_idx, values in enumerate(data_by_dimension):
        if values:
            jitter = np.random.uniform(-0.1, 0.1, len(values))
            x = np.array([dim_idx] * len(values)) + jitter

            plt.scatter(
                x,
                values,
                alpha=0.6,
                s=50,
                label=f"Dim {dim_idx}" if dim_idx == 0 else "",
            )

    plt.title("All Value Distribution (Scatter)")
    plt.xlabel("Receiver Order Index")
    plt.ylabel("Value (元)")
    plt.xticks(range(max_len))
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    all_plot_path = os.path.join(plots_dir, "all_value_scatter.png")
    plt.savefig(all_plot_path)
    print(f"Combined plot saved to {all_plot_path}")
    plt.close()

    # Overlay Plot: Vertical Boxplot (Index vs Value)
    plt.figure(figsize=(12, 6))
    plt.boxplot(data_by_dimension, tick_labels=range(max_len))

    plt.title("All Value Distribution (Boxplot)")
    plt.xlabel("Receiver Order Index")
    plt.ylabel("Value (元)")
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    boxplot_path = os.path.join(plots_dir, "all_value_boxplot.png")
    plt.savefig(boxplot_path)
    print(f"Boxplot saved to {boxplot_path}")
    plt.close()

    # Individual Plots: Histogram (Value Distribution)
    print("Generating individual histograms...")

    for dim_idx, values in enumerate(data_by_dimension):
        if not values:
            continue

        plt.figure(figsize=(10, 6))

        plt.hist(values, bins=20, color="teal", alpha=0.7, edgecolor="black")

        plt.title(f"Distribution Histogram for Receiver # {dim_idx}")
        plt.xlabel("Value (元)")
        plt.ylabel("Frequency")
        plt.grid(axis="y", linestyle="--", alpha=0.7)

        filename = os.path.join(plots_dir, f"Receiver_Index_{dim_idx}_hist.png")
        plt.savefig(filename)
        plt.close()

    print(f"Individual histograms saved to {plots_dir}")


if __name__ == "__main__":
    data_path = os.path.join(DATABASE_DIR, "Wechat_Samples.json")

    if os.path.exists(data_path):
        print(f"Loading data from {data_path}...")
    else:
        print(f"Data file not found: {data_path}")
        exit(1)
    try:
        with open(data_path, "r", encoding="utf-8") as f:
            all_data = json.load(f)
        print(f"Loaded {len(all_data)} data groups from {data_path}")
        output_path = os.path.join(PLOTS_DIR, "Raw_Wechat/")
        visualize_data(all_data, output_path)
    except Exception as e:
        print(f"Error loading data from file: {e}")
