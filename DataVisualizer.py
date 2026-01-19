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

    # Generate conditional histograms
    visualize_conditional_histograms(data, plots_dir)


def visualize_conditional_histograms(data, plots_dir):
    print("Generating conditional histograms...")

    max_len = max(len(d) for d in data) if data else 0

    # k corresponds to dimensions X_k where k > 1 (1-based) => indices 1, 2, ...
    for dim_idx in range(1, max_len):
        pairs = []
        for sample in data:
            if dim_idx < len(sample):
                # Sum of previous dimensions X_0 ... X_{k-1}
                prev_sum = sum(sample[:dim_idx])
                val = sample[dim_idx]
                pairs.append((prev_sum, val))

        if not pairs:
            continue

        sums = [p[0] for p in pairs]

        # Determine the most frequent interval (mode) for the sums
        if not sums:
            continue

        # Use bins='auto' or a fixed number large enough to validly capture the peak
        counts, bin_edges = np.histogram(sums, bins="auto")
        if len(counts) == 0:
            continue

        max_bin_idx = np.argmax(counts)
        range_min = bin_edges[max_bin_idx]
        range_max = bin_edges[max_bin_idx + 1]

        # Filter X_k where sum is in this interval
        # Note: np.histogram includes right edge for the last bin, but generally [a, b)
        # We will use the edges directly.
        filtered_values = [val for s, val in pairs if range_min <= s <= range_max]

        if not filtered_values:
            continue

        plt.figure(figsize=(10, 6))
        plt.hist(filtered_values, bins=20, color="purple", alpha=0.7, edgecolor="black")

        plt.title(
            f"Hist for X_{dim_idx} | Sum(X_0..X_{dim_idx-1}) in [{range_min:.2f}, {range_max:.2f}]"
        )
        plt.xlabel(f"X_{dim_idx} Value")
        plt.ylabel("Frequency")
        plt.grid(axis="y", linestyle="--", alpha=0.7)

        filename = os.path.join(plots_dir, f"Dim_{dim_idx}_cond_hist.png")
        plt.savefig(filename)
        plt.close()

    print(f"Conditional histograms saved to {plots_dir}")


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
