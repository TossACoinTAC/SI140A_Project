import os
import glob
import json
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]

import DataExtractor

PLOT_DIR = "Plots/Raw_Wechat/"
DATABASE_DIR = "Database/"  # Acknowledged and manually transferred long-term data storage from OUTPUT_DIR


def process_all_snapshots():
    # Cleanup: Delete all temp generated .jpgs in Snapshots/
    jpg_files = glob.glob(os.path.join("Snapshots", "*.jpg"))
    if jpg_files:
        print(f"Cleaning up {len(jpg_files)} .jpg files in Snapshots/...")
        for f in jpg_files:
            try:
                os.remove(f)
            except Exception as e:
                print(f"Failed to delete {f}: {e}")

    snapshot_files = glob.glob(os.path.join("Snapshots", "info*.png"))

    snapshot_files.sort()

    all_data = []

    print(f"Found {len(snapshot_files)} files: {snapshot_files}")

    for file_path in snapshot_files:
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue

        print(f"Processing {file_path}...")

        file_data = DataExtractor.extract_data_from_image(file_path)

        if file_data:
            all_data.append(file_data)
            print(f"Extracted {len(file_data)} items from {file_path}")
        else:
            print(f"Failed to extract data from {file_path}")

    if not all_data:
        print("No data collected.")
        return

    print(f"Collected data groups: {len(all_data)}")

    # Save all_data to a JSON
    output_json_path = os.path.join(DATABASE_DIR, "all_data.json")
    try:
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(all_data, f, ensure_ascii=False, indent=4)
        print(f"All data saved to {output_json_path}")
    except Exception as e:
        print(f"Error saving data to file: {e}")


def visualize_data(all_data):
    plt.figure(figsize=(12, 6))

    # Data logic for plotting
    # Example: all_data = [[A1, B1], [A2, B2], [A3, B3]]
    # Index 0 (A): Values (A1, A2, A3)
    # Index 1 (B): Values (B1, B2, B3)

    # Determine maximum dimension length
    max_len = max(len(d) for d in all_data) if all_data else 0

    if not os.path.exists(PLOT_DIR):
        os.makedirs(PLOT_DIR)

    # 1. Overlay Plot: Vertical Scatter (Index vs Value)
    plt.figure(figsize=(12, 6))

    # 1. Prepare data by dimension
    data_by_dimension = []
    for dim_idx in range(max_len):
        values = [snap[dim_idx] for snap in all_data if dim_idx < len(snap)]
        data_by_dimension.append(values)

    # 2. Overlay Plot: Vertical Scatter (Index vs Value)
    plt.figure(figsize=(12, 6))

    for dim_idx, values in enumerate(data_by_dimension):
        if values:
            # Add random jitter to x coordinates
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

    all_plot_path = os.path.join(PLOT_DIR, "all_value_scatter.png")
    plt.savefig(all_plot_path)
    print(f"Combined plot saved to {all_plot_path}")
    plt.close()  # Close the main figure

    # 3. Overlay Plot: Vertical Boxplot (Index vs Value)
    plt.figure(figsize=(12, 6))
    plt.boxplot(data_by_dimension, tick_labels=range(max_len))

    plt.title("All Value Distribution (Boxplot)")
    plt.xlabel("Receiver Order Index")
    plt.ylabel("Value (元)")
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    boxplot_path = os.path.join(PLOT_DIR, "all_value_boxplot.png")
    plt.savefig(boxplot_path)
    print(f"Boxplot saved to {boxplot_path}")
    plt.close()

    # 4. Individual Plots: Histogram (Value Distribution)
    print("Generating individual histograms...")

    for dim_idx, values in enumerate(data_by_dimension):
        if not values:
            continue

        plt.figure(figsize=(10, 6))

        plt.hist(values, bins=20, color="teal", alpha=0.7, edgecolor="black")

        plt.title(
            f"Distribution Histogram for Receiver # {dim_idx} (Count: {len(values)})"
        )
        plt.xlabel("Value (元)")
        plt.ylabel("Frequency")
        plt.grid(axis="y", linestyle="--", alpha=0.7)

        filename = os.path.join(PLOT_DIR, f"Receiver_Index_{dim_idx}_hist.png")
        plt.savefig(filename)
        plt.close()

    print(f"Individual histograms saved to {PLOT_DIR}")


if __name__ == "__main__":
    # Load Wechat_Samples.json if it exists, otherwise process snapshots
    data_path = os.path.join(DATABASE_DIR, "Wechat_Samples.json")

    if os.path.exists(data_path):
        print(f"Loading data from {data_path}...")
        try:
            with open(data_path, "r", encoding="utf-8") as f:
                all_data = json.load(f)
            print(f"Loaded {len(all_data)} data groups from {data_path}")
            visualize_data(all_data)
        except Exception as e:
            print(f"Error loading data from file: {e}")
            print("Falling back to processing snapshots...")
            process_all_snapshots()
    else:
        process_all_snapshots()
