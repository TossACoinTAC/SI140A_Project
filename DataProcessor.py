import os
import glob
import json
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]

import DataExtractor

OUTPUT_DIR = "Output/"


def process_all_snapshots():
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
    output_json_path = os.path.join(OUTPUT_DIR, "all_data.json")
    try:
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(all_data, f, ensure_ascii=False, indent=4)
        print(f"All data saved to {output_json_path}")
    except Exception as e:
        print(f"Error saving data to file: {e}")

    # ========= Plotting =========
    plt.figure(figsize=(12, 6))

    # Data logic for plotting
    # Example: all_data = [[A1, B1], [A2, B2], [A3, B3]]
    # Index 0 (A): Values (A1, A2, A3)
    # Index 1 (B): Values (B1, B2, B3)

    # Determine maximum dimension length
    max_len = max(len(d) for d in all_data) if all_data else 0

    # Ensure output directory exists (if not created by JSON save)
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 1. Overlay Plot: Vertical Scatter (Index vs Value)
    plt.figure(figsize=(12, 6))

    for dim_idx in range(max_len):
        values_at_dim = []
        # Collect values from all snapshots at this dimension index
        for snapshot_data in all_data:
            if dim_idx < len(snapshot_data):
                values_at_dim.append(snapshot_data[dim_idx])

        # Plot these values at x = dim_idx
        if values_at_dim:
            # Add random jitter to x coordinates
            jitter = np.random.uniform(-0.1, 0.1, len(values_at_dim))
            x = np.array([dim_idx] * len(values_at_dim)) + jitter

            plt.scatter(
                x,
                values_at_dim,
                alpha=0.6,
                s=50,
                label=f"Dim {dim_idx}" if dim_idx == 0 else "",
            )

    plt.title("All Value Distribution")
    plt.xlabel("Receiver Order Index")
    plt.ylabel("Value (元)")
    plt.xticks(range(max_len))
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    all_plot_path = os.path.join(OUTPUT_DIR, "all_value_distribution.png")
    plt.savefig(all_plot_path)
    print(f"Combined plot saved to {all_plot_path}")
    plt.close()  # Close the main figure

    # 2. Individual Plots: Horizontal Scatter (Value vs Jitter)
    print("Generating individual plots...")

    for dim_idx in range(max_len):
        values_at_dim = []
        for snapshot_data in all_data:
            if dim_idx < len(snapshot_data):
                values_at_dim.append(snapshot_data[dim_idx])

        if not values_at_dim:
            continue

        plt.figure(figsize=(10, 4))

        # Horizontal scatter: X = Values, Y = Random Jitter
        # Jitter helps separate points if they strictly overlap in value
        y_jitter = np.random.uniform(-0.1, 0.1, len(values_at_dim))

        plt.scatter(values_at_dim, y_jitter, alpha=0.7, s=80, color="teal")

        plt.title(
            f"Distribution for Receiver # {dim_idx} (Count: {len(values_at_dim)})"
        )
        plt.xlabel("Value (元)")
        plt.ylabel("")  # Jitter axis has no meaning
        plt.yticks([])
        plt.ylim(-0.5, 0.5)
        plt.grid(axis="x", linestyle="--", alpha=0.7)

        # Determine strict bounds or auto-scale
        # plt.xlim(min_val, max_val) # Optional: Standardize axis if needed

        filename = os.path.join(OUTPUT_DIR, f"Receiver_Index_{dim_idx}_scatter.png")
        plt.savefig(filename)
        plt.close()

    print(f"Individual plots saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    process_all_snapshots()
