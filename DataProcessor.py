import os
import glob
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]

import DataExtractor


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

    # ========= Plotting =========
    plt.figure(figsize=(12, 6))

    # Data logic for plotting
    # Example: all_data = [[A1, B1], [A2, B2], [A3, B3]]
    # Index 0 (A): Values (A1, A2, A3)
    # Index 1 (B): Values (B1, B2, B3)

    # Determine maximum dimension length
    max_len = max(len(d) for d in all_data) if all_data else 0

    # Iterate through each dimension index (0 to max_len-1)
    for dim_idx in range(max_len):
        values_at_dim = []
        # Collect values from all snapshots at this dimension index
        for snapshot_data in all_data:
            if dim_idx < len(snapshot_data):
                values_at_dim.append(snapshot_data[dim_idx])

        # Plot these values at x = dim_idx
        if values_at_dim:
            # Add random jitter to x coordinates to separate overlapping points
            jitter = np.random.uniform(-0.05, 0.05, len(values_at_dim))
            x = np.array([dim_idx] * len(values_at_dim)) + jitter

            plt.scatter(
                x,
                values_at_dim,
                alpha=0.6,
                s=50,
                label=f"Dim {dim_idx}" if dim_idx == 0 else "",
            )

    plt.title("Extracted Data Distribution")
    plt.xlabel("Receiver Order Index")
    plt.ylabel("Value (元)")
    plt.xticks(range(max_len))
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    plt.savefig("data_distribution.png")
    print("Plot saved to data_distribution.png")


if __name__ == "__main__":
    process_all_snapshots()
