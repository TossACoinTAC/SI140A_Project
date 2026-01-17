import os
import glob
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

    # Scatter plot: x-axis is index of the group, y-axis is the value
    for i, group_data in enumerate(all_data):
        x = [i] * len(group_data)
        plt.scatter(x, group_data, alpha=0.6, s=50)

    plt.title("Extracted Data Distribution Details")
    plt.xlabel("Snapshot Index")
    plt.ylabel("Value (元)")
    plt.xticks(range(len(all_data)))
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    plt.savefig("data_distribution.png")
    print("Plot saved to data_distribution.png")


if __name__ == "__main__":
    process_all_snapshots()
