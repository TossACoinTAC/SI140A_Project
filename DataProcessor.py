import os
import glob
import json

import DataExtractor

OUTPUT_DIR = "Output/"  # Temporary output directory for test-generated data
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
    output_json_path = os.path.join(OUTPUT_DIR, "Wechat_Samples.json")
    try:
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(all_data, f, ensure_ascii=False, indent=4)
        print(f"All data saved to {output_json_path}")
    except Exception as e:
        print(f"Error saving data to file: {e}")


if __name__ == "__main__":
    process_all_snapshots()
