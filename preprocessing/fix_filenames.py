#!/usr/bin/env python3
"""
Fix annotation filenames to match the old naming convention.
Removes apostrophes, accents, and parentheses from filenames.
"""

import os
import glob
import unicodedata

TARGET_DIRS = [
    "/n/data1/hms/dbmi/rajpurkar/lab/datasets/MAIDA-NICU/preprocessed_vish/new_annotations/annotations_latest",
    "/n/data1/hms/dbmi/rajpurkar/lab/datasets/MAIDA-NICU/preprocessed_vish/new_annotations/annotations_640x640_latest"
]

def remove_accents(text):
    """Remove accents from unicode string."""
    # Normalize to NFD (decomposed form)
    nfd = unicodedata.normalize('NFD', text)
    # Filter out combining characters (accents)
    without_accents = ''.join(c for c in nfd if unicodedata.category(c) != 'Mn')
    return without_accents

def fix_filename(filename):
    """Apply naming fixes to match old convention."""
    # Remove apostrophes
    filename = filename.replace("'", "")

    # Remove accents (ó → o, á → a, etc.)
    filename = remove_accents(filename)

    # Remove parentheses
    filename = filename.replace("(", "").replace(")", "")

    return filename

def main():
    total_renamed = 0

    for TARGET_DIR in TARGET_DIRS:
        print(f"\n{'='*80}")
        print(f"Processing: {os.path.basename(TARGET_DIR)}")
        print(f"{'='*80}")

        # Get all JSON files in the directory
        json_files = glob.glob(os.path.join(TARGET_DIR, "*.json"))

        if not json_files:
            print(f"No JSON files found in {TARGET_DIR}")
            continue

        print(f"Found {len(json_files)} files to check")
        print()

        renamed_count = 0

        for old_path in sorted(json_files):
            old_filename = os.path.basename(old_path)
            new_filename = fix_filename(old_filename)

            if old_filename != new_filename:
                new_path = os.path.join(TARGET_DIR, new_filename)

                # Check if target already exists
                if os.path.exists(new_path):
                    print(f"WARNING: Target already exists, skipping:")
                    print(f"  {old_filename} → {new_filename}")
                    continue

                # Rename
                os.rename(old_path, new_path)
                print(f"Renamed: {old_filename}")
                print(f"      → {new_filename}")
                renamed_count += 1
            else:
                # No change needed
                pass

        print()
        print(f"Renamed {renamed_count} files in this directory")
        total_renamed += renamed_count

    print()
    print(f"{'='*80}")
    print(f"TOTAL RENAMED: {total_renamed} files")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
