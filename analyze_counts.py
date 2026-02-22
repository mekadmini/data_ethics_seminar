import os
import pandas as pd
from pathlib import Path

def analyze_source_files(base_path):
    base_dir = Path(base_path)
    experiment_results_dir = base_dir / "experiment_results"
    
    total_rows = 0
    study_counts = []
    
    study_folders = [f for f in experiment_results_dir.iterdir() if f.is_dir() and f.name.startswith("study_")]
    print(f"Total study folders found: {len(study_folders)}")
    
    for study_folder in study_folders:
        study_results_count = 0
        for subfolder in study_folder.iterdir():
            if subfolder.is_dir():
                results_file = subfolder / "results_evaluated.csv"
                if results_file.exists():
                    try:
                        df = pd.read_csv(results_file)
                        study_results_count += len(df)
                    except Exception as e:
                        print(f"Error reading {results_file}: {e}")
        
        study_counts.append((study_folder.name, study_results_count))
        total_rows += study_results_count

    print(f"\nTotal rows across all source files: {total_rows}")
    print("\nRows per study:")
    for study_name, count in sorted(study_counts):
        print(f"{study_name}: {count}")

if __name__ == "__main__":
    analyze_source_files(r"c:\Users\moham\Documents\Master Data Science\data_ethics_seminar")
