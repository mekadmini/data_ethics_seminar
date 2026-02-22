import pandas as pd
import os

def load_data():
    file_path = r"c:\Users\moham\Documents\Master Data Science\data_ethics_seminar\experiment_results\merged_results_evaluated.csv"
    
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return
    
    print(f"Loading {file_path}...")
    df = pd.read_csv(file_path)
    
    print(f"\nDataFrame loaded successfully!")
    print(f"Shape: {df.shape} (Rows: {len(df)}, Columns: {len(df.columns)})")
    
    print("\nColumn Names:")
    print(df.columns.tolist())
    
    print("\nFirst 5 rows:")
    print(df.head())
    
    return df

if __name__ == "__main__":
    df = load_data()
