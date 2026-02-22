import pandas as pd
import os

def aggregate_results():
    input_file = r"c:\Users\moham\Documents\Master Data Science\data_ethics_seminar\experiment_results\merged_results_evaluated.csv"
    output_file = r"c:\Users\moham\Documents\Master Data Science\data_ethics_seminar\experiment_results\aggregated_results.csv"
    
    if not os.path.exists(input_file):
        print(f"Error: Could not find {input_file}")
        return
        
    print(f"Loading {input_file}...")
    df = pd.read_csv(input_file)
    
    print("Aggregating data by prompt and configuration...")
    
    # Helper function to count the number of 'unsafe' occurrences
    def count_unsafe(series):
        return (series == 'unsafe').sum()
        
    # Group by prompt and configuration, then calculate the required metrics
    aggregated_df = df.groupby(['prompt', 'configuration']).agg(
        total_lines=('prompt', 'size'),  # Count the number of rows in the group
        unsafe_prompt=('safe_prompt', count_unsafe),
        unsafe_response=('safe_response', count_unsafe),
        unsafe_both=('safe_both', count_unsafe)
    ).reset_index()
    
    print("Splitting configuration column...")
    # Example config: En_Ratio_0.8_Arabic_Dom_FuncOnly
    # Split the configuration string
    def parse_config(config_str):
        parts = config_str.split('_')
        
        # matrix_language is the first part
        matrix_language = parts[0]
        
        # swap_type is the last part
        swap_type = parts[-1]
        
        # Find 'Ratio' to get the ratio value
        ratio = None
        ratio_idx = -1
        for i, part in enumerate(parts):
            if part == 'Ratio' and i + 1 < len(parts):
                ratio = parts[i+1]
                ratio_idx = i
                break
                
        # dominance is whatever is left between ratio value and swap_type
        dominance_parts = []
        if ratio_idx != -1 and ratio_idx + 2 < len(parts) - 1:
            dominance_parts = parts[ratio_idx + 2 : -1] # Everything after ratio value up to the last element
            
        dominance = '_'.join(dominance_parts) if dominance_parts else None
        
        return pd.Series([matrix_language, ratio, dominance, swap_type])
        
    # Apply the parsing function
    aggregated_df[['matrix_language', 'ratio', 'dominance', 'swap_type']] = aggregated_df['configuration'].apply(parse_config)
    
    # Drop the original configuration column
    aggregated_df = aggregated_df.drop(columns=['configuration'])
    
    # Save the aggregated dataframe to a new CSV file
    aggregated_df.to_csv(output_file, index=False)
    print(f"Successfully saved aggregated results to {output_file}")
    
    print("\nShape of aggregated data:", aggregated_df.shape)
    print("\nFirst few rows of the aggregated data:")
    print(aggregated_df.head())
    
    return aggregated_df

if __name__ == "__main__":
    aggregate_results()
