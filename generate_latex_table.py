import pandas as pd
import os

def generate_latex_table(metrics_summary_path='experiment_results/metrics_summary.csv', 
                         baseline_metrics_path='experiment_results/baseline_metrics.csv'):
    
    if not os.path.exists(metrics_summary_path) or not os.path.exists(baseline_metrics_path):
        print(f"Error: Could not find required CSV files.")
        print(f"Expected: {metrics_summary_path} and {baseline_metrics_path}")
        return

    df = pd.read_csv(metrics_summary_path)
    df_baseline = pd.read_csv(baseline_metrics_path)

    # Ensure baseline matches capitalization and has swap_type
    df_baseline['matrix_language'] = df_baseline['matrix_language'].str.capitalize()
    if 'swap_type' not in df_baseline.columns:
        df_baseline['swap_type'] = 'Baseline'

    metrics = ['Pct_Confusion', 'Pct_Refusal', 'Pct_Rebuttal', 'Pct_Benign_Compliance', 'Pct_Malicious_Compliance']

    # Combine dataframes
    df_combined = pd.concat([df_baseline, df], ignore_index=True)

    # Group by matrix_language and swap_type, calculate mean and standard deviation
    df_agg = df_combined.groupby(['matrix_language', 'swap_type'])[metrics].agg(['mean', 'std'])

    lines = []
    lines.append(r'\begin{table}[h]')
    lines.append(r'\centering')
    lines.append(r'\resizebox{\textwidth}{!}{')
    lines.append(r'\begin{tabular}{llccccc}')
    lines.append(r'\toprule')
    lines.append(r'\textbf{Matrix} & \textbf{Swap Type} & \textbf{Confusion} & \textbf{Refusal} & \textbf{Rebuttal} & \textbf{Benign Comp.} & \textbf{Malicious Comp.} \\')
    lines.append(r'\midrule')

    for lang in ['En', 'It']:
        if lang not in df_agg.index.get_level_values('matrix_language'):
            continue
        
        lang_df = df_agg.xs(lang, level='matrix_language')
        lang_print = 'English' if lang == 'En' else 'Italian'
        
        for i, swap in enumerate(['Baseline', 'ContentOnly', 'FuncOnly', 'Both']):
            swap_lookup = swap
            # Handle potential naming discrepancies between script generated names and CSV names
            if swap not in lang_df.index:
                if swap == 'ContentOnly' and 'Content_only' in lang_df.index: swap_lookup = 'Content_only'
                elif swap == 'ContentOnly' and 'content_only' in lang_df.index: swap_lookup = 'content_only'
                elif swap == 'FuncOnly' and 'Func_only' in lang_df.index: swap_lookup = 'Func_only'
                elif swap == 'FuncOnly' and 'Functional_only' in lang_df.index: swap_lookup = 'Functional_only'
                elif swap == 'Both' and 'both' in lang_df.index: swap_lookup = 'both'
                else: continue
                
            row_str = (lang_print if i==0 else '') + ' & ' + swap.replace('Only', ' Only')
            for m in metrics:
                mean_val = lang_df.loc[swap_lookup, (m, 'mean')] * 100
                
                # If standard deviation is missing (like for single baseline trial runs), set it to 0.0
                std_val = lang_df.loc[swap_lookup, (m, 'std')] * 100
                if pd.isna(std_val): 
                    std_val = 0.0
                    
                row_str += f' & {mean_val:.1f}\\% $\\pm$ {std_val:.1f}\\%'
            lines.append(row_str + r' \\')
        lines.append(r'\midrule')

    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'}')
    lines.append(r'\caption{Average Response Class compositions and standard deviations (\%) across configurations.}')
    lines.append(r'\label{tab:response_classes_stats}')
    lines.append(r'\end{table}')
    
    table_str = '\n'.join(lines)
    print("\n--- GENERATED LATEX TABLE ---\n")
    print(table_str)
    print("\n-----------------------------\n")

if __name__ == '__main__':
    generate_latex_table()
