import os

import pandas as pd


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

    metrics = ['Pct_Refusal', 'Pct_Benign_Compliance', 'Pct_Malicious_Compliance', 'Pct_Confusion', 'Pct_Rebuttal',
               'ASR_both', 'ASR_resp']

    # Combine dataframes
    df_combined = pd.concat([df_baseline, df], ignore_index=True)

    # Group by matrix_language and swap_type, calculate mean and std
    df_agg = df_combined.groupby(['matrix_language', 'swap_type'])[metrics].agg(['mean', 'std']) * 100

    languages = ['En', 'It']
    swap_order = ['Baseline', 'Both', 'ContentOnly', 'FuncOnly']
    swap_headers = ['Baseline', 'Both', 'Content', 'Function']

    lines = []
    lines.append(r'\begin{table}[h]')
    lines.append(r'\centering')
    lines.append(r'\resizebox{\textwidth}{!}{')
    lines.append(r'\begin{tabular}{l' + 'c' * 4 + ' ' + 'c' * 4 + '}')
    lines.append(r'\toprule')
    lines.append(r' & \multicolumn{4}{c}{\textbf{English Matrix}} & \multicolumn{4}{c}{\textbf{Italian Matrix}} \\')
    lines.append(r'\cmidrule(lr){2-5} \cmidrule(lr){6-9}')
    lines.append(r' & ' + ' & '.join(swap_headers) + ' & ' + ' & '.join(swap_headers) + r' \\')
    lines.append(r'\midrule')
    lines.append(r'\multicolumn{9}{l}{\textit{Behavioral decomposition (\%; rows sum to $\approx$ 100 per column)}} \\')

    row_metrics = [
        ('Refusal', 'Pct_Refusal', 0),
        ('Benign Compliance', 'Pct_Benign_Compliance', 0),
        ('Malicious Compliance', 'Pct_Malicious_Compliance', 0),
        ('Confusion', 'Pct_Confusion', 0),
        ('Rebuttal', 'Pct_Rebuttal', 0),
        ('midrule', '', 0),
        (r'\multicolumn{9}{l}{\textit{Safety judge metrics (\%)}}', 'header', 0),
        (r'$\text{ASR}_{\text{both}}$', 'ASR_both', 0),
        (r'$\text{ASR}_{\text{resp}}$', 'ASR_resp', 1)
    ]

    for label, metric_key, decimals in row_metrics:
        if metric_key == '':
            lines.append(r'\midrule')
            continue
        elif metric_key == 'header':
            lines.append(label + r' \\')
            continue

        row_str = label

        for lang in languages:
            if lang not in df_agg.index.get_level_values('matrix_language'):
                row_str += ' & ' * 4
                continue

            lang_df = df_agg.xs(lang, level='matrix_language')

            vals = []
            stds = []
            for swap in swap_order:
                swap_lookup = swap
                if swap not in lang_df.index:
                    if swap == 'ContentOnly' and 'Content_only' in lang_df.index:
                        swap_lookup = 'Content_only'
                    elif swap == 'ContentOnly' and 'content_only' in lang_df.index:
                        swap_lookup = 'content_only'
                    elif swap == 'FuncOnly' and 'Func_only' in lang_df.index:
                        swap_lookup = 'Func_only'
                    elif swap == 'FuncOnly' and 'Functional_only' in lang_df.index:
                        swap_lookup = 'Functional_only'
                    elif swap == 'Both' and 'both' in lang_df.index:
                        swap_lookup = 'both'

                if swap_lookup in lang_df.index:
                    vals.append(lang_df.loc[swap_lookup, (metric_key, 'mean')])
                    stds.append(lang_df.loc[swap_lookup, (metric_key, 'std')])
                else:
                    vals.append(-1)
                    stds.append(-1)

            max_val = max(vals)

            for val, std in zip(vals, stds):
                if val == -1:
                    row_str += ' & -'
                else:
                    if pd.isna(std): std = 0.0

                    is_max = (val == max_val)
                    if decimals == 0:
                        val_str = f"{val:.0f}"
                        std_str = f"{std:.0f}"
                    else:
                        val_str = f"{val:.1f}"
                        std_str = f"{std:.1f}"

                    if is_max:
                        row_str += f' & \\textbf{{{val_str}}} $\\pm$ {std_str}'
                    else:
                        row_str += f' & {val_str} $\\pm$ {std_str}'

        lines.append(row_str + r' \\')

    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'}')
    lines.append(r'\caption{Average Response Class compositions and ASR Rates (\%) across configurations.}')
    lines.append(r'\label{tab:response_classes_stats_swapped}')
    lines.append(r'\end{table}')

    table_str = '\n'.join(lines)
    print("\n--- GENERATED LATEX TABLE ---\n")
    print(table_str)
    print("\n-----------------------------\n")


if __name__ == '__main__':
    generate_latex_table()
