import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

def visualize_means_with_cis(csv_path='experiment_results/aggregated_results.csv', 
                               baseline_path='experiment_results/aggregated_baseline.csv',
                               output_dir='experiment_results/plots'):
    if not os.path.exists(csv_path):
        print(f"Error: Could not find {csv_path}")
        return
        
    df = pd.read_csv(csv_path)
    
    # Load and append baseline if it exists
    if os.path.exists(baseline_path):
        df_base = pd.read_csv(baseline_path)
        df_base['swap_type'] = 'Baseline'
        df_base['dominance'] = 'Baseline'
        df_base['ratio'] = 0.0
        df = pd.concat([df, df_base], ignore_index=True)
    
    # Standardize matrix_language to avoid 'En' vs 'en' mismatches separating the data
    df['matrix_language'] = df['matrix_language'].str.upper()
    
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    
    # Fill nan just in case, treating 'None' as Baseline if applicable
    if 'swap_type' in df.columns:
        df['swap_type'] = df['swap_type'].fillna('Baseline')
    if 'dominance' in df.columns:
        df['dominance'] = df['dominance'].fillna('Baseline')
    
    # Calculate the normalized metrics per prompt so K=10 and K=1 are directly comparable
    df['ASR_both'] = df['unsafe_both'] / df['total_lines']
    df['ASR_resp'] = df['unsafe_response'] / df['total_lines']
    df['OneSuccess_both'] = (df['unsafe_both'] > 0).astype(float)
    df['OneSuccess_resp'] = (df['unsafe_response'] > 0).astype(float)
    df['Consistent_both'] = (df['ASR_both'] >= 0.5).astype(float)
    df['Consistent_resp'] = (df['ASR_resp'] >= 0.5).astype(float)
    
    # Calculate Lexical metric percentages (handling potential missing columns gracefully fallback)
    for col, new_col in [('compliance_count', 'Pct_Compliance'), 
                         ('benign_compliance_count', 'Pct_Benign_Compliance'),
                         ('malicious_compliance_count', 'Pct_Malicious_Compliance'),
                         ('refusal_count', 'Pct_Refusal'), 
                         ('confusion_count', 'Pct_Confusion'), 
                         ('rebuttal_count', 'Pct_Rebuttal')]:
        if col in df.columns:
            df[new_col] = df[col] / df['total_lines']
        else:
            df[new_col] = 0.0

    metrics_pairs = [
        ('ASR_both', 'ASR_resp', 'Average Attack Success Rate'),
        ('OneSuccess_both', 'OneSuccess_resp', 'Proportion of Prompts w/ >=1 Success'),
        ('Consistent_both', 'Consistent_resp', 'Proportion of Prompts w/ >=50% Success')
    ]

    # Custom order so Baseline is first if it exists
    def custom_sort(vals):
        vals = sorted(list(set(vals)))
        if 'Baseline' in vals:
            vals.insert(0, vals.pop(vals.index('Baseline')))
        return vals

    # Loop through unique matrix languages
    matrix_languages = df['matrix_language'].dropna().unique()
    for ml in matrix_languages:
        ml_label = str(ml).upper()
        print(f"\n--- Generating Plots for Matrix Language: {ml_label} ---")
        
        df_ml = df[df['matrix_language'] == ml]
        if df_ml.empty:
            continue
            
        for metric_both, metric_resp, metric_label in metrics_pairs:
            print(f"[{ml_label}] Generating RQ1.1 Plot: {metric_both} with 95% CIs...")
            # Limit y max carefully to what exists, or 1.0 (since these are all rates/proportions)
            y_max = min(1.0, max(0.3, df_ml[metric_both].max() + 0.1))

            if 'swap_type' in df_ml.columns:
                plt.figure(figsize=(8, 6))
                order = custom_sort(df_ml['swap_type'].unique())
                
                # barplot automatically computes 95% Bootstrapped Confidence Intervals
                ax = sns.barplot(data=df_ml, x='swap_type', y=metric_both, order=order, errorbar=('ci', 95), capsize=0.1, palette="viridis")
                
                plt.title(f'RQ1.1 ({ml_label}): {metric_both} by Swap Type (95% CI)', pad=20)
                plt.ylabel(metric_label)
                plt.xlabel('Swap Type')
                plt.ylim(0, y_max)
                
                for i in ax.containers:
                    try:
                        ax.bar_label(i, fmt='%.3f', padding=15)
                    except: pass
                    
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'{metric_both}_rq1_1_swap_type_ci_{ml_label}.png'), dpi=300)
                plt.close()

            print(f"[{ml_label}] Generating RQ1.2/1.3 Plot: {metric_both} by Language Dominance with 95% CIs...")
            if 'dominance' in df_ml.columns:
                plt.figure(figsize=(10, 6))
                order = custom_sort(df_ml['dominance'].unique())
                
                ax = sns.barplot(data=df_ml, x='dominance', y=metric_both, order=order, errorbar=('ci', 95), capsize=0.1, palette="magma")
                
                plt.title(f'RQ1.2 & 1.3 ({ml_label}): {metric_both} by Language Dominance', pad=20)
                plt.ylabel(metric_label)
                plt.xlabel('Dominance Strategy')
                plt.xticks(rotation=45)
                plt.ylim(0, y_max)
                
                for i in ax.containers:
                    try:
                        ax.bar_label(i, fmt='%.3f', padding=15)
                    except: pass
                    
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'{metric_both}_rq1_3_dominance_ci_{ml_label}.png'), dpi=300)
                plt.close()

            print(f"[{ml_label}] Generating RQ1.4 Plot: Contextual vs Response-Only for {metric_both}...")
            if metric_both in df_ml.columns and metric_resp in df_ml.columns:
                melted_df = pd.melt(df_ml, id_vars=['swap_type'], value_vars=[metric_both, metric_resp], 
                                    var_name='Evaluation View', value_name='Score')
                
                melted_df['Evaluation View'] = melted_df['Evaluation View'].replace({
                    metric_both: 'Contextual (Both)',
                    metric_resp: 'Response-Only'
                })
                
                plt.figure(figsize=(10, 6))
                order = custom_sort(df_ml['swap_type'].unique())
                
                ax = sns.barplot(data=melted_df, x='swap_type', y='Score', hue='Evaluation View', order=order, errorbar=('ci', 95), capsize=0.1, palette="Set2")
                
                plt.title(f'RQ1.4 ({ml_label}): Llama Guard Over-refusal Gap for {metric_both}', pad=20)
                plt.ylabel(metric_label)
                plt.xlabel('Swap Type')
                plt.xticks(rotation=45)
                plt.ylim(0, y_max)
                
                for container in ax.containers:
                    try:
                        ax.bar_label(container, fmt='%.3f', padding=15)
                    except: pass
                    
                plt.legend(title='Judge Evaluation View')
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'{metric_both}_rq1_4_judge_view_ci_{ml_label}.png'), dpi=300)
                plt.close()

        # Generate similar spread plots for the single structural/lexical metrics
        single_metrics = [
            ('Pct_Compliance', 'Total Compliance Rate (Lexical)'),
            ('Pct_Benign_Compliance', 'Benign Compliance Rate (Safe Output)'),
            ('Pct_Malicious_Compliance', 'Malicious Compliance Rate (Unsafe Output)'),
            ('Pct_Refusal', 'Refusal Rate (Lexical)'),
            ('Pct_Confusion', 'Confusion Rate (Lexical)'),
            ('Pct_Rebuttal', 'Rebuttal Rate (Lexical)')
        ]
        
        for metric, metric_label in single_metrics:
            if metric not in df_ml.columns:
                continue
            
            print(f"[{ml_label}] Generating Lexical Metric Plot: {metric} with 95% CIs...")
            y_max = min(1.0, max(0.3, df_ml[metric].max() + 0.1))

            if 'swap_type' in df_ml.columns:
                plt.figure(figsize=(8, 6))
                order = custom_sort(df_ml['swap_type'].unique())
                
                ax = sns.barplot(data=df_ml, x='swap_type', y=metric, order=order, errorbar=('ci', 95), capsize=0.1, palette="viridis")
                
                plt.title(f'({ml_label}): {metric} by Swap Type (95% CI)', pad=20)
                plt.ylabel(metric_label)
                plt.xlabel('Swap Type')
                plt.ylim(0, y_max)
                
                for i in ax.containers:
                    try: ax.bar_label(i, fmt='%.3f', padding=15)
                    except: pass
                    
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'{metric}_swap_type_ci_{ml_label}.png'), dpi=300)
                plt.close()

            if 'dominance' in df_ml.columns:
                plt.figure(figsize=(10, 6))
                order = custom_sort(df_ml['dominance'].unique())
                
                ax = sns.barplot(data=df_ml, x='dominance', y=metric, order=order, errorbar=('ci', 95), capsize=0.1, palette="magma")
                
                plt.title(f'({ml_label}): {metric} by Language Dominance', pad=20)
                plt.ylabel(metric_label)
                plt.xlabel('Dominance Strategy')
                plt.xticks(rotation=45)
                plt.ylim(0, y_max)
                
                for i in ax.containers:
                    try: ax.bar_label(i, fmt='%.3f', padding=15)
                    except: pass
                    
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'{metric}_dominance_ci_{ml_label}.png'), dpi=300)
                plt.close()

    print(f"Check the '{output_dir}' directory for the updated CI PNG files separated by Matrix Language.")

if __name__ == "__main__":
    visualize_means_with_cis()
