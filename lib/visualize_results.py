import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

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
    
    # Standardize
    df['matrix_language'] = df['matrix_language'].str.upper()
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.4)
    
    # Fill nan
    df['swap_type'] = df['swap_type'].fillna('Baseline')
    df['dominance'] = df['dominance'].fillna('Baseline')
    
    # Simple ratio calculations
    df['ASR_both'] = df['unsafe_both'] / df['total_lines']
    df['ASR_resp'] = df['unsafe_response'] / df['total_lines']
    
    for col, new_col in [('compliance_count', 'Pct_Compliance'), 
                         ('benign_compliance_count', 'Pct_Benign_Compliance'),
                         ('malicious_compliance_count', 'Pct_Malicious_Compliance'),
                         ('refusal_count', 'Pct_Refusal')]:
        if col in df.columns:
            df[new_col] = df[col] / df['total_lines']

    metrics_labels = [
        ('ASR_both', 'Average Attack Success Rate'),
        ('Pct_Compliance', 'Total Compliance Rate'),
        ('Pct_Refusal', 'Refusal Rate'),
        ('Pct_Benign_Compliance', 'Benign Compliance Rate')
    ]

    def custom_sort(vals):
        vals = sorted(list(set(vals)))
        if 'Baseline' in vals:
            vals.insert(0, vals.pop(vals.index('Baseline')))
        return vals

    # Use a vibrant palette for consistency
    palette = "husl"
    languages = ['EN', 'IT']

    # 1. Consolidated CI Spread Plots
    for metric, metric_label in metrics_labels:
        if metric in df.columns:
            y_max = df[metric].max()
            if pd.isna(y_max) or y_max == 0: y_max = 1.0
            else: y_max = min(1.0, y_max * 1.15)

            fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharey=True)
            for i, lang in enumerate(languages):
                df_lang = df[df['matrix_language'] == lang]
                if df_lang.empty: continue
                
                order = custom_sort(df_lang['swap_type'].unique())
                sns.barplot(data=df_lang, x='swap_type', y=metric, order=order, 
                            hue='swap_type', hue_order=order, dodge=False,
                            ci=95, capsize=0.05, palette=palette, ax=axes[i], alpha=0.9)
                
                axes[i].set_title(f'{lang} Matrix', fontsize=20, pad=15)
                axes[i].set_ylabel(f'{metric_label}' if i == 0 else "")
                axes[i].set_xlabel("Swap Type")
                axes[i].set_ylim(0, y_max)
                axes[i].tick_params(axis='x', rotation=15)
                axes[i].get_legend().remove()

            handles, labels = axes[1].get_legend_handles_labels()
            axes[1].legend(handles, labels, title="Swap Type", loc='lower right', 
                           frameon=True, shadow=True, fontsize=10)

            # plt.suptitle(f'{metric_label} by Strategy', fontsize=24, y=1.05)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{metric}_consolidated_ci_spread.png'), dpi=300, bbox_inches='tight')
            plt.close()

if __name__ == "__main__":
    visualize_means_with_cis()
