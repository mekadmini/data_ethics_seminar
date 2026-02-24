import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def visualize_variance(csv_path='experiment_results/metrics_summary.csv',
                       baseline_path='experiment_results/baseline_metrics.csv',
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
    
    # Fill nan just in case
    df['swap_type'] = df['swap_type'].fillna('Baseline')
    if 'dominance' in df.columns:
        df['dominance'] = df['dominance'].fillna('Baseline')
    if 'ratio' in df.columns:
        df['ratio'] = df['ratio'].fillna(0.0)

    def custom_sort(vals):
        vals = sorted(list(set(vals)))
        if 'Baseline' in vals:
            vals.insert(0, vals.pop(vals.index('Baseline')))
        return vals

    metrics_to_plot = [
        'ASR_both', 'ASR_resp', 
        'OneSuccess_both', 'OneSuccess_resp', 
        'Consistent_both', 'Consistent_resp', 
        'MaxPromptASR_both', 'MaxPromptASR_resp',
        'Pct_Compliance', 'Pct_Refusal', 
        'Pct_Confusion', 'Pct_Rebuttal',
        'Pct_Benign_Compliance', 'Pct_Malicious_Compliance'
    ]

    # Loop through unique matrix languages
    matrix_languages = df['matrix_language'].dropna().unique()
    for ml in matrix_languages:
        ml_label = str(ml).upper()
        print(f"\n--- Generating Config Variance Plots for Matrix Language: {ml_label} ---")
        
        df_ml = df[df['matrix_language'] == ml].copy()
        if df_ml.empty:
            continue
            
        for metric in metrics_to_plot:
            if metric not in df_ml.columns:
                continue
                
            print(f"[{ml_label}] Generating Config Spread for: {metric}...")
            plt.figure(figsize=(9, 6))
            order = custom_sort(df_ml['swap_type'].unique())
            
            # Boxplot for the distribution range, Swarmplot to show every single config
            sns.boxplot(data=df_ml, x='swap_type', y=metric, order=order, palette="viridis", showfliers=False, boxprops={'alpha':0.4})
            sns.swarmplot(data=df_ml, x='swap_type', y=metric, order=order, hue='dominance', palette="magma", size=6, dodge=False)
            
            plt.title(f'{metric} ({ml_label}): Distribution Across All Configurations', pad=20)
            plt.ylabel(metric)
            plt.xlabel('Swap Type')
            plt.legend(title="Dominance", bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{metric}_config_spread_{ml_label}.png'), dpi=300)
            plt.close()

            # 2. Grouped Bar Chart: Swap Type split by Ratio
            print(f"[{ml_label}] Generating {metric} x Ratio Grouped Bar Chart...")
            plt.figure(figsize=(10, 6))
            
            # Exclude Baseline from this specific grouped chart if we want to focus on Ratio scaling
            df_no_base = df_ml[df_ml['swap_type'] != 'Baseline']
            if not df_no_base.empty:
                sns.barplot(data=df_no_base, x='swap_type', y=metric, hue='ratio', palette="YlOrRd", errorbar=None)
                
                plt.title(f'Interaction ({ml_label}): How Ratio changes {metric} for each Swap Type', pad=20)
                plt.ylabel(f'Average {metric}')
                plt.xlabel('Swap Type')
                plt.legend(title='Swap Ratio')
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'{metric}_by_ratio_grouped_{ml_label}.png'), dpi=300)
                plt.close()

            # 3. Grouped Bar Chart: Swap Type split by Dominance
            print(f"[{ml_label}] Generating {metric} x Dominance Grouped Bar Chart...")
            plt.figure(figsize=(12, 6))
            if not df_no_base.empty:
                order_dom = custom_sort(df_no_base['dominance'].unique())
                sns.barplot(data=df_no_base, x='swap_type', y=metric, hue='dominance', hue_order=order_dom, palette="magma", errorbar=None)
                
                plt.title(f'Interaction ({ml_label}): How Language Dominance changes {metric} for each Swap Type', pad=20)
                plt.ylabel(f'Average {metric}')
                plt.xlabel('Swap Type')
                plt.legend(title='Dominance', bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'{metric}_by_dominance_grouped_{ml_label}.png'), dpi=300)
                plt.close()

    print("\nVisualizations saved to 'experiment_results/plots'")

if __name__ == "__main__":
    visualize_variance()
