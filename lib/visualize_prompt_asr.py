import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def visualize_prompt_asr():
    exp_file = 'experiment_results/aggregated_results.csv'
    baseline_file = 'experiment_results/aggregated_baseline.csv'
    output_dir = 'experiment_results/plots'
    
    if not os.path.exists(exp_file):
        print(f"Error: Could not find {exp_file}")
        return
        
    df = pd.read_csv(exp_file)
    
    # Load and append baseline if it exists
    if os.path.exists(baseline_file):
        df_base = pd.read_csv(baseline_file)
        df_base['swap_type'] = 'Baseline'
        df_base['dominance'] = 'Baseline'
        df_base['ratio'] = 0.0
        df = pd.concat([df, df_base], ignore_index=True)
        
    # Standardize string capitalization
    df['matrix_language'] = df['matrix_language'].str.upper()
    df['swap_type'] = df['swap_type'].fillna('Baseline')
    
    # Calculate the per-prompt percentage
    df['prompt_asr_both'] = df['unsafe_both'] / df['total_lines']
    df['prompt_asr_resp'] = df['unsafe_response'] / df['total_lines']
    
    # Convert proportions to percentages for easier reading
    df['prompt_asr_both_pct'] = df['prompt_asr_both'] * 100
    df['prompt_asr_resp_pct'] = df['prompt_asr_resp'] * 100
    
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    
    def custom_sort(vals):
        vals = sorted(list(set(vals)))
        if 'Baseline' in vals:
            vals.insert(0, vals.pop(vals.index('Baseline')))
        return vals

    matrix_languages = df['matrix_language'].dropna().unique()
    for ml in matrix_languages:
        ml_label = str(ml).upper()
        print(f"\n--- Generating Prompt Plots for Matrix Language: {ml_label} ---")
        
        df_ml = df[df['matrix_language'] == ml]
        if df_ml.empty:
            continue
            
        order = custom_sort(df_ml['swap_type'].unique())
        
        # Plot 1: Violin + Stripplot for unsafe_both
        print(f"[{ml_label}] Generating Contextual ASR Violin Plot (unsafe_both)...")
        plt.figure(figsize=(10, 6))
        
        # Violin plot to show density/shape of the distribution
        sns.violinplot(data=df_ml, x='swap_type', y='prompt_asr_both_pct', order=order, 
                       palette="viridis", inner=None, alpha=0.5, cut=0) # cut=0 keeps the violin strictly within data bounds
        
        # Stripplot to show the actual data points
        sns.stripplot(data=df_ml, x='swap_type', y='prompt_asr_both_pct', order=order, 
                      color="black", alpha=0.3, jitter=0.15, size=4)
        
        plt.title(f'Contextual ASR Distribution per Prompt ({ml_label})', pad=20)
        plt.ylabel('Prompt ASR (%) [unsafe_both / total_lines]')
        plt.xlabel('Swap Type')
        plt.ylim(-5, 105)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'prompt_asr_both_violin_{ml_label}.png'), dpi=300)
        plt.close()
        
        # Plot 2: ECDF Plot for unsafe_both
        print(f"[{ml_label}] Generating Contextual ECDF Plot (unsafe_both)...")
        plt.figure(figsize=(10, 6))
        
        sns.ecdfplot(data=df_ml, x='prompt_asr_both_pct', hue='swap_type', hue_order=order, palette="viridis", linewidth=2.5)
        
        plt.title(f'Cumulative Distribution of Contextual ASR ({ml_label})', pad=20)
        plt.xlabel('Attack Success Rate (%)')
        plt.ylabel('Proportion of Prompts')
        # Add gridlines for readability
        plt.grid(True, which='both', linestyle='--', alpha=0.6)
        plt.xlim(-5, 105)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'prompt_asr_both_ecdf_{ml_label}.png'), dpi=300)
        plt.close()
        
        # Plot 3: Violin Plot for unsafe_response
        print(f"[{ml_label}] Generating Response-Only ASR Violin Plot (unsafe_response)...")
        plt.figure(figsize=(10, 6))
        sns.violinplot(data=df_ml, x='swap_type', y='prompt_asr_resp_pct', order=order, 
                       palette="magma", inner=None, alpha=0.5, cut=0)
        sns.stripplot(data=df_ml, x='swap_type', y='prompt_asr_resp_pct', order=order, 
                      color="black", alpha=0.3, jitter=0.15, size=4)
        
        plt.title(f'Response-Only ASR Distribution per Prompt ({ml_label})', pad=20)
        plt.ylabel('Prompt ASR (%) [unsafe_response / total_lines]')
        plt.xlabel('Swap Type')
        plt.ylim(-5, 105)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'prompt_asr_resp_violin_{ml_label}.png'), dpi=300)
        plt.close()

        # Plot 4: ECDF Plot for unsafe_response
        print(f"[{ml_label}] Generating Response-Only ECDF Plot (unsafe_response)...")
        plt.figure(figsize=(10, 6))
        
        sns.ecdfplot(data=df_ml, x='prompt_asr_resp_pct', hue='swap_type', hue_order=order, palette="magma", linewidth=2.5)
        
        plt.title(f'Cumulative Distribution of Response-Only ASR ({ml_label})', pad=20)
        plt.xlabel('Attack Success Rate (%)')
        plt.ylabel('Proportion of Prompts')
        plt.grid(True, which='both', linestyle='--', alpha=0.6)
        plt.xlim(-5, 105)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'prompt_asr_resp_ecdf_{ml_label}.png'), dpi=300)
        plt.close()
        
    print(f"\n✅ All Prompt ASR scatter plots saved to {output_dir}")

if __name__ == "__main__":
    visualize_prompt_asr()
