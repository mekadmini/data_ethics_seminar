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
    
    # Calculate the per-prompt ratio
    df['prompt_asr_both'] = df['unsafe_both'] / df['total_lines']
    df['prompt_asr_resp'] = df['unsafe_response'] / df['total_lines']
    
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    
    def custom_sort(vals):
        vals = sorted(list(set(vals)))
        if 'Baseline' in vals:
            vals.insert(0, vals.pop(vals.index('Baseline')))
        return vals

    languages = ['EN', 'IT']
    metrics = [
        ('prompt_asr_both', 'Contextual ASR', 'husl'),
        ('prompt_asr_resp', 'Response-Only ASR', 'husl')
    ]

    for metric, label, cmap in metrics:
        # Determine shared Y/X max
        m_max = df[metric].max()
        if pd.isna(m_max) or m_max == 0: m_max = 1.0
        else: m_max = min(1.0, m_max * 1.15)

        # 1. Consolidated Violin Plot
        fig, axes = plt.subplots(1, 2, figsize=(18, 6), sharey=True)
        for i, lang in enumerate(languages):
            df_lang = df[df['matrix_language'] == lang]
            if df_lang.empty:
                axes[i].text(0.5, 0.5, f'No data for {lang}', ha='center')
                continue
            
            order = custom_sort(df_lang['swap_type'].unique())
            sns.violinplot(data=df_lang, x='swap_type', y=metric, order=order, 
                           palette=cmap, inner=None, alpha=0.5, cut=0, ax=axes[i])
            sns.stripplot(data=df_lang, x='swap_type', y=metric, order=order, 
                          color="black", alpha=0.3, jitter=0.15, size=4, ax=axes[i])
            
            axes[i].set_title(f'{lang} Matrix', fontsize=14)
            axes[i].set_ylim(0, m_max)
            axes[i].set_ylabel(f'Prompt {label}' if i == 0 else "")
            axes[i].set_xlabel('Swap Type')

        plt.suptitle(f'{label} Distribution per Prompt', fontsize=16, y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{metric}_consolidated_violin.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # 2. Consolidated ECDF Plot
        fig, axes = plt.subplots(1, 2, figsize=(18, 6), sharey=True)
        for i, lang in enumerate(languages):
            df_lang = df[df['matrix_language'] == lang]
            if df_lang.empty:
                axes[i].text(0.5, 0.5, f'No data for {lang}', ha='center')
                continue
            
            order = custom_sort(df_lang['swap_type'].unique())
            sns.ecdfplot(data=df_lang, x=metric, hue='swap_type', hue_order=order, palette=cmap, linewidth=2.5, ax=axes[i])
            
            axes[i].set_title(f'{lang} Matrix', fontsize=14)
            axes[i].set_xlim(0, m_max)
            axes[i].set_xlabel(f'{label} Rate')
            axes[i].set_ylabel('Proportion of Prompts' if i == 0 else "")
            axes[i].grid(True, which='both', linestyle='--', alpha=0.6)
            
            if i == 0:
                axes[i].get_legend().remove()
            else:
                axes[i].legend(title='Swap Type', loc='lower right', frameon=True, shadow=True)

        plt.suptitle(f'Cumulative Distribution of {label}', fontsize=16, y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{metric}_consolidated_ecdf.png'), dpi=300, bbox_inches='tight')
        plt.close()

    print(f"\n✅ Consolidated Prompt ASR plots saved to {output_dir}")

if __name__ == "__main__":
    visualize_prompt_asr()
