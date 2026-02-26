import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

def visualize_variance(csv_path='experiment_results/metrics_summary.csv', 
                       baseline_metrics_path='experiment_results/baseline_metrics.csv',
                       output_dir='experiment_results/plots'):
    if not os.path.exists(csv_path):
        print(f"Error: Could not find {csv_path}")
        return
        
    df = pd.read_csv(csv_path)
    
    # Load and combine baseline metrics if they exist
    if os.path.exists(baseline_metrics_path):
        df_base = pd.read_csv(baseline_metrics_path)
        df_base['swap_type'] = 'Baseline'
        df_base['dominance'] = 'Baseline'
        df = pd.concat([df, df_base], ignore_index=True)
    
    df['matrix_language'] = df['matrix_language'].str.upper()
    os.makedirs(output_dir, exist_ok=True)
    
    # Global theme matching reference (whitegrid, clean)
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.4)
    
    metrics_to_plot = ['ASR_both', 'Pct_Compliance', 'Pct_Refusal', 'Pct_Benign_Compliance']
    
    def custom_sort(vals):
        vals = sorted(list(set(vals)))
        if 'Baseline' in vals:
            vals.insert(0, vals.pop(vals.index('Baseline')))
        return vals

    # Manual color map to match the specific "purple to orange/peach" reference exactly
    color_palette = {
        'Arabic_Dom': "#1C0F4A",   # Very dark indigo/purple
        'Balanced': "#440F76",     # Deep purple
        'Greek_Dom': "#7B1B82",    # Purple/magenta
        'Japanese_Dom': "#F1495A", # Bright red
        'Spanish_Dom': "#FF804D",  # Vibrant orange
        'Baseline': "#FFBF80"      # Light peach/orange
    }

    # If any dominance strategies are missing in the map, fill with magma
    ext_strategies = df['dominance'].unique()
    for s in ext_strategies:
        if s not in color_palette:
            color_palette[s] = "#808080" # Default gray for unexpected ones

    dom_strategies = custom_sort(df['dominance'].unique())

    for metric in metrics_to_plot:
        if metric not in df.columns: continue
        
        # Shared Dynamic Y Max
        y_max = df[metric].max()
        if pd.isna(y_max) or y_max == 0:
            y_max = 1.0
        else:
            y_max = min(1.0, y_max * 1.15)

        if metric == 'ASR_both':
            m_label = 'Contextual ASR'
        else:
            m_label = metric.replace('Pct_', '').replace('_', ' ')
            if 'Rate' not in m_label: m_label += ' Rate'

        # 1. Consolidated Configuration Spread (Box/Swarm)
        fig, axes = plt.subplots(1, 2, figsize=(20, 8), sharey=True)
        languages = ['EN', 'IT']
        handles, labels = [], []
        
        for i, lang in enumerate(languages):
            df_lang = df[df['matrix_language'] == lang]
            if df_lang.empty:
                axes[i].text(0.5, 0.5, f'No data for {lang}', ha='center')
                continue
            
            order = custom_sort(df_lang['swap_type'].unique())
            
            # Boxplot with reference styling (light-gray/greenish fills)
            sns.boxplot(data=df_lang, x='swap_type', y=metric, order=order, 
                        ax=axes[i], color="#B2D8D8", boxprops=dict(alpha=0.6))
            
            # Swarmplot with beauty: colored by dominance using the manual palette
            sns.swarmplot(data=df_lang, x='swap_type', y=metric, hue='dominance', 
                          hue_order=dom_strategies, palette=color_palette, 
                          order=order, ax=axes[i], size=7, alpha=0.9, dodge=False)
            
            axes[i].set_title(f'{lang} Matrix', fontsize=20, pad=15)
            axes[i].set_ylim(0, y_max)
            axes[i].set_ylabel(m_label if i == 0 else "")
            axes[i].set_xlabel("Swap Type")
            axes[i].tick_params(axis='x', rotation=15)
            
            h, l = axes[i].get_legend_handles_labels()
            if h:
                handles, labels = h, l
            axes[i].get_legend().remove()

        # Shared legend inside the second subplot (bottom right)
        if handles:
            axes[1].legend(handles, labels, title="Dominance", loc='lower right', 
                           frameon=True, shadow=True, fontsize=10)

        # plt.suptitle(f'{m_label} Distribution Across All Configurations', fontsize=24, y=1.05)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{metric}_consolidated_spread.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # 2. Consolidated Interaction Plot
        df_inter = df[df['swap_type'] != 'Baseline']
        if not df_inter.empty and 'ratio' in df_inter.columns and 'dominance' in df_inter.columns:
            fig, axes = plt.subplots(1, 2, figsize=(20, 8), sharey=True)
            handles_inter, labels_inter = [], []
            
            for i, lang in enumerate(languages):
                df_lang = df_inter[df_inter['matrix_language'] == lang]
                if df_lang.empty:
                    axes[i].text(0.5, 0.5, f'No data for {lang}', ha='center')
                    continue
                
                sns.barplot(data=df_lang, x='ratio', y=metric, hue='dominance', 
                            hue_order=[s for s in dom_strategies if s != 'Baseline'],
                            palette=color_palette, ci=95, ax=axes[i], alpha=0.85, capsize=0.05)
                
                axes[i].set_title(f'{lang} Matrix', fontsize=20, pad=15)
                axes[i].set_ylim(0, y_max)
                axes[i].set_ylabel(m_label if i == 0 else "")
                axes[i].set_xlabel("Language Swap Ratio")
                
                h, l = axes[i].get_legend_handles_labels()
                if h:
                    handles_inter, labels_inter = h, l
                axes[i].get_legend().remove()

            if handles_inter:
                axes[1].legend(handles_inter, labels_inter, title="Dominance Strategy", loc='lower right', 
                               frameon=True, shadow=True, fontsize=10)

            # plt.suptitle(f'Interaction Effect: {m_label} by Ratio & Dominance', fontsize=24, y=1.05)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{metric}_consolidated_interaction.png'), dpi=300, bbox_inches='tight')
            plt.close()

    # 3. Consolidated Grouped Spread Plot for Response Classes
    response_metrics = [
        'Pct_Confusion', 'Pct_Rebuttal', 'Pct_Benign_Compliance', 
        'Pct_Malicious_Compliance', 'Pct_Refusal'
    ]
    
    available_response_metrics = [m for m in response_metrics if m in df.columns]
    
    if available_response_metrics:
        # Include 'dominance' in id_vars to color the swarm plots
        id_vars = ['matrix_language', 'swap_type', 'dominance']
        df_melted = df.melt(id_vars=id_vars, value_vars=available_response_metrics, 
                            var_name='Response Class', value_name='Rate')
        
        df_melted['Response Class'] = df_melted['Response Class'].str.replace('Pct_', '').str.replace('_', ' ')
        languages = ['EN', 'IT']
        
        swap_order = custom_sort(df['swap_type'].unique())
        resp_order = [m.replace('Pct_', '').replace('_', ' ') for m in available_response_metrics]
        
        x_order = []
        for swap in swap_order:
            for resp in resp_order:
                x_order.append(f"{swap}_{resp}")
                
        df_melted['Swap_Resp_Class'] = df_melted['swap_type'] + '_' + df_melted['Response Class']
        
        y_min = df_melted['Rate'].min()
        y_max = df_melted['Rate'].max()
        padding = (y_max - y_min) * 0.05 if y_max > y_min else 0.05
        
        fig, axes = plt.subplots(1, 2, figsize=(28, 10), sharey=True)
        handles, labels = [], []
        
        for i, lang in enumerate(languages):
            df_lang = df_melted[df_melted['matrix_language'] == lang]
            if df_lang.empty:
                axes[i].text(0.5, 0.5, f'No data for {lang}', ha='center')
                continue
                
            sns.boxplot(data=df_lang, x='Swap_Resp_Class', y='Rate', order=x_order, 
                        ax=axes[i], color="#B2D8D8", boxprops=dict(alpha=0.6))
            
            sns.swarmplot(data=df_lang, x='Swap_Resp_Class', y='Rate', hue='dominance', 
                          hue_order=dom_strategies, palette=color_palette, 
                          order=x_order, ax=axes[i], size=6, alpha=0.9, dodge=False)
                          
            axes[i].set_title(f'{lang} Matrix', fontsize=20, pad=15)
            axes[i].set_ylim(y_min - padding, y_max + padding)
            axes[i].set_ylabel("Rate" if i == 0 else "")
            axes[i].set_xlabel("")
            
            # Fix X-Ticks to show only response classes
            tick_labels = [resp for swap in swap_order for resp in resp_order]
            axes[i].set_xticks(range(len(x_order)))
            axes[i].set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=12)
            
            # Draw separators and Swap labels Below Axis
            num_resps = len(resp_order)
            for s_idx, swap in enumerate(swap_order):
                center_x = s_idx * num_resps + (num_resps - 1) / 2.0
                # Transform uses relative 0-1 for Y, so negative pushes below x-axis
                axes[i].text(center_x, -0.22, swap, ha='center', va='top', 
                             fontsize=16, fontweight='bold', 
                             transform=axes[i].get_xaxis_transform())
                
                if s_idx > 0:
                    sep_pos = s_idx * num_resps - 0.5
                    axes[i].axvline(sep_pos, color='gray', linestyle='--', alpha=0.5)
            
            h, l = axes[i].get_legend_handles_labels()
            if h:
                handles, labels = h, l
            axes[i].get_legend().remove()
            
        if handles:
            axes[1].legend(handles, labels, title="Dominance Strategy", loc='upper right', frameon=True, fontsize=12, title_fontsize=14)
            
        # plt.suptitle('Response Classes Distribution by Swap Type', fontsize=26, y=1.05)
        # Prevent cutoff of bottom text labels
        plt.subplots_adjust(bottom=0.25)
        plt.savefig(os.path.join(output_dir, 'Response_Classes_consolidated_spread.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # 4. 100% Stacked Bar Chart for Response Compositions
        stack_metrics = ['Pct_Benign_Compliance', 'Pct_Rebuttal', 'Pct_Confusion', 'Pct_Refusal', 'Pct_Malicious_Compliance']
        available_stack = [m for m in stack_metrics if m in df.columns]
        
        if available_stack:
            # Prepare aggregation for bar chart
            df_agg = df.groupby(['matrix_language', 'swap_type'])[available_stack].mean().reset_index()
            # Normalize to 100%
            df_agg['Total'] = df_agg[available_stack].sum(axis=1)
            for m in available_stack:
                df_agg[m] = (df_agg[m] / df_agg['Total']) * 100
            
            fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharey=True)
            
            # Semantic, professional, colorblind-friendly palette
            # Benign (Green) and Malicious (Red) are separated by other metrics
            custom_colors = {
                'Pct_Benign_Compliance': '#2a9d8f',  # Teal/Green
                'Pct_Rebuttal': '#e9c46a',           # Muted Gold/Orange
                'Pct_Confusion': '#adb5bd',          # Slate Gray
                'Pct_Refusal': '#0077b6',            # Rich Blue
                'Pct_Malicious_Compliance': '#e63946' # Deep Red
            }
            stack_colors = [custom_colors.get(m, '#000000') for m in available_stack]
            
            for i, lang in enumerate(languages):
                df_lang = df_agg[df_agg['matrix_language'] == lang].set_index('swap_type').reindex(swap_order)
                if df_lang.dropna().empty:
                    axes[i].text(0.5, 0.5, f'No data for {lang}', ha='center')
                    continue
                
                df_lang[available_stack].plot(kind='bar', stacked=True, ax=axes[i], color=stack_colors, alpha=0.9, edgecolor='white', width=0.6)
                
                axes[i].set_title(f'{lang} Matrix', fontsize=20, pad=15)
                axes[i].set_ylabel("Composition %" if i == 0 else "")
                axes[i].set_xlabel("Swap Type")
                axes[i].tick_params(axis='x', rotation=0)
                
                if i == 0:
                    axes[i].get_legend().remove()
                else:
                    labels = [m.replace('Pct_', '').replace('_', ' ') for m in available_stack]
                    axes[i].legend(labels, title="Response Class", loc='upper right', bbox_to_anchor=(1.25, 1.05), frameon=True, shadow=True, fontsize=10)
                                   
            # plt.suptitle('Average Response Class Composition by Swap Type', fontsize=24, y=1.05)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'Response_Classes_stacked_bar.png'), dpi=300, bbox_inches='tight')
            plt.close()

            # 5. Clean Heatmap of Response Rates
            fig, axes = plt.subplots(1, 2, figsize=(20, 8))
            for i, lang in enumerate(languages):
                df_lang = df_agg[df_agg['matrix_language'] == lang].set_index('swap_type').reindex(swap_order)
                if df_lang.dropna().empty:
                    axes[i].text(0.5, 0.5, f'No data for {lang}', ha='center')
                    continue
                
                # We want Response Classes as Rows, Swap Types as Columns
                heat_data = df_lang[available_stack].T
                heat_data.index = [m.replace('Pct_', '').replace('_', ' ') for m in available_stack]
                
                sns.heatmap(heat_data, annot=True, fmt=".1f", cmap="YlOrRd", ax=axes[i], 
                            cbar_kws={'label': 'Average Rate (%)'}, annot_kws={"size": 12}, 
                            linewidths=0.5, linecolor='white')
                
                axes[i].set_title(f'{lang} Matrix', fontsize=20, pad=15)
                axes[i].set_ylabel("Response Class" if i == 0 else "")
                axes[i].set_xlabel("Swap Type")
                axes[i].tick_params(axis='x', rotation=0)
                axes[i].tick_params(axis='y', rotation=0)
                
            # plt.suptitle('Heatmap: Average Response Rate across Configurations', fontsize=24, y=1.05)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'Response_Classes_heatmap.png'), dpi=300, bbox_inches='tight')
            plt.close()

            # 6. Smooth Trend Line Chart (Progression of Complexity)
            fig, axes = plt.subplots(1, 2, figsize=(20, 8), sharey=True)
            for i, lang in enumerate(languages):
                df_lang = df[df['matrix_language'] == lang]
                if df_lang.empty:
                    axes[i].text(0.5, 0.5, f'No data for {lang}', ha='center')
                    continue
                
                for j, metric in enumerate(available_stack):
                    label = metric.replace('Pct_', '').replace('_', ' ')
                    sns.lineplot(data=df_lang, x='swap_type', y=metric, color=stack_colors[j],
                                 label=label, marker='o', linewidth=3, markersize=10, 
                                 ci='sd', ax=axes[i])
                
                axes[i].set_title(f'{lang} Matrix Trend', fontsize=20, pad=15)
                axes[i].set_ylabel("Mean Rate (±1 SD)" if i == 0 else "")
                axes[i].set_xlabel("Swap Type (Complexity Progression)")
                axes[i].tick_params(axis='x', rotation=0)
                
                if i == 0:
                    axes[i].get_legend().remove()
                else:
                    axes[i].legend(title="Response Class", loc='upper left', bbox_to_anchor=(1.05, 1.0), frameon=True, shadow=True, fontsize=11)
            
            # plt.suptitle('Trends in Defense & Attack Complexity', fontsize=24, y=1.05)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'Response_Classes_trend_lines.png'), dpi=300, bbox_inches='tight')
            plt.close()

if __name__ == "__main__":
    visualize_variance()
