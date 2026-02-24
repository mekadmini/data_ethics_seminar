from collections import defaultdict
from pathlib import Path

import pandas as pd


def consolidate_experiment_results(base_path):
    base_dir = Path(base_path)
    experiment_results_dir = base_dir / "experiment_results"
    output_file = experiment_results_dir / "aggregated_results.csv"

    if not experiment_results_dir.exists():
        print(f"Error: {experiment_results_dir} does not exist.")
        return

    # Group subfolders by name across all study folders
    config_groups = defaultdict(list)
    study_folders = [f for f in experiment_results_dir.iterdir() if f.is_dir() and f.name.startswith("study_")]

    for study_folder in study_folders:
        subfolders = [f for f in study_folder.iterdir() if f.is_dir()]
        for subfolder in subfolders:
            config_groups[subfolder.name].append(subfolder)

    all_results = []
    print("Collecting and merging raw files...")

    for config_name, folders in config_groups.items():
        prompts_dfs = []
        results_dfs = []

        for folder in folders:
            prompts_file = folder / "prompts.csv"
            results_file = folder / "results_evaluated.csv"

            if prompts_file.exists():
                df = pd.read_csv(prompts_file)
                # Rename language column to 'prompt'
                # Language column is the one that is not 'csrt', 'id', or 'translation_iter'
                lang_cols = [col for col in df.columns if col not in ['csrt', 'id', 'translation_iter']]
                if lang_cols:
                    df = df.rename(columns={lang_cols[0]: 'prompt'})
                prompts_dfs.append(df)

            if results_file.exists():
                df = pd.read_csv(results_file)
                # Rename response column
                response_cols = [col for col in df.columns if col.startswith("response_")]
                if response_cols:
                    df = df.rename(columns={response_cols[0]: 'response'})
                results_dfs.append(df)

        if not prompts_dfs or not results_dfs:
            continue

        # Merge prompts
        merged_prompts = pd.concat(prompts_dfs, ignore_index=True)
        # Drop id and translation_iter
        cols_to_drop = [col for col in ["id", "translation_iter"] if col in merged_prompts.columns]
        merged_prompts = merged_prompts.drop(columns=cols_to_drop)

        # Drop duplicates
        merged_prompts = merged_prompts.drop_duplicates()

        # Merge results_evaluated
        merged_results = pd.concat(results_dfs, ignore_index=True)

        # Join
        joined_df = pd.merge(merged_results, merged_prompts, on="csrt", how="left")

        # Add configuration column
        joined_df["configuration"] = config_name

        # Drop iteration column
        if "iteration" in joined_df.columns:
            joined_df = joined_df.drop(columns=["iteration"])

        all_results.append(joined_df)

    if not all_results:
        print("No results found to merge.")
        return

    # Combine everything into one giant DataFrame
    final_df = pd.concat(all_results, ignore_index=True)
    print(f"Total merged rows collected directly from memory: {len(final_df)}")

    # -------------------------------------------------------------
    # Aggregation Phase
    # -------------------------------------------------------------
    print("\nAggregating data by prompt and configuration...")

    # Helper function to count the number of 'unsafe' occurrences
    def count_unsafe(series):
        return (series == 'unsafe').sum()

    def count_category(series, category):
        return (series == category).sum()

    if 'response_classification' not in final_df.columns:
        final_df['response_classification'] = pd.NA

    # Helper functions for complex counting
    def count_benign_compliance(df_group):
        return ((df_group['response_classification'] == 'compliance') & (df_group['safe_both'] == 'safe')).sum()

    def count_malicious_compliance(df_group):
        return ((df_group['response_classification'] == 'compliance') & (df_group['safe_both'] == 'unsafe')).sum()

    # Group by prompt and configuration
    aggregated_df = final_df.groupby(['prompt', 'configuration']).apply(
        lambda g: pd.Series({
            'total_lines': len(g),
            'unsafe_prompt': count_unsafe(g['safe_prompt']),
            'unsafe_response': count_unsafe(g['safe_response']),
            'unsafe_both': count_unsafe(g['safe_both']),
            'confusion_count': count_category(g['response_classification'], 'confusion'),
            'rebuttal_count': count_category(g['response_classification'], 'rebuttal'),
            'refusal_count': count_category(g['response_classification'], 'refusal'),
            'compliance_count': count_category(g['response_classification'], 'compliance'),
            'benign_compliance_count': count_benign_compliance(g),
            'malicious_compliance_count': count_malicious_compliance(g)
        })
    ).reset_index()

    print("Splitting configuration column...")

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
                ratio = parts[i + 1]
                ratio_idx = i
                break

        # dominance is whatever is left between ratio value and swap_type
        dominance_parts = []
        if ratio_idx != -1 and ratio_idx + 2 < len(parts) - 1:
            dominance_parts = parts[ratio_idx + 2: -1]  # Everything after ratio value up to the last element

        dominance = '_'.join(dominance_parts) if dominance_parts else None

        return pd.Series([matrix_language, ratio, dominance, swap_type])

    # Apply the parsing function
    aggregated_df[['matrix_language', 'ratio', 'dominance', 'swap_type']] = aggregated_df['configuration'].apply(
        parse_config)

    # Drop the original configuration column
    aggregated_df = aggregated_df.drop(columns=['configuration'])

    # Save the aggregated dataframe to a new CSV file
    aggregated_df.to_csv(output_file, index=False)
    print(f"Successfully saved final aggregated output directly to {output_file}")

    print("\nShape of final output:", aggregated_df.shape)
    print("First few rows:")
    print(aggregated_df.head())


def consolidate_baseline_results(base_path):
    base_dir = Path(base_path)
    experiment_results_dir = base_dir / "experiment_results"
    output_file = experiment_results_dir / "aggregated_baseline.csv"

    if not experiment_results_dir.exists():
        print(f"Error: {experiment_results_dir} does not exist.")
        return

    baseline_folders = [f for f in experiment_results_dir.iterdir() if f.is_dir() and f.name.startswith("baseline_")]
    all_results = []
    print("\nCollecting and merging raw baseline files...")

    for study_folder in baseline_folders:
        subfolders = [f for f in study_folder.iterdir() if f.is_dir() and f.name.startswith("baseline_")]
        for subfolder in subfolders:
            matrix_language = subfolder.name.split('_')[-1]

            prompts_file = subfolder / "prompts.csv"
            results_file = subfolder / "results_evaluated.csv"

            if not prompts_file.exists() or not results_file.exists():
                continue

            try:
                prompts_df = pd.read_csv(prompts_file)
                results_df = pd.read_csv(results_file)
            except pd.errors.EmptyDataError:
                print(f"⚠️ Skipping {subfolder.name}: Empty CSV file found.")
                continue

            # The original language text (e.g. 'en', 'it', 'zh') is the column name in prompts.csv
            lang_cols = [col for col in prompts_df.columns if
                         col not in ['csrt', 'id', 'iteration', 'translation_iter']]
            if lang_cols:
                prompts_df = prompts_df.rename(columns={lang_cols[0]: 'prompt'})

            response_cols = [col for col in results_df.columns if col.startswith("response_")]
            if response_cols:
                results_df = results_df.rename(columns={response_cols[0]: 'response'})

            cols_to_drop = [col for col in ["id", "iteration", "translation_iter"] if col in prompts_df.columns]
            prompts_df = prompts_df.drop(columns=cols_to_drop).drop_duplicates()

            joined_df = pd.merge(results_df, prompts_df, on="csrt", how="left")
            joined_df['matrix_language'] = matrix_language

            # Drop 'csrt' since the user requested to drop it (it's identical to the 'prompt' column now)
            if 'csrt' in joined_df.columns:
                joined_df = joined_df.drop(columns=['csrt'])

            all_results.append(joined_df)

    if not all_results:
        print("No baseline results found to merge.")
        return

    final_df = pd.concat(all_results, ignore_index=True)
    print(f"Total merged baseline rows collected directly from memory: {len(final_df)}")

    def count_unsafe(series):
        return (series == 'unsafe').sum()

    def count_category(series, category):
        return (series == category).sum()

    if 'response_classification' not in final_df.columns:
        final_df['response_classification'] = pd.NA

    # Helper functions for complex counting
    def count_benign_compliance(df_group):
        return ((df_group['response_classification'] == 'compliance') & (df_group['safe_both'] == 'safe')).sum()

    def count_malicious_compliance(df_group):
        return ((df_group['response_classification'] == 'compliance') & (df_group['safe_both'] == 'unsafe')).sum()

    print("\nAggregating baseline data by prompt and matrix_language...")
    aggregated_df = final_df.groupby(['prompt', 'matrix_language']).apply(
        lambda g: pd.Series({
            'response': g['response'].iloc[0],
            'total_lines': len(g),
            'unsafe_prompt': count_unsafe(g['safe_prompt']),
            'unsafe_response': count_unsafe(g['safe_response']),
            'unsafe_both': count_unsafe(g['safe_both']),
            'confusion_count': count_category(g['response_classification'], 'confusion'),
            'rebuttal_count': count_category(g['response_classification'], 'rebuttal'),
            'refusal_count': count_category(g['response_classification'], 'refusal'),
            'compliance_count': count_category(g['response_classification'], 'compliance'),
            'benign_compliance_count': count_benign_compliance(g),
            'malicious_compliance_count': count_malicious_compliance(g)
        })
    ).reset_index()

    aggregated_df = aggregated_df[
        ['prompt', 'response', 'total_lines', 'matrix_language', 'unsafe_prompt', 'unsafe_response', 'unsafe_both',
         'confusion_count', 'rebuttal_count', 'refusal_count', 'compliance_count', 'benign_compliance_count',
         'malicious_compliance_count']]

    aggregated_df.to_csv(output_file, index=False)
    print(f"Successfully saved final aggregated baseline output directly to {output_file}")


if __name__ == "__main__":
    # Get the directory where the script is located
    base_path = Path(__file__).parent.resolve()
    consolidate_experiment_results(base_path)
    consolidate_baseline_results(base_path)
