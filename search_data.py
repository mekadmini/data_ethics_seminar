import pandas as pd
from pathlib import Path

def search_data(base_path, search_term, column_name):
    base_dir = Path(base_path)
    experiment_results_dir = base_dir / "experiment_results"
    output_file = experiment_results_dir / "merged_all_data.csv"

    if not experiment_results_dir.exists():
        print(f"Error: {experiment_results_dir} does not exist.")
        return

    # Find all study folders (both study_ and baseline_)
    study_folders = [f for f in experiment_results_dir.iterdir() if f.is_dir() and (f.name.startswith("study_") or f.name.startswith("baseline_"))]

    all_results = []
    print("Collecting and merging raw files into one big dataframe...")

    for study_folder in study_folders:
        subfolders = [f for f in study_folder.iterdir() if f.is_dir()]
        for subfolder in subfolders:
            prompts_file = subfolder / "prompts.csv"
            results_file = subfolder / "results_evaluated.csv"

            if prompts_file.exists() and results_file.exists():
                try:
                    prompts_df = pd.read_csv(prompts_file)
                    results_df = pd.read_csv(results_file)
                except pd.errors.EmptyDataError:
                    print(f"Warning: Empty CSV file found in {subfolder.name}, skipping.")
                    continue

                # Locate the original language text (e.g. 'en', 'it', 'zh')
                lang_cols = [col for col in prompts_df.columns if col not in ['csrt', 'id', 'iteration', 'translation_iter']]
                if lang_cols:
                    prompts_df = prompts_df.rename(columns={lang_cols[0]: 'original_prompt'})
                else:
                    prompts_df['original_prompt'] = pd.NA

                # Locate the response column
                response_cols = [col for col in results_df.columns if col.startswith("response_") and col != "response_classification"]
                if response_cols:
                    results_df = results_df.rename(columns={response_cols[0]: 'response'})
                else:
                    results_df['response'] = pd.NA

                # Merge on the code switched prompt ('csrt')
                joined_df = pd.merge(results_df, prompts_df, on="csrt", how="left")
                
                # Add useful metadata columns to filter on later
                joined_df["study_name"] = study_folder.name
                joined_df["configuration"] = subfolder.name
                
                all_results.append(joined_df)

    if not all_results:
        print("No results found to merge.")
        return

    # Combine all pieces into a single large dataframe
    final_df = pd.concat(all_results, ignore_index=True)
    print(final_df.columns)
    final_df = final_df[final_df[column_name].str.contains(search_term, na=False)]
    print(f"Total merged rows collected: {len(final_df)}")

    # Reorder columns to make it easy to read (putting the most important ones up front)
    front_cols = ['study_name', 'configuration', 'id', 'translation_iter', 'original_prompt', 'csrt', 'response', 'response_classification']
    front_cols = [c for c in front_cols if c in final_df.columns]
    other_cols = [c for c in final_df.columns if c not in front_cols]
    final_df = final_df[front_cols + other_cols]

    # Save out
    final_df.to_csv(output_file, index=False)
    print(f"Successfully saved giant merged data frame to {output_file}")


if __name__ == "__main__":
    # Get the directory where the script is located
    base_path = Path(__file__).parent.resolve()
    SEARCH_TERM = "unsafe"
    COLUMN_NAME = "safe_response"
    # Available columns ['csrt', 'response', 'iteration_x', 'safe_prompt', 'safe_response',
    #        'safe_both', 'response_classification', 'id', 'original_prompt',
    #        'iteration_y', 'study_name', 'configuration', 'iteration',
    #        'translation_iter']
    search_data(base_path, SEARCH_TERM, COLUMN_NAME)
