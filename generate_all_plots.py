import os
import argparse

from lib.visualize_results import visualize_means_with_cis
from lib.visualize_config_variance import visualize_variance
from lib.visualize_prompt_asr import visualize_prompt_asr

def main():
    print("="*70)
    print("Starting Comprehensive Plot Generation Pipeline")
    print("="*70)
    
    print("\n[1/3] Generating Means and Confidence Intervals (Bar Plots)...")
    try:
        visualize_means_with_cis()
    except Exception as e:
        print(f"Error in Step 1: {e}")
        
    print("\n" + "-"*70)
    print("[2/3] Generating Configuration Variance Spread (Box/Swarm Plots)...")
    try:
        visualize_variance()
    except Exception as e:
        print(f"Error in Step 2: {e}")
        
    print("\n" + "-"*70)
    print("[3/3] Generating Prompt-Level ASR Distributions (Violin/ECDF Plots)...")
    try:
        visualize_prompt_asr()
    except Exception as e:
        print(f"Error in Step 3: {e}")
        
    print("\n" + "="*70)
    print("🎉 All plotting scripts executed successfully!")
    print("Check the 'experiment_results/plots' directory for the generated PNG files.")
    print("="*70)

if __name__ == "__main__":
    main()
