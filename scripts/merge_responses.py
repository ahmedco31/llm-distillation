import pandas as pd
from pathlib import Path

def merge_datasets():
    print("Loading datasets...")
    
    # Load original datasets
    v1_path = Path("data/responses_openai.csv")
    v2_path = Path("data/responses_v2.csv")
    output_path = Path("data/responses_combined.csv")
    
    if not v1_path.exists() or not v2_path.exists():
        print("Error: Missing input files in data/ directory.")
        return

    df_v1 = pd.read_csv(v1_path)
    df_v2 = pd.read_csv(v2_path)
    
    # Add batch identifiers to track source
    df_v1["batch"] = "v1_gpt4o_mini"
    df_v2["batch"] = "v2_gpt4o"
    
    print(f"Loaded {len(df_v1)} rows from v1 and {len(df_v2)} rows from v2.")
    
    # Combine datasets
    combined = pd.concat([df_v1, df_v2], ignore_index=True)
    
    # Sort logically
    # This groups them by category, then by the prompt ID, then by model/batch
    # This ensures related prompts stay together and categories are grouped
    combined = combined.sort_values(by=["category", "id", "batch"])
    
    # Save to a new file (leaving originals intact)
    combined.to_csv(output_path, index=False)
    
    print(f"\n✓ Successfully merged into {output_path}")
    print(f"Total rows: {len(combined)}")
    print("\nBreakdown by category and model:")
    print(combined.groupby(["category", "model"]).size().unstack(fill_value=0))

if __name__ == "__main__":
    merge_datasets()
