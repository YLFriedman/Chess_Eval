import pandas as pd
import numpy as np

def preprocess_evaluations(csv_path, output_path):
    print("Loading dataset...")
    df = pd.read_csv(csv_path)
    
    print("Parsing evaluations...")
    # 1. Extract mates vs centipawns
    # If the string contains '#', it's a mate. Give it a massive centipawn value.
    # Otherwise, it's a normal centipawn evaluation.
    
    def parse_eval_string(val):
        val = str(val) # Ensure it's a string
        if '#' in val:
            # Forced mate: #+3 or #-2
            return 10000.0 if '+' in val else -10000.0
        else:
            # Centipawns: +150 or -23
            return float(val.replace('+', ''))

    # Apply the parsing function
    # Note: .apply on 16M rows will take 30-60 seconds to run
    raw_scores = df['Evaluation'].apply(parse_eval_string)
    
    print("Scaling values...")
    # 2. Squash the scores between 0 and 1 using the standard chess sigmoid
    # 1 / (1 + 10^(-score / 400))
    df['Target'] = 1 / (1 + 10 ** (-raw_scores / 400))

    # 3. Drop the old string column to save space
    df = df.drop(columns=['Evaluation'])
    
    print(f"Saving cleaned data to {output_path}...")
    df.to_parquet(output_path, index=False)
    print("Done!")

preprocess_evaluations("Data/archive/chessData.csv", "Data/chessData_cleaned.parquet")