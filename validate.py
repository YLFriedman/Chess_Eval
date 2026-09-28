import pandas as pd
import numpy as np
import pyarrow.parquet as pq

def validate_dataset_low_ram(csv_path, parquet_path, chunk_size=100_000):
    print("Initializing chunked readers...")
    
    # Initialize PyArrow Parquet reader for streaming
    parquet_file = pq.ParquetFile(parquet_path)
    parquet_iter = parquet_file.iter_batches(batch_size=chunk_size)
    
    # Initialize Pandas CSV reader for streaming
    csv_iter = pd.read_csv(csv_path, chunksize=chunk_size)
    
    total_rows_processed = 0
    
    def parse_eval_string(val):
        val = str(val)
        if '#' in val:
            return 10000.0 if '+' in val else -10000.0
        else:
            return float(val.replace('+', ''))

    print(f"Starting validation in chunks of {chunk_size:,} rows...\n")
    
    while True:
        # 1. Fetch the next chunk from both files
        try:
            csv_chunk = next(csv_iter)
            csv_has_data = True
        except StopIteration:
            csv_has_data = False
            
        try:
            pq_batch = next(parquet_iter)
            pq_chunk = pq_batch.to_pandas()
            pq_has_data = True
        except StopIteration:
            pq_has_data = False
            
        # 2. Check for mismatched file lengths
        if csv_has_data != pq_has_data:
            print(f"❌ FAILED: File lengths do not match at row {total_rows_processed:,}.")
            return
            
        if not csv_has_data and not pq_has_data:
            break # Both files finished successfully
            
        if len(csv_chunk) != len(pq_chunk):
            print(f"❌ FAILED: Chunk sizes don't match at row {total_rows_processed:,}.")
            return

        # 3. Validate FEN Strings
        # reset_index ensures pandas doesn't fail the comparison due to mismatched row indices
        fens_match = (csv_chunk['FEN'].reset_index(drop=True) == pq_chunk['FEN'].reset_index(drop=True)).all()
        if not fens_match:
            print(f"❌ FAILED: FEN strings mismatch in chunk starting at row {total_rows_processed:,}!")
            return
            
        # 4. Validate Target Values
        raw_scores = csv_chunk['Evaluation'].apply(parse_eval_string)
        expected_targets = 1 / (1 + 10 ** (-raw_scores / 400))
        
        targets_match = np.allclose(
            pq_chunk['Target'].values, 
            expected_targets.values, 
            atol=1e-6
        )
        
        if not targets_match:
            print(f"❌ FAILED: Target values mismatch in chunk starting at row {total_rows_processed:,}!")
            return
            
        total_rows_processed += len(csv_chunk)
        
        # Print a progress update every 1 million rows
        if total_rows_processed % 1_000_000 == 0:
            print(f"Successfully validated {total_rows_processed:,} rows...")

    print(f"\n✅ Validation Complete! All {total_rows_processed:,} rows match perfectly.")

# Run the low-RAM validation
validate_dataset_low_ram("Data/archive/chessData.csv", "Data/chessData_cleaned.parquet")