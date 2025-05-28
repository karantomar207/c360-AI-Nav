import pandas as pd
import numpy as np
import pickle
import os
import json
import faiss
from sentence_transformers import SentenceTransformer

def generate_embeddings_from_folder(csv_folder="static/csv_data", embeddings_dir="static/embeddings", model_name='all-MiniLM-L6-v2', batch_size=200, max_rows=1000):
    os.makedirs(embeddings_dir, exist_ok=True)

    csv_files_dict = {
        os.path.splitext(file)[0]: os.path.join(csv_folder, file)
        for file in os.listdir(csv_folder)
        if file.endswith(".csv")
    }

    print(f"Found {len(csv_files_dict)} CSV files in '{csv_folder}': {list(csv_files_dict.keys())}")
    print(f"Loading SentenceTransformer model: {model_name}")
    model = SentenceTransformer(model_name)

    professional_keywords = [
        'senior', 'lead', 'manager', 'director', 'principal', 'architect',
        'expert', 'specialist', 'consultant', 'experienced', 'advanced',
        'head of', 'chief', 'vp', 'vice president', 'executive',
        '5+ years', '3+ years', 'leadership', 'team lead', 'supervise',
        'mentoring', 'strategy', 'stakeholder', 'budget', 'enterprise'
    ]

    student_keywords = [
        'internship', 'entry level', 'junior', 'graduate', 'trainee',
        'apprentice', 'associate', 'fresher', 'new grad', 'recent graduate',
        'bootcamp', 'certification', 'learning', 'beginner', 'starter',
        '0-2 years', 'no experience', 'tutorial', 'basics', 'introduction'
    ]

    def classify_entry_type(row_text):
        text = str(row_text).lower()
        prof_score = sum(1 for kw in professional_keywords if kw in text)
        student_score = sum(1 for kw in student_keywords if kw in text)
        if prof_score > student_score:
            return 'professional'
        elif student_score > prof_score:
            return 'student'
        else:
            complexity_indicators = ['advanced', 'complex', 'enterprise', 'production']
            if any(indicator in text for indicator in complexity_indicators):
                return 'professional'
            else:
                return 'student'

    print("=== Starting Embedding Generation ===")
    summary = {}

    for csv_type, csv_path in csv_files_dict.items():
        print(f"\n--- Processing {csv_type} CSV ---")
        try:
            df = pd.read_csv(csv_path)
            original_rows = len(df)

            if len(df) > max_rows:
                print(f"⚠️  Limiting from {len(df)} to {max_rows} rows")
                df = df.head(max_rows)

            print(f"✓ Processing {len(df)} rows, {len(df.columns)} columns")

            # Clean the dataframe to handle NaN, inf values
            df = df.replace([np.inf, -np.inf], np.nan)
            df = df.fillna('')  # Replace NaN with empty strings

            df['combined_text'] = df.apply(
                lambda row: ' '.join([f"{col}: {str(val)}" for col, val in row.items() if str(val).strip()]), axis=1
            )

            print("Classifying entries as student/professional...")
            df['entry_type'] = df['combined_text'].apply(classify_entry_type)

            print(f"Generating embeddings in batches of {batch_size}...")
            all_texts = df['combined_text'].tolist()
            all_embeddings = []

            for i in range(0, len(all_texts), batch_size):
                batch_texts = all_texts[i:i+batch_size]
                batch_end = min(i+batch_size, len(all_texts))
                print(f"  Processing batch {i//batch_size + 1}/{(len(all_texts)-1)//batch_size + 1} (rows {i+1}-{batch_end})")
                batch_embeddings = model.encode(batch_texts, show_progress_bar=False)
                all_embeddings.extend(batch_embeddings)

            all_embeddings = np.array(all_embeddings, dtype=np.float32)
            
            # Check for invalid values in embeddings
            if np.any(np.isnan(all_embeddings)) or np.any(np.isinf(all_embeddings)):
                print("⚠️  Found NaN or inf values in embeddings, replacing with zeros")
                all_embeddings = np.nan_to_num(all_embeddings, nan=0.0, posinf=0.0, neginf=0.0)
            
            print(f"✓ Generated embeddings shape: {all_embeddings.shape}")

            dim = all_embeddings.shape[1]
            unified_index = faiss.IndexFlatL2(dim)
            unified_index.add(all_embeddings)

            mode_mapping = {
                'student': {'indices': [], 'metadata': []},
                'professional': {'indices': [], 'metadata': []}
            }

            print("Creating mode mappings...")
            for idx, row in enumerate(df.itertuples(index=False)):
                etype = df.iloc[idx]['entry_type']
                mode_mapping[etype]['indices'].append(idx)
                # Clean metadata to remove problematic columns
                metadata = df.iloc[idx].drop(['combined_text', 'entry_type'])
                
                # Convert metadata to dict and clean values
                metadata_dict = {}
                for k, v in metadata.to_dict().items():
                    if pd.isna(v) or (isinstance(v, float) and (np.isinf(v) or np.isnan(v))):
                        metadata_dict[k] = None
                    else:
                        metadata_dict[k] = v
                
                mode_mapping[etype]['metadata'].append(metadata_dict)

            mode_mapping['student']['indices'] = np.array(mode_mapping['student']['indices'])
            mode_mapping['professional']['indices'] = np.array(mode_mapping['professional']['indices'])

            index_path = os.path.join(embeddings_dir, f"{csv_type}.index")
            pkl_path = os.path.join(embeddings_dir, f"{csv_type}.pkl")

            faiss.write_index(unified_index, index_path)
            with open(pkl_path, 'wb') as f:
                pickle.dump({
                    'mode_mapping': mode_mapping,
                    'total_entries': len(df),
                    'embedding_dimension': dim,
                    'entry_type_counts': {
                        'student': len(mode_mapping['student']['indices']),
                        'professional': len(mode_mapping['professional']['indices'])
                    }
                }, f)

            print(f"✓ Saved unified FAISS index: {index_path}")
            print(f"✓ Saved unified mapping data: {pkl_path}")
            print(f"  - Total entries: {len(df)}")
            print(f"  - Student entries: {len(mode_mapping['student']['indices'])}")
            print(f"  - Professional entries: {len(mode_mapping['professional']['indices'])}")

            summary[csv_type] = {
                'original_rows': int(original_rows),
                'rows_processed': int(len(df)),
                'embedding_dimension': int(dim),
                'batch_size': int(batch_size),
                'student_count': int(len(mode_mapping['student']['indices'])),
                'professional_count': int(len(mode_mapping['professional']['indices'])),
                'files': {
                    'index': index_path,
                    'metadata': pkl_path
                }
            }

        except Exception as e:
            print(f"❌ Error processing {csv_type}: {str(e)}")
            summary[csv_type] = {'error': str(e)}

    # Improved sanitize function
    def sanitize(obj):
        if isinstance(obj, dict):
            return {k: sanitize(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [sanitize(v) for v in obj]
        elif isinstance(obj, (np.floating, float)):
            if np.isnan(obj) or np.isinf(obj):
                return None
            # Check for extremely large values that might cause JSON issues
            if abs(obj) > 1e308:
                return None
            return float(obj)
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif pd.isna(obj):
            return None
        else:
            return obj

    summary_clean = sanitize(summary)

    summary_path = os.path.join(embeddings_dir, "embedding_summary.json")
    
    # Use standard json instead of json_tricks for better error handling
    try:
        with open(summary_path, 'w') as f:
            json.dump(summary_clean, f, indent=2, allow_nan=False)
    except (ValueError, TypeError) as e:
        print(f"❌ Error saving JSON: {e}")
        # Save a minimal summary if the full one fails
        minimal_summary = {
            'datasets_processed': len([k for k in summary.keys() if 'error' not in summary.get(k, {})]),
            'total_datasets': len(summary),
            'timestamp': str(pd.Timestamp.now())
        }
        with open(summary_path, 'w') as f:
            json.dump(minimal_summary, f, indent=2)

    print(f"\n=== Generation Complete ===")
    print(f"Summary saved: {summary_path}")
    print(f"Total files created: {len([f for f in summary.keys() if 'error' not in summary.get(f, {})]) * 2}")

    return summary_clean


if __name__ == "__main__":
    generate_embeddings_from_folder()