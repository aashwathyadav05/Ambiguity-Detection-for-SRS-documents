import pandas as pd
from pathlib import Path

def load_data(data_path: str = "data/dataset.csv"):
    """Load the Fault-prone SRS dataset with encoding handling."""
    
    path = Path(data_path)
    
    # Try multiple paths
    if not path.exists():
        possible_paths = [
            Path.cwd() / "data" / "dataset.csv",
            Path.cwd().parent / "data" / "dataset.csv",
            Path("data/dataset.csv"),
        ]
        for p in possible_paths:
            if p and p.exists():
                path = p
                break
        else:
            raise FileNotFoundError(
                f"Dataset not found. Please place dataset.csv inside the 'data' folder.\n"
                f"Current path tried: {path.absolute()}"
            )

    # Try different encodings - this is the key fix
    encodings = ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']
    
    df = None
    for encoding in encodings:
        try:
            df = pd.read_csv(path, encoding=encoding)
            print(f"Successfully loaded with encoding: {encoding}")
            break
        except UnicodeDecodeError:
            continue
        except Exception as e:
            print(f"Error with encoding {encoding}: {e}")
            continue
    
    if df is None:
        raise UnicodeDecodeError("Failed to read the CSV with common encodings. The file may be corrupted.")

    print(f"Loaded {len(df):,} rows")
    print(f"File path: {path.absolute()}")
    print(f"Columns: {df.columns.tolist()}\n")

    # Column detection
    text_col = label_col = None
    for col in df.columns:
        cl = str(col).lower().strip()
        if any(x in cl for x in ['requirement', 'req', 'text', 'srs', 'statement', 'description']):
            text_col = col
        if any(x in cl for x in ['label', 'class', 'ambiguity', 'type', 'category', 'fault', 'target']):
            label_col = col

    if text_col is None or label_col is None:
        print("Warning: Using first two columns as fallback.")
        text_col = df.columns[0]
        label_col = df.columns[1]

    df = df.rename(columns={text_col: "text", label_col: "label"})

    # Clean data
    df = df.dropna(subset=["text", "label"]).drop_duplicates(subset=["text"]).reset_index(drop=True)

    print(f"Final cleaned shape: {df.shape}")
    print(f"Unique labels: {sorted(df['label'].unique())}")
    return df