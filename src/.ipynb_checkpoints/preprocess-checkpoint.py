# src/preprocess.py
import os
import numpy as np
import pandas as pd
import ipaddress
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# ==============================
# Helper Functions
# ==============================

def detect_header_issues(file_path, expected_cols=None):
    """Detect if a CSV file likely lacks headers using heuristic checks."""
    try:
        sample = pd.read_csv(file_path, nrows=5, header=None, low_memory=False)
        first_row = sample.iloc[0].astype(str).tolist()
    except Exception as e:
        print(f"⚠️ Could not read {file_path}: {e}")
        return True  # assume header issue if unreadable

    # Simple heuristics: IPs or many numeric values suggest data, not headers
    has_ips = any(val.count('.') == 3 for val in first_row[:5])
    has_numbers = any(val.replace('.', '', 1).isdigit() for val in first_row[:5])
    likely_data_row = has_ips or (has_numbers and len(first_row) > 40)

    # Optional column count check for extra reliability
    if expected_cols is not None and len(first_row) != len(expected_cols):
        print(f"⚠️ Column count mismatch: expected {len(expected_cols)}, found {len(first_row)}")

    return not likely_data_row


def load_complete_dataset(data_path="../data/", columns=None):
    """Load and combine all UNSW-NB15 CSV files with proper header handling."""
    files = ["UNSW-NB15_1.csv", "UNSW-NB15_2.csv", "UNSW-NB15_3.csv", "UNSW-NB15_4.csv"]
    dfs = []
    
    for fname in files:
        fp = os.path.join(data_path, fname)
        if not os.path.exists(fp):
            continue
        
        has_proper_header = detect_header_issues(fp)
        
        if has_proper_header:
            df = pd.read_csv(fp, sep=",", skipinitialspace=True, low_memory=False, header=0)
            df.columns = normalize_cols(df.columns)
        else:
            df = pd.read_csv(fp, sep=",", skipinitialspace=True, low_memory=False, header=None)
            # Use provided columns parameter
            if columns is None:
                raise ValueError("columns parameter must be provided when CSV files don't have proper headers")
            df.columns = columns[:len(df.columns)]
        
        # Clean
        df = df.dropna(how='all').loc[:, ~df.isnull().all()]
        df = df.loc[:, ~df.columns.duplicated(keep='first')]
        dfs.append(df)
    
    if not dfs:
        raise FileNotFoundError("No valid CSV files found!")
    
    combined = pd.concat(dfs, ignore_index=True, sort=False).drop_duplicates()
    
    # Drop unnamed junk
    unnamed_cols = [col for col in combined.columns if 'unnamed' in str(col).lower()]
    if unnamed_cols:
        combined = combined.drop(columns=unnamed_cols)
    
    # Convert datatypes
    combined = combined.apply(pd.to_numeric, errors='ignore')
    
    return combined

# ==============================
# Feature Engineering Functions
# ==============================

def extract_ip_features(df, ip_column):
    """Extract meaningful features from IP addresses instead of one-hot encoding."""
    print(f"  Extracting features from {ip_column}...")
    
    def get_ip_features(ip_str):
        try:
            # Handle missing or invalid IPs
            if pd.isna(ip_str) or ip_str == '' or ip_str == 'unknown':
                return {
                    f'{ip_column}_is_private': False,
                    f'{ip_column}_is_multicast': False,
                    f'{ip_column}_is_loopback': False,
                    f'{ip_column}_network_class': 'unknown',
                    f'{ip_column}_first_octet': 0
                }
            
            ip = ipaddress.ip_address(str(ip_str))
            first_octet = int(str(ip).split('.')[0])
            
            return {
                f'{ip_column}_is_private': ip.is_private,
                f'{ip_column}_is_multicast': ip.is_multicast,
                f'{ip_column}_is_loopback': ip.is_loopback,
                f'{ip_column}_network_class': get_network_class(str(ip)),
                f'{ip_column}_first_octet': first_octet
            }
        except:
            return {
                f'{ip_column}_is_private': False,
                f'{ip_column}_is_multicast': False,
                f'{ip_column}_is_loopback': False,
                f'{ip_column}_network_class': 'unknown',
                f'{ip_column}_first_octet': 0
            }
    
    # Apply IP feature extraction
    ip_features = df[ip_column].apply(get_ip_features)
    ip_features_df = pd.DataFrame(ip_features.tolist())
    
    # Combine with original dataframe
    df_combined = pd.concat([df, ip_features_df], axis=1)
    
    return df_combined

def get_network_class(ip_str):
    """Determine network class of IP address."""
    try:
        first_octet = int(ip_str.split('.')[0])
        if 1 <= first_octet <= 126:
            return 'A'
        elif 128 <= first_octet <= 191:
            return 'B'
        elif 192 <= first_octet <= 223:
            return 'C'
        elif 224 <= first_octet <= 239:
            return 'D'
        else:
            return 'E'
    except:
        return 'unknown'

def categorize_ports(port_series):
    """Categorize ports into meaningful groups instead of one-hot encoding each port."""
    def port_category(port):
        try:
            port_num = int(port)
            if port_num == 0:
                return 'zero'
            elif 1 <= port_num <= 1023:
                return 'well_known'
            elif 1024 <= port_num <= 49151:
                return 'registered'
            elif 49152 <= port_num <= 65535:
                return 'dynamic'
            else:
                return 'invalid'
        except:
            return 'unknown'
    
    return port_series.apply(port_category)

def handle_high_cardinality_features(df, max_categories=50):
    """
    Handle high-cardinality categorical features by either dropping or grouping them.
    
    Args:
        df: DataFrame to process
        max_categories: Maximum number of unique values before considering high-cardinality
    
    Returns:
        DataFrame with high-cardinality features handled appropriately
    """
    df_processed = df.copy()
    
    # Check each object/categorical column
    for col in df_processed.select_dtypes(include=['object', 'category']).columns:
        unique_count = df_processed[col].nunique()
        
        if unique_count > max_categories:
            print(f"⚠️  High-cardinality feature '{col}': {unique_count} unique values")
            
            # Special handling for known problematic columns
            if col in ['srcip', 'dstip']:
                print(f"   → Converting {col} to derived IP features")
                df_processed = extract_ip_features(df_processed, col)
                df_processed.drop(col, axis=1, inplace=True)
                
            elif col in ['sport', 'dsport']:
                print(f"   → Converting {col} to port categories")
                df_processed[f'{col}_category'] = categorize_ports(df_processed[col])
                df_processed.drop(col, axis=1, inplace=True)
                
            else:
                print(f"   → Dropping high-cardinality feature '{col}'")
                df_processed.drop(col, axis=1, inplace=True)
        else:
            print(f"✅ Keeping categorical feature '{col}': {unique_count} unique values")
    
    return df_processed

# ==============================
# Main Preprocessing Functions
# ==============================

def prepare_features(df, task="binary", handle_high_cardinality=True):
    """
    Prepare features & labels for ML pipeline with proper handling of high-cardinality features.
    
    Args:
        df: Input dataframe
        task: "binary" or "multiclass"
        handle_high_cardinality: Whether to handle high-cardinality categorical features
    """
    print(f"🔧 Preparing features for {task} classification...")
    print(f"Input shape: {df.shape}")
    
    # Define target
    if task == "binary":
        y = df["label"]
        print(f"Binary target distribution: {y.value_counts().to_dict()}")
    elif task == "multiclass":
        df["attack_cat"] = df["attack_cat"].fillna("Normal").astype(str)
        y = df["attack_cat"]
        print(f"Multiclass target classes: {y.nunique()}")
    else:
        raise ValueError("task must be 'binary' or 'multiclass'")
    
    # Drop target columns from features
    X = df.drop(columns=["label", "attack_cat"], errors="ignore")
    
    # Handle high-cardinality features if requested
    if handle_high_cardinality:
        print("\n🔍 Checking for high-cardinality features...")
        X = handle_high_cardinality_features(X, max_categories=50)
    
    # Identify feature types after high-cardinality handling
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()
    
    print(f"\n📊 Final feature summary:")
    print(f"   Numeric features: {len(numeric_features)}")
    print(f"   Categorical features: {len(categorical_features)}")
    print(f"   Total features: {len(numeric_features) + len(categorical_features)}")
    
    # Ensure categorical features are strings
    for col in categorical_features:
        X[col] = X[col].astype(str)
    
    # Build preprocessing pipeline
    transformers = []
    
    if numeric_features:
        numeric_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler())
        ])
        transformers.append(("num", numeric_transformer, numeric_features))
    
    if categorical_features:
        categorical_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])
        transformers.append(("cat", categorical_transformer, categorical_features))
    
    if not transformers:
        raise ValueError("No valid features found after preprocessing!")
    
    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder='drop'
    )
    
    # Apply preprocessing
    print("\n🔄 Applying preprocessing pipeline...")
    X_processed = preprocessor.fit_transform(X)
    
    print(f"✅ Preprocessing complete!")
    print(f"   Final shape: {X_processed.shape}")
    print(f"   Memory usage: {X_processed.nbytes / (1024**2):.2f} MB")
    
    return X_processed, y, preprocessor

def clean_dataset(df, missing_threshold=0.5, verbose=True):
    """
    Clean dataset before feature preparation.

    Steps:
    1. Detect and fix mixed-type columns.
    2. Handle missing values:
       - Drop columns with too many NaNs.
       - Fill important UNSW columns based on semantic meaning.
       - Fill numeric/categorical columns with appropriate strategies.
    3. Drop constant or duplicate columns.
    4. Return cleaned dataframe.

    Args:
        df (pd.DataFrame): Input dataframe
        missing_threshold (float): Drop columns with more than this ratio of missing values
        verbose (bool): Whether to print detailed cleaning steps

    Returns:
        pd.DataFrame: Cleaned dataframe
    """

    if verbose: print("🧹 Starting data cleaning...")
    df = df.copy()

    # -----------------------------------------------------------------------------
    # Step 1: Detect mixed data types
    # -----------------------------------------------------------------------------
    if verbose: print("\n🔍 Detecting mixed data types...")
    mixed_cols = {}
    for col in df.select_dtypes(include=["object"]).columns:
        types = df[col].map(type).unique()
        if len(types) > 1:
            mixed_cols[col] = types
            if verbose:
                print(f"⚠️ Column '{col}' has mixed types: {types}")

    # Fix known mixed-type columns
    if 'sport' in df.columns:
        df['sport'] = df['sport'].astype(str)
        if verbose: print("✅ Fixed 'sport' column type → string")

    if 'attack_cat' in df.columns:
        df['attack_cat'] = (
            df['attack_cat']
            .fillna('normal')
            .astype(str)
            .str.lower()
            .str.strip()
        )
        if verbose:
            print("✅ Fixed 'attack_cat' column type → string and filled NaN with 'normal'")

    # -----------------------------------------------------------------------------
    # Step 2: Handle missing values
    # -----------------------------------------------------------------------------
    if verbose: print("\n📊 Handling missing values...")

    # Report missing ratios
    missing_ratio = df.isnull().mean()
    if verbose and missing_ratio.any():
        print("Top missing columns:")
        print(missing_ratio[missing_ratio > 0].sort_values(ascending=False).head(10))

    # Drop columns with too many missing values
    cols_to_drop = missing_ratio[missing_ratio > missing_threshold].index.tolist()
    if cols_to_drop:
        if verbose:
            print(f"\n🚫 Dropping {len(cols_to_drop)} columns (> {missing_threshold*100:.0f}% missing): {cols_to_drop}")
        df = df.drop(columns=cols_to_drop)

    # Fill known missing columns (domain-based)
    for col in ["ct_flw_http_mthd", "is_ftp_login", "ct_ftp_cmd"]:
        if col in df.columns:
            df[col] = df[col].fillna(0)
            if verbose:
                print(f"🔧 Filled '{col}' NaN with 0 (interpreted as 'no activity')")

    # Fill remaining missing values
    num_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = df.select_dtypes(include=["object"]).columns

    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    df[cat_cols] = df[cat_cols].fillna("unknown")


    return df

# ==============================
# Legacy Functions (for backward compatibility)
# ==============================

def load_and_preprocess_data(path, task="binary"):
    """
    Load UNSW-NB15 dataset and preprocess features.
    
    Args:
        path (str): Path to CSV file.
        task (str): "binary" for benign/attack, "multiclass" for attack categories.

    Returns:
        X_processed (array-like): Preprocessed feature matrix.
        y (Series): Target labels.
        preprocessor (Pipeline): Fitted preprocessing pipeline.
    """
    print("⚠️  Using legacy load_and_preprocess_data function.")
    print("   Consider using load_complete_dataset + prepare_features instead.")

    # Load data
    df = pd.read_csv(path)

    # Use the new preprocessing function
    return prepare_features(df, task=task, handle_high_cardinality=True)





    
# ==========================================================================================
# Advanced Models pipeline functions
# ==========================================================================================


# preprocess.py
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.base import clone

def create_unfitted_preprocessor(X):
    if isinstance(X, pd.DataFrame):
        num_cols = X.select_dtypes(include='number').columns.tolist()
        cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    else:
        if hasattr(X, 'shape'):
            num_cols = list(range(X.shape[1]))
            cat_cols = []
        else:
            raise ValueError("Input must be a pandas DataFrame or numpy array with shape attribute")
    
    if not isinstance(X, pd.DataFrame):
        num_pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        preproc = ColumnTransformer([('num', num_pipe, slice(None))], remainder='drop')
        return preproc, num_cols, cat_cols
    
    if num_cols:
        num_pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
    
    if cat_cols:
        cat_pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
    
    transformers = []
    if num_cols:
        transformers.append(('num', num_pipe, num_cols))
    if cat_cols:
        transformers.append(('cat', cat_pipe, cat_cols))
    
    preproc = ColumnTransformer(transformers, remainder='drop')
    return preproc, num_cols, cat_cols

def create_simple_numeric_preprocessor():
    """Simple preprocessor for all-numeric numpy arrays"""
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])


def build_pipeline(preprocessor, clf):
    """Builds a scikit-learn pipeline from a preprocessor and a classifier."""
    return Pipeline([('preprocessor', clone(preprocessor)), ('classifier', clf)])

