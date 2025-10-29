#!/usr/bin/env python
# -*- coding:utf-8 -*-
import csv
import math # For ceil, log2
from typing import List, Tuple, Dict, Optional

from Crypto.Cipher import AES
import base64
import pandas as pd
import numpy as np
# sklearn.preprocessing.OneHotEncoder will be removed for "other attributes"
# import sklearn 
from attribute_tree import AgeTree

BLOCK_SIZE = 16  # Bytes
pad = lambda s: s + (BLOCK_SIZE - len(s) % BLOCK_SIZE) * \
                chr(BLOCK_SIZE - len(s) % BLOCK_SIZE)
unpad = lambda s: s[:-ord(s[len(s) - 1:])]

# CSV Path (used for diag and other attribute meta preparation)
DATA_CSV_PATH = "doc/dataset/dia-100K/dia_10k.csv"

# --- Cached Dictionaries and Metadata ---
_CACHED_AGE_DICT = None
_CACHED_NUM_DICT = None
_CACHED_DIAG_META = None # Expected: {'map': {diag_val_str: code_np_array}, 'num_bits': X, 'ordered_keys': [...]}
_CACHED_OTHER_ATTR_META = None # Expected: {'attr_name': {'value_map': {val_str: idx}, 'num_bits': X, 'ordered_unique_values': [...]}, ...}
_CACHED_COLUMN_NAMES = None # Full list of column names from CSV
_CACHED_SPECIAL_ATTR_NAMES = ['age', 'diag_1', 'diag_2', 'diag_3', 'rand1']
_CACHED_OTHER_ATTR_NAMES = None # Names of attributes not in _CACHED_SPECIAL_ATTR_NAMES
_CACHED_FULL_DF = None # Optionally cache the dataframe if read multiple times
_CACHED_ICD_PATH_MAP: Optional[Dict[str, List[Tuple[str, str]]]] = None # For ICD code to path mapping

# --- ICD Path Mapping Loading ---
ICD_JSON_PATH = "doc/dataset/diagnosis_codes.json"
ICD_PICKLE_PATH = "doc/dataset/icd_path_map.pkl" # Location of the persisted path map
# Specify the top-level ICD codes we are interested in for the hierarchical index
# These are typically chapter codes, e.g., E00-E90 for Endocrine, I00-I99 for Circulatory
TARGET_ICD_CHAPTERS_FOR_HIERARCHY = ["E00-E90", "I00-I99"]

def get_icd_path_map(force_recreate_pickle: bool = False) -> Dict[str, List[Tuple[str, str]]]:
    """
    Loads and returns the ICD code to hierarchical path mapping.
    Uses icd_loader to either load from a pickle file or generate from JSON and then save.
    Caches the loaded map in _CACHED_ICD_PATH_MAP for subsequent calls.

    Args:
        force_recreate_pickle: If True, tells icd_loader to ignore existing pickle 
                               and regenerate from the JSON source.

    Returns:
        A dictionary mapping ICD codes to their paths.
    """
    global _CACHED_ICD_PATH_MAP
    if _CACHED_ICD_PATH_MAP is not None and not force_recreate_pickle:
        # print("DEBUG: Returning cached ICD path map.") # Optional debug print
        return _CACHED_ICD_PATH_MAP

    # Ensure icd_loader is available. Assuming it's in the same directory or Python path.
    try:
        import icd_loader # Dynamically import to avoid circular dependency if tool.py is imported by icd_loader
    except ImportError:
        print("ERROR: icd_loader.py not found. Cannot load ICD path map.")
        # Fallback to an empty map to prevent crashes, though functionality will be limited.
        _CACHED_ICD_PATH_MAP = {}
        return _CACHED_ICD_PATH_MAP

    print(f"INFO: Attempting to load/create ICD path map. force_recreate_pickle={force_recreate_pickle}")
    _CACHED_ICD_PATH_MAP = icd_loader.get_or_create_icd_path_map(
        json_file_path=ICD_JSON_PATH,
        target_top_level_codes=TARGET_ICD_CHAPTERS_FOR_HIERARCHY,
        pickle_file_path=ICD_PICKLE_PATH,
        force_recreate=force_recreate_pickle
    )
    
    if not _CACHED_ICD_PATH_MAP:
        print("Warning: ICD path map is empty after attempting to load/create.")
        # Ensure it's an empty dict, not None, if loading failed and returned empty.
        _CACHED_ICD_PATH_MAP = {} 
        
    # print(f"DEBUG: ICD path map loaded/created. Size: {len(_CACHED_ICD_PATH_MAP)} entries.") # Optional
    return _CACHED_ICD_PATH_MAP

def generate_compact_code(value_index, num_bits):
    """
    Generates a compact binary code for a given value index.
    Args:
        value_index: The 0-based index of the value in its attribute's unique list.
        num_bits: The number of bits required for the encoding.
    Returns:
        A numpy array representing the num_bits binary form of value_index.
    """
    if num_bits == 0:
        return np.array([], dtype=int) if value_index == 0 else np.array([], dtype=int) # Should be empty for 0 bits.

    if value_index < 0 or (value_index >= (1 << num_bits) and num_bits > 0) : # check num_bits > 0 for 1 << num_bits
        print(f"Warning: value_index {value_index} is out of range for {num_bits} bits. Returning zeros.")
        return np.zeros(num_bits, dtype=int) 

    binary_string = bin(value_index)[2:]
    padded_binary_string = binary_string.zfill(num_bits)
    return np.array([int(bit) for bit in padded_binary_string], dtype=int)

# --- Helper to load and cache DataFrame ---
def _get_data_df():
    global _CACHED_FULL_DF
    if _CACHED_FULL_DF is None:
        try:
            _CACHED_FULL_DF = pd.read_csv(DATA_CSV_PATH)
        except FileNotFoundError:
            print(f"Error: Data CSV file not found at {DATA_CSV_PATH}")
            raise
    return _CACHED_FULL_DF

# --- Dictionary and Metadata Preparation Functions ---
def get_age_dict():
    global _CACHED_AGE_DICT
    if _CACHED_AGE_DICT is None:
        age_processing_tree = AgeTree()
        # This structure MUST align with how AgeTree.age_to_tree generates paths
        # and how AgeTree.to_compact_vect traverses the Trie.
        # Leaf segments are tuples representing the 10-year age groups.
        age_tree_data_for_trie = [
            ['1-100', '0-30', '0-10', tuple(range(0,10))],
            ['1-100', '0-30', '10-20', tuple(range(10,20))],
            ['1-100', '0-30', '20-30', tuple(range(20,30))],
            ['1-100', '30-60', '30-40', tuple(range(30,40))],
            ['1-100', '30-60', '40-50', tuple(range(40,50))],
            ['1-100', '30-60', '50-60', tuple(range(50,60))],
            ['1-100', '60-100', '60-70', tuple(range(60,70))],
            ['1-100', '60-100', '70-80', tuple(range(70,80))],
            ['1-100', '60-100', '80-90', tuple(range(80,90))],
            ['1-100', '60-100', '90-100', tuple(range(90,100))]
        ]
        # '*' entries from original age_tree are omitted as they are for range queries,
        # not specific document encoding. build_dict in AgeTree is for specific ages.
        age_processing_tree.add_keywords_from_list(age_tree_data_for_trie)
        _CACHED_AGE_DICT = age_processing_tree.build_dict(gencode_func=generate_compact_code)
    return _CACHED_AGE_DICT

def get_num_dict():
    global _CACHED_NUM_DICT
    if _CACHED_NUM_DICT is None:
        # Pass the actual generate_compact_code function from this module
        _CACHED_NUM_DICT = num_tree.num_dict(10000, gencode_func=generate_compact_code)
    return _CACHED_NUM_DICT

def get_diag_meta():
    global _CACHED_DIAG_META
    if _CACHED_DIAG_META is None:
        df = _get_data_df()
        diag_values = pd.concat([df['diag_1'], df['diag_2'], df['diag_3']]).astype(str).unique()
        # Sort for deterministic index assignment
        ordered_unique_diag_str = sorted(list(diag_values))
        
        if not ordered_unique_diag_str: # Handle empty list
             _CACHED_DIAG_META = {'map': {}, 'num_bits': 0, 'ordered_keys': []}
             return _CACHED_DIAG_META

        num_bits_diag = math.ceil(math.log2(len(ordered_unique_diag_str))) if len(ordered_unique_diag_str) > 0 else 0
        
        diag_map = {}
        for idx, diag_str in enumerate(ordered_unique_diag_str):
            diag_map[diag_str] = generate_compact_code(idx, num_bits_diag)
        
        _CACHED_DIAG_META = {'map': diag_map, 'num_bits': num_bits_diag, 'ordered_keys': ordered_unique_diag_str}
    return _CACHED_DIAG_META

def _get_column_names_and_types():
    global _CACHED_COLUMN_NAMES, _CACHED_OTHER_ATTR_NAMES
    if _CACHED_COLUMN_NAMES is None:
        df = _get_data_df()
        _CACHED_COLUMN_NAMES = df.columns.tolist()
        _CACHED_OTHER_ATTR_NAMES = [col for col in _CACHED_COLUMN_NAMES if col not in _CACHED_SPECIAL_ATTR_NAMES]
    return _CACHED_COLUMN_NAMES, _CACHED_OTHER_ATTR_NAMES

def prepare_other_attributes_meta():
    global _CACHED_OTHER_ATTR_META
    if _CACHED_OTHER_ATTR_META is None:
        _get_column_names_and_types() # Ensures _CACHED_OTHER_ATTR_NAMES is populated
        df = _get_data_df()
        
        meta_dict = {}
        for attr_name in _CACHED_OTHER_ATTR_NAMES:
            unique_vals_for_attr = df[attr_name].astype(str).unique()
            ordered_unique_vals_str = sorted(list(unique_vals_for_attr))
            
            if not ordered_unique_vals_str: # Handle empty list for an attribute
                meta_dict[attr_name] = {'value_map': {}, 'num_bits': 0, 'ordered_unique_values': []}
                continue

            num_bits = math.ceil(math.log2(len(ordered_unique_vals_str))) if len(ordered_unique_vals_str) > 0 else 0
            val_map = {val_str: idx for idx, val_str in enumerate(ordered_unique_vals_str)}
            meta_dict[attr_name] = {
                'value_map': val_map, 
                'num_bits': num_bits,
                'ordered_unique_values': ordered_unique_vals_str
            }
        _CACHED_OTHER_ATTR_META = meta_dict
    return _CACHED_OTHER_ATTR_META

# Ensure metadata is prepared once, can be called explicitly or lazily.
# For simplicity, let's assume they are called before main vector functions.

# --- Main Vectorization Functions ---
def get_attrvect(df_input): # df_input is the dataframe passed from mian.py
    # Ensure all necessary dicts and metas are loaded/prepared
    age_d = get_age_dict()
    num_d = get_num_dict()
    diag_m = get_diag_meta()
    other_attr_m = prepare_other_attributes_meta() 
    # _get_column_names_and_types() # already called by prepare_other_attributes_meta

    document_vectors = []
    
    # Convert relevant columns to string to match dictionary keys if necessary, esp. for diag
    # Age and rand1 are typically int keys in their dicts.
    df_processed = df_input.copy()
    df_processed['diag_1'] = df_processed['diag_1'].astype(str)
    df_processed['diag_2'] = df_processed['diag_2'].astype(str)
    df_processed['diag_3'] = df_processed['diag_3'].astype(str)
    # Other attributes are already converted to str in prepare_other_attributes_meta for map creation

    for _, row in df_processed.iterrows():
        vector_parts = []
        
        # 1. Age
        age_val = int(row['age']) # Age is expected as int
        age_code = age_d.get(age_val, generate_compact_code(0, len(next(iter(age_d.values()))))) # Fallback
        vector_parts.append(age_code)
        
        # 2. Diag_1, Diag_2, Diag_3
        diag_map = diag_m['map']
        diag_num_bits = diag_m['num_bits']
        default_diag_code = generate_compact_code(0, diag_num_bits) # Fallback for unknown diag values

        vector_parts.append(diag_map.get(row['diag_1'], default_diag_code))
        vector_parts.append(diag_map.get(row['diag_2'], default_diag_code))
        vector_parts.append(diag_map.get(row['diag_3'], default_diag_code))

        # 3. Rand1
        rand1_val = int(row['rand1']) # Rand1 is expected as int
        rand1_code = num_d.get(rand1_val, generate_compact_code(0, len(next(iter(num_d.values()))))) # Fallback
        vector_parts.append(rand1_code)
        
        # 4. Other Attributes
        for attr_name in _CACHED_OTHER_ATTR_NAMES:
            attr_meta = other_attr_m[attr_name]
            val_str = str(row[attr_name])
            idx = attr_meta['value_map'].get(val_str, 0) # Default to index 0 if value not in map (e.g. first value)
            code = generate_compact_code(idx, attr_meta['num_bits'])
            vector_parts.append(code)
            
        document_vectors.append(np.concatenate(vector_parts))
        
    return np.array(document_vectors, dtype=int)


def get_attrtrapvect(search_word): # path argument removed, data meta loaded from global cache
    # Ensure all necessary dicts and metas are loaded/prepared
    age_d = get_age_dict()
    num_d = get_num_dict()
    diag_m = get_diag_meta()
    other_attr_m = prepare_other_attributes_meta()
    # _get_column_names_and_types()

    trapdoor_parts = []

    # 1. Age
    # Determine num_bits for age from a sample code, or store it.
    # Assuming all age codes have same length.
    age_num_bits = len(next(iter(age_d.values()))) if age_d else 0 
    if 'age' in search_word:
        age_query_val = search_word['age']
        # Handling for age ranges in trapdoor:
        # If age_query_val is a string like '60-100', this requires special logic.
        # For now, assume specific age int or that age_d might contain pre-coded ranges.
        # The current AgeTree build_dict does not add ranges.
        # So, this part needs refinement if range queries on age are to be compact-coded hierarchically.
        # A simple approach for now: if it's a key in age_d, use it. Else, wildcard.
        if isinstance(age_query_val, int) and age_query_val in age_d:
             trapdoor_parts.append(age_d[age_query_val])
        elif isinstance(age_query_val, str): # Attempt to support range queries from original search_word if they match age_tree levels
            # This is a placeholder for more complex hierarchical range trapdoor generation.
            # E.g. search_word['age'] = '60-100'. The code should represent '1-100' then '60-100', then wildcards.
            # This requires a function in AgeTree like `to_compact_vect_for_range_query`.
            # For now, if not an int key, treat as wildcard.
            print(f"Warning: Age range query '{age_query_val}' needs specific trapdoor logic. Using wildcard.")
            trapdoor_parts.append(generate_compact_code(0, age_num_bits)) # Wildcard/all_match
        else:
            trapdoor_parts.append(generate_compact_code(0, age_num_bits)) # Wildcard/all_match for age
    else:
        trapdoor_parts.append(generate_compact_code(0, age_num_bits)) # Wildcard/all_match for age

    # 2. Diag_1, Diag_2, Diag_3
    diag_map = diag_m['map']
    diag_num_bits = diag_m['num_bits']
    diag_wildcard = generate_compact_code(0, diag_num_bits)

    for diag_key in ['diag_1', 'diag_2', 'diag_3']:
        if diag_key in search_word:
            diag_val_str = str(search_word[diag_key])
            trapdoor_parts.append(diag_map.get(diag_val_str, diag_wildcard)) # Use wildcard if unknown diag value in query
        else:
            trapdoor_parts.append(diag_wildcard)

    # 3. Rand1
    rand1_num_bits = len(next(iter(num_d.values()))) if num_d else 0
    if 'rand1' in search_word:
        rand1_val = int(search_word['rand1'])
        trapdoor_parts.append(num_d.get(rand1_val, generate_compact_code(0,rand1_num_bits)))
    else:
        trapdoor_parts.append(generate_compact_code(0, rand1_num_bits)) # Wildcard

    # 4. Other Attributes
    for attr_name in _CACHED_OTHER_ATTR_NAMES:
        attr_meta = other_attr_m[attr_name]
        attr_num_bits = attr_meta['num_bits']
        if attr_name in search_word:
            val_str = str(search_word[attr_name])
            idx = attr_meta['value_map'].get(val_str, 0) # Default to index 0 for unknown query values
            trapdoor_parts.append(generate_compact_code(idx, attr_num_bits))
        else:
            trapdoor_parts.append(generate_compact_code(0, attr_num_bits)) # Wildcard
            
    return np.concatenate(trapdoor_parts)

def vect_len(): # df argument no longer needed if metadata is cached
    # Ensure all metas are prepared to get correct num_bits
    age_d = get_age_dict()
    num_d = get_num_dict()
    diag_m = get_diag_meta()
    other_attr_m = prepare_other_attributes_meta()

    total_len = 0
    if age_d: total_len += len(next(iter(age_d.values())))
    if diag_m: total_len += 3 * diag_m['num_bits']
    if num_d: total_len += len(next(iter(num_d.values())))
    
    if other_attr_m:
        for attr_name in _CACHED_OTHER_ATTR_NAMES:
            total_len += other_attr_m[attr_name]['num_bits']
            
    return total_len

# --- Original functions (to be removed or refactored further if still used elsewhere) ---
# def cvsEncrypt(path, key): ... remains for now
# def get_feature(path): ... remains for now, used by original get_attrkeyword
# def get_keyword(path): ... to be deprecated
# def get_attrkeyword(path): ... to be deprecated
# def get_vect(df): ... (original OneHotEncoder) to be deprecated
# def get_trapvect(path, search_word): ... (original) to be deprecated


# --- AES Encryption/Decryption (unmodified) ---
def aesEncrypt(key, data):
    key = key.encode('utf8')
    data = pad(data)
    cipher = AES.new(key, AES.MODE_ECB)
    result = cipher.encrypt(data.encode())
    encodestrs = base64.b64encode(result)
    enctext = encodestrs.decode('utf8')
    return enctext

def aesDecrypt(key, data):
    key = key.encode('utf8')
    data = base64.b64decode(data)
    cipher = AES.new(key, AES.MODE_ECB)
    text_decrypted = unpad(cipher.decrypt(data))
    text_decrypted = text_decrypted.decode('utf8')
    return text_decrypted

# cvsEncrypt is not directly part of the vectorization but uses AES.
# It's original file path for output 'D:\\ProgramData\\...' is hardcoded and problematic.
# For now, just keeping it as is if it's used by EDMS.enc_file or mian.py directly.
def cvsEncrypt(path, key):
    df = pd.read_csv(path)
    data = np.array(df.loc[:, :])
    for val in data:
        to_str = ','.join(str(i) for i in val)
        # For safety, I'm commenting out the file write if this function is indeed dead code regarding main flow.
        ecdata = aesEncrypt(key, to_str)
        # f.write(ecdata + '\n') # Commented out to avoid accidental writes with hardcoded/unclear path

# Original get_feature, get_keyword, get_attrkeyword, get_vect, get_trapvect
# These are now superseded by the new compact encoding logic.
# Keeping them here temporarily for reference or if any minor part is reused,
# but they should ideally be removed once the new system is fully integrated and tested.

def get_feature(path): # Still used by get_attrkeyword
    # This should ideally use the cached column names if available
    global _CACHED_COLUMN_NAMES
    if _CACHED_COLUMN_NAMES is None:
        _get_column_names_and_types() # Load them
    # Fallback if CSV reading failed or not yet called
    if _CACHED_COLUMN_NAMES is not None:
        return _CACHED_COLUMN_NAMES
    # Original implementation if cache fails:
    try:
        with open(path, 'r', encoding='utf-8') as f: # Added encoding
            reader = csv.reader(f)
            result = list(reader)
            if result:
                return result[0] # header row
            return [] # Fallback if result is empty
    except FileNotFoundError:
        print(f"Error: File {path} not found in get_feature.")
        return []


# These functions are deprecated in favor of the new metadata preparation and vectorization
def get_keyword(path):
    print("Warning: get_keyword is deprecated.")
    df = pd.read_csv(path) # This path might be DATA_CSV_PATH
    keyword = []
    dicts = {}
    index = 0
    for key in get_feature(path): # Uses the (potentially modified) get_feature
        keyword1 = df[key].drop_duplicates().values.tolist()
        keyword1 = sorted(keyword1)
        sort_dict = {}
        for key_dict in keyword1:
            sort_dict1 = {key_dict: index}
            index = index + 1
            sort_dict.update(sort_dict1)
        dict1 = {key: sort_dict}
        dicts.update(dict1)
        keyword = keyword + keyword1
    return keyword, dicts

def get_attrkeyword(path):
    print("Warning: get_attrkeyword is deprecated.")
    df = pd.read_csv(path) # This path might be DATA_CSV_PATH
    all_keyword = [] 
    keyword = [] 
    dicts = {} 
    index = 0
    # Use cached special attribute names
    list_special = _CACHED_SPECIAL_ATTR_NAMES 
    
    current_features = get_feature(path) # Get features from the file directly

    for key in current_features:
        if key in list_special:
            all_key = df[key].drop_duplicates().values.tolist()
            all_keyword = all_keyword + all_key
            continue
        keyword1 = df[key].drop_duplicates().values.tolist()
        keyword1 = sorted(keyword1)
        sort_dict = {}
        for key_dict in keyword1:
            sort_dict1 = {key_dict: index} # This global index is problematic for trapdoor
            index = index + 1
            sort_dict.update(sort_dict1)
        dict1 = {key: sort_dict}
        dicts.update(dict1)
        keyword = keyword + keyword1
        all_keyword = all_keyword + keyword1
    return all_keyword, keyword, dicts


def get_vect(df_input_deprecated): # Renamed df to df_input_deprecated
    print("Warning: get_vect is deprecated (used OneHotEncoder).")
    raw_convert_data = np.array(df_input_deprecated)
    # This requires sklearn, which we are trying to phase out for this part
    from sklearn.preprocessing import OneHotEncoder 
    model_enc = OneHotEncoder() 
    df_new2 = model_enc.fit_transform(raw_convert_data).toarray().astype(int)
    return df_new2


def get_trapvect(path_deprecated, search_word_deprecated): # Renamed args
    print("Warning: get_trapvect is deprecated.")
    # This uses the problematic get_keyword and its global indexing
    keyword, dic = get_keyword(path_deprecated) 
    trap_vect = [1] * len(keyword) # Default to 1 (non-match after XOR)
    for key, val in search_word_deprecated.items():
        if key in dic and val in dic[key]: # Check if key and val exist
            loc = dic[key][val]
            # Original logic:
            # trap_vect[loc] = 0 # This was for a different scheme
            # The loop below creates a one-hot like segment for 'key'
            for k_attr_val, i_idx in dic[key].items(): # dic[key] is {value: global_index}
                if val == k_attr_val:
                    trap_vect[i_idx] = 1 # Query value gets 1 (becomes 0 after XOR in mian.py)
                else:
                    trap_vect[i_idx] = 0 # Other values for this attribute get 0 (becomes 1 after XOR)
        else:
            print(f"Warning: Query key '{key}' or value '{val}' not found in dictionary for get_trapvect.")
    return np.array(trap_vect)

