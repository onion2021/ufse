#!/usr/bin/env python
# -*- coding:utf-8 -*-
import pandas as pd
import time
import os
import pickle # Required for LayeredIndex, and potentially for direct size estimation if needed
import sys # For sys.getsizeof if we were to use it directly (pickle.dumps is better for tree)
import numpy as np # For ToolPyEncoder
from typing import List, Dict, Any, Optional, Tuple, Set, Union # Added Set and Union
import logging
from datetime import datetime

from cryptography.fernet import Fernet
# from layered_index import LayeredIndex, LayeredIndexNode, FIXED_BUCKET_FLAG
import layered_index # Import the module directly
import tool # Import the actual tool.py for metadata preparation
from tool import (
    generate_compact_code, 
    get_age_dict, 
    get_diag_meta, 
    prepare_other_attributes_meta
)

# 设置日志级别为 ERROR,这样就不会显示 DEBUG 信息
logging.basicConfig(level=logging.ERROR)

# ATTRIBUTE_ORDER 将在 main 函数中从加载的 DataFrame 动态设置
ATTRIBUTE_ORDER: Optional[List[str]] = None


def load_and_prepare_dataframe(file_path: str) -> pd.DataFrame:
    """
    加载并预处理数据
    Args:
        file_path: CSV 文件路径
    Returns:
        处理后的数据框
    """
    df = pd.read_csv(file_path)
    # 确保所有 diag_X 字段为字符串
    for col_name in df.columns:
        if col_name.startswith('diag_'):
            df[col_name] = df[col_name].astype(str)

    # 添加Record_ID列
    df['Record_ID'] = df.index
    return df

def create_encoders(df: pd.DataFrame) -> Dict[str, Any]:
    """
    创建编码器
    
    Args:
        df: 数据框
        
    Returns:
        编码器字典
    """
    # 获取年龄字典
    age_dict = tool.get_age_dict()
    
    # 获取诊断元数据
    diag_meta = tool.get_diag_meta()
    
    # 获取其他属性元数据
    other_attributes_meta = tool.prepare_other_attributes_meta()
    
    return {
        'age_dict': age_dict,
        'diag_meta': diag_meta,
        'other_attributes_meta': other_attributes_meta
    }

def main():
    # 1. 加载数据
    print("1. Loading data")
    DATA_FILE_PATH = 'doc\dataset\data\data60D-10k.csv'  # 当前使用的文件
    df = load_and_prepare_dataframe(DATA_FILE_PATH)
    print(f"Data loading completed. There are a total of {len(df)} records, sourced from the file: {DATA_FILE_PATH}")

    # 从加载的 DataFrame 动态设置 ATTRIBUTE_ORDER
    global ATTRIBUTE_ORDER
    if df is not None:
        ATTRIBUTE_ORDER = [col for col in df.columns.tolist() if col != 'Record_ID']
        print(f"Order of dynamically obtained attributes: {ATTRIBUTE_ORDER[:5]}... (Total of {len(ATTRIBUTE_ORDER)})")
    else:
        print("Error: The DataFrame failed to load, so ATTRIBUTE_ORDER cannot be set")
        return

    # 2. 创建编码器
    print("\n2. Create an encoder")
    encoders = create_encoders(df)
    print("The encoder has been created successfully")
    
    # 3. 构建索引
    print("\n3.Build an index ")
    start_time = time.time()
    index = layered_index.LayeredIndex(
        df=df,
        attribute_order=ATTRIBUTE_ORDER,
        age_dict=encoders['age_dict'],
        diag_meta=encoders['diag_meta'],
        other_attributes_meta=encoders['other_attributes_meta']
    )
    index.build_index()
    end_time = time.time()
    build_time_seconds = end_time - start_time
    print(f"Index building completed, Time elapsed: {build_time_seconds:.8f} seconds")

    # 持久化索引到文件
    if not os.path.exists("saved_indexes"):
        os.makedirs("saved_indexes")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    index_filename = f"layered_index_{timestamp}.pkl"
    index_filepath = os.path.join("saved_indexes", index_filename)
    
    # 生成文件加密密钥
    file_encryption_key = os.urandom(32)  # 32字节AES密钥
    
    try:
        # 使用新的save_encrypted方法保存索引
        index.save_encrypted(index_filepath, file_encryption_key)
        print(f"The index has been successfully saved to: {index_filepath}")
        
        # 保存加密密钥到单独文件
        key_filename = f"encryption_key_{timestamp}.key"
        key_filepath = os.path.join("saved_indexes", key_filename)
        with open(key_filepath, "wb") as f:
            f.write(file_encryption_key)
        print(f"The encryption key has been saved to: {key_filepath}")
        
    except Exception as e:
        print(f"An error occurred while saving the index to {index_filepath}: {e}")

    # # 3.5. 测试索引加载功能
    # print("\n3.5. 测试索引加载功能")
    # try:
    #     # 加载保存的索引
    #     loaded_index = layered_index.LayeredIndex.load_encrypted(
    #         index_filepath, 
    #         file_encryption_key, 
    #         df
    #     )
        
    #     if loaded_index is not None:
    #         print("索引加载成功！")
            
    #         # 验证加载的索引是否与原索引一致
    #         test_query = {'race': 'Caucasian', 'gender': 'Female'}
    #         original_results = index.search(test_query)
    #         loaded_results = loaded_index.search(test_query)
            
    #         print(f"原始索引查询结果数: {len(original_results)}")
    #         print(f"加载索引查询结果数: {len(loaded_results)}")
    #         print(f"结果是否一致: {original_results == loaded_results}")
            
    #         if original_results == loaded_results:
    #             print("✅ 索引保存和加载功能正常工作！")
    #         else:
    #             print("❌ 索引保存和加载功能存在问题！")
    #     else:
    #         print("❌ 索引加载失败！")
            
    # except Exception as e:
    #     print(f"测试索引加载功能时发生错误: {e}")

       
    # 4. 测试性能
    print("\n4.Test performance ")
    
    # 原有的性能测试
    num_queries = 100
    query_perf_orig = {
        'race': 'Caucasian',
        'gender': 'Female',
        'age': 65,
    }
    print(f"\nTest Performance (Original Query): {query_perf_orig}")

    # 陷门生成时间 (原始查询 - 单次)
    start_time_trapdoor_orig = time.perf_counter()
    _ = index._get_query_path_for_test(query_perf_orig)
    end_time_trapdoor_orig = time.perf_counter()
    trapdoor_time_ms_orig = (end_time_trapdoor_orig - start_time_trapdoor_orig) * 1000
    print(f"Trapdoor Generation Time (Original Query): {trapdoor_time_ms_orig:.8f} ms")

    start_time = time.time()
    results1_perf_orig = index.search(query_perf_orig) # Get results once for count
    for _ in range(num_queries -1): # adjust loop for the one search already done
        index.search(query_perf_orig)
    end_time = time.time()
    avg_time = (end_time - start_time) / num_queries
    print(f"Index query found {len(results1_perf_orig)} matching records")
    print(f"Average Index Query Time (Original Query): {avg_time*1000:.8f} ms")
    
    # print("\n执行Pandas搜索 (原始查询, 重复100次)...") # Comment out or remove Pandas timing section header
    # pandas_times_orig = [] # Removed
    # Pandas search for original query - executed once for comparison, no timing
    df['age_range_orig'] = df['age'].apply(
        lambda x: '[60-70)' if isinstance(x, str) and x == '[60-70)' 
        else ('[60-70)' if 60 <= int(x) < 70 
              else str(x)) # Fallback for other string or int values
    )
    pandas_results_perf_orig = set(df[
        (df['race'] == query_perf_orig['race']) &
        (df['gender'] == query_perf_orig['gender']) &
        (df['age'] == query_perf_orig['age']) # Now compares '[60-70)' with '[60-70)'
    ]['Record_ID'])

    # for _ in range(100): # Removed loop for timing
    #     start_time_pd_orig = time.perf_counter_ns()
    #     _ = set(df[
    #         (df['race'] == query_perf_orig['race']) &
    #         (df['gender'] == query_perf_orig['gender']) &
    #         (df['age_range_orig'] == query_perf_orig['age'])
    #     ]['Record_ID'].astype(str))
    #     end_time_pd_orig = time.perf_counter_ns()
    #     pandas_times_orig.append((end_time_pd_orig - start_time_pd_orig) / 1_000_000)
    
    # avg_pandas_time_orig = sum(pandas_times_orig) / len(pandas_times_orig) # Removed
    print(f"Pandas query found {len(pandas_results_perf_orig)} matching records (for comparison)") # Clarified purpose
    # print(f"Pandas平均查询时间 (原始查询): {avg_pandas_time_orig:.6f} 毫秒") # Removed

    # 新增: 测试所有数据的完整匹配性能
    print("\n\nTest the full matching performance of all data:")
    
    # Define age mapping for query construction
    age_query_mapping = {
        '[0-10)': '[0-10)', '[10-20)': '[10-20)', '[20-30)': '[20-30)', '[30-40)': '[30-40)', '[40-50)': '[40-50)',
        '[50-60)': '[50-60)',
        '[60-70)': '[60-70)',
        '[70-80)': '[70-80)',
        '[80-90)': '[80-90)',
        '[90-100+)': '[90-100)', # Assuming data might have 90-100+ and index uses [90-100)
        '[90-100)': '[90-100)', # Explicitly map [90-100) to [90-100)
    }

    # REMOVED random sampling, will iterate over the whole DataFrame
    # if len(df) >= 10:
    #     random_records_perf = df.sample(n=10, random_state=42) # Use random_state for reproducibility
    # else:
    #     print("数据集记录少于10条，使用所有记录进行测试。")
    #     random_records_perf = df.copy()
        
    total_trapdoor_time_ms_fm = 0.0
    total_search_time_ms_fm = 0.0
    num_processed_records_fm = len(df) # Use the total number of records

    if num_processed_records_fm > 0:
        for _, record_perf in df.iterrows(): # Iterate over the entire DataFrame
            full_match_query_perf = {}
            for attr_perf in ATTRIBUTE_ORDER:
                value_perf = record_perf[attr_perf]
                if attr_perf == 'age':
                    # Use the mapping; if a specific age string isn't in the map,
                    # it implies it might be an integer or a format the index's _get_attribute_value handles.
                    # For consistency with indexing, we ensure it's mapped to the '[X-Y)' format.
                    query_age_value = age_query_mapping.get(str(value_perf), str(value_perf))
                    full_match_query_perf[attr_perf] = query_age_value
                elif attr_perf == 'diag_1':
                    full_match_query_perf[attr_perf] = str(value_perf)
                else:
                    full_match_query_perf[attr_perf] = value_perf

            # 陷门生成时间 (完整匹配 - 单次)
            start_time_trapdoor_fm = time.perf_counter()
            _ = index._get_query_path_for_test(full_match_query_perf)
            end_time_trapdoor_fm = time.perf_counter()
            trapdoor_time_ms_fm = (end_time_trapdoor_fm - start_time_trapdoor_fm) * 1000
            total_trapdoor_time_ms_fm += trapdoor_time_ms_fm

            # 索引查询性能 (完整匹配 - 单次)
            start_idx_search_fm = time.perf_counter()
            _ = index.search(full_match_query_perf) # Results not stored as per "只打印时间"
            end_idx_search_fm = time.perf_counter()
            current_search_time_ms_fm = (end_idx_search_fm - start_idx_search_fm) * 1000
            total_search_time_ms_fm += current_search_time_ms_fm

        avg_trapdoor_time_fm = total_trapdoor_time_ms_fm / num_processed_records_fm
        avg_search_time_fm = total_search_time_ms_fm / num_processed_records_fm
        print(f"Full matching performance based on {num_processed_records_fm} records:")
        print(f"Average Trapdoor Generation Time (Full Matching): {avg_trapdoor_time_fm:.8f} ms")
        print(f"Average Index Query Time (Full Matching, Single Query): {avg_search_time_fm:.8f} ms")
    else:
        print("There are no records available for testing full matching performance.")

    # Pandas查询性能 (完整匹配) - REMOVED as per request
    # pandas_condition_str_fm = " & ".join(pandas_conditions_list_perf)
    # print("\n执行Pandas搜索 (完整匹配, 重复100次)...")
    # pandas_times_fm = []
    # pandas_results_val_fm = set(df[eval(pandas_condition_str_fm, {'df': df, 'pd': pd, 'np': np})]['Record_ID'].astype(str))
    # for _ in range(num_queries_fm):
    #     start_pd_fm = time.perf_counter_ns()
    #     _ = set(df[eval(pandas_condition_str_fm, {'df': df, 'pd': pd, 'np': np})]['Record_ID'].astype(str))
    #     end_pd_fm = time.perf_counter_ns()
    #     pandas_times_fm.append((end_pd_fm - start_pd_fm) / 1_000_000)
    # avg_pandas_time_val_fm = sum(pandas_times_fm) / len(pandas_times_fm)
    # print(f"Pandas查询 (完整匹配) 找到 {len(pandas_results_val_fm)} 条匹配记录 (用于比较)")
    # print(f"Pandas平均查询时间 (完整匹配): {avg_pandas_time_val_fm:.6f} 毫秒")
    
    # 结果集比较 (原始查询):
    print("\n\nResultSet Comparison (Original Query):") # Changed header for clarity
    print("---Original Query Performance Results---")
    print(f"Number of Index Query Results (Original): {len(results1_perf_orig)}")
    print(f"Number of Pandas Query Results (Original): {len(pandas_results_perf_orig)}")
    is_subset_orig = results1_perf_orig.issubset(pandas_results_perf_orig)
    print(f"Is the Index Result a Subset of the Pandas Result (Original): {is_subset_orig}")
    if not is_subset_orig:
        print(f"Number of Records Present in the Index but Not in Pandas (Original): {len(results1_perf_orig - pandas_results_perf_orig)}")
    print(f"Number of Records Present in Pandas but Not in the Index (Original): {len(pandas_results_perf_orig - results1_perf_orig)}")

    # 5. 新增功能测试 ( 范围年龄查询, 插入, 删除)
    print("\n\n5. Update Test")

    # 确保 DataFrame 有 'Record_ID' 并且是唯一的，以便查找用于测试的记录
    if 'Record_ID' not in df.columns:
        df['Record_ID'] = df.index

    
    # --- 测试记录插入 ---
    print("\n--- Test Record Insertion ---")
    # 需要一个与 ATTRIBUTE_ORDER 完全匹配的记录字典
    # 从现有数据中取一个样本，然后修改它，确保Record_ID是新的
    if not df.empty and ATTRIBUTE_ORDER is not None:
        sample_record_for_insert = df.iloc[0].to_dict()
        new_record_id = df['Record_ID'].max() + 1 if not df.empty else 0
        sample_record_for_insert['Record_ID'] = new_record_id # Not used by insert_record method, but for tracking
        
        # 修改一些值以使其独特，例如年龄或诊断
        sample_record_for_insert['age'] = 45 # Ensure this age is an int
        sample_record_for_insert['diag_1'] = '996.0' # Example, ensure it's a str
        # Ensure all fields from ATTRIBUTE_ORDER are present (df.iloc[0] should cover this)
        for attr_key in ATTRIBUTE_ORDER:
            if attr_key not in sample_record_for_insert:
                 # Fallback for any missing keys, though df.iloc[0] should have all original columns
                sample_record_for_insert[attr_key] = 'Unknown' 
                if attr_key == 'age': sample_record_for_insert[attr_key] = 0 # default age

        print(f"Records to Be Inserted (ID: {new_record_id}): { {k: sample_record_for_insert[k] for k in ATTRIBUTE_ORDER[:5]} }...")
        
        # 插入记录 (record_id to insert_record is the actual ID for storage)
        # The 'record' dict itself should not contain 'Record_ID' if it's not an attribute in ATTRIBUTE_ORDER
        record_data_for_index = {k: v for k, v in sample_record_for_insert.items() if k != 'Record_ID' and k in ATTRIBUTE_ORDER}
        # Ensure all ATTRIBUTE_ORDER keys are in record_data_for_index for insertion path
        # This might have already been handled above.
        start_idx_insert_time = time.perf_counter()
        insert_success = index.insert_record(record_data_for_index, new_record_id)
        end_idx_insert_time = time.perf_counter()
        insert_time_ms = (end_idx_insert_time - start_idx_insert_time) * 1000
        print(f"Record Insertion Time: {insert_time_ms:.8f} ms")
        print(f"Was the Record Insertion Successful?: {insert_success}")

        if insert_success:
            print(f"Verify the Inserted Records (age {sample_record_for_insert['age']}):")
            query_inserted = {'age': sample_record_for_insert['age'], 'diag_1': sample_record_for_insert['diag_1']}
            results_inserted = index.search(query_inserted)
            print(f"Search Results for the Inserted Records: {results_inserted}")
            if new_record_id in results_inserted:
                print("Successfully found the newly inserted records!")
            else:
                print("Error: The newly inserted records were not found!")
            
            # --- 测试记录删除 ---
            print("\n--- Test Record Deletion ---")
            # 使用插入记录的属性进行删除查询 (需要精确匹配，特别是年龄)
            query_for_delete = {
                'race': record_data_for_index['race'], 
                'gender': record_data_for_index['gender'], 
                'age': record_data_for_index['age'], # Specific age
                'diag_1': record_data_for_index['diag_1']
            }
            # To be safe, one might want to use all attributes from ATTRIBUTE_ORDER in query_for_delete
            # if the path to the record depends on all of them.
            # For this test, using a subset that should make it unique enough.
            
            print(f"Records to Be Deleted (ID: {new_record_id}) Using Query: {query_for_delete}")
            start_idx_delete_time = time.perf_counter()
            delete_success = index.delete_record(query_for_delete, new_record_id)
            end_idx_delete_time = time.perf_counter()
            delete_time_ms = (end_idx_delete_time - start_idx_delete_time) * 1000
            print(f"Record Deletion Time: {delete_time_ms:.8f} ms")
            print(f"Was the Record Deletion Successful?: {delete_success}")
            if delete_success:
                results_after_delete = index.search(query_inserted) # Search again with the same query
                print(f"Search Results After Deletion: {results_after_delete}")
                if new_record_id not in results_after_delete:
                    print("Successfully deleted the record!")
                else:
                    print("Error: The record was found after deletion!")
            else:
                print("Deletion operation failed, cannot verify deletion.")
        else:
            print("Record insertion failed, skipping deletion and update tests.")

    else:
        print("DataFrame is empty, skipping insertion/deletion/update tests.")

if __name__ == "__main__":
    main() 
