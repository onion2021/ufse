#!/usr/bin/env python
# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import tool
from typing import Dict, List, Set, Tuple, Union, Optional, Any
from medical_codes import get_codes_for_category, DIABETES_ICD9_HIERARCHY
import pickle
import hashlib
import hmac
import sys
# from cryptography.fernet import Fernet # Removed Fernet
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding as sym_padding # For potential CBC use, GCM handles padding.
from cryptography.hazmat.backends import default_backend
from cryptography.exceptions import InvalidTag
import os # For os.urandom

from attribute_tree import AgeTree
import logging
from icd_encoder import ICDEncoder

class LayeredIndexNode:
    def __init__(self):
        self.children: Dict[str, 'LayeredIndexNode'] = {}
        self.record_ids: Set[int] = set()
        self.first_hash: Optional[str] = None  # 存储第一层哈希值

class LayeredIndex:
    WILDCARD_FLAG = object()
    RANGE_QUERY_FLAG = object()
    FIXED_BUCKET_FLAG = object()

    # AES-GCM constants
    AES_KEY_SIZE = 32  # 256 bits
    GCM_IV_SIZE = 12   # 96 bits is recommended for GCM
    GCM_TAG_SIZE = 16  # 128 bits tag

    def __init__(self, df: pd.DataFrame, attribute_order: List[str], 
                 age_dict: Dict[str, int], diag_meta: Dict[str, Any],
                 other_attributes_meta: Dict[str, Any]):
        self.df = df
        self.attribute_order = attribute_order
        self.age_dict = age_dict # May not be used directly for age indexing per new logic
        self.diag_meta = diag_meta
        self.other_attributes_meta = other_attributes_meta
        self.root = LayeredIndexNode()
        
        self.first_key = os.urandom(32) 
        
        self.build_index()
    
    def _first_hash(self, value: str) -> str:
        """使用固定的self.first_key通过HMAC-SHA256生成哈希值，并截断到32位。"""
        mac = hmac.new(self.first_key, value.encode('utf-8'), hashlib.sha256)
        # 截断到32位 (4字节) -> 8个十六进制字符
        return mac.hexdigest()[:8] 
    
    def _second_hash(self, input_data_str: str, query_specific_key: bytes) -> str:
        """使用特定于查询的密钥通过HMAC-SHA256生成哈希值，并截断到32位。"""
        # input_data_str 通常是 _first_hash 的输出或类似的哈希值
        mac = hmac.new(query_specific_key, input_data_str.encode('utf-8'), hashlib.sha256)
        # 截断到32位 (4字节) -> 8个十六进制字符
        return mac.hexdigest()[:8]
    
    def _determine_age_range_str(self, age_int: int) -> str:
        """Determines the 10-year age range string for a given integer age."""
        if age_int < 0: return '[0-10)' 
        elif age_int < 10: return '[0-10)'
        elif age_int < 20: return '[10-20)'
        elif age_int < 30: return '[20-30)'
        elif age_int < 40: return '[30-40)'
        elif age_int < 50: return '[40-50)'
        elif age_int < 60: return '[50-60)'
        elif age_int < 70: return '[60-70)'
        elif age_int < 80: return '[70-80)'
        elif age_int < 90: return '[80-90)'
        else: return '[90-100)' # Covers 90 and above, assuming 100 is exclusive upper for this bucket

    def _get_attribute_value(self, record_value_map: Dict[str, Any], attr: str) -> str:
        """
        Gets the string representation of an attribute's value.
        For 'age', expects integer from data or int/str from query; returns simple str conversion.
        For 'diag_*', uses diag_meta for encoding.
        For other attributes, uses other_attributes_meta for encoding if available.
        """
        value = record_value_map[attr]
        if attr == 'age':
            # If age comes as int (from data or query like {'age':65}), convert to "65"
            # If age comes as str (from query like {'age':"[60-70)"} or {'age':"65"}), use as is.
            return str(value)
        elif attr.startswith('diag_'):
            if 'map' in self.diag_meta and isinstance(self.diag_meta['map'], dict):
                return str(self.diag_meta['map'].get(str(value), -1))
            else:
                logging.warning(f"diag_meta['map'] is not a dict or 'map' key is missing for attribute {attr}")
                return str(-1) 
        else:
            # Handle other attributes with encoding if other_attributes_meta is available
            if (self.other_attributes_meta and 
                attr in self.other_attributes_meta and 
                'value_map' in self.other_attributes_meta[attr]):
                attr_meta = self.other_attributes_meta[attr]
                val_str = str(value)
                idx = attr_meta['value_map'].get(val_str, 0)  # Default to index 0 if value not in map
                return str(idx)  # Return the encoded index as string
            else:
                # Fallback to original value if no encoding metadata available
                return str(value)

    def build_index(self):
        """构建分层索引"""
        for _, record in self.df.iterrows():
            record_dict = record.to_dict()
            record_id = int(record_dict['Record_ID'])
            current_node = self.root
            
            for attr in self.attribute_order:
                if attr == 'age':
                    actual_age_int = record_dict[attr]
                    if not isinstance(actual_age_int, int):
                        try: # If age in DataFrame is string like "65"
                            actual_age_int = int(actual_age_int)
                        except ValueError:
                            logging.error(f"Age '{actual_age_int}' for Record_ID {record_id} is not an int. Skipping record for age indexing.")
                            # Decide how to handle: skip attribute, skip record, or default?
                            # For now, let's assume we can't proceed with this attr for this record
                            # and break inner loop or set a flag to not add record_id.
                            # A robust way: mark current_node as invalid for this record or skip record.
                            # Simplified: If age is bad, this record effectively won't be findable by age.
                            # To make it fully robust, one might need to ensure record_dict[attr] is cleaned earlier.
                            continue # Skip this attribute for this record if age is not parseable to int

                    age_range_s = self._determine_age_range_str(actual_age_int)
                    specific_age_s = str(actual_age_int)
                    
                    # Level 1: Range
                    hash_val_range = self._first_hash(age_range_s)
                    if hash_val_range not in current_node.children:
                        current_node.children[hash_val_range] = LayeredIndexNode()
                        current_node.children[hash_val_range].first_hash = hash_val_range
                    current_node = current_node.children[hash_val_range]
                    
                    # Level 2: Specific Age
                    hash_val_specific = self._first_hash(specific_age_s)
                    if hash_val_specific not in current_node.children:
                        current_node.children[hash_val_specific] = LayeredIndexNode()
                        current_node.children[hash_val_specific].first_hash = hash_val_specific
                    current_node = current_node.children[hash_val_specific]
                else:
                    # Standard handling for other attributes
                    value_s = self._get_attribute_value(record_dict, attr) # Pass record_dict directly
                    hash_value = self._first_hash(value_s)
                    if hash_value not in current_node.children:
                        current_node.children[hash_value] = LayeredIndexNode()
                        current_node.children[hash_value].first_hash = hash_value
                    current_node = current_node.children[hash_value]
            
            current_node.record_ids.add(record_id)
    
    def search(self, query: Dict[str, Any]) -> Set[int]:
        """搜索记录"""
        query_key = os.urandom(32)
        current_nodes_at_level = [self.root] # Renamed for clarity
        results: Set[int] = set()
        
        for attr in self.attribute_order:
            # Handle wildcard for the current attribute
            if attr not in query:
                next_nodes_after_wildcard = []
                if attr == 'age': # Age wildcard spans two conceptual levels
                    for node_before_age in current_nodes_at_level:
                        for age_range_child in node_before_age.children.values(): # All age ranges
                            next_nodes_after_wildcard.extend(age_range_child.children.values()) # All specific ages under them
                else: # Standard wildcard for other attributes (one level)
                    for node in current_nodes_at_level:
                        next_nodes_after_wildcard.extend(node.children.values())
                current_nodes_at_level = list(set(next_nodes_after_wildcard))
                if not current_nodes_at_level: break
                continue

            # Attribute is in query - proceed with specific/range matching
            query_value_for_attr = query[attr]

            if attr == 'age':
                # Age attribute processing (potentially two levels)
                # This will replace current_nodes_at_level with nodes AFTER age processing
                
                # Buffer for nodes found after traversing the age hierarchy for this query
                nodes_after_age_processed = []

                if isinstance(query_value_for_attr, str) and query_value_for_attr.startswith('['):
                    # Age Range Query, e.g., "[60-70)"
                    age_range_s = query_value_for_attr 
                    
                    # Level 1: Traverse to the range node(s)
                    first_hash_range = self._first_hash(age_range_s)
                    query_hash_range = self._second_hash(first_hash_range, query_key)
                    
                    candidate_range_nodes = []
                    for node in current_nodes_at_level: # Nodes from previous attribute
                        for child_hash, child_node in node.children.items():
                            if self._second_hash(child_hash, query_key) == query_hash_range:
                                candidate_range_nodes.append(child_node)
                    
                    # Level 2: Collect all specific age children from these range nodes
                    for range_node in candidate_range_nodes:
                        nodes_after_age_processed.extend(range_node.children.values())
                    current_nodes_at_level = list(set(nodes_after_age_processed))

                else: # Specific Age Query, e.g., 65 (int) or "65" (str)
                    specific_age_int = -1
                    try:
                        specific_age_int = int(query_value_for_attr) # Handles int or str like "65"
                    except (ValueError, TypeError):
                        logging.warning(f"Invalid specific age query value: {query_value_for_attr}. No results for this path.")
                        current_nodes_at_level = [] # Stop search for this path
                        # break # This break would exit the outer attributes loop. Set to empty and let outer loop check.
                    
                    if not current_nodes_at_level : # if parsing failed and list is empty
                         if specific_age_int == -1: # check if parsing failed
                            current_nodes_at_level = [] # Ensure it's empty if error occurred
                         # else: proceed if specific_age_int is valid

                    if specific_age_int != -1: # Proceed if specific_age_int is valid
                        age_range_s = self._determine_age_range_str(specific_age_int)
                        specific_age_s = str(specific_age_int)

                        # Level 1: Traverse to range node(s)
                        first_hash_range = self._first_hash(age_range_s)
                        query_hash_range = self._second_hash(first_hash_range, query_key)
                        
                        candidate_range_nodes = []
                        for node in current_nodes_at_level:
                            for child_hash, child_node in node.children.items():
                                if self._second_hash(child_hash, query_key) == query_hash_range:
                                    candidate_range_nodes.append(child_node)
                        
                        # Level 2: Traverse to specific age node(s) from the found range_nodes
                        for range_node in candidate_range_nodes:
                            first_hash_specific = self._first_hash(specific_age_s)
                            query_hash_specific = self._second_hash(first_hash_specific, query_key)
                            for child_hash, specific_age_node in range_node.children.items():
                                if self._second_hash(child_hash, query_key) == query_hash_specific:
                                    nodes_after_age_processed.append(specific_age_node)
                        current_nodes_at_level = list(set(nodes_after_age_processed))
                    else: # specific_age_int remained -1, meaning error in parsing
                        current_nodes_at_level = []


            else: # Standard attribute processing (one level)
                # _get_attribute_value expects a map {attr: value}
                # Here query_value_for_attr is the actual value.
                value_s = self._get_attribute_value({attr: query_value_for_attr}, attr)
                first_hash = self._first_hash(value_s)
                current_query_hash = self._second_hash(first_hash, query_key)
                
                next_nodes_for_exact_match = []
                for node in current_nodes_at_level:
                    for child_hash, child_node in node.children.items():
                        if self._second_hash(child_hash, query_key) == current_query_hash:
                            next_nodes_for_exact_match.append(child_node)
                current_nodes_at_level = next_nodes_for_exact_match
            
            if not current_nodes_at_level: # If any attribute processing results in no nodes, stop.
                break 
                
        for node in current_nodes_at_level: # These are the leaf nodes (or relevant intermediate for ranges)
            results.update(node.record_ids)
        
        return results
    
    def get_index_size(self) -> int:
        """获取索引大小（字节）"""
        return len(pickle.dumps(self))
    
    def get_total_nodes(self) -> int:
        """获取索引中的总节点数"""
        def count_nodes(node: LayeredIndexNode) -> int:
            count = 1
            for child in node.children.values():
                count += count_nodes(child)
            return count
        return count_nodes(self.root)

    def get_encrypted_index_size(self) -> int: # This might be misleading now, it's size of python objects in memory
        return sys.getsizeof(pickle.dumps(self.root))

    def save_encrypted(self, filepath: str, encryption_key_for_file: bytes):
        """
        保存加密的索引，包括索引树和first_key
        Args:
            filepath: 保存文件路径
            encryption_key_for_file: 文件加密密钥
        """
        if len(encryption_key_for_file) != self.AES_KEY_SIZE:
            raise ValueError(f"File encryption key must be {self.AES_KEY_SIZE} bytes.")
        
        # 保存索引树和first_key
        data_to_save = {
            'root': self.root,
            'first_key': self.first_key,
            'attribute_order': self.attribute_order,
            'age_dict': self.age_dict,
            'diag_meta': self.diag_meta,
            'other_attributes_meta': self.other_attributes_meta
        }
        
        pickled_data = pickle.dumps(data_to_save)
        with open(filepath, 'wb') as f:
            f.write(pickled_data)

    @classmethod
    def load_encrypted(cls, filepath: str, encryption_key_for_file: bytes, 
                       df_for_init: pd.DataFrame) -> Optional['LayeredIndex']:
        """
        加载加密的索引，包括索引树和first_key
        Args:
            filepath: 索引文件路径
            encryption_key_for_file: 文件加密密钥
            df_for_init: 数据框（用于初始化）
        Returns:
            加载的LayeredIndex实例，如果失败则返回None
        """
        if len(encryption_key_for_file) != cls.AES_KEY_SIZE:
            raise ValueError(f"File encryption key must be {cls.AES_KEY_SIZE} bytes.")
        try:
            with open(filepath, 'rb') as f:
                pickled_data = f.read()
            
            loaded_data = pickle.loads(pickled_data)
            
            # 检查加载的数据是否包含必要的字段
            required_keys = ['root', 'first_key', 'attribute_order', 'age_dict', 'diag_meta', 'other_attributes_meta']
            for key in required_keys:
                if key not in loaded_data:
                    raise ValueError(f"Missing required key '{key}' in loaded data")
            
            # 创建实例，使用加载的元数据
            instance = cls(
                df=df_for_init,
                attribute_order=loaded_data['attribute_order'],
                age_dict=loaded_data['age_dict'],
                diag_meta=loaded_data['diag_meta'],
                other_attributes_meta=loaded_data['other_attributes_meta']
            )
            
            # 恢复保存的索引树和first_key
            instance.root = loaded_data['root']
            instance.first_key = loaded_data['first_key']
            
            return instance
        except Exception as e:
            logging.error(f"Error loading encrypted index from {filepath}: {str(e)}")
            return None

    def get_approximate_tree_size(self) -> int:
        def get_node_size(node: LayeredIndexNode) -> int:
            size = sys.getsizeof(node)
            if node.record_ids: # record_ids is bytes
                size += sys.getsizeof(node.record_ids)
            for child in node.children.values():
                size += get_node_size(child)
            return size
        return get_node_size(self.root)

    def insert_record(self, record: Dict[str, Any], record_id: Any) -> bool:
        record_id_int = int(record_id)
        logging.info(f"Attempting to insert record_id: {record_id_int} with data: {record}")
        try:
            current_node = self.root
            for attr in self.attribute_order:
                # Ensure the attribute from attribute_order is present in the record for insertion.
                # build_index implicitly has all columns from df.iterrows().
                if attr not in record:
                    logging.error(f"Attribute '{attr}' (from attribute_order) is missing in the record provided for insertion. Record ID: {record_id_int}")
                    return False # Cannot form a complete path for insertion.

                if attr == 'age':
                    actual_age_int = -1
                    try:
                        actual_age_int = int(record[attr]) # Expects int or string parseable to int.
                    except (ValueError, TypeError) as e:
                        logging.error(f"Error converting age '{record[attr]}' to int for record_id {record_id_int}: {e}. Cannot insert.")
                        return False
                    
                    age_range_s = self._determine_age_range_str(actual_age_int)
                    specific_age_s = str(actual_age_int)
                    
                    # Level 1: Range node
                    hash_val_range = self._first_hash(age_range_s)
                    if hash_val_range not in current_node.children:
                        current_node.children[hash_val_range] = LayeredIndexNode()
                        current_node.children[hash_val_range].first_hash = hash_val_range
                    current_node = current_node.children[hash_val_range]
                    
                    # Level 2: Specific Age node
                    hash_val_specific = self._first_hash(specific_age_s)
                    if hash_val_specific not in current_node.children:
                        current_node.children[hash_val_specific] = LayeredIndexNode()
                        current_node.children[hash_val_specific].first_hash = hash_val_specific
                    current_node = current_node.children[hash_val_specific]
                else:
                    # Standard handling for other attributes
                    value_s = self._get_attribute_value({attr: record[attr]}, attr)
                    hash_value = self._first_hash(value_s)
                    if hash_value not in current_node.children:
                        current_node.children[hash_value] = LayeredIndexNode()
                        current_node.children[hash_value].first_hash = hash_value
                    current_node = current_node.children[hash_value]
            
            current_node.record_ids.add(record_id_int)
            logging.info(f"Successfully inserted record_id: {record_id_int}")
            return True
            
        except Exception as e:
            logging.error(f"Generic error inserting record {record_id_int}: {e}", exc_info=True)
            return False

    def delete_record(self, record_query: Dict[str, Any], record_id: Any) -> bool:
        record_id_int = int(record_id)
        logging.info(f"Attempting to delete record_id: {record_id_int} using query: {record_query}")

        # For deletion, if 'age' is specified in record_query, it must be a specific age, not a range.
        if 'age' in record_query and isinstance(record_query['age'], str) and record_query['age'].startswith('['):
            logging.error(f"Deletion failed for record_id {record_id_int}: record_query for age must be a specific age, not a range. Got: {record_query['age']}")
            return False

        # Traversal logic to find the node based on record_query (using first_hash only for path finding)
        # This part does not use the query_key or second_hash, as we are finding the actual storage path.
        current_path_nodes = [self.root]
        
        for attr in self.attribute_order:
            next_path_nodes = []
            if attr not in record_query: # Wildcard: Traverse all children for this attribute level
                if attr == 'age': # Age wildcard spans two conceptual levels
                    temp_nodes_after_age_range_wildcard = []
                    for node in current_path_nodes:
                        temp_nodes_after_age_range_wildcard.extend(node.children.values()) # All age_range children
                    for range_node in temp_nodes_after_age_range_wildcard:
                        next_path_nodes.extend(range_node.children.values()) # All specific_age children under them
                else: # Standard wildcard for other attributes
                    for node in current_path_nodes:
                        next_path_nodes.extend(node.children.values())
            else: # Attribute is specified in the record_query
                query_val_for_attr = record_query[attr]
                if attr == 'age':
                    actual_age_int = -1
                    try:
                        actual_age_int = int(query_val_for_attr) # Expects int or str parseable to int
                    except (ValueError, TypeError) as e:
                        logging.error(f"Deletion failed for record_id {record_id_int}: Age '{query_val_for_attr}' in record_query is not a valid specific age int: {e}")
                        return False
                    
                    age_range_s = self._determine_age_range_str(actual_age_int)
                    specific_age_s = str(actual_age_int)
                    
                    # Path for age: Level 1 (Range), then Level 2 (Specific)
                    first_hash_range = self._first_hash(age_range_s)
                    first_hash_specific = self._first_hash(specific_age_s)

                    for node in current_path_nodes:
                        range_child_node = node.children.get(first_hash_range)
                        if range_child_node:
                            specific_age_child_node = range_child_node.children.get(first_hash_specific)
                            if specific_age_child_node:
                                next_path_nodes.append(specific_age_child_node)
                else: # Other attributes (single level)
                    value_s = self._get_attribute_value({attr: query_val_for_attr}, attr)
                    hash_val = self._first_hash(value_s)
                    for node in current_path_nodes:
                        child_node = node.children.get(hash_val)
                        if child_node:
                            next_path_nodes.append(child_node)
            
            current_path_nodes = list(set(next_path_nodes)) # Deduplicate if paths converge due to wildcards
            if not current_path_nodes:
                logging.warning(f"Deletion path for record_id {record_id_int} broke at attribute '{attr}'. Query: {record_query}. Record not found at expected path.")
                return False # Path does not fully exist to the leaf node
        
        # At this point, current_path_nodes contains the potential leaf node(s) matching the record_query.
        # For a well-defined record, this should ideally be one node.
        found_and_removed = False
        if not current_path_nodes:
             logging.warning(f"No final node found for record_id {record_id_int} with query {record_query}.")
             return False

        for final_node in current_path_nodes:
            if record_id_int in final_node.record_ids:
                final_node.record_ids.remove(record_id_int)
                found_and_removed = True
                logging.info(f"Successfully removed record_id {record_id_int} from node's record_ids set.")
                # Node cleanup (optional): If final_node.record_ids and final_node.children are empty,
                # one could implement logic to prune this node from its parent. This is complex.
                break # Assume record_id is unique and thus found in only one leaf node if path is specific.
        
        if not found_and_removed:
            logging.warning(f"Record_id {record_id_int} was not found in the record_ids set of the final node(s) reached by query {record_query}.")

        return found_and_removed

    def _get_query_path_for_test(self, query: Dict[str, Any]) -> list:
        """生成本次查询的hash路径，仅用于测试不可链接性"""
        # 为本次查询生成新的随机密钥
        query_key = os.urandom(32)
        path = []
        for attr in self.attribute_order:
            if attr not in query:
                # For wildcard attributes, their values are not part of the query path.
                # The original _get_query_path_for_test skipped them, which is correct
                # as the path is defined by *queried* attributes.
                continue 
            
            # 对查询值进行与 _get_attribute_value 相同的处理
            # If query[attr] is an int for age, _get_attribute_value converts to str.
            # If query[attr] is a range string like "[60-70)", it's returned as is.
            # This means the path for a range query is based on hashing the range string itself,
            # which is fine for testing unlinkability of two identical range query objects.
            value = self._get_attribute_value({attr: query[attr]}, attr)
            
            # 先用第一层 hash
            first_hash = self._first_hash(value)
            # 再用 query_key 对 first_hash 做第二层 hash
            query_hash = self._second_hash(first_hash, query_key)
            path.append(query_hash)
        return path 