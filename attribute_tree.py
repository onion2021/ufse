#!/usr/bin/env python
# -*- coding:utf-8 -*-
import numpy as np
import math # For ceil
from typing import List, Tuple, Optional, Any

# Placeholder for gencode_func for standalone testing
def generate_compact_code_placeholder(value_index, num_bits):
    if num_bits == 0:
        return np.array([], dtype=int) if value_index == 0 else np.array([-1], dtype=int) # Error/special case for non-zero index with 0 bits
    if value_index < 0 or value_index >= (1 << num_bits):
        return np.zeros(num_bits, dtype=int) # Fallback for out of range
    return np.array([int(bit) for bit in bin(value_index)[2:].zfill(num_bits)], dtype=int)

class TrieNode:
    def __init__(self):
        self.child = {} # content is {path_segment: TrieNode}
        # path_segment can be strings like '1-100' or tuples like (0,1,...,9) or int for specific age.

class AgeTree:
    def __init__(self):
        self.algorithm = "age_attribute_tree"
        self.root = TrieNode()

    def add_keyword(self, keyword_path): # keyword_path is a list of segments
        node_curr = self.root
        for segment in keyword_path:
            # Ensure segment is hashable for dict key
            current_segment_key = segment
            if isinstance(segment, list): current_segment_key = tuple(segment)
            elif isinstance(segment, np.ndarray): current_segment_key = tuple(segment)

            if node_curr.child.get(current_segment_key) is None:
                node_next = TrieNode()
                node_curr.child[current_segment_key] = node_next
            node_curr = node_curr.child[current_segment_key]
        return self.root

    def add_keywords_from_list(self, attribute_split_list):
        for path_list in attribute_split_list:
            self.add_keyword(path_list)

    def age_to_tree(self, age_int: int) -> Optional[List[Tuple[str, ...]]]:
        """
        Determines the hierarchical path for a given integer age.
        Returns a list of string tuples, where each tuple is a segment in the path.
        Example: age_int = 25 -> [('1-100',), ('0-30',), ('20-30',)]
        Returns None if the age is out of the supported range (0-99).
        """
        if not (0 <= age_int < 100):
            # print(f"Warning: Age {age_int} is out of the supported range (0-99).")
            return None

        path_str_segments = []
        path_str_segments.append('1-100')

        if age_int < 30:
            path_str_segments.append('0-30')
            if age_int < 10: path_str_segments.append('0-10')
            elif age_int < 20: path_str_segments.append('10-20')
            else: path_str_segments.append('20-30') # 20-29
        elif age_int < 60:
            path_str_segments.append('30-60')
            if age_int < 40: path_str_segments.append('30-40')
            elif age_int < 50: path_str_segments.append('40-50')
            else: path_str_segments.append('50-60') # 50-59
        else:  # 60-99
            path_str_segments.append('60-100')
            if age_int < 70: path_str_segments.append('60-70')
            elif age_int < 80: path_str_segments.append('70-80')
            elif age_int < 90: path_str_segments.append('80-90')
            else: path_str_segments.append('90-100') # 90-99
        
        # Append the exact age as the final segment
        path_str_segments.append(str(age_int))

        return [(s,) for s in path_str_segments]

    def to_compact_vect(self, age_int_value, gencode_func):
        """
        Generates compact vector parts for an age.
        Returns a dictionary:
        {
            'full_path_codes': [code_L1, code_L2, ..., code_Ln, code_specific_age_in_bucket],
            'bucket_path_codes': [code_L1, code_L2, ..., code_Ln_bucket_itself]
        }
        Returns empty lists in dict if age is out of range or error occurs.
        """
        hierarchical_path_to_tuple = self.age_to_tree(age_int_value) # age_to_tree now returns only the path list or None
        # final_age_val is the original age_int_value, used for specific age encoding part if applicable

        default_return = {'full_path_codes': [], 'bucket_path_codes': []}

        if not hierarchical_path_to_tuple:
            # print(f"Warning: Age {age_int_value} could not be mapped to a tree path by age_to_tree for compact_vect.")
            return default_return

        bucket_path_codes_list = []
        current_node = self.root

        for i, path_segment_key_tuple in enumerate(hierarchical_path_to_tuple):
            # path_segment_key_tuple is like ('1-100',), we need the string '1-100'
            if not (isinstance(path_segment_key_tuple, tuple) and len(path_segment_key_tuple) > 0):
                # print(f"Warning: Expected tuple segment from age_to_tree, got {path_segment_key_tuple}")
                return default_return # Path is malformed for to_compact_vect logic
            
            path_segment_key = path_segment_key_tuple[0] # Extract the string like '1-100'

            children_keys = list(current_node.child.keys())
            num_siblings = len(children_keys)

            if num_siblings == 0:
                print(f"Warning: Node for {path_segment_key} has no children, path traversal issue.")
                return default_return
            
            try:
                # Ensure path_segment_key (string) is compared against str keys in current_node.child
                # Children keys in the Trie built by add_keywords_from_list are typically strings or tuples based on age_tree_data_from_tool
                # Let's assume children_keys are strings for age ranges like '0-10', '10-20'
                # The original age_tree_data_from_tool used strings for ranges, and tuples like tuple(range(0,10)) for leaf groups.
                # The new age_to_tree returns path like [('1-100',), ('0-30',), ('0-10',)].
                # So, path_segment_key will be a string e.g. '0-10'.
                # current_node.child keys should match this if Trie is built consistently.
                
                # The Trie built by add_keywords_from_list using age_tree_data_from_tool uses strings as keys for these levels.
                # e.g., age_tree_data_from_tool = [['1-100', '0-30', '0-10', tuple(range(0,10))]]
                # Here, '1-100', '0-30', '0-10' are string keys. The last one tuple(range(0,10)) is a tuple key.
                # Our new age_to_tree returns [('1-100',), ('0-30',), ('0-10',)].
                # So, path_segment_key variable correctly becomes '1-100', then '0-30', then '0-10'.
                # This assumes the Trie (self.root) is built with string keys like '1-100', '0-30', etc. at these levels.

                sorted_children_keys = sorted([str(k) for k in children_keys]) # Sort as strings for consistent indexing
                current_segment_idx = sorted_children_keys.index(str(path_segment_key)) # Compare string form

            except ValueError:
                # print(f"Warning: Path segment {path_segment_key} not found in stringified children {sorted_children_keys} of current node.")
                return default_return # Path segment not found in children

            num_bits_for_level = math.ceil(math.log2(num_siblings)) if num_siblings > 1 else (1 if num_siblings == 1 else 0)
            level_code_np = gencode_func(current_segment_idx, num_bits_for_level)
            bucket_path_codes_list.append(tuple(level_code_np.tolist())) # Store as tuple of ints
            
            current_node = current_node.child[path_segment_key]

        full_path_codes_list = list(bucket_path_codes_list) # Copy bucket path codes

        if isinstance(hierarchical_path_to_tuple[-1], tuple) and len(hierarchical_path_to_tuple[-1]) > 0:
            # This block was for handling the old age_to_tree format where the last element was tuple(range(X,Y)).
            # The new age_to_tree returns path like [('1-100',), ('0-30',), ('0-10',)].
            # The specific age encoding (e.g. for '5' in '0-10') is not part of this path.
            # The to_compact_vect logic was for a different kind of encoding where specific age within bucket had bits.
            # For now, to make it run, we can assume the bucket_path_codes_list is the full path for this old vector model.
            # The concept of 'full_path_codes' adding more bits for specific age might be obsolete or need rethink
            # if age_to_tree path represents the bucket only.
            # Let's assume for now full_path_codes is same as bucket_path_codes if leaf isn't a tuple group.
            final_age_val_for_specific_encoding = age_int_value # Use the original age for this part
            try:
                # This logic assumes a final tuple group in the path, which is no longer the case.
                # For compatibility, let's assume a fixed 4 bits for the specific age within its 10-year bucket.
                # This part of to_compact_vect needs careful review if it's still critical.
                val_index_in_assumed_bucket = final_age_val_for_specific_encoding % 10 
                final_segment_code_np = gencode_func(val_index_in_assumed_bucket, 4) # Assuming 4 bits for 0-9 index
                full_path_codes_list.append(tuple(final_segment_code_np.tolist()))
            except ValueError: # Should not happen with % 10 and gencode_func handling range
                 # print(f"Warning: Error encoding specific age {final_age_val_for_specific_encoding} for full_path_codes.")
                 full_path_codes_list.append(tuple(np.zeros(4, dtype=int).tolist())) # Fallback
        # else:
            # print(f"Debug: Last path segment for age {age_int_value} was {hierarchical_path_to_tuple[-1]}, not a detailed age group tuple for specific age encoding in to_compact_vect.")
            # This means full_path_codes might be considered same as bucket_path_codes by this old logic.
            pass # full_path_codes_list remains same as bucket_path_codes_list if no specific age bits added

        if not bucket_path_codes_list:
             return default_return
             
        return {'full_path_codes': full_path_codes_list, 'bucket_path_codes': bucket_path_codes_list}

    def encode(self, value: Any, attribute_name: Optional[str] = None) -> Optional[List[Tuple[str, ...]]]:
        """
        Encoder interface for LayeredIndex compatibility.
        Args:
            value: The age value (expected to be an integer).
            attribute_name: The name of the attribute (e.g., 'age'). Ignored by this specific encoder
                          as it only handles ages, but included for interface consistency.
        Returns:
            A list of path segment tuples, e.g., for age 67: [('1-100',), ('60-100',), ('60-70',)],
            or None if the age cannot be processed (e.g., not an int or out of range).
        """
        if not isinstance(value, int):
            # print(f"Warning: AgeTree.encode expects an integer value, got {type(value)}: {value}")
            return None
        
        # attribute_name is ignored here as AgeTree is specific to age.
        return self.age_to_tree(value)

    def build_dict(self, gencode_func):
        age_compact_dict = {}
        # tool.py's age_tree data, used to build the Trie structure first
        # This is usually done by get_age_dict in tool.py before build_dict is called.
        # Here, we assume the Trie is already populated based on age_tree structure.

        for age_val in range(100): # Process ages 0-99
            # Get the dictionary containing 'full_path_codes' and 'bucket_path_codes'
            # to_compact_vect has been updated to use the new age_to_tree and takes age_val directly
            path_codes_dict = self.to_compact_vect(age_val, gencode_func)
            age_compact_dict[age_val] = path_codes_dict
        
        # Handling for range strings like '60-100' needs specific logic if to be included.
        # For now, they are omitted. Trapdoors for ranges will be built differently.
        # Example: age_compact_dict['60-100'] = generate_compact_code_for_range(...) 
        return age_compact_dict

# Example of how tool.py would use this:
if __name__ == '__main__':
    # 1. Define the age_tree structure (as in tool.py)
    # This structure defines the Trie paths and node keys.
    age_tree_data_from_tool = [
        ['1-100', '0-30', '0-10', tuple(range(0,10))],
        ['1-100', '0-30', '10-20', tuple(range(10,20))],
        ['1-100', '0-30', '20-30', tuple(range(20,30))],
        # ['1-100', '0-30', '*_range_placeholder'], # How '*' was handled needs review for compact
        ['1-100', '30-60', '30-40', tuple(range(30,40))],
        ['1-100', '30-60', '40-50', tuple(range(40,50))],
        ['1-100', '30-60', '50-60', tuple(range(50,60))],
        # ['1-100', '30-60', '*_range_placeholder'],
        ['1-100', '60-100', '60-70', tuple(range(60,70))],
        ['1-100', '60-100', '70-80', tuple(range(70,80))],
        ['1-100', '60-100', '80-90', tuple(range(80,90))],
        ['1-100', '60-100', '90-100', tuple(range(90,100))]
        # The original also had entries like ['1-100', '0-30', '*_range_placeholder'], 
        # these represent a range query at that level. Encoding them needs thought.
        # For compact encoding, a trapdoor for "'0-30' range" might mean specific codes for '1-100', '0-30',
        # and then "all zeros" for deeper levels if that means wildcard.
    ]

    age_processing_tree = AgeTree()
    age_processing_tree.add_keywords_from_list(age_tree_data_from_tool) # Build the Trie

    # Now use the build_dict method with a gencode_func
    # tool.py would pass its actual generate_compact_code function
    generated_age_dict = age_processing_tree.build_dict(gencode_func=generate_compact_code_placeholder)

    for age_example in [0, 9, 10, 25, 65, 99]:
        age_data = generated_age_dict.get(age_example)
        if age_data:
            print(f"Age {age_example}:")
            print(f"  Full Path Codes:   {age_data['full_path_codes']}")
            # print(f"    (lengths: {[len(c) for c in age_data['full_path_codes']]})")
            print(f"  Bucket Path Codes: {age_data['bucket_path_codes']}")
            # print(f"    (lengths: {[len(c) for c in age_data['bucket_path_codes']]})")
        else:
            print(f"Age {age_example}: No data generated.")
    
    # Example for age 25:
    # Expected hierarchical_path_to_tuple for 25: ['1-100', '0-30', '20-30', tuple(range(20,30))]
    # Codes for bucket_path_codes:
    # L1 ('1-100'): idx 0 of 1 (sorted ['1-100']) -> 1 bit -> (0,)
    # L2 ('0-30'): idx 0 of 3 (sorted ['0-30', '30-60', '60-100']) -> 2 bits -> (0,0)
    # L3 ('20-30'): idx 2 of 3 (sorted ['0-10', '10-20', '20-30']) -> 2 bits -> (1,0)
    # L4 (tuple(range(20,30))): idx 2 of 3 (sorted [tuple(range(0,10)), tuple(range(10,20)), tuple(range(20,30))]) assuming these are the children of '20-30' in the test data used for trie
    # Wait, the age_tree_data_from_tool implies tuple(range(X,Y)) IS the leaf segment of the path.
    # So, the path is ['1-100', '0-30', '20-30', tuple(range(20,30))].
    # The to_compact_vect iterates this path.
    # Let's re-check the example from before.
    # Path: ['1-100', '0-30', '20-30', tuple(range(20,30))]
    # Codes for this path (bucket_path_codes):
    # Code for '1-100' (root child)
    # Code for '0-30' (child of '1-100')
    # Code for '20-30' (child of '0-30')
    # Code for tuple(range(20,30)) (child of '20-30')
    # Then, full_path_codes adds:
    # Code for 25 within tuple(range(20,30)) (index 5) -> (0,1,0,1)

    # With the example age_tree_data_from_tool structure:
    # Path for 25: ['1-100', '0-30', '20-30', (20, 21, ..., 29)]
    # Code for '1-100' (node_curr=root, children=['1-100']) -> seg_idx=0, bits=1 -> (0,)
    # Code for '0-30' (node_curr=node for '1-100', children=['0-30','30-60','60-100']) -> seg_idx=0, bits=2 -> (0,0)
    # Code for '20-30' (node_curr=node for '0-30', children=['0-10','10-20','20-30']) -> seg_idx=2, bits=2 -> (1,0)
    # Code for (20..29) (node_curr=node for '20-30', children=[(0..9), (10..19), (20..29)]) -> seg_idx=2, bits=2 -> (1,0)
    # bucket_path_codes = [(0,), (0,0), (1,0), (1,0)]
    # specific age code for 25 in (20..29) -> idx 5, 4 bits -> (0,1,0,1)
    # full_path_codes = [(0,), (0,0), (1,0), (1,0), (0,1,0,1)]
    
    # Test an out-of-range age for path generation
    # print(f"Compact vector for age 101: {age_processing_tree.to_compact_vect(101, generate_compact_code_placeholder)}")

    # Expected path for age 25: ['1-100', '0-30', '20-30', (20,21,...,29)], final_age_val=25
    # Level 1 ('1-100'): 1 sibling ('1-100') -> idx 0, 1 bit -> [0]
    # Level 2 ('0-30'): 3 siblings ('0-30', '30-60', '60-100') under '1-100' -> idx 0 for '0-30', ceil(log2(3))=2 bits -> [0,0]
    # Level 3 ('20-30'): 3 siblings ('0-10', '10-20', '20-30') under '0-30' -> idx 2 for '20-30', ceil(log2(3))=2 bits -> [1,0]
    # Level 4 (tuple (20,..,29)): path ends here as per revised age_to_tree. current_node is node for (20,..,29)
    # Final encoding for 25 within (20,..,29) (idx 5 for value 25): 4 bits -> [0,1,0,1]
    # Total: [0] + [0,0] + [1,0] + [0,1,0,1] = [0,0,0,1,0,0,1,0,1] (9 bits)
    # This assumes the Trie structure in add_keywords_from_list correctly mirrors age_to_tree paths.
    # Specifically, `age_tree_data_from_tool` should define these paths for the Trie to be built for traversal.

