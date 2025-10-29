# medical_codes.py

def _get_all_leaf_codes(node):
    """
    Recursively collects all specific_codes from a node and its children.
    """
    leaf_codes = []
    if "specific_codes" in node:
        leaf_codes.extend(node.get("specific_codes", []))
    
    if "children" in node:
        for child_key in node["children"]:
            leaf_codes.extend(_get_all_leaf_codes(node["children"][child_key]))
    return list(set(leaf_codes)) # Return unique codes


def get_codes_for_category(category_identifier, hierarchy_dict):
    """
    Retrieves all specific ICD codes for a given category identifier from a hierarchy.
    The identifier can be a top-level key (e.g., '250') or a 'name' within the hierarchy.
    It performs a case-insensitive search for names.
    """
    
    # Check if identifier is a direct key in the main children, or a key within a child (e.g. '250.0')
    path_parts = category_identifier.split('.')
    
    current_level_dict = hierarchy_dict.get("children", {})
    node_found = None

    # Traverse by path parts if it looks like a code key (e.g., "250" or "250.0")
    temp_node = current_level_dict
    for i, part in enumerate(path_parts):
        current_key_part = ".".join(path_parts[:i+1])
        if temp_node and isinstance(temp_node, dict) and current_key_part in temp_node:
            if i == len(path_parts) - 1: # Last part, this is the node
                 node_found = temp_node[current_key_part]
                 break
            elif "children" in temp_node[current_key_part]:
                temp_node = temp_node[current_key_part]["children"]
            else: # Path exists but doesn't lead to further children as expected
                temp_node = None # Stop traversal
                break
        # Check if the part itself is a key at the current children level
        # This handles cases like "250" being a key in hierarchy_dict["children"]
        # or "0" being a key in hierarchy_dict["children"]["250"]["children"] (if 250.0 was keyed as "0")
        elif temp_node and isinstance(temp_node, dict) and part in temp_node:
            if i == len(path_parts) - 1: # Last part
                node_found = temp_node[part]
                break
            elif "children" in temp_node[part]:
                temp_node = temp_node[part]["children"]
            else:
                temp_node = None
                break
        else:
            temp_node = None # Key not found
            break
            
    if node_found:
        return _get_all_leaf_codes(node_found)

    # If not found by key path, search by name recursively (case-insensitive)
    queue = [hierarchy_dict]
    nodes_to_visit = [hierarchy_dict] # Use a list for BFS/DFS style traversal

    visited_nodes_for_name_search = [] # To handle potential cycles if any, though not expected with trees

    while nodes_to_visit:
        current_node = nodes_to_visit.pop(0)
        visited_nodes_for_name_search.append(current_node)

        if "name" in current_node and current_node["name"].lower() == category_identifier.lower():
            return _get_all_leaf_codes(current_node)
        
        if "children" in current_node:
            for child_key in current_node["children"]:
                child_node = current_node["children"][child_key]
                if child_node not in visited_nodes_for_name_search:
                    nodes_to_visit.append(child_node)
    return []


DIABETES_ICD9_HIERARCHY = {
    "name": "Diabetes Mellitus and related conditions (ICD-9: 249, 250)",
    "description": "Includes primary diabetes mellitus (250) and secondary diabetes mellitus (249).",
    "children": {
        "250": {
            "name": "Diabetes mellitus (Primary)",
            "description": "ICD-9 Code 250: Diabetes mellitus",
            "specific_codes": [], # '250' itself is usually a category, not a billable code.
            "children": {
                "250.0": {
                    "name": "Diabetes mellitus without mention of complication",
                    "specific_codes": ["250.00", "250.01", "250.02", "250.03"]
                },
                "250.1": {
                    "name": "Diabetes with ketoacidosis",
                    "specific_codes": ["250.10", "250.11", "250.12", "250.13"]
                },
                "250.2": {
                    "name": "Diabetes with hyperosmolarity",
                    "specific_codes": ["250.20", "250.21", "250.22", "250.23"]
                },
                "250.3": {
                    "name": "Diabetes with other coma",
                    "specific_codes": ["250.30", "250.31", "250.32", "250.33"]
                },
                "250.4": {
                    "name": "Diabetes with renal manifestations",
                    "specific_codes": ["250.40", "250.41", "250.42", "250.43"]
                },
                "250.5": {
                    "name": "Diabetes with ophthalmic manifestations",
                    "specific_codes": ["250.50", "250.51", "250.52", "250.53"]
                },
                "250.6": {
                    "name": "Diabetes with neurological manifestations",
                    "specific_codes": ["250.60", "250.61", "250.62", "250.63"]
                },
                "250.7": {
                    "name": "Diabetes with peripheral circulatory disorders",
                    "specific_codes": ["250.70", "250.71", "250.72", "250.73"]
                },
                "250.8": {
                    "name": "Diabetes with other specified manifestations",
                    "specific_codes": ["250.80", "250.81", "250.82", "250.83"]
                },
                "250.9": {
                    "name": "Diabetes with unspecified complication",
                    "specific_codes": ["250.90", "250.91", "250.92", "250.93"]
                }
            }
        },
        "249": {
            "name": "Secondary diabetes mellitus",
            "description": "ICD-9 Code 249: Secondary diabetes mellitus",
            "specific_codes": [], # '249' itself is usually a category.
            "children": {
                # Based on ICD-9 structure, these would be 249.0x, 249.1x etc.
                # For now, leaving these more specific codes empty, to be filled.
                "249.0": {"name": "Secondary DM without mention of complication", "specific_codes": ["249.00", "249.01"]}, # Example
                "249.1": {"name": "Secondary DM with ketoacidosis", "specific_codes": ["249.10", "249.11"]}, # Example
                "249.2": {"name": "Secondary DM with hyperosmolarity", "specific_codes": ["249.20", "249.21"]}, # Example
                "249.3": {"name": "Secondary DM with other coma", "specific_codes": ["249.30", "249.31"]}, # Example
                "249.4": {"name": "Secondary DM with renal manifestations", "specific_codes": ["249.40", "249.41"]}, # Example
                "249.5": {"name": "Secondary DM with ophthalmic manifestations", "specific_codes": ["249.50", "249.51"]}, # Example
                "249.6": {"name": "Secondary DM with neurological manifestations", "specific_codes": ["249.60", "249.61"]}, # Example
                "249.7": {"name": "Secondary DM with peripheral circulatory disorders", "specific_codes": ["249.70", "249.71"]}, # Example
                "249.8": {"name": "Secondary DM with other specified manifestations", "specific_codes": ["249.80", "249.81"]}, # Example
                "249.9": {"name": "Secondary DM with unspecified complication", "specific_codes": ["249.90", "249.91"]}  # Example
            }
        }
    }
}

# Example usage (can be tested by running this file directly if __main__ block is added)
# if __name__ == '__main__':
# print("--- Testing DIABETES_ICD9_HIERARCHY ---")
# print(f"Codes for '250.0': {get_codes_for_category('250.0', DIABETES_ICD9_HIERARCHY)}")
# print(f"Codes for 'Diabetes with ketoacidosis': {get_codes_for_category('Diabetes with ketoacidosis', DIABETES_ICD9_HIERARCHY)}")
# print(f"Codes for '250': {get_codes_for_category('250', DIABETES_ICD9_HIERARCHY)}")
# print(f"Codes for 'Diabetes mellitus (Primary)': {get_codes_for_category('Diabetes mellitus (Primary)', DIABETES_ICD9_HIERARCHY)}")
# print(f"Codes for '249.1': {get_codes_for_category('249.1', DIABETES_ICD9_HIERARCHY)}")
# print(f"All codes for Diabetes hierarchy: {get_codes_for_category('Diabetes Mellitus and related conditions (ICD-9: 249, 250)', DIABETES_ICD9_HIERARCHY)}")
# print(f"Non-existent category: {get_codes_for_category('NonExistent', DIABETES_ICD9_HIERARCHY)}")
# print(f"Non-existent code key: {get_codes_for_category('999.9', DIABETES_ICD9_HIERARCHY)}")

# To add more hierarchies:
# ICD9_HIERARCHIES = {
# "diabetes": DIABETES_ICD9_HIERARCHY,
#     # e.g. "blood_disorders": BLOOD_DISORDERS_ICD9_HIERARCHY,
# } 