import json
import os
from typing import Dict, List, Optional, Union
import re

class ICDEncoder:
    def __init__(self, icd_file: str = "doc/icd/icd9cm_structure.json"):
        """
        初始化ICD编码器
        
        Args:
            icd_file: ICD编码结构文件路径
        """
        self.icd_structure = self._load_icd_structure(icd_file)
        self.code_to_name = self._build_code_mapping()
        
    def _load_icd_structure(self, icd_file: str) -> Dict:
        """
        加载ICD编码结构
        
        Args:
            icd_file: ICD编码结构文件路径
            
        Returns:
            ICD编码结构字典
        """
        if not os.path.exists(icd_file):
            raise FileNotFoundError(f"ICD编码结构文件不存在: {icd_file}")
            
        with open(icd_file, "r", encoding="utf-8") as f:
            return json.load(f)
            
    def _build_code_mapping(self) -> Dict[str, str]:
        """
        构建代码到名称的映射
        
        Returns:
            代码到名称的映射字典
        """
        code_mapping = {}
        
        def process_category(category: Dict, prefix: str = ""):
            if "name" in category:
                code_mapping[prefix] = category["name"]
                
            if "subcategories" in category:
                for code, subcategory in category["subcategories"].items():
                    if isinstance(subcategory, dict):
                        process_category(subcategory, code)
                    else:
                        code_mapping[code] = subcategory
                        
        for code_range, category in self.icd_structure.items():
            process_category(category, code_range)
            
        return code_mapping
        
    def encode(self, code: str) -> str:
        """
        将ICD代码编码为名称
        
        Args:
            code: ICD代码
            
        Returns:
            对应的疾病名称
        """
        # 标准化代码格式
        code = str(code).strip()
        if not code:
            return "未知"
            
        # 尝试直接匹配
        if code in self.code_to_name:
            return self.code_to_name[code]
            
        # 尝试匹配范围
        for code_range, name in self.code_to_name.items():
            if "-" in code_range:
                start, end = map(int, code_range.split("-"))
                try:
                    code_num = int(code.replace(".", ""))
                    if start <= code_num <= end:
                        return name
                except ValueError:
                    continue
                    
        return "未知"
        
    def decode(self, name: str) -> List[str]:
        """
        将疾病名称解码为ICD代码
        
        Args:
            name: 疾病名称
            
        Returns:
            对应的ICD代码列表
        """
        return [code for code, n in self.code_to_name.items() if n == name]
        
    def get_all_codes(self) -> List[str]:
        """
        获取所有ICD代码
        
        Returns:
            ICD代码列表
        """
        return list(self.code_to_name.keys())
        
    def get_all_names(self) -> List[str]:
        """
        获取所有疾病名称
        
        Returns:
            疾病名称列表
        """
        return list(set(self.code_to_name.values()))
        
    def get_code_range(self, code: str) -> Optional[str]:
        """
        获取代码所属的范围
        
        Args:
            code: ICD代码
            
        Returns:
            代码范围
        """
        code = str(code).strip()
        if not code:
            return None
        code_digits = code.replace('.', '')
        if code_digits.isdigit():
            if len(code_digits) >= 5:
                code_num = int(code_digits)
            else:
                code_num = int(code_digits[:3])
        else:
            return None
        for code_range in self.icd_structure.keys():
            if "-" in code_range:
                start, end = map(int, code_range.split("-"))
                if start <= code_num <= end:
                    return code_range
        return None 