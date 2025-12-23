# -*- coding: utf-8 -*-
"""
代码验证脚本 - 简化版
=====================

验证所有Python文件的语法正确性和可导入性
"""

import sys
from pathlib import Path


def check_syntax():
    """检查所有Python文件的语法"""
    print("="*70)
    print("检查Python文件语法...")
    print("="*70)
    
    base_dir = Path(__file__).parent
    py_files = list(base_dir.glob("hetao_ag/**/*.py"))
    
    errors = []
    for py_file in py_files:
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                compile(f.read(), py_file, 'exec')
        except SyntaxError as e:
            errors.append((py_file, e))
            print(f"[ERROR] {py_file.name}: {e}")
    
    if errors:
        print(f"\n发现 {len(errors)} 个语法错误")
        return False
    else:
        print(f"\n[OK] 所有 {len(py_files)} 个文件语法正确")
        return True


def check_imports():
    """检查核心模块是否可以导入"""
    print("\n"+"="*70)
    print("检查模块导入...")
    print("="*70)
    
    modules = [
        "hetao_ag",
        "hetao_ag.core",
        "hetao_ag.soil",
        "hetao_ag.water",
        "hetao_ag.crop",
    ]
    
    errors = []
    for module in modules:
        try:
            __import__(module)
            print(f"[OK] {module}")
        except Exception as e:
            errors.append((module, e))
            print(f"[ERROR] {module}: {e}")
    
    if errors:
        print(f"\n发现 {len(errors)} 个导入错误")
        return False
    else:
        print(f"\n[OK] 所有模块导入成功")
        return True


def main():
    """主函数"""
    print("\n河套智慧农牧业库 - 代码验证\n")
    
    syntax_ok = check_syntax()
    import_ok = check_imports()
    
    print("\n"+"="*70)
    if syntax_ok and import_ok:
        print("[SUCCESS] 代码验证通过!")
        print("="*70)
        return 0
    else:
        print("[FAILED] 代码验证失败")
        print("="*70)
        return 1


if __name__ == "__main__":
    sys.exit(main())
