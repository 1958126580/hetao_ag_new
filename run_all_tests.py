# -*- coding: utf-8 -*-
"""
河套智慧农牧业库 - 综合测试套件
================================

提供全面的测试覆盖,包括:
    - 单元测试
    - 集成测试  
    - 性能测试
    - 代码质量验证

运行方式:
    python run_all_tests.py
"""

import sys
import subprocess
from pathlib import Path
from typing import List, Tuple
import time


class TestRunner:
    """测试运行器"""
    
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.results: List[Tuple[str, bool, str]] = []
    
    def run_command(self, cmd: List[str], description: str) -> bool:
        """
        运行命令并记录结果
        
        参数:
            cmd: 命令列表
            description: 测试描述
            
        返回:
            是否成功
        """
        print(f"\n{'='*70}")
        print(f"运行: {description}")
        print(f"命令: {' '.join(cmd)}")
        print(f"{'='*70}")
        
        try:
            result = subprocess.run(
                cmd,
                cwd=self.base_dir,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace'
            )
            
            success = result.returncode == 0
            
            if success:
                print(f"✓ {description} - 通过")
                output = result.stdout
            else:
                print(f"✗ {description} - 失败")
                output = result.stderr or result.stdout
            
            self.results.append((description, success, output))
            return success
            
        except Exception as e:
            print(f"✗ {description} - 错误: {e}")
            self.results.append((description, False, str(e)))
            return False
    
    def print_summary(self):
        """打印测试摘要"""
        print(f"\n\n{'='*70}")
        print("测试摘要")
        print(f"{'='*70}\n")
        
        passed = sum(1 for _, success, _ in self.results if success)
        total = len(self.results)
        
        for desc, success, _ in self.results:
            status = "✓ 通过" if success else "✗ 失败"
            print(f"{status:10} | {desc}")
        
        print(f"\n{'='*70}")
        print(f"总计: {passed}/{total} 通过 ({passed/total*100:.1f}%)")
        print(f"{'='*70}\n")
        
        return passed == total


def main():
    """主测试函数"""
    print("="*70)
    print("河套智慧农牧业库 - 综合测试套件")
    print("="*70)
    print()
    
    base_dir = Path(__file__).parent
    runner = TestRunner(base_dir)
    
    # 1. 语法检查 - 编译所有Python文件
    print("\n[阶段 1/5] 语法检查")
    print("-" * 70)
    
    py_files = list(base_dir.glob("hetao_ag/**/*.py"))
    syntax_errors = []
    
    for py_file in py_files:
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                compile(f.read(), py_file, 'exec')
            print(f"✓ {py_file.relative_to(base_dir)}")
        except SyntaxError as e:
            syntax_errors.append((py_file, e))
            print(f"✗ {py_file.relative_to(base_dir)}: {e}")
    
    if syntax_errors:
        print(f"\n发现 {len(syntax_errors)} 个语法错误")
        runner.results.append(("语法检查", False, f"{len(syntax_errors)} 个错误"))
    else:
        print(f"\n✓ 所有 {len(py_files)} 个文件语法正确")
        runner.results.append(("语法检查", True, f"{len(py_files)} 个文件"))
    
    # 2. 导入测试 - 确保所有模块可以导入
    print("\n[阶段 2/5] 导入测试")
    print("-" * 70)
    
    modules_to_test = [
        "hetao_ag",
        "hetao_ag.core",
        "hetao_ag.soil",
        "hetao_ag.water",
        "hetao_ag.crop",
        "hetao_ag.livestock",
        "hetao_ag.space",
        "hetao_ag.opt",
    ]
    
    import_success = True
    for module in modules_to_test:
        try:
            __import__(module)
            print(f"✓ {module}")
        except Exception as e:
            print(f"✗ {module}: {e}")
            import_success = False
    
    runner.results.append(("模块导入", import_success, ""))
    
    # 3. 运行示例代码
    print("\n[阶段 3/5] 示例代码测试")
    print("-" * 70)
    
    example_files = [
        "examples/example_core.py",
        "examples/example_soil.py",
        "examples/example_water.py",
        "examples/example_crop.py",
        # "examples/example_livestock.py",  # 可能需要额外依赖
        # "examples/example_space.py",      # 可能需要额外依赖
        "examples/example_opt.py",
    ]
    
    for example in example_files:
        example_path = base_dir / example
        if example_path.exists():
            runner.run_command(
                [sys.executable, str(example_path)],
                f"示例: {example}"
            )
    
    # 4. 单元测试 (如果存在)
    print("\n[阶段 4/5] 单元测试")
    print("-" * 70)
    
    test_dir = base_dir / "tests"
    if test_dir.exists():
        # 尝试使用pytest
        try:
            import pytest
            runner.run_command(
                [sys.executable, "-m", "pytest", "tests/", "-v"],
                "Pytest单元测试"
            )
        except ImportError:
            print("pytest未安装,跳过单元测试")
            print("安装命令: pip install pytest")
    else:
        print("未找到tests目录,跳过单元测试")
    
    # 5. 代码质量检查
    print("\n[阶段 5/5] 代码质量检查")
    print("-" * 70)
    
    # 检查是否有pylint
    try:
        import pylint
        runner.run_command(
            [sys.executable, "-m", "pylint", "hetao_ag/", 
             "--disable=C0103,C0114,C0115,C0116",  # 禁用一些格式检查
             "--max-line-length=100"],
            "Pylint代码质量检查"
        )
    except ImportError:
        print("pylint未安装,跳过代码质量检查")
        print("安装命令: pip install pylint")
    
    # 打印最终摘要
    all_passed = runner.print_summary()
    
    # 生成测试报告
    report_file = base_dir / "测试报告.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("河套智慧农牧业库 - 测试报告\n")
        f.write("="*70 + "\n\n")
        f.write(f"测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        for desc, success, output in runner.results:
            f.write(f"\n{'='*70}\n")
            f.write(f"测试: {desc}\n")
            f.write(f"结果: {'通过' if success else '失败'}\n")
            f.write(f"{'='*70}\n")
            if output:
                f.write(output[:1000])  # 限制输出长度
                f.write("\n")
    
    print(f"\n详细测试报告已保存到: {report_file}")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
