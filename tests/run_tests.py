#!/usr/bin/env python3
"""
测试运行脚本 - 提供不同级别的测试运行选项

使用方法：
    python tests/run_tests.py --all          # 运行所有测试
    python tests/run_tests.py --unit         # 只运行单元测试
    python tests/run_tests.py --integration  # 只运行集成测试
    python tests/run_tests.py --fast         # 只运行快速测试
    python tests/run_tests.py --cov          # 运行测试并生成覆盖率报告
"""

import sys
import subprocess
import argparse
from pathlib import Path


def run_pytest_command(args_list):
    """运行pytest命令"""
    cmd = ['python', '-m', 'pytest'] + args_list
    print(f"运行命令: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"测试失败！错误代码: {e.returncode}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(description='运行ThermoElasticSim测试套件')
    
    # 测试类型选项
    test_group = parser.add_mutually_exclusive_group()
    test_group.add_argument('--all', action='store_true', 
                          help='运行所有测试（默认）')
    test_group.add_argument('--unit', action='store_true',
                          help='只运行单元测试')
    test_group.add_argument('--integration', action='store_true',
                          help='只运行集成测试')
    test_group.add_argument('--fast', action='store_true',
                          help='只运行快速测试（排除slow标记）')
    
    # 输出选项
    parser.add_argument('--cov', action='store_true',
                       help='生成测试覆盖率报告')
    parser.add_argument('--html', action='store_true',
                       help='生成HTML测试报告')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='详细输出')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='简洁输出')
    
    # 特定测试选项
    parser.add_argument('--file', type=str,
                       help='运行特定测试文件')
    parser.add_argument('--pattern', type=str,
                       help='运行匹配模式的测试')
    
    args = parser.parse_args()
    
    # 构建pytest参数列表
    pytest_args = []
    
    # 添加基本路径
    pytest_args.append('tests/')
    
    # 测试选择
    if args.unit:
        pytest_args.extend(['-m', 'unit'])
    elif args.integration:
        pytest_args.extend(['-m', 'integration'])
    elif args.fast:
        pytest_args.extend(['-m', 'not slow'])
    
    # 特定文件或模式
    if args.file:
        pytest_args = [f'tests/{args.file}']
    elif args.pattern:
        pytest_args.extend(['-k', args.pattern])
    
    # 输出控制
    if args.verbose:
        pytest_args.append('-v')
    elif args.quiet:
        pytest_args.append('-q')
    
    # 覆盖率报告
    if args.cov:
        pytest_args.extend([
            '--cov=thermoelasticsim',
            '--cov-report=term-missing',
            '--cov-report=html:htmlcov'
        ])
    
    # HTML报告
    if args.html:
        pytest_args.extend([
            '--html=tests/reports/test_report.html',
            '--self-contained-html'
        ])
    
    # 确保报告目录存在
    if args.html:
        Path('tests/reports').mkdir(exist_ok=True)
    
    # 运行测试
    success = run_pytest_command(pytest_args)
    
    if success:
        print("\n✅ 所有测试通过！")
        if args.cov:
            print("📊 覆盖率报告已生成在 htmlcov/ 目录")
        if args.html:
            print("📄 HTML测试报告已生成在 tests/reports/test_report.html")
    else:
        print("\n❌ 测试失败！")
        sys.exit(1)


if __name__ == '__main__':
    main()