#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
51MCM - 交通流量分析系统
统一运行脚本 - 用于运行所有问题的解决方案
"""

import os
import time
import argparse
import importlib.util
import sys

def import_module_from_file(module_name, file_path):
    """从文件导入模块

    Args:
        module_name: 模块名称
        file_path: 文件路径

    Returns:
        导入的模块
    """
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

def run_problem(problem_id, args):
    """运行指定问题的解决方案

    Args:
        problem_id: 问题ID (1-5)
        args: 命令行参数
    """
    file_path = f"./P{problem_id}.py"

    if not os.path.exists(file_path):
        print(f"错误: 未找到问题{problem_id}的解决方案文件 ({file_path})")
        return False

    print(f"\n{'='*50}")
    print(f"开始求解问题 {problem_id}")
    print(f"{'='*50}")

    try:
        # 导入问题模块
        module = import_module_from_file(f"P{problem_id}", file_path)

        # 为P5问题处理特殊参数
        if problem_id == 5:
            if hasattr(module, 'main'):
                if args.method:
                    module.main(method=args.method)
                else:
                    module.main()
            else:
                print(f"错误: P{problem_id}.py 文件中未找到 main() 函数")
                return False
        else:
            # 运行其他问题的main函数
            if hasattr(module, 'main'):
                module.main()
            else:
                print(f"错误: P{problem_id}.py 文件中未找到 main() 函数")
                return False

        print(f"\n问题 {problem_id} 求解完成!")
        return True

    except Exception as e:
        print(f"运行问题 {problem_id} 时出错: {str(e)}")
        return False

def main():
    """主函数"""
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='51MCM - 交通流量分析系统')
    parser.add_argument('--problems', nargs='+', type=int, default=list(range(1, 6)),
                      help='要运行的问题ID (1-5)，默认运行所有问题')
    parser.add_argument('--skip', nargs='+', type=int, default=[],
                      help='要跳过的问题ID')
    parser.add_argument('--method', type=str, default='combined',
                      choices=['derivative', 'peaks', 'breakpoints', 'combined'],
                      help='P5问题采样点识别方法')

    args = parser.parse_args()

    # 过滤要运行的问题
    problems_to_run = [p for p in args.problems if p not in args.skip]
    problems_to_run = sorted(list(set(problems_to_run)))  # 去重并排序

    if not problems_to_run:
        print("错误: 没有要运行的问题")
        return

    # 创建输出目录
    for p in problems_to_run:
        os.makedirs(f"./P{p}", exist_ok=True)

    # 记录开始时间
    start_time = time.time()

    # 运行所有问题
    results = {}
    for problem_id in problems_to_run:
        problem_start = time.time()
        success = run_problem(problem_id, args)
        problem_end = time.time()
        results[problem_id] = {
            'success': success,
            'time': problem_end - problem_start
        }

    # 计算总耗时
    total_time = time.time() - start_time

    # 打印运行结果摘要
    print("\n")
    print("="*50)
    print("运行结果摘要")
    print("="*50)
    print(f"总耗时: {total_time:.2f} 秒")
    print("\n问题运行状态:")

    for problem_id, result in results.items():
        status = "成功" if result['success'] else "失败"
        print(f"问题 {problem_id}: {status} (耗时: {result['time']:.2f} 秒)")

    print("\n所有问题运行完成!")

if __name__ == "__main__":
    main()