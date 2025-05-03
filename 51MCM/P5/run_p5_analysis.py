#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
P5问题求解脚本 - 关键采样点分析

此脚本提供了一个简单的界面来运行P5问题的求解，
用户可以选择分析P2或P3问题，以及选择采样点识别方法。
"""

import os
import sys
import subprocess
import argparse

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    print("\n===== P5问题求解 - 关键采样点分析 =====\n")

    # 解析命令行参数
    parser = argparse.ArgumentParser(description='P5问题求解工具')
    parser.add_argument('--problem', type=str, default=None, choices=['P2', 'P3', 'both'],
                        help='选择要分析的问题（P2、P3或both）')
    parser.add_argument('--method', type=str, default='combined',
                        choices=['derivative', 'peaks', 'breakpoints', 'combined'],
                        help='采样点识别方法')

    args = parser.parse_args()

    # 如果没有指定问题，让用户选择
    if args.problem is None:
        print("请选择要分析的问题：")
        print("1. P2问题（表2数据）")
        print("2. P3问题（表3数据）")
        print("3. 两者都分析")

        choice = input("请输入选项（1/2/3）: ")
        if choice == '1':
            problems = ['P2']
        elif choice == '2':
            problems = ['P3']
        elif choice == '3':
            problems = ['P2', 'P3']
        else:
            print("无效选项，默认分析P2问题")
            problems = ['P2']
    else:
        if args.problem == 'both':
            problems = ['P2', 'P3']
        else:
            problems = [args.problem]

    # 如果没有指定方法，让用户选择
    method = args.method
    if method == 'combined':
        print("\n请选择采样点识别方法：")
        print("1. 基于导数变化（derivative）")
        print("2. 基于峰值检测（peaks）")
        print("3. 基于模型转折点（breakpoints）")
        print("4. 结合所有方法（combined）")

        choice = input("请输入选项（1/2/3/4），默认为4: ")
        if choice == '1':
            method = 'derivative'
        elif choice == '2':
            method = 'peaks'
        elif choice == '3':
            method = 'breakpoints'
        else:
            method = 'combined'

    # 确保P5目录存在
    os.makedirs('./P5', exist_ok=True)

    # 运行分析
    for problem in problems:
        print(f"\n===== 开始分析{problem}问题 =====\n")
        cmd = [sys.executable, f"{problem}.py", "--p5", "--method", method]
        subprocess.run(cmd, cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        print(f"\n===== {problem}问题分析完成 =====\n")

    print("\n所有分析已完成！结果保存在P5目录下。")
    print("请查看以下文件获取详细结果：")
    for problem in problems:
        print(f"- ./P5/{problem}/关键采样点分析报告.md")
        print(f"- ./P5/{problem}/关键采样点分析.png")

if __name__ == "__main__":
    main()