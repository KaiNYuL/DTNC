import os
import sys

# 获取当前文件所在目录的父目录（即项目根目录）
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# 导入 Flask 应用实例
# 注意：确保 python_algorithms 包里没有语法错误，且依赖已安装
from python_algorithms.app import app

# Vercel Serverless Function 入口
# Vercel 会自动寻找名为 `app` 或 `handler` 的变量
