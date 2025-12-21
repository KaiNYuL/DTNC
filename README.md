# 数据分析平台

## 项目简介
这是一个基于Flask的数据分析平台，支持多种算法进行数据分析和因果关系发现。

## 功能特性
- 数据上传和管理
- 多种算法选择：
  - 相关系数
  - 偏相关系数
  - 贪婪等价搜索算法(GES)
  - 最大最小爬山算法(MMHC)
  - 改进的增量关联Markov边界算法(INTER-IAMB)
  - NOTEARS算法
- 网络可视化
- 邻接矩阵保存

## 技术栈
- 后端：Flask
- 前端：HTML, CSS, JavaScript
- 数据处理：numpy, pandas, scikit-learn, scipy
- 网络可视化：networkx, matplotlib

## 安装和运行

### 安装依赖
```bash
pip install -r requirements.txt
```

### 运行应用
```bash
python python_algorithms/app.py
```

### 访问应用
打开浏览器访问：http://localhost:3000

## 算法说明

### NOTEARS算法
基于L1正则化的因果发现算法，通过光滑的矩阵函数精确表征DAG的无环性，避免传统算法的组合搜索问题。

## 项目结构
```
python_algorithms/
├── app.py                # Flask应用主文件
├── algorithms.py         # 算法实现
├── db.py                 # 数据库操作
├── utils.py              # 工具函数
└── testdata/             # 测试数据
public/
├── index.html           # 前端页面
├── css/
│   └── styles.css       # 样式文件
└── js/
    └── script.js        # 前端脚本
```