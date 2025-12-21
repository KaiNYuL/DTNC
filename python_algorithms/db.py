import sqlite3
import json
import os

class Database:
    def __init__(self):
        self.db_path = os.path.join(os.path.dirname(__file__), '../data/database.db')
        self.connection = sqlite3.connect(self.db_path, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self.create_tables()
    
    def create_tables(self):
        cursor = self.connection.cursor()
        
        # 创建数据集表
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS datasets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            path TEXT NOT NULL,
            upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # 检查并添加缺少的列（如果表已存在但结构不同）
        try:
            cursor.execute("ALTER TABLE datasets ADD COLUMN upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP")
        except sqlite3.OperationalError:
            # 列已存在，忽略错误
            pass
        
        # 创建特征表
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS features (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            dataset_id INTEGER NOT NULL,
            name TEXT NOT NULL,
            type TEXT NOT NULL,
            FOREIGN KEY (dataset_id) REFERENCES datasets (id) ON DELETE CASCADE
        )
        ''')
        
        # 创建分类表
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS classifications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            dataset_id INTEGER NOT NULL,
            feature_name TEXT NOT NULL,
            FOREIGN KEY (dataset_id) REFERENCES datasets (id) ON DELETE CASCADE
        )
        ''')
        
        # 创建分析结果表
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS analysis_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            dataset_id INTEGER NOT NULL,
            algorithm TEXT NOT NULL,
            result_json TEXT NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (dataset_id) REFERENCES datasets (id) ON DELETE CASCADE
        )
        ''')
        
        self.connection.commit()
    
    def save_analysis_result(self, dataset_id, algorithm, result_json):
        cursor = self.connection.cursor()
        
        # 删除旧的分析结果
        cursor.execute("DELETE FROM analysis_results WHERE dataset_id = ? AND algorithm = ?", 
                      (dataset_id, algorithm))
        
        # 插入新的分析结果
        cursor.execute("INSERT INTO analysis_results (dataset_id, algorithm, result_json) VALUES (?, ?, ?)", 
                      (dataset_id, algorithm, json.dumps(result_json)))
        
        self.connection.commit()
        return cursor.lastrowid
    
    def get_analysis_result(self, dataset_id, algorithm):
        cursor = self.connection.cursor()
        cursor.execute("SELECT * FROM analysis_results WHERE dataset_id = ? AND algorithm = ?", 
                      (dataset_id, algorithm))
        result = cursor.fetchone()
        if result:
            return {
                'id': result['id'],
                'dataset_id': result['dataset_id'],
                'algorithm': result['algorithm'],
                'result_json': json.loads(result['result_json']),
                'timestamp': result['timestamp']
            }
        return None
    
    def save_dataset(self, name, path):
        cursor = self.connection.cursor()
        cursor.execute("INSERT INTO datasets (name, path) VALUES (?, ?)", (name, path))
        self.connection.commit()
        return cursor.lastrowid
    
    def get_dataset(self, dataset_id):
        cursor = self.connection.cursor()
        cursor.execute("SELECT * FROM datasets WHERE id = ?", (dataset_id,))
        result = cursor.fetchone()
        if result:
            return {
                'id': result['id'],
                'name': result['name'],
                'path': result['path'],
                'upload_time': result['upload_time']
            }
        return None
    
    def get_all_datasets(self):
        cursor = self.connection.cursor()
        cursor.execute("SELECT * FROM datasets ORDER BY upload_time DESC")
        results = cursor.fetchall()
        datasets = []
        for row in results:
            # 计算文件大小
            try:
                size = os.path.getsize(row['path'])
            except Exception:
                size = 0
            
            datasets.append({
                'id': row['id'],
                'name': row['name'],
                'path': row['path'],
                'upload_time': row['upload_time'],
                'size': size
            })
        return datasets
    
    def delete_dataset(self, dataset_id):
        cursor = self.connection.cursor()
        cursor.execute("DELETE FROM datasets WHERE id = ?", (dataset_id,))
        self.connection.commit()
        return cursor.rowcount > 0
    
    def close(self):
        self.connection.close()