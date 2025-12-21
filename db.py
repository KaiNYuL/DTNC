import sqlite3
import os
import json
from pathlib import Path

class Database:
    def __init__(self):
        # 数据库文件路径
        self.db_path = os.path.join(os.path.dirname(__file__), 'data', 'database.db')
        
        # 确保数据目录存在
        data_dir = os.path.dirname(self.db_path)
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        
        # 创建数据库连接
        self.connection = sqlite3.connect(self.db_path, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row  # 使查询结果可以通过列名访问
        
        # 创建必要的表结构
        self.create_tables()
        
    def create_tables(self):
        cursor = self.connection.cursor()
        
        # 数据集表
        datasets_table = '''
            CREATE TABLE IF NOT EXISTS datasets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                filename TEXT NOT NULL,
                upload_date TEXT NOT NULL,
                file_path TEXT NOT NULL,
                size TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );
        '''

        # 数据特征表
        features_table = '''
            CREATE TABLE IF NOT EXISTS features (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                dataset_id INTEGER NOT NULL,
                name TEXT NOT NULL,
                FOREIGN KEY (dataset_id) REFERENCES datasets(id) ON DELETE CASCADE
            );
        '''

        # 数据分类表
        classifications_table = '''
            CREATE TABLE IF NOT EXISTS classifications (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                dataset_id INTEGER NOT NULL,
                name TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (dataset_id) REFERENCES datasets(id) ON DELETE CASCADE
            );
        '''

        # 分类特征关联表
        classification_features_table = '''
            CREATE TABLE IF NOT EXISTS classification_features (
                classification_id INTEGER NOT NULL,
                feature_id INTEGER NOT NULL,
                PRIMARY KEY (classification_id, feature_id),
                FOREIGN KEY (classification_id) REFERENCES classifications(id) ON DELETE CASCADE,
                FOREIGN KEY (feature_id) REFERENCES features(id) ON DELETE CASCADE
            );
        '''

        # 分析结果表
        analysis_results_table = '''
            CREATE TABLE IF NOT EXISTS analysis_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                dataset_id INTEGER NOT NULL,
                algorithm TEXT NOT NULL,
                result_json TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (dataset_id) REFERENCES datasets(id) ON DELETE CASCADE
            );
        '''

        # 执行创建表的SQL语句
        cursor.execute(datasets_table)
        cursor.execute(features_table)
        cursor.execute(classifications_table)
        cursor.execute(classification_features_table)
        cursor.execute(analysis_results_table)
        
        self.connection.commit()
        cursor.close()
    
    # 数据集操作方法
    
    def add_dataset(self, name, filename, upload_date, file_path, size):
        cursor = self.connection.cursor()
        sql = '''INSERT INTO datasets (name, filename, upload_date, file_path, size) 
                 VALUES (?, ?, ?, ?, ?)''' 
        cursor.execute(sql, (name, filename, upload_date, file_path, size))
        self.connection.commit()
        dataset_id = cursor.lastrowid
        cursor.close()
        return dataset_id
    
    def get_all_datasets(self):
        cursor = self.connection.cursor()
        sql = '''SELECT * FROM datasets ORDER BY created_at DESC''' 
        cursor.execute(sql)
        rows = cursor.fetchall()
        cursor.close()
        return [dict(row) for row in rows]
    
    def get_dataset_by_id(self, id):
        cursor = self.connection.cursor()
        sql = '''SELECT * FROM datasets WHERE id = ?''' 
        cursor.execute(sql, (id,))
        row = cursor.fetchone()
        cursor.close()
        return dict(row) if row else None
    
    def delete_dataset(self, id):
        cursor = self.connection.cursor()
        sql = '''DELETE FROM datasets WHERE id = ?''' 
        cursor.execute(sql, (id,))
        changes = cursor.rowcount
        self.connection.commit()
        cursor.close()
        return changes
    
    # 特征操作方法
    
    def add_feature(self, dataset_id, name):
        cursor = self.connection.cursor()
        sql = '''INSERT INTO features (dataset_id, name) VALUES (?, ?)''' 
        cursor.execute(sql, (dataset_id, name))
        self.connection.commit()
        feature_id = cursor.lastrowid
        cursor.close()
        return feature_id
    
    def add_features(self, dataset_id, feature_names):
        cursor = self.connection.cursor()
        sql = '''INSERT INTO features (dataset_id, name) VALUES (?, ?)''' 
        
        cursor.execute('BEGIN TRANSACTION')
        try:
            for name in feature_names:
                cursor.execute(sql, (dataset_id, name))
            cursor.execute('COMMIT')
            return True
        except Exception as e:
            cursor.execute('ROLLBACK')
            return False
        finally:
            cursor.close()
    
    def get_features_by_dataset_id(self, dataset_id):
        cursor = self.connection.cursor()
        sql = '''SELECT * FROM features WHERE dataset_id = ?''' 
        cursor.execute(sql, (dataset_id,))
        rows = cursor.fetchall()
        cursor.close()
        return [dict(row) for row in rows]
    
    # 分类操作方法
    
    def add_classification(self, dataset_id, name):
        cursor = self.connection.cursor()
        sql = '''INSERT INTO classifications (dataset_id, name) VALUES (?, ?)''' 
        cursor.execute(sql, (dataset_id, name))
        self.connection.commit()
        classification_id = cursor.lastrowid
        cursor.close()
        return classification_id
    
    def get_classifications_by_dataset_id(self, dataset_id):
        cursor = self.connection.cursor()
        sql = '''SELECT * FROM classifications WHERE dataset_id = ?''' 
        cursor.execute(sql, (dataset_id,))
        rows = cursor.fetchall()
        cursor.close()
        return [dict(row) for row in rows]
    
    def add_classification_feature(self, classification_id, feature_id):
        cursor = self.connection.cursor()
        sql = '''INSERT OR IGNORE INTO classification_features (classification_id, feature_id) 
                 VALUES (?, ?)''' 
        cursor.execute(sql, (classification_id, feature_id))
        self.connection.commit()
        changes = cursor.rowcount
        cursor.close()
        return changes
    
    def add_classification_features(self, classification_id, feature_ids):
        cursor = self.connection.cursor()
        sql = '''INSERT OR IGNORE INTO classification_features (classification_id, feature_id) 
                 VALUES (?, ?)''' 
        
        cursor.execute('BEGIN TRANSACTION')
        try:
            for feature_id in feature_ids:
                cursor.execute(sql, (classification_id, feature_id))
            cursor.execute('COMMIT')
            return True
        except Exception as e:
            cursor.execute('ROLLBACK')
            return False
        finally:
            cursor.close()
    
    def get_features_by_classification_id(self, classification_id):
        cursor = self.connection.cursor()
        sql = '''
            SELECT f.* FROM features f
            JOIN classification_features cf ON f.id = cf.feature_id
            WHERE cf.classification_id = ?
        ''' 
        cursor.execute(sql, (classification_id,))
        rows = cursor.fetchall()
        cursor.close()
        return [dict(row) for row in rows]
    
    # 分析结果操作方法
    
    def save_analysis_result(self, dataset_id, algorithm, result_json):
        cursor = self.connection.cursor()
        
        # 先删除同一数据集和算法的旧结果
        delete_sql = '''DELETE FROM analysis_results WHERE dataset_id = ? AND algorithm = ?''' 
        cursor.execute(delete_sql, (dataset_id, algorithm))
        
        # 然后插入新结果
        insert_sql = '''INSERT INTO analysis_results (dataset_id, algorithm, result_json) 
                        VALUES (?, ?, ?)''' 
        cursor.execute(insert_sql, (dataset_id, algorithm, json.dumps(result_json)))
        
        self.connection.commit()
        result_id = cursor.lastrowid
        cursor.close()
        return result_id
    
    def get_analysis_results(self, dataset_id):
        cursor = self.connection.cursor()
        sql = '''SELECT * FROM analysis_results WHERE dataset_id = ? ORDER BY created_at DESC''' 
        cursor.execute(sql, (dataset_id,))
        rows = cursor.fetchall()
        cursor.close()
        
        # 解析JSON结果
        results = []
        for row in rows:
            row_dict = dict(row)
            row_dict['result_json'] = json.loads(row_dict['result_json'])
            results.append(row_dict)
        
        return results
    
    def get_latest_analysis_result(self, dataset_id, algorithm):
        cursor = self.connection.cursor()
        sql = '''
            SELECT * FROM analysis_results 
            WHERE dataset_id = ? AND algorithm = ? 
            ORDER BY created_at DESC 
            LIMIT 1
        ''' 
        cursor.execute(sql, (dataset_id, algorithm))
        row = cursor.fetchone()
        cursor.close()
        
        if row:
            row_dict = dict(row)
            row_dict['result_json'] = json.loads(row_dict['result_json'])
            return row_dict
        return None

# 创建数据库实例
db = Database()
