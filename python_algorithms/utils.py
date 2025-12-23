import os
import pandas as pd
import numpy as np
from werkzeug.utils import secure_filename

class FileUtils:
    def __init__(self):
        # 使用绝对路径
        self.UPLOAD_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), '../uploads'))
        self.ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}
        
        # 确保上传目录存在
        if not os.path.exists(self.UPLOAD_FOLDER):
            os.makedirs(self.UPLOAD_FOLDER)
    
    def allowed_file(self, filename):
        """检查文件是否允许上传"""
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in self.ALLOWED_EXTENSIONS
    
    def save_uploaded_file(self, file):
        """保存上传的文件"""
        if file and self.allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(self.UPLOAD_FOLDER, filename)
            file.save(file_path)
            return file_path, filename
        return None, None
    
    def parse_file(self, file_path, filename):
        """解析CSV或Excel文件"""
        try:
            # 获取文件扩展名，处理文件名不包含'.'的情况
            if '.' in filename:
                file_ext = filename.rsplit('.', 1)[1].lower()
            else:
                # 如果文件名没有扩展名，尝试根据文件内容判断或使用默认方式
                file_ext = 'csv'  # 默认假设是CSV格式
                
            if file_ext == 'csv':
                # 尝试使用不同编码读取CSV文件
                encodings = ['utf-8', 'gbk', 'utf-16', 'latin-1']
                df = None
                for encoding in encodings:
                    try:
                        df = pd.read_csv(file_path, encoding=encoding)
                        break
                    except UnicodeDecodeError:
                        continue
                    except Exception as e:
                        raise Exception(f"使用编码{encoding}读取CSV文件失败: {str(e)}")
                
                if df is None:
                    raise Exception("所有尝试的编码格式都无法解析该CSV文件")
            else:
                # 读取Excel文件
                df = pd.read_excel(file_path)
            
            # 预处理数据
            df = self._preprocess_data(df)
            
            # 获取特征名
            feature_names = df.columns.tolist()
            
            # 转换为numpy数组
            data = df.values
            
            return {
                'data': data,
                'feature_names': feature_names,
                'num_samples': data.shape[0],
                'num_features': data.shape[1]
            }
        except Exception as e:
            raise Exception(f"文件解析错误: {str(e)}")
    
    def _preprocess_data(self, df):
        """预处理数据"""
        # 处理缺失值
        df = df.dropna()
        
        # 确保所有数据都是数值型
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 再次删除包含缺失值的行
        df = df.dropna()
        
        # 重置索引
        df = df.reset_index(drop=True)
        
        return df
    
    def normalize_data(self, data):
        """标准化数据"""
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        # 避免除以0
        std = np.where(std == 0, 1, std)
        return (data - mean) / std
    
    def get_file_info(self, file_path):
        """获取文件信息"""
        if not os.path.exists(file_path):
            return None
        
        file_size = os.path.getsize(file_path)
        file_name = os.path.basename(file_path)
        
        return {
            'file_name': file_name,
            'file_size': file_size,
            'file_path': file_path
        }
    
    def delete_file(self, file_path):
        """删除文件"""
        if os.path.exists(file_path):
            os.remove(file_path)
            return True
        return False

class DataUtils:
    def __init__(self):
        pass
    
    def split_features(self, data, feature_names, target_index=-1):
        """将特征和目标变量分离"""
        features = np.delete(data, target_index, axis=1)
        feature_names = feature_names.copy()
        target_name = feature_names.pop(target_index)
        target = data[:, target_index]
        
        return features, feature_names, target, target_name
    
    def get_feature_statistics(self, data, feature_names):
        """获取特征统计信息"""
        statistics = []
        for i, name in enumerate(feature_names):
            feature_data = data[:, i]
            stats = {
                'feature_name': name,
                'mean': float(np.mean(feature_data)),
                'std': float(np.std(feature_data)),
                'min': float(np.min(feature_data)),
                'max': float(np.max(feature_data)),
                'median': float(np.median(feature_data))
            }
            statistics.append(stats)
        return statistics
    
    def filter_features_by_correlation(self, data, feature_names, threshold=0.9):
        """根据相关系数过滤特征"""
        corr_matrix = np.corrcoef(data, rowvar=False)
        n = len(feature_names)
        
        # 找出高度相关的特征对
        to_remove = set()
        for i in range(n):
            for j in range(i + 1, n):
                if abs(corr_matrix[i, j]) > threshold:
                    to_remove.add(j)
        
        # 保留不高度相关的特征
        features_to_keep = [i for i in range(n) if i not in to_remove]
        filtered_data = data[:, features_to_keep]
        filtered_feature_names = [feature_names[i] for i in features_to_keep]
        
        return filtered_data, filtered_feature_names