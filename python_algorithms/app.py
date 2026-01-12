from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import sys
import pandas as pd

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from db import Database
from algorithms import Algorithms
from utils import FileUtils, DataUtils

# 创建应用实例
app = Flask(__name__, 
            static_folder=os.path.join(os.path.dirname(__file__), '../public'),
            static_url_path='')
CORS(app)  # 启用CORS

# 配置文件上传
if os.environ.get('VERCEL'):
    app.config['UPLOAD_FOLDER'] = '/tmp/uploads'
else:
    app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), '../uploads')

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

# 初始化工具类
db = Database()
algos = Algorithms()
file_utils = FileUtils()
data_utils = DataUtils()

# 辅助函数：确保目录存在
def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# 辅助函数：保存分析结果

def save_analysis_results(adjacency_matrix, graph_base64, feature_names, dataset_id, algorithm, save_path=None):
    try:
        # 确定保存目录
        if save_path:
            testdata_dir = os.path.abspath(save_path)
            # 如果提供了save_path但不在/tmp下且我们在vercel上，可能还是会有问题，但用户通常不会在云端提供自定义路径。
            # 为安全起见，如果在Vercel，忽略自定义路径强制使用/tmp
            if os.environ.get('VERCEL'):
                testdata_dir = os.path.join('/tmp', 'testdata')
        elif os.environ.get('VERCEL'):
            testdata_dir = os.path.join('/tmp', 'testdata')
        else:
            testdata_dir = os.path.join(os.path.dirname(__file__), 'testdata')
            
        ensure_directory_exists(testdata_dir)
        
        # 获取数据集名称
        dataset = db.get_dataset(dataset_id)
        dataset_name = dataset['name'].replace(' ', '_') if dataset else f'dataset_{dataset_id}'
        
        # 保存邻接矩阵为CSV
        import csv
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        adj_path = os.path.join(testdata_dir, f'{dataset_name}_{algorithm}_adj_matrix_{timestamp}.csv')
        
        with open(adj_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow([''] + feature_names)  # 表头
            for i, row in enumerate(adjacency_matrix):
                writer.writerow([feature_names[i]] + [round(val, 6) for val in row])
        
        # 保存PNG网络图
        if graph_base64:
            import base64
            if ',' in graph_base64:
                graph_base64 = graph_base64.split(',')[1]
            image_bytes = base64.b64decode(graph_base64)
            png_path = os.path.join(testdata_dir, f'{dataset_name}_{algorithm}_network_{timestamp}.png')
            with open(png_path, 'wb') as f:
                f.write(image_bytes)
        
        return True, adj_path, png_path if graph_base64 else None
    except Exception as e:
        print(f"保存分析结果失败: {str(e)}")
        return False, None, None

# 主页路由
@app.route('/')
def index():
    return app.send_static_file('index.html')

# 文件上传路由
@app.route('/api/datasets/upload', methods=['POST'])
def upload_file():
    try:
        if 'dataFile' not in request.files:
            return jsonify({'error': '没有文件上传'}), 400
        
        file = request.files['dataFile']
        dataset_name = request.form.get('datasetName', file.filename)
        
        if file.filename == '':
            return jsonify({'error': '没有选择文件'}), 400
        
        # 检查文件类型（前端也有验证，后端再做一层保障）
        allowed_extensions = {'.csv', '.xls', '.xlsx'}
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in allowed_extensions:
            return jsonify({'error': '仅支持CSV、XLS和XLSX格式的文件'}), 400
        
        # 保存文件
        file_path, filename = file_utils.save_uploaded_file(file)
        if not file_path:
            return jsonify({'error': '不支持的文件格式'}), 400
        
        try:
            # 解析文件
            parsed_data = file_utils.parse_file(file_path, filename)
            
            # 保存数据集信息到数据库
            dataset_id = db.save_dataset(dataset_name, file_path)
            
            return jsonify({
                'success': True,
                'data': {
                    'id': dataset_id,
                    'name': dataset_name,
                    'filename': filename,
                    'path': file_path,
                    'upload_time': db.get_dataset(dataset_id)['upload_time'],
                    'size': os.path.getsize(file_path)
                },
                'message': '数据集上传成功'
            }), 200
        except Exception as parse_error:
            # 解析失败，删除已保存的文件
            if os.path.exists(file_path):
                os.remove(file_path)
            return jsonify({'error': f'文件解析失败：{str(parse_error)}', 'success': False, 'message': f'文件解析失败：{str(parse_error)}'}), 400
        
    except Exception as e:
        return jsonify({'error': str(e), 'success': False, 'message': str(e)}), 500

# 获取所有数据集
@app.route('/api/datasets', methods=['GET'])
def get_datasets():
    try:
        datasets = db.get_all_datasets()
        return jsonify({
            'success': True,
            'data': datasets
        }), 200
    except Exception as e:
        return jsonify({'error': str(e), 'success': False, 'message': str(e)}), 500

# 获取数据集详情
@app.route('/api/datasets/<int:dataset_id>', methods=['GET'])
def get_dataset(dataset_id):
    try:
        dataset = db.get_dataset(dataset_id)
        if not dataset:
            return jsonify({'error': '数据集不存在', 'success': False, 'message': '数据集不存在'}), 404
        
        return jsonify({
            'success': True,
            'data': dataset
        }), 200
    except Exception as e:
        return jsonify({'error': str(e), 'success': False, 'message': str(e)}), 500

# 删除数据集
@app.route('/api/datasets/<int:dataset_id>', methods=['DELETE'])
def delete_dataset(dataset_id):
    try:
        # 获取数据集信息
        dataset = db.get_dataset(dataset_id)
        if not dataset:
            return jsonify({'error': '数据集不存在', 'success': False, 'message': '数据集不存在'}), 404
        
        # 保存文件路径用于后续删除
        file_path = dataset['path']
        
        # 先从数据库中删除记录
        db.delete_dataset(dataset_id)
        
        # 再删除文件
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception as file_error:
                # 文件删除失败不影响API响应，但记录错误
                print(f"删除文件失败 {file_path}: {str(file_error)}")
        
        return jsonify({
            'success': True,
            'message': '数据集删除成功'
        }), 200
    except Exception as e:
        return jsonify({'error': str(e), 'success': False, 'message': str(e)}), 500


# 获取数据集特征
@app.route('/api/datasets/<int:dataset_id>/features', methods=['GET'])
def get_dataset_features(dataset_id):
    try:
        dataset = db.get_dataset(dataset_id)
        if not dataset:
            return jsonify({'error': '数据集不存在', 'success': False, 'message': '数据集不存在'}), 404
        
        # 解析文件内容
        parsed_data = file_utils.parse_file(dataset['path'], dataset['name'])
        
        return jsonify({
            'success': True,
            'data': parsed_data['feature_names']
        }), 200
    except Exception as e:
        import traceback
        error_msg = f"Error in get_dataset_features: {str(e)}\nTraceback: {traceback.format_exc()}"
        print(error_msg)
        return jsonify({'error': str(e), 'success': False, 'message': str(e)}), 500

# 运行算法分析
@app.route('/api/analyze', methods=['POST'])
def analyze():
    try:
        data = request.json
        dataset_id = data.get('datasetId')
        algorithm = data.get('algorithm')
        save_path = data.get('savePath')  # 获取保存路径参数
        
        if not dataset_id or not algorithm:
            return jsonify({'error': '缺少必要参数', 'success': False, 'message': '缺少必要参数'}), 400
        
        # 获取数据集信息
        dataset = db.get_dataset(dataset_id)
        if not dataset:
            return jsonify({'error': '数据集不存在', 'success': False, 'message': '数据集不存在'}), 404
        
        # 解析文件内容
        parsed_data = file_utils.parse_file(dataset['path'], dataset['name'])
        data_matrix = parsed_data['data']
        feature_names = parsed_data['feature_names']
        
        # 选择算法
        result = None
        algorithm_mapping = {
            'correlation': algos.correlation_algorithm,
            'partial_correlation': algos.partial_correlation_algorithm,
            'ges': algos.ges_algorithm,
            'mmhc': algos.mmhc_algorithm,
            'interiamb': algos.inter_iamb_algorithm,
            'notears': algos.notears_algorithm
        }
        
        if algorithm in algorithm_mapping:
            result = algorithm_mapping[algorithm](data_matrix, feature_names)
        else:
            return jsonify({'error': '不支持的算法', 'success': False, 'message': '不支持的算法'}), 400
        
        # 保存分析结果到数据库
        # 保存分析结果到数据库（忽略返回值）
        db.save_analysis_result(dataset_id, algorithm, result)
        
        # 构建返回结果 - 支持所有算法的矩阵类型
        # 检查所有可能的矩阵类型
        correlation_matrix = result.get('correlation_matrix') or \
                           result.get('partial_correlation_matrix') or \
                           result.get('precision_matrix') or \
                           result.get('adjacency_matrix')
        graph_base64 = result.get('graph_base64')
        
        return_result = {
            'success': True,
            'data': {
                'network': {
                    'nodes': result['nodes'],
                    'links': result['links']
                },
                'featureNames': feature_names,
                'correlationMatrix': correlation_matrix,
                'graph_base64': graph_base64
            },
            'message': '数据分析完成'
        }
        
        # 自动保存分析结果
        if correlation_matrix:
            save_analysis_results(correlation_matrix, graph_base64, feature_names, dataset_id, algorithm, save_path)
        
        return jsonify(return_result), 200
        
    except Exception as e:
        import traceback
        error_msg = f"Error: {str(e)}\nTraceback: {traceback.format_exc()}"
        print(error_msg)
        return jsonify({'error': str(e), 'success': False, 'message': str(e)}), 500

# 获取分析结果
@app.route('/api/result/<int:dataset_id>/<string:algorithm>', methods=['GET'])
def get_result(dataset_id, algorithm):
    try:
        result = db.get_analysis_result(dataset_id, algorithm)
        if not result:
            return jsonify({'error': '分析结果不存在', 'success': False, 'message': '分析结果不存在'}), 404
        
        return jsonify({
            'success': True,
            'data': result['result_json']
        }), 200
    except Exception as e:
        return jsonify({'error': str(e), 'success': False, 'message': str(e)}), 500

# 获取特征统计信息
@app.route('/api/datasets/<int:dataset_id>/statistics', methods=['GET'])
def get_statistics(dataset_id):
    try:
        # 获取数据集信息
        dataset = db.get_dataset(dataset_id)
        if not dataset:
            return jsonify({'error': '数据集不存在', 'success': False, 'message': '数据集不存在'}), 404
        
        # 解析文件内容
        parsed_data = file_utils.parse_file(dataset['path'], dataset['name'])
        data_matrix = parsed_data['data']
        feature_names = parsed_data['feature_names']
        
        # 计算统计信息
        statistics = data_utils.get_feature_statistics(data_matrix, feature_names)
        
        return jsonify({
            'success': True,
            'data': statistics
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e), 'success': False, 'message': str(e)}), 500

# 保存图片
@app.route('/api/save/image', methods=['POST'])
def save_image():
    try:
        data = request.json
        image_data = data.get('image_data')
        filename = data.get('filename', 'network-graph')
        file_type = data.get('file_type', 'png')
        
        if not image_data:
            return jsonify({'error': '缺少图片数据', 'success': False, 'message': '缺少图片数据'}), 400
        
        # 解析Base64图片数据
        import base64
        
        # 移除Base64前缀
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        # 解码Base64数据
        image_bytes = base64.b64decode(image_data)
        
        # 确保上传目录存在
        ensure_directory_exists(app.config['UPLOAD_FOLDER'])
        
        # 保存图片
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], f'{filename}.{file_type}')
        
        with open(file_path, 'wb') as f:
            f.write(image_bytes)
        
        return jsonify({
            'success': True,
            'message': '图片保存成功',
            'file_path': file_path
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e), 'success': False, 'message': str(e)}), 500

# 保存邻接矩阵
@app.route('/api/save/adjacency_matrix', methods=['POST'])
def save_adjacency_matrix():
    try:
        data = request.json
        adjacency_matrix = data.get('adjacency_matrix')
        feature_names = data.get('feature_names')
        filename = data.get('filename', 'adjacency_matrix')
        
        if not adjacency_matrix or not feature_names:
            return jsonify({'error': '缺少邻接矩阵数据或特征名称', 'success': False, 'message': '缺少邻接矩阵数据或特征名称'}), 400
        
        # 创建CSV内容
        import csv
        
        # 确保上传目录存在
        ensure_directory_exists(app.config['UPLOAD_FOLDER'])
        
        # 保存邻接矩阵
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], f'{filename}.csv')
        
        with open(file_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            
            # 写入表头
            writer.writerow([''] + feature_names)
            
            # 写入矩阵数据
            for i, row in enumerate(adjacency_matrix):
                writer.writerow([feature_names[i]] + [round(val, 6) for val in row])
        
        return jsonify({
            'success': True,
            'message': '邻接矩阵保存成功',
            'file_path': file_path
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e), 'success': False, 'message': str(e)}), 500

# 健康检查路由
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'}), 200

# 获取testdata文件夹中的文件列表
@app.route('/api/testdata/files/<int:dataset_id>', methods=['GET'])
def get_testdata_files(dataset_id):
    try:
        testdata_dir = os.path.join(os.path.dirname(__file__), 'testdata')
        if not os.path.exists(testdata_dir):
            return jsonify({'success': False, 'message': 'testdata文件夹不存在'}), 404
        
        # 获取所有与指定dataset_id相关的文件
        import glob
        files = []
        
        # 查找邻接矩阵文件
        adj_files = glob.glob(os.path.join(testdata_dir, f'adj_matrix_{dataset_id}_*.csv'))
        for file_path in adj_files:
            files.append({
                'type': 'adjacency_matrix',
                'name': os.path.basename(file_path),
                'path': file_path
            })
        
        # 查找网络图文件
        png_files = glob.glob(os.path.join(testdata_dir, f'network_{dataset_id}_*.png'))
        for file_path in png_files:
            files.append({
                'type': 'network_graph',
                'name': os.path.basename(file_path),
                'path': file_path
            })
        
        return jsonify({'success': True, 'data': files}), 200
    except Exception as e:
        return jsonify({'error': str(e), 'success': False, 'message': str(e)}), 500

# 下载testdata文件
@app.route('/api/testdata/download/<path:filename>', methods=['GET'])
def download_testdata_file(filename):
    try:
        testdata_dir = os.path.join(os.path.dirname(__file__), 'testdata')
        file_path = os.path.join(testdata_dir, filename)
        
        if not os.path.exists(file_path):
            return jsonify({'success': False, 'message': '文件不存在'}), 404
        
        return app.send_from_directory(testdata_dir, filename, as_attachment=True)
    except Exception as e:
        return jsonify({'error': str(e), 'success': False, 'message': str(e)}), 500

if __name__ == '__main__':
    # 启动Flask应用
    app.run(debug=True, host='0.0.0.0', port=3000)
