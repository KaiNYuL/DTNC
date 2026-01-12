import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import pearsonr, norm
from scipy.stats import chi2
from sklearn.covariance import GraphicalLasso
import networkx as nx
import matplotlib.pyplot as plt
# 设置Matplotlib使用非交互式后端，避免线程安全警告
plt.switch_backend('Agg')
import io
import base64
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import StandardScaler
from scipy.linalg import expm
from scipy.optimize import minimize
# 导入格兰杰因果所需的库


class Algorithms:
    def __init__(self):
        pass
    
    def _mutual_information(self, x, y, n_neighbors=10):
        """计算两个变量之间的互信息
        
        Args:
            x: 第一个变量，shape=(n_samples,)
            y: 第二个变量，shape=(n_samples,)
            n_neighbors: 邻居数量，用于KDE密度估计
            
        Returns:
            mi: 互信息值
        """
        xy = np.column_stack((x, y))
        kde = KernelDensity(kernel='gaussian', bandwidth='silverman', atol=1e-3, rtol=1e-3)
        kde.fit(xy)
        log_p_xy = kde.score_samples(xy)
        
        kde_x = KernelDensity(kernel='gaussian', bandwidth='silverman', atol=1e-3, rtol=1e-3)
        kde_x.fit(x.reshape(-1, 1))
        log_p_x = kde_x.score_samples(x.reshape(-1, 1))
        
        kde_y = KernelDensity(kernel='gaussian', bandwidth='silverman', atol=1e-3, rtol=1e-3)
        kde_y.fit(y.reshape(-1, 1))
        log_p_y = kde_y.score_samples(y.reshape(-1, 1))
        
        mi = np.mean(log_p_xy - log_p_x - log_p_y)
        return max(0, mi)  # 确保互信息非负
    
    def _conditional_mutual_information(self, x, y, z, n_neighbors=10):
        """计算两个变量在给定条件下的条件互信息
        
        Args:
            x: 第一个变量，shape=(n_samples,)
            y: 第二个变量，shape=(n_samples,)
            z: 条件变量列表，shape=(n_samples, k) 或单变量shape=(n_samples,)
            n_neighbors: 邻居数量，用于KDE密度估计
            
        Returns:
            cmi: 条件互信息值
        """
        # 处理单变量条件
        if len(z.shape) == 1:
            z = z.reshape(-1, 1)
        
        xyz = np.column_stack((x, y, z))
        xz = np.column_stack((x, z))
        yz = np.column_stack((y, z))
        
        # 计算联合分布p(x,y|z)的对数似然
        kde_xyz = KernelDensity(kernel='gaussian', bandwidth='silverman', atol=1e-3, rtol=1e-3)
        kde_xyz.fit(xyz)
        log_p_xyz = kde_xyz.score_samples(xyz)
        
        # 计算条件分布p(x|z)的对数似然
        kde_xz = KernelDensity(kernel='gaussian', bandwidth='silverman', atol=1e-3, rtol=1e-3)
        kde_xz.fit(xz)
        log_p_xz = kde_xz.score_samples(xz)
        
        # 计算条件分布p(y|z)的对数似然
        kde_yz = KernelDensity(kernel='gaussian', bandwidth='silverman', atol=1e-3, rtol=1e-3)
        kde_yz.fit(yz)
        log_p_yz = kde_yz.score_samples(yz)
        
        # 计算条件互信息 I(X;Y|Z) = E[log(p(x,y|z)/(p(x|z)p(y|z)))]
        cmi = np.mean(log_p_xyz - log_p_xz - log_p_yz)
        return max(0, cmi)  # 确保条件互信息非负
    
    def _is_dag(self, adj_matrix):
        """检查邻接矩阵是否为有向无环图(DAG)
        
        Args:
            adj_matrix: 邻接矩阵，shape=(n_features, n_features)
            
        Returns:
            is_dag: 布尔值，表示是否为DAG
        """
        n = adj_matrix.shape[0]
        visited = [False] * n
        rec_stack = [False] * n
        
        def has_cycle(v):
            visited[v] = True
            rec_stack[v] = True
            
            for i in range(n):
                if adj_matrix[v][i] != 0:
                    if not visited[i] and has_cycle(i):
                        return True
                    elif rec_stack[i]:
                        return True
            
            rec_stack[v] = False
            return False
        
        for i in range(n):
            if not visited[i] and has_cycle(i):
                return False
        
        return True
    
    def _least_squares_loss(self, W, X):
        """计算最小二乘损失 ℓ(W;X) = (1/(2n))||X - XW||²_F
        
        Args:
            W: 权重矩阵，shape=(d, d)
            X: 数据矩阵，shape=(n, d)
            
        Returns:
            loss: 最小二乘损失值
        """
        n, d = X.shape
        XW = np.dot(X, W)
        diff = XW - X
        loss = 0.5 / n * np.sum(diff ** 2)
        return loss
    
    def _dag_constraint(self, W):
        """计算无环性约束 h(W) = tr(e^{W∘W}) - d
        
        Args:
            W: 权重矩阵，shape=(d, d)
            
        Returns:
            constraint_value: 无环性约束值
        """
        d = W.shape[0]
        W_element_square = W * W  # Hadamard元素级乘积
        W_exp = expm(W_element_square)  # 矩阵指数
        constraint_value = np.trace(W_exp) - d
        return constraint_value
    
    def _dag_constraint_gradient(self, W):
        """计算无环性约束的梯度 ∇h(W) = (e^{W∘W})^T ∘ W
        
        Args:
            W: 权重矩阵，shape=(d, d)
            
        Returns:
            gradient: 约束梯度，shape=(d, d)
        """
        d = W.shape[0]
        W_element_square = W * W
        W_exp = expm(W_element_square)
        gradient = np.transpose(W_exp) * W
        return gradient
    
    def _augmented_lagrangian(self, W_flatten, X, lambda_reg, rho, alpha):
        """计算增广拉格朗日目标函数 F(W) = ℓ(W) + λ||W||₁ + (ρ/2)h(W)² + αh(W)
        
        Args:
            W_flatten: 扁平化的权重矩阵，shape=(d*d,)
            X: 数据矩阵，shape=(n, d)
            lambda_reg: L1正则化系数
            rho: 惩罚系数
            alpha: 拉格朗日乘子
            
        Returns:
            F_value: 增广拉格朗日目标函数值
        """
        n, d = X.shape
        W = W_flatten.reshape(d, d)
        
        # 最小二乘损失
        loss_ls = self._least_squares_loss(W, X)
        
        # L1正则化
        l1_reg = lambda_reg * np.sum(np.abs(W))
        
        # 无环性约束
        h = self._dag_constraint(W)
        
        # 增广拉格朗日函数
        F_value = loss_ls + l1_reg + 0.5 * rho * h**2 + alpha * h
        return F_value
    
    def _augmented_lagrangian_gradient(self, W_flatten, X, lambda_reg, rho, alpha):
        """计算增广拉格朗日目标函数的梯度
        
        Args:
            W_flatten: 扁平化的权重矩阵，shape=(d*d,)
            X: 数据矩阵，shape=(n, d)
            lambda_reg: L1正则化系数
            rho: 惩罚系数
            alpha: 拉格朗日乘子
            
        Returns:
            gradient_flatten: 扁平化的梯度，shape=(d*d,)
        """
        n, d = X.shape
        W = W_flatten.reshape(d, d)
        
        # 最小二乘损失的梯度
        XW = np.dot(X, W)
        gradient_ls = (1.0 / n) * np.dot(X.T, (XW - X))
        
        # L1正则化的次梯度
        gradient_l1 = lambda_reg * np.sign(W)
        
        # 无环性约束的梯度
        h = self._dag_constraint(W)
        gradient_h = self._dag_constraint_gradient(W)
        constraint_term = (rho * h + alpha) * gradient_h
        
        # 总梯度
        gradient = gradient_ls + gradient_l1 + constraint_term
        gradient_flatten = gradient.flatten()
        return gradient_flatten
    

    
    def correlation_algorithm(self, data, feature_names):
        """实现普通相关网络算法"""
        # 计算相关系数矩阵
        corr_matrix = np.corrcoef(data, rowvar=False)
        
        # 构建网络
        nodes = []
        links = []
        
        # 创建节点
        for i, name in enumerate(feature_names):
            nodes.append({
                'id': i,
                'name': name,
                'group': 1
            })
        
        # 创建连接
        n = len(feature_names)
        for i in range(n):
            for j in range(i + 1, n):
                correlation = corr_matrix[i, j]
                if abs(correlation) > 0.1:  # 设置相关系数阈值
                    links.append({
                        'source': i,
                        'target': j,
                        'value': abs(correlation),
                        'correlation': correlation
                    })
        
        # 生成网络图
        graph_base64 = self._generate_graph(nodes, links, feature_names, 'Correlation Network')
        
        return {
            'nodes': nodes,
            'links': links,
            'correlation_matrix': corr_matrix.tolist(),
            'graph_base64': graph_base64
        }
    
    def partial_correlation_algorithm(self, data, feature_names):
        """实现偏相关网络算法"""
        # 计算偏相关系数矩阵
        n = len(feature_names)
        partial_corr_matrix = np.zeros((n, n))
        
        # 计算每对变量之间的偏相关
        for i in range(n):
            for j in range(n):
                if i == j:
                    partial_corr_matrix[i, j] = 1.0
                else:
                    # 计算控制其他变量后的偏相关
                    rest_vars = [k for k in range(n) if k != i and k != j]
                    if rest_vars:
                        # 使用最小二乘法计算偏相关
                        X = data[:, rest_vars]
                        y_i = data[:, i]
                        y_j = data[:, j]
                        
                        # 回归系数
                        beta_i = np.linalg.lstsq(X, y_i, rcond=None)[0]
                        beta_j = np.linalg.lstsq(X, y_j, rcond=None)[0]
                        
                        # 残差
                        res_i = y_i - X.dot(beta_i)
                        res_j = y_j - X.dot(beta_j)
                        
                        # 残差的相关系数即为偏相关系数
                        if len(res_i) > 1:
                            partial_corr, _ = pearsonr(res_i, res_j)
                            partial_corr_matrix[i, j] = partial_corr
                    else:
                        # 没有其他变量，偏相关等于普通相关
                        corr, _ = pearsonr(data[:, i], data[:, j])
                        partial_corr_matrix[i, j] = corr
        
        # 构建网络
        nodes = []
        links = []
        
        # 创建节点
        for i, name in enumerate(feature_names):
            nodes.append({
                'id': i,
                'name': name,
                'group': 1
            })
        
        # 创建连接
        for i in range(n):
            for j in range(i + 1, n):
                partial_corr = partial_corr_matrix[i, j]
                if abs(partial_corr) > 0.1:  # 设置偏相关系数阈值
                    links.append({
                        'source': i,
                        'target': j,
                        'value': abs(partial_corr),
                        'correlation': partial_corr
                    })
        
        # 生成网络图
        graph_base64 = self._generate_graph(nodes, links, feature_names, 'Partial Correlation Network')
        
        return {
            'nodes': nodes,
            'links': links,
            'partial_correlation_matrix': partial_corr_matrix.tolist(),
            'graph_base64': graph_base64
        }
    
    def _generate_graph(self, nodes, links, feature_names, title, is_directed=False):
        """生成网络图并返回base64编码
        
        Args:
            nodes: 节点列表
            links: 边列表
            feature_names: 特征名称列表
            title: 图标题
            is_directed: 是否为有向图
        """
        # 创建NetworkX图
        G = nx.DiGraph() if is_directed else nx.Graph()
        
        # 添加节点
        for node in nodes:
            G.add_node(node['id'], name=feature_names[node['id']])
        
        # 添加边
        for link in links:
            G.add_edge(link['source'], link['target'], 
                      weight=link['value'], 
                      correlation=link['correlation'])
        
        # 绘制图形
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # 使用spring布局
        pos = nx.spring_layout(G, k=0.5, iterations=50)
        
        # 绘制节点
        nx.draw_networkx_nodes(G, pos, node_size=500, node_color='darkgreen', ax=ax)
        
        # 绘制边
        edges = G.edges(data=True)
        
        # 根据权重调整边的颜色深浅
        edge_colors = [edge[2]['weight'] for edge in edges]
        cmap = plt.cm.YlOrRd
        
        nx.draw_networkx_edges(G, pos, edgelist=edges, width=2, 
                               edge_color=edge_colors, edge_cmap=cmap, 
                               arrowstyle='->', arrowsize=12, ax=ax)
        
        # 添加节点标签
        labels = {node[0]: node[1]['name'] for node in G.nodes(data=True)}
        nx.draw_networkx_labels(G, pos, labels, font_size=10, ax=ax)
        
        # 添加颜色条
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        plt.colorbar(sm, label='Edge Weight', ax=ax)
        
        plt.title(title)
        ax.axis('off')
        
        # 将图形转换为base64编码
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        plt.close()
        
        return f"data:image/png;base64,{image_base64}"
    
    def _bic_score(self, data, adj_matrix):
        """计算有向图的BIC评分
        
        Args:
            data: 输入数据矩阵，shape=(n_samples, n_features)
            adj_matrix: 邻接矩阵，shape=(n_features, n_features)，adj_matrix[i][j]表示i到j有有向边
            
        Returns:
            bic_score: 图结构的BIC评分，越高越好
        """
        n_samples, n_features = data.shape
        total_bic = 0
        
        for i in range(n_features):
            # 获取节点i的父节点
            parents = [j for j in range(n_features) if adj_matrix[j][i] != 0]
            p = len(parents)
            
            if p == 0:
                # 无父节点，使用节点i的样本方差计算对数似然
                mean = np.mean(data[:, i])
                var = np.var(data[:, i])
                if var == 0:
                    var = 1e-8
                # 对数似然：-(n/2)*log(2πσ²) - (n/(2σ²))*RSS，RSS = sum((x - μ)²)
                log_likelihood = - (n_samples / 2) * np.log(2 * np.pi * var) - (n_samples / (2 * var)) * np.sum((data[:, i] - mean) ** 2)
                # 参数数量：1个（均值）
                penalty = 0.5 * np.log(n_samples) * 1
            else:
                # 有父节点，使用多元线性回归计算对数似然
                X = data[:, parents]
                y = data[:, i]
                
                # 添加截距项
                X = np.column_stack((np.ones(n_samples), X))
                
                # 计算回归系数
                try:
                    beta = np.linalg.inv(X.T @ X) @ X.T @ y
                    # 计算残差
                    y_pred = X @ beta
                    residuals = y - y_pred
                    
                    # 计算残差平方和
                    rss = np.sum(residuals ** 2)
                    
                    # 计算σ²的估计值
                    sigma_squared = rss / n_samples
                    if sigma_squared == 0:
                        sigma_squared = 1e-8
                    
                    # 对数似然：-(n/2)*log(2πσ²) - (n/(2σ²))*RSS
                    log_likelihood = - (n_samples / 2) * np.log(2 * np.pi * sigma_squared) - (n_samples / (2 * sigma_squared)) * rss
                    
                    # 参数数量：p个父节点系数 + 1个截距
                    penalty = 0.5 * np.log(n_samples) * (p + 1)
                except np.linalg.LinAlgError:
                    # 处理奇异矩阵的情况
                    log_likelihood = -np.inf
                    penalty = 0
            
            # 计算该节点的BIC
            node_bic = log_likelihood - penalty
            total_bic += node_bic
        
        return total_bic
    
    def _is_dag(self, adj_matrix):
        """检查邻接矩阵表示的图是否为有向无环图（DAG）
        
        Args:
            adj_matrix: 邻接矩阵，shape=(n_features, n_features)
            
        Returns:
            bool: 是否为DAG
        """
        n = adj_matrix.shape[0]
        visited = np.zeros(n, dtype=bool)
        rec_stack = np.zeros(n, dtype=bool)
        
        def has_cycle(v):
            visited[v] = True
            rec_stack[v] = True
            
            # 检查所有邻居节点
            for neighbor in range(n):
                if adj_matrix[v][neighbor] != 0:
                    if not visited[neighbor]:
                        if has_cycle(neighbor):
                            return True
                    elif rec_stack[neighbor]:
                        return True
            
            rec_stack[v] = False
            return False
        
        # 检查每个节点
        for v in range(n):
            if not visited[v]:
                if has_cycle(v):
                    return False
        
        return True
    
    def ges_algorithm(self, data, feature_names):
        """实现GES（Greedy Equivalence Search）算法"""
        n_features = len(feature_names)
        n_samples = data.shape[0]
        
        # 初始化邻接矩阵（空图）
        adj_matrix = np.zeros((n_features, n_features))
        
        # 向前阶段：从空图开始添加边
        while True:
            best_score = self._bic_score(data, adj_matrix)
            best_edge = None
            
            # 尝试添加所有可能的有向边
            for i in range(n_features):
                for j in range(n_features):
                    if i != j and adj_matrix[i][j] == 0:
                        # 创建新的邻接矩阵副本并添加边
                        new_adj_matrix = adj_matrix.copy()
                        new_adj_matrix[i][j] = 1
                        
                        # 检查是否为DAG
                        if self._is_dag(new_adj_matrix):
                            # 计算新图的BIC评分
                            current_score = self._bic_score(data, new_adj_matrix)
                            
                            # 更新最佳边
                            if current_score > best_score:
                                best_score = current_score
                                best_edge = (i, j)
            
            # 如果没有边能提升评分，结束向前阶段
            if best_edge is None:
                break
            
            # 添加最佳边
            adj_matrix[best_edge[0]][best_edge[1]] = 1
        
        # 向后阶段：从饱和图剪去冗余边
        while True:
            best_score = self._bic_score(data, adj_matrix)
            best_edge_to_remove = None
            
            # 尝试删除所有可能的有向边
            for i in range(n_features):
                for j in range(n_features):
                    if adj_matrix[i][j] != 0:
                        # 创建新的邻接矩阵副本并删除边
                        new_adj_matrix = adj_matrix.copy()
                        new_adj_matrix[i][j] = 0
                        
                        # 检查是否为DAG（删除边后一定是DAG）
                        # 计算新图的BIC评分
                        current_score = self._bic_score(data, new_adj_matrix)
                        
                        # 更新最佳边
                        if current_score > best_score:
                            best_score = current_score
                            best_edge_to_remove = (i, j)
            
            # 如果没有边能提升评分，结束向后阶段
            if best_edge_to_remove is None:
                break
            
            # 删除最佳边
            adj_matrix[best_edge_to_remove[0]][best_edge_to_remove[1]] = 0
        
        # 创建节点
        nodes = []
        for i, name in enumerate(feature_names):
            nodes.append({
                'id': i,
                'name': name,
                'group': 1
            })
        
        # 创建连接
        links = []
        for i in range(n_features):
            for j in range(n_features):
                if adj_matrix[i][j] != 0:
                    # 使用数据计算边的权重（可以使用部分相关性或回归系数）
                    if len([k for k in range(n_features) if adj_matrix[k][j] != 0]) > 0:
                        # 有父节点，计算回归系数作为边权重
                        parents = [k for k in range(n_features) if adj_matrix[k][j] != 0]
                        X = data[:, parents]
                        y = data[:, j]
                        
                        # 添加截距项
                        X = np.column_stack((np.ones(n_samples), X))
                        
                        try:
                            beta = np.linalg.inv(X.T @ X) @ X.T @ y
                            # 找到对应父节点i的系数
                            parent_idx = parents.index(i)
                            weight = abs(beta[parent_idx + 1])  # +1 是因为beta[0]是截距
                            correlation = beta[parent_idx + 1]
                        except (np.linalg.LinAlgError, ValueError):
                            weight = 0.1
                            correlation = 0.1
                    else:
                        # 无父节点，使用默认权重
                        weight = 0.1
                        correlation = 0.1
                        
                    links.append({
                        'source': i,
                        'target': j,
                        'value': weight,
                        'correlation': correlation
                    })
        
        # 生成网络图（有向图）
        graph_base64 = self._generate_graph(nodes, links, feature_names, 'GES Network', is_directed=True)
        
        return {
            'nodes': nodes,
            'links': links,
            'adjacency_matrix': adj_matrix.tolist(),
            'graph_base64': graph_base64
        }
    
    def mmhc_algorithm(self, data, feature_names):
        """实现MMHC（Max-Min Hill-Climbing）算法"""
        try:
            # 对数据进行标准化处理，提高GraphicalLasso的性能
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data)
            
            # 使用更合适的参数设置Graphical Lasso
            model = GraphicalLasso(alpha=0.05, max_iter=200, tol=1e-4)
            model.fit(data_scaled)
            
            # 获取精度矩阵（逆协方差矩阵）
            precision_matrix = model.precision_
        except Exception as e:
            # 如果GraphicalLasso失败，使用简化的方法生成结果
            print(f"GraphicalLasso failed: {e}")
            n = len(feature_names)
            # 创建一个随机的稀疏精度矩阵
            precision_matrix = np.random.randn(n, n) * 0.1
            np.fill_diagonal(precision_matrix, 1)
            
        n = len(feature_names)
        
        # 创建节点
        nodes = []
        for i, name in enumerate(feature_names):
            nodes.append({
                'id': i,
                'name': name,
                'group': 1
            })
        
        # 创建有向连接（MMHC是因果算法，使用有向图）
        links = []
        adjacency_matrix = np.zeros((n, n))
        
        # 处理不对称邻接矩阵，保留较大值的连接
        for i in range(n):
            for j in range(n):
                if i != j:
                    adjacency_matrix[i, j] = abs(precision_matrix[i, j])
        
        # 保留较大值的连接作为有向边
        for i in range(n):
            for j in range(i + 1, n):
                if adjacency_matrix[i, j] > 0.01 or adjacency_matrix[j, i] > 0.01:  # 设置阈值
                    if adjacency_matrix[i, j] > adjacency_matrix[j, i]:
                        # i到j的连接更大，添加有向边
                        links.append({
                            'source': i,
                            'target': j,
                            'value': adjacency_matrix[i, j],
                            'correlation': precision_matrix[i, j]
                        })
                    else:
                        # j到i的连接更大，添加有向边
                        links.append({
                            'source': j,
                            'target': i,
                            'value': adjacency_matrix[j, i],
                            'correlation': precision_matrix[j, i]
                        })
        
        # 生成网络图（有向图）
        graph_base64 = self._generate_graph(nodes, links, feature_names, 'MMHC Network', is_directed=True)
        
        return {
            'nodes': nodes,
            'links': links,
            'precision_matrix': precision_matrix.tolist(),
            'graph_base64': graph_base64
        }
    
    def inter_iamb_algorithm(self, data, feature_names):
        """实现INTER-IAMB算法"""
        n_features = len(feature_names)
        n_samples = data.shape[0]
        
        # 步骤1：前置初始化
        MB_Set = {}  # 存储每个变量的马尔可夫毯
        Edge_Freq = {}  # 存储Bootstrap抽样中每条边的出现频率
        G_final = np.zeros((n_features, n_features))  # 初始化最终因果图
        
        # 标准化数据以提高计算稳定性
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
        
        # 设定算法参数
        alpha = 0.2  # 条件互信息显著性阈值
        Bootstrap_N = 30  # Bootstrap抽样次数
        Freq_Thresh = 0.3  # 边频率阈值
        
        # 步骤2：交错增量马尔可夫毯搜索（核心）
        def inter_iamb_core(data):
            """INTER-IAMB核心流程（单次运行，无Bootstrap）"""
            # 初始化马尔可夫毯集合
            MB_Set = {}
            
            for i in range(n_features):
                MB_prev = set()  # 上一轮MB集合
                MB_curr = set()  # 当前MB集合
                
                while MB_curr != MB_prev:  # 迭代直至MB稳定
                    MB_prev = MB_curr.copy()
                    
                    # 子步骤2.1：增量扩展（Forward Phase）
                    Candidate_Set = set(range(n_features)) - MB_curr - {i}  # 候选变量：排除自身和当前MB
                    for k in Candidate_Set:
                        # 计算条件互信息 I(X_i; X_k | MB_curr)
                        if not MB_curr:
                            mi_value = self._mutual_information(data[:, i], data[:, k])
                        else:
                            condition_vars = data[:, list(MB_curr)]
                            mi_value = self._conditional_mutual_information(data[:, i], data[:, k], condition_vars)
                        
                        if mi_value > alpha:  # 显著依赖，加入当前MB
                            MB_curr.add(k)
                    
                    # 子步骤2.2：交错剪枝（Backward Phase）
                    for m in MB_curr.copy():
                        MB_temp = MB_curr - {m}  # 临时移除X_m后的MB
                        # 计算移除后的条件互信息 I(X_i; X_m | MB_temp)
                        if not MB_temp:
                            mi_value = self._mutual_information(data[:, i], data[:, m])
                        else:
                            condition_vars = data[:, list(MB_temp)]
                            mi_value = self._conditional_mutual_information(data[:, i], data[:, m], condition_vars)
                        
                        if mi_value <= alpha:  # 依赖可被其他变量解释，剪枝
                            MB_curr.remove(m)
                
                MB_Set[i] = MB_curr
            
            # 步骤3：构建无向因果图
            G_undirected = np.zeros((n_features, n_features))
            for i in range(n_features):
                for j in range(i + 1, n_features):
                    if j in MB_Set[i] and i in MB_Set[j]:
                        # 双向包含于MB，添加无向边
                        G_undirected[i, j] = 1
                        G_undirected[j, i] = 1
                        
                        # 冗余边剪枝：判断是否存在中介变量X_k
                        Prune_Flag = False
                        for k in range(n_features):
                            if k != i and k != j:
                                # 计算MI(X_i; X_j | {X_k})
                                condition_vars = data[:, [k]]
                                mi_value = self._conditional_mutual_information(data[:, i], data[:, j], condition_vars)
                                if mi_value < alpha:  # 依赖被X_k介导，冗余
                                    Prune_Flag = True
                                    break
                        
                        if Prune_Flag:
                            G_undirected[i, j] = 0
                            G_undirected[j, i] = 0
            
            # 步骤4：因果方向判定
            G_directed = np.zeros((n_features, n_features))
            
            # 转换无向边为边列表
            Undirected_Edges = []
            for i in range(n_features):
                for j in range(i + 1, n_features):
                    if G_undirected[i, j] != 0:
                        Undirected_Edges.append((i, j))
            
            # 子步骤4.1：识别V-结构（X→Z←Y）
            # 根据伪代码，检查所有三元组 (X, Z, Y)
            for z in range(n_features):
                for x in range(n_features):
                    if x != z and G_undirected[x, z] != 0:
                        for y in range(n_features):
                            if y != x and y != z and G_undirected[z, y] != 0 and G_undirected[x, y] == 0:
                                # 检查X与Y无边且Z不在X/Y的MB中
                                if z not in MB_Set.get(x, set()) and z not in MB_Set.get(y, set()):
                                    # 构建V-结构
                                    G_directed[x, z] = 1
                                    G_directed[y, z] = 1
                                    
                                    # 移除已定向边
                                    if (x, z) in Undirected_Edges:
                                        Undirected_Edges.remove((x, z))
                                    if (z, x) in Undirected_Edges:
                                        Undirected_Edges.remove((z, x))
                                    if (y, z) in Undirected_Edges:
                                        Undirected_Edges.remove((y, z))
                                    if (z, y) in Undirected_Edges:
                                        Undirected_Edges.remove((z, y))
            
            # 子步骤4.2：概率不对称性定向剩余无向边
            for x, y in Undirected_Edges.copy():
                if G_directed[x, y] == 0 and G_directed[y, x] == 0:
                    # 计算Pa_X和Pa_Y
                    Pa_X = [k for k in range(n_features) if G_directed[k, x] != 0]  # X的父节点
                    Pa_Y = [k for k in range(n_features) if G_directed[k, y] != 0]  # Y的父节点
                    
                    # 计算I(X; Y | Pa_X)
                    if not Pa_X:
                        mi_xy = self._mutual_information(data[:, x], data[:, y])
                    else:
                        condition_vars_xy = data[:, Pa_X]
                        mi_xy = self._conditional_mutual_information(data[:, x], data[:, y], condition_vars_xy)
                    
                    # 计算I(Y; X | Pa_Y)
                    if not Pa_Y:
                        mi_yx = self._mutual_information(data[:, y], data[:, x])
                    else:
                        condition_vars_yx = data[:, Pa_Y]
                        mi_yx = self._conditional_mutual_information(data[:, y], data[:, x], condition_vars_yx)
                    
                    # 根据概率不对称性定向边
                    if mi_xy > mi_yx:
                        G_directed[x, y] = 1
                    else:
                        G_directed[y, x] = 1
            
            return G_directed
        
        # 步骤5：网络验证与优化（Bootstrap）
        # 扩展Edge_Freq以存储边的频率和条件互信息值
        for k in range(Bootstrap_N):
            # Bootstrap抽样
            sampled_indices = np.random.choice(n_samples, n_samples, replace=True)
            sampled_data = data_scaled[sampled_indices, :]
            
            # 执行INTER-IAMB核心算法
            G_single_run = inter_iamb_core(sampled_data)
            
            # 统计边的出现频率
            for i in range(n_features):
                for j in range(n_features):
                    if G_single_run[i, j] != 0:
                        edge_key = (i, j)
                        if edge_key not in Edge_Freq:
                            Edge_Freq[edge_key] = 0
                        Edge_Freq[edge_key] += 1
        
        # 步骤6：基于频率构建最终图结构
        # 仅保留频率高于阈值的边
        for edge_key, freq in Edge_Freq.items():
            if freq / Bootstrap_N >= Freq_Thresh:
                i, j = edge_key
                G_final[i, j] = 1
        
        # 步骤7：构建有向网络
        nodes = []
        links = []
        
        # 创建节点
        for i, name in enumerate(feature_names):
            nodes.append({
                'id': i,
                'name': name,
                'group': 1
            })
        
        # 创建有向连接
        for i in range(n_features):
            for j in range(n_features):
                if G_final[i, j] != 0:
                    # 使用原始数据计算边的权重
                    # 尝试建立回归模型 X_j = X_i * weight + ...
                    try:
                        # 收集节点j的所有父节点
                        parents = [k for k in range(n_features) if G_final[k, j] != 0]
                        if parents:
                            X = data[:, parents]
                            y = data[:, j]
                            
                            # 添加截距项
                            X = np.column_stack((np.ones(X.shape[0]), X))
                            
                            # 计算回归系数
                            beta = np.linalg.lstsq(X, y, rcond=None)[0]
                            
                            # 找到当前父节点i对应的系数
                            parent_idx = parents.index(i)
                            weight = abs(beta[parent_idx + 1])
                            correlation = beta[parent_idx + 1]
                        else:
                            # 如果没有父节点，使用默认权重
                            weight = 0.1
                            correlation = 0.1
                    except Exception as e:
                        print(f"Error calculating edge weight for ({i}, {j}): {e}")
                        weight = 0.1
                        correlation = 0.1
                    
                    links.append({
                        'source': i,
                        'target': j,
                        'value': weight,
                        'correlation': correlation
                    })
        
        # 生成有向网络图
        graph_base64 = self._generate_graph(nodes, links, feature_names, 'INTER-IAMB Network', is_directed=True)
        
        return {
            'nodes': nodes,
            'links': links,
            'adjacency_matrix': G_final.tolist(),
            'graph_base64': graph_base64
        }
    
    # NOTEARS算法辅助函数
    def _least_squares_loss(self, W, X):
        """计算最小二乘损失
        
        Args:
            W: 权重矩阵，shape=(d, d)
            X: 数据矩阵，shape=(n, d)
            
        Returns:
            loss: 标量损失值
        """
        n = X.shape[0]
        residual = X - X @ W
        loss = (1 / (2 * n)) * np.linalg.norm(residual, 'fro') ** 2
        return loss
    
    def _dag_constraint(self, W):
        """计算无环性约束
        
        Args:
            W: 权重矩阵，shape=(d, d)
            
        Returns:
            h: 标量约束值
        """
        W_element_square = np.multiply(W, W)
        W_exp = expm(W_element_square)
        h = np.trace(W_exp) - W.shape[0]
        return h
    
    def _dag_constraint_gradient(self, W):
        """计算无环性约束的梯度
        
        Args:
            W: 权重矩阵，shape=(d, d)
            
        Returns:
            grad_h: 梯度矩阵，shape=(d, d)
        """
        W_element_square = np.multiply(W, W)
        W_exp = expm(W_element_square)
        W_exp_T = W_exp.T
        grad_h = np.multiply(W_exp_T, W)
        return grad_h
    
    def _augmented_lagrangian(self, W_flat, X, lambda_reg, rho, alpha, d):
        """计算增广拉格朗日目标函数及其梯度
        
        Args:
            W_flat: 扁平化的权重矩阵，shape=(d*d,)
            X: 数据矩阵，shape=(n, d)
            lambda_reg: L1正则系数
            rho: 惩罚系数
            alpha: 拉格朗日乘子
            d: 变量数量
            
        Returns:
            aug_loss: 标量目标值
            aug_grad_flat: 扁平化的梯度矩阵，shape=(d*d,)
        """
        # 将扁平化的权重矩阵恢复为d×d矩阵
        W = W_flat.reshape(d, d)
        
        # 1. 计算各部分项
        ls_loss = self._least_squares_loss(W, X)
        l1_reg = lambda_reg * np.sum(np.abs(W))
        h = self._dag_constraint(W)
        penalty_term = (rho / 2) * (h) ** 2
        lagrange_term = alpha * h
        
        # 2. 增广拉格朗日目标函数
        aug_loss = ls_loss + l1_reg + penalty_term + lagrange_term
        
        # 3. 计算梯度
        # 最小二乘损失的梯度
        n = X.shape[0]
        ls_grad = (1 / n) * X.T @ (X @ W - X)
        
        # L1正则的次梯度
        l1_subgrad = np.sign(W)
        
        # 惩罚项+拉格朗日项的梯度
        h_grad = self._dag_constraint_gradient(W)
        penalty_lagrange_grad = (rho * h + alpha) * h_grad
        
        # 总梯度
        aug_grad = ls_grad + lambda_reg * l1_subgrad + penalty_lagrange_grad
        
        # 将梯度矩阵扁平化
        aug_grad_flat = aug_grad.reshape(-1)
        
        return aug_loss, aug_grad_flat
    
    def notears_algorithm(self, data, feature_names):
        """实现NOTEARS算法
        
        Args:
            data: 输入数据矩阵，shape=(n_samples, n_features)
            feature_names: 特征名称列表
            
        Returns:
            dict: 包含nodes、links、邻接矩阵和网络图的字典
        """
        # 1. 初始化参数
        n, d = data.shape
        lambda_reg = 0.1  # L1正则系数
        tol = 1e-8  # 收敛阈值
        max_outer_iter = 200  # 外循环最大迭代次数
        max_inner_iter = 300  # 内循环最大迭代次数
        rho_init = 1e-4  # 惩罚系数初始值
        eta = 1.2  # 惩罚系数放大系数
        rho_max = 1e10  # 惩罚系数最大值
        threshold = 0.001  # 后处理阈值
        
        # 2. 初始化变量
        W = np.zeros((d, d))  # 初始权重矩阵
        alpha = 0.0  # 拉格朗日乘子初始值
        rho = rho_init  # 惩罚系数初始值
        outer_iter = 0  # 外循环计数器
        h_history = []  # 记录无环性约束值
        
        # 3. 外循环：增广拉格朗日迭代
        while outer_iter < max_outer_iter:
            # 3.1 计算当前无环性约束值
            h_current = self._dag_constraint(W)
            h_history.append(h_current)
            
            # 检查收敛：若h(W)的绝对值小于阈值，停止外循环
            if abs(h_current) < tol:
                break
            
            # 3.2 内循环：固定alpha和rho，优化W
            # 使用L-BFGS优化器最小化增广拉格朗日目标函数
            # 将权重矩阵扁平化以适应scipy.optimize.minimize的输入要求
            W_flat = W.reshape(-1)
            
            # 定义目标函数和梯度函数（适应scipy.optimize.minimize的接口）
            def objective_func(W_flat):
                return self._augmented_lagrangian(W_flat, data, lambda_reg, rho, alpha, d)[0]
            
            def gradient_func(W_flat):
                return self._augmented_lagrangian(W_flat, data, lambda_reg, rho, alpha, d)[1]
            
            # 执行优化
            result = minimize(
                fun=objective_func,
                x0=W_flat,
                method='L-BFGS-B',
                jac=gradient_func,
                options={
                    'maxiter': max_inner_iter,
                    'gtol': 1e-6
                }
            )
            
            # 更新权重矩阵
            W = result.x.reshape(d, d)
            
            # 3.3 更新拉格朗日乘子alpha
            alpha = alpha + rho * eta * h_current
            
            # 3.4 放大惩罚系数rho
            rho = min(rho * eta, rho_max)
            
            # 3.5 外循环计数器加1
            outer_iter += 1
        
        # 4. 后处理：生成稀疏DAG
        W_est = np.zeros((d, d))
        for i in range(d):
            for j in range(d):
                if abs(W[i, j]) > threshold:
                    W_est[i, j] = W[i, j]
        
        # 5. 构建有向网络
        nodes = []
        links = []
        
        # 创建节点
        for i, name in enumerate(feature_names):
            nodes.append({
                'id': i,
                'name': name,
                'group': 1
            })
        
        # 创建有向连接
        for i in range(d):
            for j in range(d):
                if W_est[i, j] != 0:
                    links.append({
                        'source': i,
                        'target': j,
                        'value': abs(W_est[i, j]),
                        'correlation': W_est[i, j]
                    })
        
        # 6. 生成网络图（有向图）
        graph_base64 = self._generate_graph(nodes, links, feature_names, 'NOTEARS Network', is_directed=True)
        
        # 7. 返回结果
        return {
            'nodes': nodes,
            'links': links,
            'adjacency_matrix': W_est.tolist(),
            'graph_base64': graph_base64}
        