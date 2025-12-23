// 页面加载完成后执行
document.addEventListener('DOMContentLoaded', function() {
    // 初始化所有功能
    initSmoothScrolling();
    initScrollAnimations();
    initDataUpload();
    initDatasetList();
    initAnalysisControls();
    initAlgorithmInfo();
    
    // 添加保存路径设置
    addSavePathSetting();
    
    // 加载历史数据集
    loadDatasets();
});

// 导航栏平滑滚动
function initSmoothScrolling() {
    const navLinks = document.querySelectorAll('.nav-links a');
    
    navLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            const targetId = this.getAttribute('href');
            const targetSection = document.querySelector(targetId);
            
            window.scrollTo({
                top: targetSection.offsetTop - 80,
                behavior: 'smooth'
            });
        });
    });
}

// 页面滚动动画
function initScrollAnimations() {
    gsap.registerPlugin(ScrollTrigger);
    
    const sections = document.querySelectorAll('section');
    
    sections.forEach(section => {
        gsap.from(section, {
            opacity: 0,
            y: 50,
            duration: 0.8,
            scrollTrigger: {
                trigger: section,
                start: 'top 80%',
                end: 'top 20%',
                toggleActions: 'play none none reverse'
            }
        });
    });
}

// 数据上传功能
function initDataUpload() {
    const uploadBtn = document.getElementById('upload-btn');
    const datasetNameInput = document.getElementById('dataset-name');
    const dataFileInput = document.getElementById('data-file');
    
    uploadBtn.addEventListener('click', function() {
        const datasetName = datasetNameInput.value.trim();
        const dataFile = dataFileInput.files[0];
        
        if (!datasetName) {
            showMessage('请输入数据集名称', 'error');
            return;
        }
        
        if (!dataFile) {
            showMessage('请选择要上传的数据文件', 'error');
            return;
        }
        
        // 检查文件类型
        const allowedTypes = ['.csv', '.xls', '.xlsx'];
        const fileExtension = dataFile.name.slice(dataFile.name.lastIndexOf('.'));
        
        if (!allowedTypes.includes(fileExtension.toLowerCase())) {
            showMessage('只支持上传CSV、XLS或XLSX格式的文件', 'error');
            return;
        }
        
        // 显示上传中状态
        uploadBtn.innerHTML = '<span class="loading"></span> 上传中...';
        uploadBtn.disabled = true;
        
        // 创建FormData对象
        const formData = new FormData();
        formData.append('datasetName', datasetName);
        formData.append('dataFile', dataFile);
        
        // 发送POST请求到后端API
        fetch('http://localhost:3000/api/datasets/upload', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                // 上传成功
                showMessage('数据集上传成功', 'success');
                
                // 重新加载数据集列表
                loadDatasets();
                
                // 重置表单
                datasetNameInput.value = '';
                dataFileInput.value = '';
            } else {
                // 上传失败
                showMessage('数据集上传失败: ' + (data.message || '未知错误'), 'error');
            }
        })
        .catch(error => {
            showMessage('数据集上传失败: ' + error.message, 'error');
        })
        .finally(() => {
            // 恢复按钮状态
            uploadBtn.innerHTML = '上传数据';
            uploadBtn.disabled = false;
        });
    });
}


// 加载历史数据集
function loadDatasets() {
    const datasetsList = document.getElementById('datasets-list');
    const selectDataset = document.getElementById('select-dataset');
    
    // 清空现有内容
    datasetsList.innerHTML = '';
    selectDataset.innerHTML = '<option value="">请选择数据集</option>';
    
    // 显示加载状态
    datasetsList.innerHTML = '<div class="loading">加载中...</div>';
    
    // 从后端API获取数据集列表
    fetch('http://localhost:3000/api/datasets')
        .then(response => response.json())
        .then(data => {
            // 清空加载状态
            datasetsList.innerHTML = '';
            
            if (data.success) {
                const datasets = data.data || [];
                
                if (datasets.length === 0) {
                    // 数据集为空时显示友好提示
                    datasetsList.innerHTML = '<div class="empty-message">暂未包含任意数据集</div>';
                } else {
                    // 添加数据集到列表和下拉框
                    datasets.forEach(dataset => {
                        // 添加到数据集列表
                        const datasetCard = document.createElement('div');
                        datasetCard.className = 'dataset-card';
                        
                        // 格式化文件大小
                        const formattedSize = formatFileSize(dataset.size);
                        
                        // 格式化上传时间
                        const uploadDate = new Date(dataset.upload_time).toLocaleString();
                        
                        datasetCard.innerHTML = `
                            <h3>${dataset.name}</h3>
                            <p>上传日期: ${uploadDate}</p>
                            <p>文件大小: ${formattedSize}</p>
                            <div class="dataset-actions">
                                <button class="btn btn-secondary" onclick="selectDatasetForAnalysis(${dataset.id})">分析</button>
                                <button class="btn btn-secondary" onclick="deleteDataset(${dataset.id})">删除</button>
                            </div>
                        `;
                        datasetsList.appendChild(datasetCard);
                        
                        // 添加到下拉框
                        const option = document.createElement('option');
                        option.value = dataset.id;
                        option.textContent = dataset.name;
                        selectDataset.appendChild(option);
                    });
                }
            } else {
                showMessage('加载数据集失败: ' + (data.message || '未知错误'), 'error');
                datasetsList.innerHTML = '<div class="empty-message">加载数据集失败</div>';
            }
        })
        .catch(error => {
            console.error('获取数据集失败:', error);
            showMessage('加载数据集失败: ' + error.message, 'error');
            datasetsList.innerHTML = '<div class="empty-message">加载数据集失败</div>';
        });
}

// 初始化数据集列表
function initDatasetList() {
    // 数据集列表功能初始化
}

// 选择数据集进行分析
function selectDatasetForAnalysis(datasetId) {
    // 滚动到分析区域
    document.getElementById('analysis').scrollIntoView({
        behavior: 'smooth',
        block: 'start'
    });
    
    // 设置选中的数据集
    document.getElementById('select-dataset').value = datasetId;
    
    // 加载数据集特征
    loadDatasetFeatures(datasetId);
}

// 删除数据集
function deleteDataset(datasetId) {
    if (confirm('确定要删除这个数据集吗？')) {
        // 调用后端API删除数据集
        fetch(`http://localhost:3000/api/datasets/${datasetId}`, {
            method: 'DELETE'
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                // 删除完成后重新加载数据集列表
                loadDatasets();
                showMessage('数据集删除成功', 'success');
            } else {
                showMessage('数据集删除失败: ' + (data.message || '未知错误'), 'error');
            }
        })
        .catch(error => {
            showMessage('数据集删除失败: ' + error.message, 'error');
        });
    }
}

// 加载数据集特征
function loadDatasetFeatures(datasetId) {
    // 调用后端API获取数据集特征
    fetch(`http://localhost:3000/api/datasets/${datasetId}/features`)
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                // 存储特征数据
                window.currentFeatures = data.data || [];
            } else {
                showMessage('获取特征失败: ' + (data.message || '未知错误'), 'error');
                // 使用空数组作为特征数据
                window.currentFeatures = [];
            }
        })
        .catch(error => {
            showMessage('获取特征失败: ' + error.message, 'error');
            // 使用空数组作为特征数据
            window.currentFeatures = [];
        });
}

// 添加保存路径设置
function addSavePathSetting() {
    const analysisControls = document.querySelector('.analysis-controls');
    if (!analysisControls) return;
    
    // 检查是否已经存在保存路径设置
    if (document.getElementById('save-path')) return;
    
    const savePathContainer = document.createElement('div');
    savePathContainer.className = 'form-group';
    savePathContainer.innerHTML = `
        <label for="save-path">保存路径:</label>
        <input type="text" id="save-path" placeholder="请输入分析结果的保存路径（默认为testdata文件夹）">
        <div class="input-hint">设置分析结果保存的目录路径，例如: C:\\analysis_results 或 /home/user/analysis_results</div>
    `;
    
    // 将保存路径设置添加到分析控制区域
    const algorithmSelect = document.getElementById('select-algorithm');
    if (algorithmSelect) {
        algorithmSelect.parentElement.parentElement.insertAdjacentElement('afterend', savePathContainer);
    } else {
        analysisControls.appendChild(savePathContainer);
    }
}

// 初始化分析控制
function initAnalysisControls() {
    const analyzeBtn = document.getElementById('analyze-btn');
    const selectDataset = document.getElementById('select-dataset');
    
    // 数据集选择事件监听（用于加载特征）
    selectDataset.addEventListener('change', function() {
        const datasetId = this.value;
        if (datasetId) {
            loadDatasetFeatures(datasetId);
        }
    });
    
    analyzeBtn.addEventListener('click', function() {
        const datasetId = document.getElementById('select-dataset').value;
        const algorithm = document.getElementById('select-algorithm').value;
        
        if (!datasetId) {
            showMessage('请选择数据集', 'error');
            return;
        }
        
        if (!algorithm) {
            showMessage('请选择算法', 'error');
            return;
        }
        
        // 显示分析中状态
        analyzeBtn.innerHTML = '<span class="loading"></span> 分析中...';
        analyzeBtn.disabled = true;
        
        // 获取保存路径
        const savePath = document.getElementById('save-path')?.value || '';
        
        // 调用后端API进行数据分析
        fetch('http://localhost:3000/api/analyze', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    datasetId: datasetId,
                    algorithm: algorithm,
                    savePath: savePath
                })
            })
            .then(response => response.json())
            .then(data => {
                // 恢复按钮状态
                analyzeBtn.innerHTML = '开始分析';
                analyzeBtn.disabled = false;
                
                if (data.success) {
                    // 保存当前数据集ID用于下载功能
                    window.currentDatasetId = datasetId;
                    
                    // 显示分析成功信息
                    showMessage(data.message || '数据分析完成', 'success');
                    
                    // 显示分析图像
                    if (data.data.graph_base64) {
                        displayAnalysisImage(data.data.graph_base64);
                    }
                    
                    // 显示分析完成弹窗
                    showCompletionPopup(savePath, datasetId);
                } else {
                    showMessage('数据分析失败: ' + (data.message || '未知错误'), 'error');
                }
            })
        .catch(error => {
            // 恢复按钮状态
            analyzeBtn.innerHTML = '开始分析';
            analyzeBtn.disabled = false;
            
            showMessage('数据分析失败: ' + error.message, 'error');
        });
    });
}

// 格式化文件大小
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

// 初始化算法信息
function initAlgorithmInfo() {
    const algorithmSelect = document.getElementById('select-algorithm');
    
    // 静态定义算法列表（与后端算法管理器保持一致）
    const algorithms = [
        { id: 'correlation', name: '相关系数 (Correlation)', description: '计算变量间的相关系数，用于构建关联网络。适用于连续变量的线性关系分析。' },
        { id: 'partial_correlation', name: '偏相关系数 (Partial Correlation)', description: '计算变量间的偏相关系数，控制其他变量影响，用于构建更准确的关联网络。' },
        { id: 'ges', name: '贪婪等价搜索算法 (GES)', description: '通过搜索等价类的方式构建因果网络，适用于大型数据集的因果发现。' },
        { id: 'mmhc', name: '最大最小爬山算法 (MMHC)', description: '结合最大最小父母算法和爬山算法，用于高效发现变量间的因果关系。' },
        { id: 'interiamb', name: '改进的增量关联Markov边界算法 (INTER-IAMB)', description: '通过发现变量的Markov边界来构建因果网络，适用于变量间关系较复杂的数据集。' },
        { id: 'notears', name: 'NOTears算法 (NOTEARS)', description: '基于L1正则化的因果发现算法，适用于高维数据集的因果关系分析。' },
    ];
    
    // 存储算法信息到全局变量
    globalAlgorithms = algorithms;
    
    // 为算法选择添加事件监听
    algorithmSelect.addEventListener('change', function() {
        showAlgorithmInfo(this.value);
    });
}

// 全局变量，用于存储算法信息
let globalAlgorithms = [];

// 显示算法信息
function showAlgorithmInfo(algorithm) {
    const algorithmInfoDiv = document.getElementById('algorithm-info');
    
    if (algorithm && globalAlgorithms.length > 0) {
        const info = globalAlgorithms.find(a => a.id == algorithm);
        if (info) {
            algorithmInfoDiv.innerHTML = `
                <h3>${info.name}</h3>
                <p>${info.description}</p>
            `;
            algorithmInfoDiv.style.display = 'block';
            return;
        }
    }
    
    // 如果没有找到算法信息，使用默认的硬编码信息作为后备
    const defaultAlgorithmInfo = {
        'correlation': {
            name: '相关系数 (Correlation)',
            description: '计算变量间的相关系数，用于构建关联网络。适用于连续变量的线性关系分析。'
        },
        'partial_correlation': {
            name: '偏相关系数 (Partial Correlation)',
            description: '计算变量间的偏相关系数，控制其他变量影响，用于构建更准确的关联网络。'
        },
        'ges': {
            name: '贪婪等价搜索算法 (GES)',
            description: '通过搜索等价类的方式构建因果网络，适用于大型数据集的因果发现。'
        },
        'mmhc': {
            name: '最大最小爬山算法 (MMHC)',
            description: '结合最大最小父母算法和爬山算法，用于高效发现变量间的因果关系。'
        },
        'interiamb': {
            name: '改进的增量关联Markov边界算法 (INTER-IAMB)',
            description: '通过发现变量的Markov边界来构建因果网络，适用于变量间关系较复杂的数据集。'
        },
        'notears': {
            name: 'NOTears算法 (NOTEARS)',
            description: '基于L1正则化的因果发现算法，适用于高维数据集的因果关系分析。'
        }
    };
    
    if (algorithm && defaultAlgorithmInfo[algorithm]) {
        const info = defaultAlgorithmInfo[algorithm];
        algorithmInfoDiv.innerHTML = `
            <h3>${info.name}</h3>
            <p>${info.description}</p>
        `;
        algorithmInfoDiv.style.display = 'block';
    } else {
        algorithmInfoDiv.style.display = 'none';
    }
}

// 保存邻接矩阵为CSV文件
function saveAdjacencyMatrix() {
    const datasetId = window.currentDatasetId;
    
    if (!datasetId) {
        showMessage('没有找到当前数据集ID', 'error');
        return;
    }
    
    // 从后端获取testdata中的邻接矩阵文件
    fetch(`http://localhost:3000/api/testdata/files/${datasetId}`)
        .then(response => response.json())
        .then(data => {
            if (data.success && data.data.length > 0) {
                // 查找最新的邻接矩阵文件
                const adjFiles = data.data.filter(f => f.type === 'adjacency_matrix');
                if (adjFiles.length > 0) {
                    // 按文件名排序，获取最新的文件
                    adjFiles.sort((a, b) => {
                        // 比较文件名中的时间戳
                        const timeA = parseInt(a.name.match(/\d+/g)[1]);
                        const timeB = parseInt(b.name.match(/\d+/g)[1]);
                        return timeB - timeA; // 降序排序，最新的在前面
                    });
                    
                    const latestFile = adjFiles[0];
                    // 触发下载
                    const link = document.createElement('a');
                    link.href = `http://localhost:3000/api/testdata/download/${latestFile.name}`;
                    link.download = latestFile.name;
                    link.style.visibility = 'hidden';
                    
                    document.body.appendChild(link);
                    link.click();
                    document.body.removeChild(link);
                    
                    showMessage('邻接矩阵保存成功', 'success');
                } else {
                    showMessage('未找到邻接矩阵文件', 'error');
                }
            } else {
                showMessage('获取文件列表失败: ' + (data.message || '未知错误'), 'error');
            }
        })
        .catch(error => {
            showMessage('保存邻接矩阵失败: ' + error.message, 'error');
        });
}

// 保存PNG
function savePng() {
    const datasetId = window.currentDatasetId;
    
    if (!datasetId) {
        showMessage('没有找到当前数据集ID', 'error');
        return;
    }
    
    // 从后端获取testdata中的网络图像文件
    fetch(`http://localhost:3000/api/testdata/files/${datasetId}`)
        .then(response => response.json())
        .then(data => {
            if (data.success && data.data.length > 0) {
                // 查找最新的网络图像文件
                const pngFiles = data.data.filter(f => f.type === 'network_graph');
                if (pngFiles.length > 0) {
                    // 按文件名排序，获取最新的文件
                    pngFiles.sort((a, b) => {
                        // 比较文件名中的时间戳
                        const timeA = parseInt(a.name.match(/\d+/g)[1]);
                        const timeB = parseInt(b.name.match(/\d+/g)[1]);
                        return timeB - timeA; // 降序排序，最新的在前面
                    });
                    
                    const latestFile = pngFiles[0];
                    // 触发下载
                    const link = document.createElement('a');
                    link.href = `http://localhost:3000/api/testdata/download/${latestFile.name}`;
                    link.download = latestFile.name;
                    link.style.visibility = 'hidden';
                    
                    document.body.appendChild(link);
                    link.click();
                    document.body.removeChild(link);
                    
                    showMessage('PNG文件保存成功', 'success');
                } else {
                    showMessage('未找到网络图像文件', 'error');
                }
            } else {
                showMessage('获取文件列表失败: ' + (data.message || '未知错误'), 'error');
            }
        })
        .catch(error => {
            showMessage('保存PNG文件失败: ' + error.message, 'error');
        });
}

// 显示分析图像
function displayAnalysisImage(base64Image) {
    const networkGraphDiv = document.getElementById('network-graph');
    if (!networkGraphDiv) return;
    
    // 清除现有的内容
    networkGraphDiv.innerHTML = '';
    
    // 创建图片元素
    const img = document.createElement('img');
    img.src = base64Image;
    img.style.maxWidth = '100%';
    img.style.maxHeight = '100%';
    img.style.objectFit = 'contain';
    
    // 添加到展示窗
    networkGraphDiv.appendChild(img);
}

// 显示分析完成弹窗
function showCompletionPopup(savePath, datasetId) {
    // 检查是否已经存在弹窗
    if (document.getElementById('completion-popup')) {
        return;
    }
    
    // 创建弹窗容器
    const popup = document.createElement('div');
    popup.id = 'completion-popup';
    popup.style.cssText = `
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-color: rgba(0, 0, 0, 0.5);
        display: flex;
        justify-content: center;
        align-items: center;
        z-index: 1000;
    `;
    
    // 弹窗内容
    const popupContent = document.createElement('div');
    popupContent.style.cssText = `
        background-color: white;
        padding: 20px;
        border-radius: 8px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        width: 400px;
        max-width: 90%;
    `;
    
    // 弹窗标题
    const title = document.createElement('h3');
    title.textContent = '分析完成';
    title.style.marginTop = '0';
    
    // 弹窗消息
    const message = document.createElement('p');
    message.textContent = `分析结果已成功保存到路径: ${savePath || 'testdata文件夹'}`;
    
    // 操作按钮
    const buttonContainer = document.createElement('div');
    buttonContainer.style.cssText = `
        display: flex;
        justify-content: flex-end;
        gap: 10px;
        margin-top: 20px;
    `;
    
    // 关闭按钮
    const closeBtn = document.createElement('button');
    closeBtn.textContent = '关闭';
    closeBtn.className = 'btn btn-primary';
    closeBtn.addEventListener('click', () => {
        popup.remove();
    });
    
    // 组装弹窗
    buttonContainer.appendChild(closeBtn);
    
    popupContent.appendChild(title);
    popupContent.appendChild(message);
    popupContent.appendChild(buttonContainer);
    popup.appendChild(popupContent);
    
    // 添加到页面
    document.body.appendChild(popup);
    
    // 点击弹窗外部关闭
    popup.addEventListener('click', (e) => {
        if (e.target === popup) {
            popup.remove();
        }
    });
}

// 显示消息
function showMessage(text, type) {
    // 检查是否已存在消息元素
    let messageElement = document.querySelector('.message');
    
    // 如果不存在，创建新元素
    if (!messageElement) {
        messageElement = document.createElement('div');
        messageElement.className = 'message';
        document.body.insertBefore(messageElement, document.body.firstChild);
    }
    
    // 设置消息内容和类型
    messageElement.textContent = text;
    messageElement.className = `message ${type}`;
    messageElement.style.display = 'block';
    
    // 3秒后隐藏消息
    setTimeout(() => {
        messageElement.style.display = 'none';
    }, 3000);
}