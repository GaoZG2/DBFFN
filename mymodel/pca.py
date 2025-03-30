import numpy as np
from sklearn.decomposition import PCA

def custom_pca(data, output_file_path, contribution_ratio):
    print('data shape: ', data.shape)
    # 将数据重塑为二维数组，每行代表一个样本，每列代表一个特征
    reshaped_data = data.reshape(data.shape[0] * data.shape[1], data.shape[2])
    print('reshaped_data shape: ', reshaped_data.shape)

    # 进行PCA分析
    pca = PCA(n_components=contribution_ratio)
    new_data = pca.fit_transform(reshaped_data)
    print('new_data before shape: ', new_data.shape)

    # 获取实际保留的主成分数量
    num_components = pca.n_components_

    # 将数据恢复到原始形状
    new_data = new_data.reshape(data.shape[0], data.shape[1], num_components)
    print('new_data after shape: ', new_data.shape)

    np.save(output_file_path, new_data)
    # 计算累计贡献率
    cumulative_contribution_ratio = np.cumsum(pca.explained_variance_ratio_)
    print(f"保留的主成分数量: {num_components}")
    print(f"保留的主成分的累计贡献率: {cumulative_contribution_ratio[num_components - 1]}")