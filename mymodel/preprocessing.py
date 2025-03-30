import numpy as np
import scipy.io as sio
from sklearn.model_selection import train_test_split

import config
from pca import custom_pca

# 对单个像素周围提取 patch 时，边缘像素就无法取了，因此，给这部分像素进行 padding 操作
def padWithZeros(X, margin=12):
    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2* margin, X.shape[2]))
    x_offset = margin
    y_offset = margin
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X
    return newX

# 在每个像素周围提取 patch
def createImageCubes(X, y, windowSize=25, removeZeroLabels = True):
    margin = int((windowSize - 1) / 2)
    zeroPaddedX = padWithZeros(X, margin=margin)
    # 存储最终的 patches 数据和标签
    final_patches_data = []
    final_patches_labels = []
    for r in range(margin, zeroPaddedX.shape[0] - margin):
        for c in range(margin, zeroPaddedX.shape[1] - margin):
            patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]
            label = y[r - margin, c - margin]
            if removeZeroLabels:
                if label > 0:
                    # 只存储标签不为 0 的 patch 数据和标签
                    final_patches_data.append(patch)
                    final_patches_labels.append(label - 1)
            else:
                final_patches_data.append(patch)
                final_patches_labels.append(label)

    # 将列表转换为 numpy 数组
    patchesData = np.array(final_patches_data)
    patchesLabels = np.array(final_patches_labels)

    return patchesData, patchesLabels

def splitTrainTestSet(X, y, testRatio, randomState=config.randomState):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testRatio, random_state=randomState, stratify=y)
    return X_train, X_test, y_train, y_test

def preprocess_dataset(selected_data, data_name, data_gt_name, Xkey, ykey, processed_data_depth, validateNtest_ratio, test_ratio, contribution_ratio):
    print('dataset: ', selected_data)
    Xpath = config.all_data_dir + '/' + selected_data + '/' + data_name
    ypath = config.all_data_dir + '/' + selected_data + '/' + data_gt_name
    data = sio.loadmat(Xpath)[Xkey]
    y = sio.loadmat(ypath)[ykey]

    Xpca_path = config.all_data_dir + '/' + selected_data + '/' + 'X.npy'
    output_file_path = Xpca_path
    custom_pca(data, output_file_path, contribution_ratio)
    X = np.load(output_file_path)
    print(X.shape)
    print(y.shape)
    X_patch, y = createImageCubes(X, y, windowSize=config.patch_size, removeZeroLabels=True)
    Xtrain, XvalidateNtest, ytrain, yvalidateNtest = splitTrainTestSet(X_patch, y, validateNtest_ratio)
    Xvalidate, Xtest, yvalidate, ytest = splitTrainTestSet(XvalidateNtest, yvalidateNtest, test_ratio, randomState=678)
    print('Xtrain shape: ', Xtrain.shape)
    print('Xvali shape: ', Xvalidate.shape)
    print('Xtest  shape: ', Xtest.shape)

    # 改变 Xtrain, Ytrain 的形状
    Xtrain = Xtrain.reshape(-1, config.patch_size, config.patch_size, processed_data_depth, 1)
    Xvalidate = Xvalidate.reshape(-1, config.patch_size, config.patch_size, processed_data_depth, 1)
    Xtest  = Xtest.reshape(-1, config.patch_size, config.patch_size, processed_data_depth, 1)
    print('before transpose: Xtrain shape: ', Xtrain.shape) 
    print('before transpose: Xvalidate shape: ', Xvalidate.shape) 
    print('before transpose: Xtest  shape: ', Xtest.shape) 

    # 为了适应 pytorch 结构，数据要做 transpose
    Xtrain = Xtrain.transpose(0, 4, 1, 2, 3)
    Xvalidate = Xvalidate.transpose(0, 4, 1, 2, 3)
    Xtest  = Xtest.transpose(0, 4, 1, 2, 3)
    print('after transpose: Xtrain shape: ', Xtrain.shape) 
    print('after transpose: Xvalidate shape: ', Xvalidate.shape) 
    print('after transpose: Xtest  shape: ', Xtest.shape) 

    Xtrain_path = config.all_data_dir + '/' + selected_data + '/' + 'Xtrain.npy'
    ytrain_path = config.all_data_dir + '/' + selected_data + '/' + 'ytrain.npy'
    Xvalidate_path = config.all_data_dir + '/' + selected_data + '/' + 'Xvalidate.npy'
    yvalidate_path = config.all_data_dir + '/' + selected_data + '/' + 'yvalidate.npy'
    Xtest_path = config.all_data_dir + '/' + selected_data + '/' + 'Xtest.npy'
    ytest_path = config.all_data_dir + '/' + selected_data + '/' + 'ytest.npy'
    np.save(Xtrain_path, Xtrain)
    np.save(Xvalidate_path, Xvalidate)
    np.save(Xtest_path, Xtest)
    np.save(ytrain_path, ytrain)
    np.save(yvalidate_path, yvalidate)
    np.save(ytest_path, ytest)