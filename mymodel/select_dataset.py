def get_dataset(dataset):
    if dataset == 'BS': # 4866 条数据
        selected_data = 'Botswana'
        data_name = 'Botswana.mat'
        data_gt_name = 'Botswana_gt.mat'
        Xkey = 'Botswana' # .mat文件中选择的键
        ykey = 'Botswana_gt' # .mat文件中选择的键
        processed_data_depth = 60 # 存储模型使用的数据的光谱数（即depth）
        class_num = 14
        contribution_ratio = 0.9995
        validateNtest_ratio = 0.90 # 训练集，验证集，测试集中，测试集和验证集的占比
        test_ratio = 0.94 # 测试集、验证集中，测试集占比
    elif dataset == 'HU': # 15029 条数据
        selected_data = 'Houston'
        data_name = 'Houstondata.mat'
        data_gt_name = 'Houstonlabel.mat'
        Xkey = 'Houstondata' # .mat文件中选择的键
        ykey = 'Houstonlabel' # .mat文件中选择的键
        processed_data_depth = 10 # 存储模型使用的数据的光谱数（即depth）
        class_num = 15
        contribution_ratio = 0.9995
        validateNtest_ratio = 0.93 # 训练集，验证集，测试集中，测试集和验证集的占比
        test_ratio = 0.96 # 测试集、验证集中，测试集占比
    elif dataset == 'IP': # 10243 条数据
        selected_data = 'Indian_Pines'
        data_name = 'Indian_pines_corrected.mat'
        data_gt_name = 'Indian_pines_gt.mat'
        Xkey = 'indian_pines_corrected' # .mat文件中选择的键
        ykey = 'indian_pines_gt' # .mat文件中选择的键
        processed_data_depth = 80 # 存储模型使用的数据的光谱数（即depth）
        class_num = 16 # 类别数量
        contribution_ratio = 0.9995
        validateNtest_ratio = 0.93 # 训练集，验证集，测试集中，测试集和验证集的占比
        test_ratio = 0.96 # 测试集、验证集中，测试集占比
    elif dataset == 'KSC': # 4734 条数据
        selected_data = 'KSC'
        data_name = 'KSC.mat'
        data_gt_name = 'KSC_gt.mat'
        Xkey = 'KSC' # .mat文件中选择的键
        ykey = 'KSC_gt' # .mat文件中选择的键
        processed_data_depth = 95 # 存储模型使用的数据的光谱数（即depth）
        class_num = 13
        contribution_ratio = 0.9995
        validateNtest_ratio = 0.85 # 训练集，验证集，测试集中，测试集和验证集的占比
        test_ratio = 0.94 # 测试集、验证集中，测试集占比
    elif dataset == 'PU': # 42776 条数据
        selected_data = 'Pavia'
        data_name = 'PaviaU.mat'
        data_gt_name = 'PaviaU_gt.mat'
        Xkey = 'paviaU' # .mat文件中选择的键
        ykey = 'paviaU_gt' # .mat文件中选择的键
        processed_data_depth = 25 # 存储模型使用的数据的光谱数（即depth）
        class_num = 9
        contribution_ratio = 0.9995
        validateNtest_ratio = 0.97 # 训练集，验证集，测试集中，测试集和验证集的占比
        test_ratio = 0.98 # 测试集、验证集中，测试集占比
    elif dataset == 'SV': # 50616 条数据
        selected_data = 'Salinas'
        data_name = 'Salinas_corrected.mat'
        data_gt_name = 'Salinas_gt.mat'
        Xkey = 'salinas_corrected' # .mat文件中选择的键
        ykey = 'salinas_gt' # .mat文件中选择的键
        processed_data_depth = 9 # 存储模型使用的数据的光谱数（即depth）
        class_num = 16
        contribution_ratio = 0.9995
        validateNtest_ratio = 0.97 # 训练集，验证集，测试集中，测试集和验证集的占比
        test_ratio = 0.98 # 测试集、验证集中，测试集占比
    elif dataset == 'HC': # 267623 条数据
        selected_data = 'WHU-Hi-HanChuan'
        data_name = 'WHU_Hi_HanChuan.mat'
        data_gt_name = 'WHU_Hi_HanChuan_gt.mat'
        Xkey = 'WHU_Hi_HanChuan' # .mat文件中选择的键
        ykey = 'WHU_Hi_HanChuan_gt' # .mat文件中选择的键
        processed_data_depth = 17 # 存储模型使用的数据的光谱数（即depth）
        class_num = 16 # 类别数量
        contribution_ratio = 0.997
        validateNtest_ratio = 0.99 # 训练集，验证集，测试集中，测试集和验证集的占比
        test_ratio = 0.99 # 测试集、验证集中，测试集占比
    elif dataset == 'LK': # 204542 条数据
        selected_data = 'WHU-Hi-LongKou'
        data_name = 'WHU_Hi_LongKou.mat'
        data_gt_name = 'WHU_Hi_LongKou_gt.mat'
        Xkey = 'WHU_Hi_LongKou' # .mat文件中选择的键
        ykey = 'WHU_Hi_LongKou_gt' # .mat文件中选择的键
        processed_data_depth = 8 # 存储模型使用的数据的光谱数（即depth）
        class_num = 9 # 类别数量
        contribution_ratio = 0.9995
        validateNtest_ratio = 0.99 # 训练集，验证集，测试集中，测试集和验证集的占比
        test_ratio = 0.99 # 测试集、验证集中，测试集占比
    else:
        print("选择的数据集不在可以处理的范围内！")
        return None
    
    return selected_data, data_name, data_gt_name, Xkey, ykey, processed_data_depth, class_num, contribution_ratio, validateNtest_ratio, test_ratio
    