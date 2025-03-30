# 数据地址
all_data_dir = '../data'

# 数据预处理设置
is_preprocess_dataset = True # 是否需要预处理数据，处理过一次之后，后续再使用不用再预处理
patch_size = 15
randomState=345 #划分训练集、验证集、测试集的随机参数
width = patch_size # patch的宽
height = patch_size # patch的高

# 和训练相关的参数
submodel_num = 2
epoch_num_min = 150
epoch_num_max = 150
batch_size = 128
learning_rate = 0.0001
kernel_spatial_list = [5, 7, 9, 11]
net_num = 2 # 训练的模型数量，使用验证集选出准确率最好的来进行测试集的测试
se_ratio = 16 # spectral_attention 模块中用于控制网络SE参数
gpu_used = True # 若 GPU 不可用，则该参数无效
print_epoch_loss = True # 是否每个轮次结束后打印损失
record_infos_lossNaccuracy = False # 是否记录训练过程中训练集和验证集的损失和准确率

# 模型保存
model_dir_path = './models'
model_list = ['model1', 'model2']