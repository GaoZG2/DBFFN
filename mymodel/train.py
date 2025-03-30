from all_module import model_train
from select_dataset import get_dataset
from preprocessing import preprocess_dataset
import config

if __name__ == '__main__':
    dataset = input('Please input the name of Dataset(IP, PU, HU, BS, SV, HC, LK or KSC):')
    Dataset = dataset.upper()
    selected_data, data_name, data_gt_name, Xkey, ykey, processed_data_depth, class_num, contribution_ratio, validateNtest_ratio, test_ratio = get_dataset(Dataset)
    if config.is_preprocess_dataset == True:
        preprocess_dataset(selected_data, data_name, data_gt_name, Xkey, ykey, processed_data_depth, validateNtest_ratio, test_ratio, contribution_ratio)
    model_train(selected_data, processed_data_depth, class_num, validateNtest_ratio, test_ratio)