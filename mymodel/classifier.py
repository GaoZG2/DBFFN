import torch.nn as nn
import torch.nn.functional as F

class Classifier(nn.Module):
    def __init__(self, class_num, width, height, depth):
        super(Classifier, self).__init__()
        self.length = height * width * depth
        self.fc1 = nn.Linear(self.length, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, class_num)
        self.dropout = nn.Dropout(p = 0.4)
    
    def forward(self, x):
        batch_size, _, width, height, depth = x.size()
        assert self.length == width * height * depth, "输入张量的 width * height * depth 与模块初始化时 self.length 不一致"
        out = x.reshape(batch_size,-1)
        out = F.relu(self.dropout(self.fc1(out)))
        out = F.relu(self.dropout(self.fc2(out)))
        out = self.fc3(out)
        return out