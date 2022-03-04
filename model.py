import torch.nn as nn
import torch.nn.functional as F
import torch

class CNN(nn.Module):

    def __init__(self,num_classes=4):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 25, kernel_size=(1,20))
        self.conv2 = nn.Conv1d(25, 25, kernel_size=(22,1))
        self.conv3 = nn.Conv1d(1, 50, kernel_size=(25,20))
        self.conv4 = nn.Conv1d(1, 100, kernel_size=(50,20))
        self.maxpool1 = nn.MaxPool1d(5)
        self.maxpool2 = nn.MaxPool1d(4)
        self.maxpool3 = nn.MaxPool1d(3)
        self.linear1 = nn.Linear(800,200)
        self.linear2 = nn.Linear(200,4)

    def forward(self, x):
        # x = torch.flatten(x, 1).float()
        x = torch.unsqueeze(x.float(),dim=1)
        # 32,1,22,1000
        x = self.conv1(x)
        # 32,25,22,991
        x = self.conv2(x)
        # 32,25,1,991
        x = torch.squeeze(x,dim=2)
        x = self.maxpool1(x)
        x = torch.unsqueeze(x,dim=1)
        # 32, 1, 25, 198
        x = self.conv3(x)
        # 32, 50, 1, 189
        x = torch.squeeze(x,dim=2)
        x = self.maxpool2(x)
        x = torch.unsqueeze(x,dim=1)
        # 32, 1, 50, 47
        x = self.conv4(x)
        # 32, 100, 1, 98
        x = torch.squeeze(x,dim=2)
        x = self.maxpool3(x)
        x = torch.unsqueeze(x,dim=1)
        # 32, 1, 100, 8
        x = torch.squeeze(x,dim=1)
        x = torch.flatten(x,start_dim=1)
        x = self.linear1(x)
        x = self.linear2(x) 
        return x