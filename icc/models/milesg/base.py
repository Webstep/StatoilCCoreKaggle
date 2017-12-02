# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F


class GangstaNetBase(nn.Module):

    def __init__(self):
        super(GangstaNetBase, self).__init__()

        torch.manual_seed(0)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Input 1 branch
        self.input1_layer1_conv2d = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=7, padding=5)
        self.input1_layer2_conv2d = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5, padding=3)
        self.input1_layer3_conv2d = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.input1_layer4_fc = nn.Linear(in_features=3200, out_features=512)

        # Input 2 branch
        self.input2_layer1_conv2d = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=7, padding=5)
        self.input2_layer2_conv2d = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5, padding=3)
        self.input2_layer3_conv2d = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.input2_layer4_fc = nn.Linear(in_features=3200, out_features=512)

        # Input 3 branch "inc_angle"
        self.input3_layer1_fc1 = nn.Linear(in_features=2, out_features=24)

        # Concatenated branch, fully connected stream
        self.fc1 = nn.Linear(in_features=1048, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=512)
        self.fc3 = nn.Linear(in_features=512, out_features=1)

    def forward(self, input1, input2, input3):
        input1 = self.pool(F.relu(self.input1_layer1_conv2d(input1)))
        input1 = self.pool(F.relu(self.input1_layer2_conv2d(input1)))
        input1 = self.pool(F.relu(self.input1_layer3_conv2d(input1)))
        input1 = F.relu(self.input1_layer4_fc(input1.view(-1, 3200)))

        input2 = self.pool(F.relu(self.input2_layer1_conv2d(input2)))
        input2 = self.pool(F.relu(self.input2_layer2_conv2d(input2)))
        input2 = self.pool(F.relu(self.input2_layer3_conv2d(input2)))
        input2 = F.relu(self.input2_layer4_fc(input2.view(-1, 3200)))

        input3 = self.input3_layer1_fc1(input3)

        combined = torch.cat((input1, input2, input3), 1)

        combined = (self.fc1(combined))
        combined = (self.fc2(combined))
        combined = F.sigmoid(self.fc3(combined))

        return combined