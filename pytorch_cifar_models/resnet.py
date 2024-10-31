class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, layer_name=None, block_index=None):
        global write_count  # 在方法中声明为全局变量

        identity = x
        out = self.conv1(x)
        out = self.bn1(out)

        # 增加计数器并生成新的文件名，包含层和块的信息
        write_count += 1
        filename = f"{layer_name}_block{block_index}_data{write_count}.txt"
        # 将结果写入新文件，以逗号分隔，保留三位小数
        with open(filename, "w") as f:
            f.write(",".join(f"{value.item():.3f}" for value in out.flatten()))

        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        # 增加计数器并生成新的文件名，包含层和块的信息
        write_count += 1
        filename = f"{layer_name}_block{block_index}_data{write_count}.txt"
        # 将结果写入新文件，以逗号分隔，保留三位小数
        with open(filename, "w") as f:
            f.write(",".join(f"{value.item():.3f}" for value in out.flatten()))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class CifarResNet(nn.Module):

    def __init__(self, block, layers, num_classes=10):
        super(CifarResNet, self).__init__()
        self.inplanes = 16
        self.conv1 = conv3x3(3, 16)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        # 执行第一个 block，传递层和块的索引
        for i, block in enumerate(self.layer1):
            x = block(x, "layer1", i + 1)  # layer1的block索引从1开始
        for i, block in enumerate(self.layer2):
            x = block(x, "layer2", i + 1)  # layer2的block索引从1开始
        for i, block in enumerate(self.layer3):
            x = block(x, "layer3", i + 1)  # layer3的block索引从1开始

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
