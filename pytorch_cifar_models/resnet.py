import torch
import torchvision
import torchvision.transforms as transforms

# 定义数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

# 加载 CIFAR-10 测试集
full_testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                             download=True, transform=transform)

# 创建测试集的 DataLoader
testloader = torch.utils.data.DataLoader(full_testset, batch_size=4,
                                         shuffle=False, num_workers=2)


# 加载预训练模型
model = torch.hub.load("sdudyl/pytorch-cifar-models", "cifar10_resnet20", pretrained=True, force_reload=True)
#model = torch.hub.load("/home/dyl/241114/pytorch-cifar-models/", "cifar10_resnet20", pretrained=True, force_reload=True)
model.eval()


# 评估模型的正确性，只测试前1000张图片
correct = 0
total = 0
max_images = 1000  # 设定要测试的图片数量上限

with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # 检查已处理图片是否达到上限
        if total >= max_images:
            total = max_images  # 确保总数精确为1000
            break

accuracy = 100 * correct / total
print(f'Accuracy of the model on the first {max_images} CIFAR-10 test images: {accuracy:.2f}%')



