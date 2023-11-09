# 导入torchattacks库
import sys
import torchattacks
import matplotlib.pyplot as plt
from torchvision import models
import torchvision.transforms as transforms
import torchvision.datasets as datasets

sys.path.insert(0, '..')

transform = transforms.Compose([
    transforms.Resize((224, 224)), # 缩放图像到224x224
    transforms.ToTensor(), # 转换为Tensor对象
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # 归一化到0-1之间
])

# 读取images文件夹中的图像文件，并且应用transform对象
dataset = datasets.ImageFolder(root='./data/val', transform=transform)

# 获取第一张图像和它的标签
image, label = dataset[0]

# 定义一个PyTorch模型，例如ResNet
model = models.resnet18(pretrained=True).to('cpu').eval()

# 定义一个对抗性攻击，例如PGD
atk = torchattacks.PGD(model, eps=8/255, alpha=2/255, steps=4)

# 读取一张图像
# image = ...

# 生成对抗性样本
adv_image = atk(image)

# 显示对抗性样本
plt.imshow(adv_image)
plt.show()
