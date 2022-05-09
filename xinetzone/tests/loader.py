from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms


def get_val_loader(batch_size=4,
                   valdir='/media/pc/data/4tb/zzy/zzy/imagenet_data/val'):
    # imagenet 数据集预处理方式, 来源 pytorch 官方
    trans = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )])

    # imagenet 验证集地址(5w张),
    # 论文里面关于imagenet的准确率指标都是指的这个验证集
    return DataLoader(datasets.ImageFolder(valdir, trans),
                      batch_size=batch_size, shuffle=False,
                      num_workers=4, pin_memory=True)
