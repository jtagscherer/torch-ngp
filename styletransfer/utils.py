from PIL.Image import Image
from torchvision import transforms
import torch.nn.functional as F

data_transform = transforms.Compose([
    transforms.RandomResizedCrop(256, scale=(256 / 480, 1), ratio=(1, 1)),
    transforms.ToTensor(),
])


def load_style_image():
    ori_style_img = Image.open('../inputdata/style.jpg').convert('RGB')
    return data_transform(ori_style_img)


def get_style_loss(style_feats, transformed_feats, style_weight=15):
    return F.mse_loss(transformed_feats, style_feats) * style_weight


def get_content_loss(content_feat, transformed_feat, content_weight=1):
    return F.mse_loss(transformed_feat, content_feat) * content_weight
