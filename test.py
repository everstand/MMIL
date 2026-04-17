import torch
import open_clip

# 加载你指定的模型
model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='dfn2b')

# 伪造一张输入图片
dummy_image = torch.randn(1, 3, 224, 224)

# 提取图像和文本特征
with torch.no_grad():
    image_features = model.encode_image(dummy_image)
    
print(f"图像对齐特征维度: {image_features.shape[-1]}") 
# 终端一定会打印: 图像对齐特征维度: 768