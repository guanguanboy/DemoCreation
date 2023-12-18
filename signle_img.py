# Check Pytorch installation
import torch, torchvision
print(torch.__version__, torch.cuda.is_available())

# Check MMSegmentation installation
import mmseg
print(mmseg.__version__)


# 设置配置文件和参数文件路径
config_file = '/data/liguanlin/codes/mmsegmentation/configs/pspnet/pspnet_r50-d8_4xb2-40k_cityscapes-512x1024.py'
checkpoint_file = '/data/liguanlin/codes/mmsegmentation/checkpoints/pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth'

# 使用 Python API 构建模型
from mmseg.apis import init_model
model = init_model(config_file, checkpoint_file, device='cuda:0')

# 准备图像
from PIL import Image
Image.open('demo/demo.png')

# 调用 Python API，使用预训练模型完成推理任务
from mmseg.apis import inference_model

img = 'demo/demo.png'
result = inference_model(model, img)


from mmseg.apis import show_result_pyplot

show_result_pyplot(model, img, result,  0.5, 'cityscapes',out_file='output/psp_city.jpg')
