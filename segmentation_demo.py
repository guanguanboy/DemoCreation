import numpy as np
import time
import shutil

import torch

from PIL import Image
import cv2

import mmcv
import mmengine
from mmseg.apis import init_model, inference_model,show_result_pyplot
from mmseg.utils import register_all_modules
register_all_modules()

from mmseg.datasets import CityscapesDataset
# 获取 Cityscapes 街景数据集 类别名和调色板
from mmseg.datasets import cityscapes
classes = cityscapes.CityscapesDataset.METAINFO['classes']
palette = cityscapes.CityscapesDataset.METAINFO['palette']

# 设置配置文件和参数文件路径
#config_file = '/data/liguanlin/codes/mmsegmentation/configs/pspnet/pspnet_r50-d8_4xb2-40k_cityscapes-512x1024.py'
#checkpoint_file = '/data/liguanlin/codes/mmsegmentation/checkpoints/pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth'
config_file = '/data/liguanlin/codes/mmsegmentation/configs/segformer/segformer_mit-b5_8xb1-160k_cityscapes-1024x1024.py'
checkpoint_file = '/data/liguanlin/codes/mmsegmentation/checkpoints/segformer_mit-b5_8x1_1024x1024_160k_cityscapes_20211206_072934-87a052ec.pth'

model = init_model(config_file, checkpoint_file, device='cuda:0')

def predict_single_frame(img, opacity=0.2):
    
    result = inference_model(model, img)
    
    # 将类别标签显示在分割区域的正中央并对区域进行着色
    #img = mmcv.imread(img)
    #vis_img = overlay_mask(img, result, model.CLASSES)
    #colored_img = colorize(result, palette=model.PALETTE)

    # 将分割图按调色板染色
    #seg_map = np.array(result.pred_sem_seg.data[0].detach().cpu().numpy()).astype('uint8')
    #seg_img = Image.fromarray(seg_map).convert('P')
    #seg_img.putpalette(np.array(palette, dtype=np.uint8))
    
    #show_img = (np.array(seg_img.convert('RGB')))*(1-opacity) + img*opacity
    
    show_img = show_result_pyplot(model, img, result, show=False)
    return show_img


# input_video = 'data/traffic.mp4'

input_video = 'input/DJI_0286_enhanced.mp4'

temp_out_dir = '/data/liguanlin/codes/mmseg_inference/tmp_results/'
import os
if not os.path.exists(temp_out_dir):

    os.mkdir(temp_out_dir)

# 读入待预测视频
imgs = mmcv.VideoReader(input_video)

prog_bar = mmengine.ProgressBar(len(imgs))

# 对视频逐帧处理
for frame_id, img in enumerate(imgs):
    
    ## 处理单帧画面
    show_img = predict_single_frame(img, opacity=0.5)
    temp_path = f'{temp_out_dir}/{frame_id:06d}.jpg' # 保存语义分割预测结果图像至临时文件夹
    cv2.imwrite(temp_path, show_img)

    prog_bar.update() # 更新进度条

# 把每一帧串成视频文件
mmcv.frames2video(temp_out_dir, 'output/DJI_0286_enhanced_seged_by_segformer_pyplot.mp4', fps=imgs.fps, fourcc='mp4v')

shutil.rmtree(temp_out_dir) # 删除存放每帧画面的临时文件夹
print('删除临时文件夹', temp_out_dir)