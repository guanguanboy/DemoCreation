from mmdet.apis import init_detector, inference_detector
import mmcv
import mmengine


from PIL import Image
import cv2
import shutil

# 指定模型的配置文件和 checkpoint 文件路径
config_file = '/data/liguanlin/codes/mmdetection/configs/yolox/yolox_x_8x8_300e_coco.py'
checkpoint_file = '/data/liguanlin/codes/mmdetection/yolox_x_8x8_300e_coco_20211126_140254-1ef88d67.pth'
#config_file = '/data/liguanlin/codes/mmdetection/configs/dino/dino-5scale_swin-l_8xb2-36e_coco.py'
#checkpoint_file = '/data/liguanlin/codes/mmdetection/yolov3_mobilenetv2_320_300e_coco_20210719_215349-d18dff72.pth'
#config_file = '/data/liguanlin/codes/codes_from_github/mmdetection/configs/deformable_detr/deformable-detr-refine-twostage_r50_16xb2-50e_coco.py'
#checkpoint_file = '/data/liguanlin/codes/mmdetection/deformable-detr-refine-twostage_r50_16xb2-50e_coco_20221021_184714-acc8a5ff.pth'

# 根据配置文件和 checkpoint 文件构建模型
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# 测试单张图片并展示结果
#img = 'test.jpg'  # 或者 img = mmcv.imread(img)，这样图片仅会被读一次
#result = inference_detector(model, img)
# 在一个新的窗口中将结果可视化
#model.show_result(img, result)
# 或者将可视化结果保存为图片
#model.show_result(img, result, out_file='result.jpg')

temp_out_dir = '/data/liguanlin/codes/mmseg_inference/tmp_results/'
import os
if not os.path.exists(temp_out_dir):

    os.mkdir(temp_out_dir)

# 测试视频并展示结果
video = mmcv.VideoReader('input/DJI_0286_enhanced.mp4')
# 读入待预测视频
#imgs = mmcv.VideoReader(input_video)

prog_bar = mmengine.ProgressBar(len(video))

# 对视频逐帧处理
for frame_id, img in enumerate(video):
    
    ## 处理单帧画面
    result = inference_detector(model, img)
    temp_path = f'{temp_out_dir}/{frame_id:06d}.jpg' # 保存语义分割预测结果图像至临时文件夹
    model.show_result(img, result, wait_time=1,out_file=temp_path,thickness=3,bbox_color=(255,0,0))

    #cv2.imwrite(temp_path, show_img)

    prog_bar.update() # 更新进度条

"""
for frame in video:
    result = inference_detector(model, frame)
    model.show_result(frame, result, wait_time=1,out_file=)
"""
# 把每一帧串成视频文件
mmcv.frames2video(temp_out_dir, 'output/DJI_0286_enhanced_detected_yolox_new_red.mp4', fps=video.fps, fourcc='mp4v')

shutil.rmtree(temp_out_dir) # 删除存放每帧画面的临时文件夹
print('删除临时文件夹', temp_out_dir)