from mmdet.apis import init_detector, inference_detector,DetInferencer
import mmcv
import mmengine
import pickle

from PIL import Image
import cv2
import shutil

# 指定模型的配置文件和 checkpoint 文件路径
#config_file = '/data/liguanlin/codes/mmdetection/configs/yolox/yolox_x_8x8_300e_coco.py'
#checkpoint_file = '/data/liguanlin/codes/mmdetection/yolox_x_8x8_300e_coco_20211126_140254-1ef88d67.pth'
#config_file = '/data1/liguanlin/codes/codes_from_github/mmdetection/configs/yolo/yolov3_mobilenetv2_8xb24-320-300e_coco.py'
#checkpoint_file = '/data1/liguanlin/codes/codes_from_github/mmdetection/checkpoints/yolov3_mobilenetv2_320_300e_coco_20210719_215349-d18dff72.pth'
config_file = '/data1/liguanlin/codes/codes_from_github/mmdetection/configs/dino/dino-5scale_swin-l_8xb2-36e_coco.py'
checkpoint_file = '/data1/liguanlin/codes/codes_from_github/mmdetection/checkpoints/dino-5scale_swin-l_8xb2-36e_coco-5486e051.pth'

# 根据配置文件和 checkpoint 文件构建模型
model = DetInferencer(config_file, checkpoint_file, device='cuda:0')

# 测试单张图片并展示结果
#img = 'test.jpg'  # 或者 img = mmcv.imread(img)，这样图片仅会被读一次
#result = inference_detector(model, img)
# 在一个新的窗口中将结果可视化
#model.show_result(img, result)
# 或者将可视化结果保存为图片
#model.show_result(img, result, out_file='result.jpg')

temp_out_dir = './tmp_results/'
import os
if not os.path.exists(temp_out_dir):

    os.mkdir(temp_out_dir)

# 测试视频并展示结果
#video = mmcv.VideoReader('input/DJI_0286.MP4')
video = mmcv.VideoReader('input/DJI_0286_enhanced.mp4')
# 读入待预测视频
#imgs = mmcv.VideoReader(input_video)

prog_bar = mmengine.ProgressBar(len(video))

detected_person_list = []
detected_car_list = []

# 对视频逐帧处理
for frame_id, img in enumerate(video):
    
    ## 处理单帧画面
    temp_path = f'{temp_out_dir}/{frame_id:06d}.jpg' # 保存语义分割预测结果图像至临时文件夹
    result = model(img,out_dir=temp_out_dir)

    #model.show_result(img, result, wait_time=1,out_file=temp_path,thickness=3,bbox_color=(255,0,0))

    #cv2.imwrite(temp_path, show_img)

    prog_bar.update() # 更新进度条

    pred_labels_list = result['predictions'][0]['labels']
    #print(pred_labels_list)

    #统计人的个数
    detected_person_count = pred_labels_list.count(1)
    detected_person_list.append(detected_person_count)
    #统计车的个数
    detected_car_count = pred_labels_list.count(3) + pred_labels_list.count(6)
    detected_car_list.append(detected_car_count)


"""
for frame in video:
    result = inference_detector(model, frame)
    model.show_result(frame, result, wait_time=1,out_file=)
"""
# 把每一帧串成视频文件
temp_image_save_dir = temp_out_dir + 'vis/'
#mmcv.frames2video(temp_image_save_dir, 'output/DJI_0286_enhanced_detected_dino.mp4', fps=video.fps, fourcc='mp4v',filename_tmpl='{:08d}.jpg')
#mmcv.frames2video(temp_image_save_dir, 'output/DJI_0286_detected_yolov3.mp4', fps=video.fps, fourcc='mp4v',filename_tmpl='{:08d}.jpg')

shutil.rmtree(temp_image_save_dir) # 删除存放每帧画面的临时文件夹
print('删除临时文件夹', temp_image_save_dir)

#detected_person_list_path = 'output/DJI_0286_detected_person_list_yolov3.pkl'
detected_person_list_path = 'output/DJI_0286_enhanced_detected_person_list_dino.pkl'

with open(detected_person_list_path, 'wb') as file:
    pickle.dump(detected_person_list, file)

#detected_car_list_path = 'output/DJI_0286_detected_car_list_yolov3.pkl'
detected_car_list_path = 'output/DJI_0286_enhanced_detected_car_list_dino.pkl'

with open(detected_car_list_path, 'wb') as file:
    pickle.dump(detected_car_list, file)