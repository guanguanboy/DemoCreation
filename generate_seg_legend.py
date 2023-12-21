import numpy as np
import cv2
#from mmseg.datasets import CityscapesDataset

# 加载Cityscapes数据集的调色板
from mmseg.datasets import CityscapesDataset
# 获取 Cityscapes 街景数据集 类别名和调色板
from mmseg.datasets import cityscapes
classes = cityscapes.CityscapesDataset.METAINFO['classes']
palette = cityscapes.CityscapesDataset.METAINFO['palette']

# 创建一个空白图像，用于绘制图例
legend_img = np.zeros((200, 600, 3), dtype=np.uint8)

# 在图例上绘制每个类别的颜色块和标签
for i, color in enumerate(palette):
    start_x = i % 3 * 200
    start_y = i // 3 * 50
    end_x = start_x + 200
    end_y = start_y + 50

    # 绘制颜色块
    legend_img[start_y:end_y, start_x:end_x, :] = color

    # 绘制类别标签
    label = classes[i]
    text_pos = (start_x + 5, start_y + 25)
    cv2.putText(legend_img, label, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

# 显示和保存图例
#cv2.imshow('Cityscapes Legend', legend_img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
cv2.imwrite('./output/cityscapes_legend.png', legend_img)