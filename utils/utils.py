from PyQt5.QtGui import QImage, QPixmap, QIcon
from PIL import Image, ImageDraw, ImageFont
import json
import os
from PIL import Image, ImageFont, ImageDraw
from utils.move_pcd import move_pcd_with_xyzrpy
import numpy as np


def pil2qicon(char):
    pil_image = create_char_image(char[0])
    data = pil_image.tobytes("raw", "RGBA")
    qimage = QImage(data, pil_image.width, pil_image.height, QImage.Format_RGBA8888)
    qpixmap = QPixmap.fromImage(qimage)
    return QIcon(qpixmap)


def create_char_image(char, size=(32, 32)):
    image = Image.new('RGBA', size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(image)
    for i in range(2):  # 最外两圈
        draw.rectangle([i, i, size[0] - 1 - i, size[1] - 1 - i], outline=(0, 0, 0, 255))  # 黑色边框
    font = ImageFont.load_default(size=20)
    # 使用 getbbox 计算文本的边界框
    bbox = draw.textbbox((0, 0), char, font=font)
    text_width = bbox[2] - bbox[0]  # 文本宽度
    text_height = bbox[3] - bbox[1]  # 文本高度
    # 计算字符的位置，使其在除去边框的区域中居中
    # 边框占用了最外两圈像素，因此有效区域为 (2, 2) 到 (30, 30)
    inner_size = (size[0] - 4, size[1] - 4)  # 减去边框的宽度
    text_x = (inner_size[0] - text_width) // 2 - bbox[0] + 2  # 调整 x 坐标
    text_y = (inner_size[1] - text_height) // 2 - bbox[1] + 2  # 调整 y 坐标

    draw.text((text_x, text_y), char, font=font, fill=(0, 0, 0, 255))  # 黑色字符

    return image

def text_3d(text, density=10, font= os.path.join(os.path.dirname(os.path.abspath(__file__)),'../icons/fengguangming.ttf'), font_size=10):
    font_obj = ImageFont.truetype(os.path.join(os.path.dirname(os.path.abspath(__file__)),'../icons/fengguangming.ttf'), int(font_size * density))
    font_dim = font_obj.getbbox(text)
    text_width = int((font_dim[2] - font_dim[0]))  # 计算宽度
    text_height = int((font_dim[3] - font_dim[1])*1.2)  # 计算高度
    img = Image.new('RGB', (text_width, text_height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((0, 0), text, font=font_obj, fill=(0, 0, 0))
    img = np.asarray(img)
    img_mask = img[:, :, 0] < 128
    indices = np.indices([*img.shape[0:2], 1])[:, img_mask, 0].reshape(3, -1).T
    indices = move_pcd_with_xyzrpy(indices,[0,-text_width/2,text_height/2,0,90,0],degrees=True)
    points = indices / 10 / density
    return points

def load_json(path):
    with open(path, 'r', encoding='UTF-8') as f:
        json_datas = json.loads(f.read())
    return json_datas

def wataprint(content,type):
    color_map = {
        "r": "\x1b[31m{}\x1b[0m",
        "rr": "\x1b[1;31m{}\x1b[0m",
        "r_": "\x1b[4;31m{}\x1b[0m",
        "rr_": "\x1b[1;4;31m{}\x1b[0m",
        "rx": "\x1b[3;31m{}\x1b[0m",
        "rx_": "\x1b[3;4;31m{}\x1b[0m",
        "rrx": "\x1b[1;3;31m{}\x1b[0m",
        "rrx_": "\x1b[1;3;4;31m{}\x1b[0m",
        
        "g": "\x1b[32m{}\x1b[0m",
        "gg": "\x1b[1;32m{}\x1b[0m",
        "g_": "\x1b[4;32m{}\x1b[0m",
        "gg_": "\x1b[1;4;32m{}\x1b[0m",
        
        "y": "\x1b[33m{}\x1b[0m",
        "yy": "\x1b[1;33m{}\x1b[0m",
        "y_": "\x1b[4;33m{}\x1b[0m",
        "yy_": "\x1b[1;4;33m{}\x1b[0m",
        
        "p": "\x1b[35m{}\x1b[0m",
        "pp": "\x1b[1;35m{}\x1b[0m",
        "p_": "\x1b[4;35m{}\x1b[0m",
        "pp_": "\x1b[1;4;35m{}\x1b[0m",
        
        "b": "\x1b[36m{}\x1b[0m",
        "bb": "\x1b[1;36m{}\x1b[0m",
        "b_": "\x1b[4;36m{}\x1b[0m",
        "bb_": "\x1b[1;4;36m{}\x1b[0m",
    }
    print(color_map[type].format(content))