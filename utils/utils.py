from PyQt5.QtGui import QImage, QPixmap, QIcon
from PIL import Image, ImageDraw, ImageFont

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
