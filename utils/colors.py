from matplotlib.colors import LinearSegmentedColormap

def get_colors_16():
    colors_16 = [(0, 0, 255),     # 蓝色
                (0, 255, 0),      # 绿色
                (255, 255, 0),    # 黄色
                (255, 0, 0),      # 红色
                (0, 255, 255),    # 青色
                (128, 0, 128),    # 紫色
                (255, 165, 0),    # 橙色
                (255, 192, 203),  # 粉色
                (165, 42, 42),    # 棕色
                (128, 128, 128),  # 灰色
                (0, 0, 0),        # 黑色
                (255, 255, 255),  # 白色
                (255, 0, 127),    # 玫瑰红
                (127, 255, 0),    # 柠檬绿
                (0, 191, 255),    # 天蓝色
                (238, 130, 238)]   # 紫罗兰色
    return colors_16

def get_class_map():
    class_map ={
          'Car': (255, 165, 0,255),
          'Van': (255, 0, 255,255),
          'Bus': (255, 0, 255,255),
          'Truck': (0, 255, 255,255),
          'Semitrailer': (165, 42, 42,255),
          'Special_vehicles':  (135, 206, 235,255),
          'Special_ vehicles': (135, 206, 235,255),
          'Tricyclist': (51, 161, 201,255),
          'Cycle': (0,255,0,255),
          'Cyclist':(0,255,0,255),
          'Pedestrian': (255,0,0,255),
          'Animal': (128,128,128,255),
          'others': (255, 255, 255,255),
    }
    return class_map

def get_bgyord_bar():
    colors = [(0, 'lightblue'), (0.2, 'blue'), (0.4, 'green'), (0.7, 'yellow'), (0.9, 'orange'), (0.95, 'red'), (1, 'darkred')]
    bgyord = LinearSegmentedColormap.from_list('blue_green_yellow_orange_red_darkred', colors, N=256)
    return bgyord