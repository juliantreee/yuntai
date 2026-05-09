import time
import os
import sys
import math
from media.sensor import *
from media.display import *
from media.media import *
from machine import FPIOA
from machine import Pin
from machine import UART
from machine import TOUCH

sensor = None
black = (0, 95)

# 物理世界矩形尺寸（厘米）
RECT_WIDTH_CM = 27.6
RECT_HEIGHT_CM = 19.1
CIRCLE_RADIUS_CM = 6.0

# 目标点坐标
TARGET_POINT = (161, 115)

# 全局状态变量
current_base_point_index = 0
base_point_counter = 0
detect_counter = 0
lost_counter = 0
min_detect_frames = 3
min_lost_frames = 6
flag_detected = False

# 阈值调节相关变量
adjusting_threshold = False
current_threshold = list(black)
touch_counter = 0
tp = None

def vector_angle_diff(v1, v2):
    dot = v1[0]*v2[0] + v1[1]*v2[1]
    det = v1[0]*v2[1] - v1[1]*v2[0]
    angle = math.atan2(det, dot) * (180 / math.pi)
    return abs(angle)

def get_line_intersection(line1, line2):
    (x1, y1), (x2, y2) = line1
    (x3, y3), (x4, y4) = line2

    A1 = y2 - y1
    B1 = x1 - x2
    C1 = A1 * x1 + B1 * y1

    A2 = y4 - y3
    B2 = x3 - x4
    C2 = A2 * x3 + B2 * y3

    det = A1 * B2 - A2 * B1

    if det == 0:
        return ((x1 + x3) / 2, (y1 + y3) / 2)
    else:
        x = (B2 * C1 - B1 * C2) / det
        y = (A1 * C2 - A2 * C1) / det
        return (x, y)

def calculate_perspective_circle(center, corners, radius_cm):
    width_px = math.sqrt((corners[1][0] - corners[0][0])**2 + (corners[1][1] - corners[0][1])**2)
    height_px = math.sqrt((corners[2][0] - corners[1][0])**2 + (corners[2][1] - corners[1][1])**2)

    px_per_cm_x = width_px / RECT_WIDTH_CM
    px_per_cm_y = height_px / RECT_HEIGHT_CM

    radius_x = radius_cm * px_per_cm_x
    radius_y = radius_cm * px_per_cm_y
    avg_radius = (radius_x + radius_y) / 2

    circle_points = []
    num_points = 16

    for i in range(num_points):
        angle = 2 * math.pi * i / num_points
        x = center[0] + radius_x * math.cos(angle)
        y = center[1] + radius_y * math.sin(angle)
        circle_points.append((int(x), int(y)))

    return circle_points, avg_radius

def sort_corners(corners, center):
    top_points = []
    bottom_points = []

    y_values = [p[1] for p in corners]
    median_y = sum(y_values) / len(y_values)

    for p in corners:
        if p[1] < median_y:
            top_points.append(p)
        else:
            bottom_points.append(p)

    if len(top_points) != 2 or len(bottom_points) != 2:
        angles = []
        for point in corners:
            dx = point[0] - center[0]
            dy = point[1] - center[1]
            angle = math.atan2(dy, dx)
            angles.append(angle)

        sorted_indices = sorted(range(len(angles)), key=lambda i: angles[i])
        sorted_corners = [corners[i] for i in sorted_indices]
        return sorted_corners

    top_points.sort(key=lambda p: p[0])
    bottom_points.sort(key=lambda p: p[0])

    top_left = top_points[0]
    top_right = top_points[1]

    vec1 = (top_right[0] - top_left[0], top_right[1] - top_left[1])
    vec2 = (bottom_points[0][0] - top_left[0], bottom_points[0][1] - top_left[1])
    cross = vec1[0] * vec2[1] - vec1[1] * vec2[0]

    if cross > 0:
        bottom_left = bottom_points[0]
        bottom_right = bottom_points[1]
    else:
        bottom_left = bottom_points[1]
        bottom_right = bottom_points[0]

    return [top_left, top_right, bottom_right, bottom_left]

def sending_data(flag, sign_dx_center, dx_center, sign_dy_center, dy_center,
                 base_index, sign_dx_base, dx_base, sign_dy_base, dy_base):
    global uart2

    if uart2 is None:
        return

    packet = [
        0xAA, 0x0D,
        flag, sign_dx_center, dx_center, sign_dy_center, dy_center,
        base_index, sign_dx_base, dx_base, sign_dy_base, dy_base
    ]

    checksum = sum(packet) & 0xFF

    try:
        uart2.write(bytes(packet) + bytes([checksum]))
    except:
        pass

def handle_threshold_adjustment():
    global adjusting_threshold, current_threshold, black, sensor

    min_val = current_threshold[0]
    max_val = current_threshold[1]

    print("阈值调节 - Min:", min_val, "Max:", max_val)

    while adjusting_threshold:
        try:
            img = sensor.snapshot()

            # 二值化预览
            binary_img = img.copy()
            binary_img = binary_img.to_grayscale()
            binary_img = binary_img.binary([(min_val, max_val)])

            # UI背景
            img.draw_rectangle(0, 0, 800, 480, color=(30, 30, 30), fill=True)

            # 使用 draw_string_advanced
            img.draw_string_advanced(250, 5, 30, "Threshold", color=(255, 255, 0))
            img.draw_string_advanced(30, 50, 25, "Min:" + str(min_val), color=(255, 255, 255))
            img.draw_string_advanced(500, 50, 25, "Max:" + str(max_val), color=(255, 255, 255))

            # Min按钮
            img.draw_rectangle(20, 100, 120, 60, color=(150, 50, 50), fill=True)
            img.draw_string_advanced(50, 115, 25, "Min-", color=(255, 255, 255))
            img.draw_rectangle(20, 180, 120, 60, color=(50, 150, 50), fill=True)
            img.draw_string_advanced(50, 195, 25, "Min+", color=(255, 255, 255))

            # Max按钮
            img.draw_rectangle(660, 100, 120, 60, color=(150, 50, 50), fill=True)
            img.draw_string_advanced(690, 115, 25, "Max-", color=(255, 255, 255))
            img.draw_rectangle(660, 180, 120, 60, color=(50, 150, 50), fill=True)
            img.draw_string_advanced(690, 195, 25, "Max+", color=(255, 255, 255))

            # 预览
            img.draw_image(binary_img, 200, 80)

            # 底部按钮
            img.draw_rectangle(200, 420, 150, 50, color=(100, 100, 150), fill=True)
            img.draw_string_advanced(235, 430, 25, "Ret", color=(255, 255, 255))
            img.draw_rectangle(450, 420, 150, 50, color=(100, 150, 100), fill=True)
            img.draw_string_advanced(485, 430, 25, "Save", color=(255, 255, 255))

            Display.show_image(img)

            # 触摸处理
            if tp:
                try:
                    points = tp.read()
                    if points and len(points) > 0:
                        tx = points[0].x
                        ty = points[0].y

                        if 20 <= tx <= 140 and 100 <= ty <= 160:
                            min_val = max(0, min_val - 1)
                            time.sleep_ms(150)
                        elif 20 <= tx <= 140 and 180 <= ty <= 240:
                            min_val = min(255, min_val + 1)
                            time.sleep_ms(150)
                        elif 660 <= tx <= 780 and 100 <= ty <= 160:
                            max_val = max(0, max_val - 1)
                            time.sleep_ms(150)
                        elif 660 <= tx <= 780 and 180 <= ty <= 240:
                            max_val = min(255, max_val + 1)
                            time.sleep_ms(150)
                        elif 200 <= tx <= 350 and 420 <= ty <= 470:
                            adjusting_threshold = False
                        elif 450 <= tx <= 600 and 420 <= ty <= 470:
                            current_threshold[0] = min_val
                            current_threshold[1] = max_val
                            black = (min_val, max_val)
                            adjusting_threshold = False
                            print("保存阈值:", black)
                except:
                    pass

            time.sleep_ms(30)

        except Exception as e:
            print("阈值调节错误:", e)
            adjusting_threshold = False

# ============ 主程序 ============
print("视觉跟踪程序启动")

# 1. 初始化媒体管理器
MediaManager.init()
time.sleep_ms(200)

# 2. 初始化FPIOA
fpioa = FPIOA()
fpioa.set_function(33, FPIOA.GPIO33)
fpioa.set_function(53, FPIOA.GPIO53)
fpioa.set_function(11, FPIOA.UART2_TXD)
fpioa.set_function(12, FPIOA.UART2_RXD)

# 3. 初始化UART
try:
    uart2 = UART(UART.UART2, 115200)
    print("UART2 OK")
except:
    uart2 = None

# 4. 初始化触摸屏
try:
    tp = TOUCH(0)
    print("触摸屏 OK")
except:
    tp = None
    print("触摸屏失败")

# 5. 初始化传感器
sensor = Sensor()
sensor.reset()
time.sleep_ms(500)
sensor.set_framesize(Sensor.QVGA)
sensor.set_pixformat(Sensor.RGB565)
time.sleep_ms(500)
print("摄像头 OK")

# 6. 初始化显示 - 使用to_ide=True
Display.init(Display.ST7701, width=800, height=480, to_ide=True)
time.sleep_ms(200)
print("显示 OK")

# 7. 启动摄像头
sensor.run()
time.sleep_ms(500)
print("摄像头运行中")

clock = time.clock()

# 存储上一帧信息
prev_min_corners = None
prev_center = None
prev_circle_points = None
prev_avg_radius = 0
prev_has_rect = False

print("进入主循环...")

while True:
    try:
        clock.tick()

        # 检测触摸长按
        if tp:
            try:
                points = tp.read()
                if points and len(points) > 0:
                    touch_counter += 1
                    if touch_counter > 30:
                        adjusting_threshold = True
                        current_threshold = list(black)
                        handle_threshold_adjustment()
                        touch_counter = 0
                else:
                    touch_counter = max(0, touch_counter - 1)
            except:
                touch_counter = 0

        if adjusting_threshold:
            time.sleep_ms(10)
            continue

        # 捕获图像
        img = sensor.snapshot()
        img_binary = img.copy()
        img_binary = img_binary.to_grayscale()
        img_binary = img_binary.binary([black])
        img_binary.dilate(1)

        rects = img_binary.find_rects(threshold=8000)

        min_rect = None
        min_area = float('inf')
        min_corners = None
        min_black_ratio = 0
        survivors = []

        if rects:
            for rect in rects:
                try:
                    corners = rect.corners()
                    if len(corners) != 4:
                        continue

                    angles = []
                    max_angle_error = 0
                    for i in range(4):
                        p0 = corners[(i-1) % 4]
                        p1 = corners[i]
                        p2 = corners[(i+1) % 4]
                        vec1 = (p0[0]-p1[0], p0[1]-p1[1])
                        vec2 = (p2[0]-p1[0], p2[1]-p1[1])
                        angle_error = abs(vector_angle_diff(vec1, vec2) - 90)
                        angles.append(angle_error)
                        if angle_error > max_angle_error:
                            max_angle_error = angle_error

                    if max_angle_error > 45 or sum(angles)/len(angles) > 30:
                        continue

                    current_area = rect.w() * rect.h()
                    if current_area < 5000:
                        continue

                    center = get_line_intersection([corners[0], corners[2]], [corners[1], corners[3]])
                    cx, cy = int(center[0]), int(center[1])

                    half = max(5, min(20, max(5, min(15, int(math.sqrt(current_area)/20)))) // 2)

                    xs = max(cx - half, 0)
                    xe = min(cx + half, img.width()-1)
                    ys = max(cy - half, 0)
                    ye = min(cy + half, img.height()-1)

                    vp, tp = 0, 0
                    for y in range(ys, ye):
                        for x in range(xs, xe):
                            try:
                                pv = img_binary.get_pixel(x, y)
                                if isinstance(pv, tuple):
                                    pv = pv[0]
                                if pv == 0:
                                    vp += 1
                                tp += 1
                            except:
                                continue

                    br = vp/tp if tp > 0 else 0
                    if br < 0.25:
                        continue

                    survivors.append((rect, corners, br))

                    if current_area < min_area:
                        min_area = current_area
                        min_corners = corners
                        min_black_ratio = br
                except:
                    continue

        if survivors:
            min_area = float('inf')
            min_corners = None
            for rect, corners, br in survivors:
                area = rect.w() * rect.h()
                if area < min_area:
                    min_area = area
                    min_corners = corners
                    min_black_ratio = br

        cur_has = min_corners is not None
        use_prev = False

        if not cur_has and prev_min_corners is not None:
            min_corners = prev_min_corners
            use_prev = True

        if cur_has:
            if not flag_detected:
                detect_counter += 1
                if detect_counter >= min_detect_frames:
                    flag_detected = True
                    detect_counter = 0
            else:
                lost_counter = 0
        else:
            if flag_detected:
                lost_counter += 1
                if lost_counter >= min_lost_frames:
                    flag_detected = False
                    lost_counter = 0
            else:
                detect_counter = 0

        flag_byte = 0xBB if flag_detected else 0xCC
        sign_dx_center = 0
        dx_center_val = 0
        sign_dy_center = 0
        dy_center_val = 0
        sign_dx_base = 0
        dx_base_val = 0
        sign_dy_base = 0
        dy_base_val = 0

        if min_corners is not None and flag_detected:
            if not use_prev:
                prev_min_corners = min_corners
                try:
                    for i in range(4):
                        x1, y1 = min_corners[i]
                        x2, y2 = min_corners[(i+1)%4]
                        img.draw_line(x1, y1, x2, y2, color=(0,255,0), thickness=2)
                except:
                    pass

            try:
                center = get_line_intersection([min_corners[0], min_corners[2]],
                                               [min_corners[1], min_corners[3]])
                cx, cy = int(center[0]), int(center[1])
                img.draw_circle(cx, cy, 2, color=(255,255,0), thickness=1)

                sorted_c = sort_corners(min_corners, (cx, cy))

                if not use_prev or (not prev_has_rect and cur_has):
                    cp, ar = calculate_perspective_circle((cx, cy), sorted_c, CIRCLE_RADIUS_CM)
                    prev_circle_points = cp
                    prev_avg_radius = ar
                    prev_center = (cx, cy)

                if prev_circle_points:
                    tv = max(5, int(prev_avg_radius/8))

                    for i in range(len(prev_circle_points)):
                        x1, y1 = prev_circle_points[i]
                        x2, y2 = prev_circle_points[(i+1)%len(prev_circle_points)]
                        img.draw_line(x1, y1, x2, y2, color=(0,255,0), thickness=2)

                    for i, pt in enumerate(prev_circle_points):
                        x, y = pt
                        c = (255,0,0) if i == current_base_point_index else (255,255,0)
                        img.draw_circle(x, y, 3, color=c, thickness=1)

                    cpt = prev_circle_points[current_base_point_index]

                    if abs(TARGET_POINT[0]-cpt[0]) < tv and abs(TARGET_POINT[1]-cpt[1]) < tv:
                        base_point_counter += 1
                    else:
                        base_point_counter = 0

                    if base_point_counter >= 2:
                        current_base_point_index = (current_base_point_index + 1) % 16
                        base_point_counter = 0

                    dx_c = TARGET_POINT[0] - cx
                    dy_c = TARGET_POINT[1] - cy
                    dx_b = TARGET_POINT[0] - cpt[0]
                    dy_b = TARGET_POINT[1] - cpt[1]

                    sign_dx_center = 1 if dx_c < 0 else 0
                    dx_center_val = min(255, abs(int(dx_c)))
                    sign_dy_center = 1 if dy_c < 0 else 0
                    dy_center_val = min(255, abs(int(dy_c)))
                    sign_dx_base = 1 if dx_b < 0 else 0
                    dx_base_val = min(255, abs(int(dx_b)))
                    sign_dy_base = 1 if dy_b < 0 else 0
                    dy_base_val = min(255, abs(int(dy_b)))
            except:
                pass

        if min_corners is None:
            prev_min_corners = None
            prev_center = None
            prev_circle_points = None

        prev_has_rect = cur_has

        # 使用 draw_string_advanced 显示信息
        try:
            img.draw_string_advanced(10, 10, 15, "FPS:" + str(clock.fps()), color=(255,0,0))
            status_str = "Tracking" if flag_detected else "Lost"
            status_color = (0,255,0) if flag_detected else (255,0,0)
            img.draw_string_advanced(10, 30, 15, status_str, color=status_color)
            img.draw_string_advanced(10, 50, 15, "TH:" + str(black), color=(0,0,255))
        except:
            pass

        sending_data(flag_byte, sign_dx_center, dx_center_val, sign_dy_center,
                    dy_center_val, current_base_point_index, sign_dx_base,
                    dx_base_val, sign_dy_base, dy_base_val)

        # 使用compressed_for_ide并传入正确的坐标
        try:
            img.compress_for_ide()
            Display.show_image(img, x=(800-320)//2, y=(480-240)//2)
        except:
            pass

    except KeyboardInterrupt:
        break
    except Exception as e:
        print("错误:", e)
        time.sleep_ms(100)

# 清理
print("清理...")
try:
    sensor.stop()
except:
    pass
try:
    Display.deinit()
except:
    pass
try:
    MediaManager.deinit()
except:
    pass
print("结束")
