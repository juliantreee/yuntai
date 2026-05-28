import time
import os
import sys
import math
from media.sensor import *
from media.display import *
from media.media import *
from time import ticks_ms
from machine import FPIOA
from machine import Pin
from machine import Timer
from machine import UART
from machine import PWM

x_motor_dir = Pin(6, Pin.OUT)
y_motor_dir = Pin(46, Pin.OUT)

def x_motor_speed(rpm):
    if rpm > 0:
        x_motor_dir.value(1)
    elif rpm < 0:
        x_motor_dir.value(0)
    freq = int(abs(rpm) * 3200 / 60)
    if freq == 0:
        freq = 1
    pwm_x = PWM(Pin(47), freq=freq, duty=50)

def y_motor_speed(rpm):
    if rpm > 0:
        y_motor_dir.value(1)
    elif rpm < 0:
        y_motor_dir.value(0)
    freq = int(abs(rpm) * 3200 / 60)
    if freq == 0:
        freq = 1
    pwm_y = PWM(Pin(42), freq=freq, duty=50)


'''PID类，从我写的C移植过来的'''
class PID:
    def __init__(self, kp, ki, kd, ctl_max, ctl_min, dt):
        self.Kp = kp
        self.Ki = ki
        self.Kd = kd
        self.error = 0.0
        self.ierror = 0.0
        self.dvalue = 0.0
        self.now_value = 0.0
        self.last_value = 0.0
        self.target_value = 0.0
        self.ctl_value = 0.0
        self.ctl_max = ctl_max
        self.ctl_min = ctl_min
        self.dt = dt
        self.first = True

    def clear(self):
        self.last_value = 0.0
        self.ierror = 0.0
        self.first = True

    def set_limit(self, max_val, min_val):
        self.ctl_max = max_val
        self.ctl_min = min_val

    def step(self, value, target):
        if self.first:
            self.last_value = value
            self.first = False
        self.now_value = value
        self.target_value = target
        self.error = self.target_value - self.now_value
        self.ierror += self.error * self.dt
        self.dvalue = (self.now_value - self.last_value) / self.dt
        self.ctl_value = (self.Kp * self.error +
                          self.Ki * self.ierror -
                          self.Kd * self.dvalue)
        if self.ctl_value > self.ctl_max:
            self.ctl_value = self.ctl_max
            self.ierror -= self.error * self.dt
        if self.ctl_value < self.ctl_min:
            self.ctl_value = self.ctl_min
            self.ierror -= self.error * self.dt
        self.last_value = self.now_value
        return self.ctl_value

sensor = None
blue = 90, 100, -11, 10, -60, 43
black = (0,95)

# 物理世界矩形尺寸（厘米）
RECT_WIDTH_CM = 27.6
RECT_HEIGHT_CM = 19.1
CIRCLE_RADIUS_CM = 6.0

# 目标点坐标 (160, 116) - 图像中心
TARGET_POINT = (135, 96)

# 全局状态变量
current_base_point_index = 0  # 当前识别的圆形基准点编号
base_point_counter = 0        # 满足条件帧数计数
detect_counter = 0            # 矩形识别计数器
lost_counter = 0              # 矩形丢失计数器
min_detect_frames = 3         # 连续检测阈值
min_lost_frames = 6           # 连续丢失阈值
flag_detected = False         # 滤波后的检测状态

# PID控制器 - X轴和Y轴
PID_DT = 0.033
pid_x = PID(kp=6, ki=3, kd=0.06, ctl_max=3000, ctl_min=-3000, dt=PID_DT)
pid_y = PID(kp=3.5, ki=3, kd=0.06, ctl_max=3000, ctl_min=-3000, dt=PID_DT)

def vector_angle_diff(v1, v2):
    """计算两个向量之间的角度差（单位：度）"""
    dot = v1[0]*v2[0] + v1[1]*v2[1]
    det = v1[0]*v2[1] - v1[1]*v2[0]
    angle = math.atan2(det, dot) * (180 / math.pi)
    return abs(angle)

def get_line_intersection(line1, line2):
    """计算两条直线的交点"""
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
    """计算考虑透视失真的圆点"""
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
    """优化角点排序逻辑"""
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
    """串口数据包发送函数"""
    global uart2

    PACKET_LENGTH = 0x0D

    packet = [
        0xAA,
        PACKET_LENGTH,
        flag,
        sign_dx_center,
        dx_center,
        sign_dy_center,
        dy_center,
        base_index,
        sign_dx_base,
        dx_base,
        sign_dy_base,
        dy_base
    ]

    checksum = sum(packet) & 0xFF
    uart2.write(bytes(packet) + bytes([checksum]))
    #print(bytes(packet) + bytes([checksum]))

try:
    flag_key = 0
    print("camera_test")
    fpioa = FPIOA()
    fpioa.set_function(33, FPIOA.GPIO33)
    fpioa.set_function(53, FPIOA.GPIO53)
    fpioa.set_function(11, FPIOA.UART2_TXD)
    fpioa.set_function(12, FPIOA.UART2_RXD)
    pin = Pin(33, Pin.OUT)
    pin.value(0)

    uart2 = UART(UART.UART2,115200)

    # 初始化传感器
    sensor = Sensor()
    sensor.reset()
    sensor.set_framesize(Sensor.QVGA)
    sensor.set_pixformat(Sensor.GRAYSCALE)
    time.sleep(1)

    Display.init(Display.ST7701, width=800, height=480, to_ide=True)
    MediaManager.init()
    sensor.run()
    clock = time.clock()

    # 用于存储上一帧的矩形信息
    prev_min_corners = None
    prev_center = None
    prev_circle_points = None
    prev_avg_radius = 0
    prev_has_rect = False

    while True:
        clock.tick()
        os.exitpoint()

        img = sensor.snapshot(chn=CAM_CHN_ID_0)
        img_binary = img.to_grayscale(copy=True)
        img_binary = img_binary.binary([black])
        # 修复：dilate需要参数，传入迭代次数
        img_binary.dilate(1)

        rects = img_binary.find_rects(threshold=12000)

        # 初始化最小矩形变量
        min_rect = None
        min_area = float('inf')
        min_corners = None
        min_black_ratio = 0
        survivors = []

        if rects is not None:
            for rect in rects:
                corners = rect.corners()
                if len(corners) != 4:
                    continue

                # 计算所有内角误差
                angles = []
                max_angle_error = 0
                for i in range(4):
                    p0 = corners[(i-1) % 4]
                    p1 = corners[i]
                    p2 = corners[(i+1) % 4]

                    vec1 = (p0[0]-p1[0], p0[1]-p1[1])
                    vec2 = (p2[0]-p1[0], p2[1]-p1[1])

                    angle_diff = vector_angle_diff(vec1, vec2)
                    angle_error = abs(angle_diff - 90)
                    angles.append(angle_error)
                    if angle_error > max_angle_error:
                        max_angle_error = angle_error

                avg_angle_error = sum(angles) / len(angles)

                if max_angle_error > 45 or avg_angle_error > 30:
                    continue

                current_area = rect.w() * rect.h()
                if current_area < 5000:
                    continue

                # 中心区域滤波
                center = get_line_intersection([corners[0], corners[2]], [corners[1], corners[3]])
                center_x, center_y = int(center[0]), int(center[1])

                rect_width = max(5, min(15, int(math.sqrt(current_area) / 20)))
                check_size = max(5, min(20, rect_width))
                half_size = check_size // 2

                x_start = max(center_x - half_size, 0)
                x_end = min(center_x + half_size, img.width() - 1)
                y_start = max(center_y - half_size, 0)
                y_end = min(center_y + half_size, img.height() - 1)

                valid_pixels = 0
                total_pixels = 0

                for y in range(y_start, y_end):
                    for x in range(x_start, x_end):
                        pixel_value = img_binary.get_pixel(x, y)
                        if isinstance(pixel_value, tuple):
                            pixel_value = pixel_value[0]
                        if pixel_value == 0:
                            valid_pixels += 1
                        total_pixels += 1

                if total_pixels > 0:
                    black_ratio = valid_pixels / total_pixels
                else:
                    black_ratio = 0.0

                if black_ratio < 0.25:
                    continue

                survivors.append((rect, corners, black_ratio, max_angle_error, avg_angle_error))

                if current_area < min_area:
                    min_area = current_area
                    min_rect = rect
                    min_corners = corners
                    min_black_ratio = black_ratio

        # 从幸存者中选择最小面积矩形
        if len(survivors) > 0:
            min_area = float('inf')
            min_rect = None
            min_corners = None
            min_black_ratio = 0
            for rect, corners, black_ratio, max_angle_error, avg_angle_error in survivors:
                area = rect.w() * rect.h()
                if area < min_area:
                    min_area = area
                    min_rect = rect
                    min_corners = corners
                    min_black_ratio = black_ratio
        else:
            min_rect = None
            min_corners = None

        current_has_rect = min_corners is not None

        if not current_has_rect and prev_min_corners is not None:
            min_corners = prev_min_corners
            use_prev = True
        else:
            use_prev = False

        # 检测标志滤波
        if current_has_rect:
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
        base_index = current_base_point_index
        sign_dx_center, dx_center_val, sign_dy_center, dy_center_val = 0, 0, 0, 0
        sign_dx_base, dx_base_val, sign_dy_base, dy_base_val = 0, 0, 0, 0

        if min_corners is not None and flag_detected:
            if not use_prev:
                prev_min_corners = min_corners

            if not use_prev:
                img.draw_line(min_corners[0][0], min_corners[0][1], min_corners[1][0], min_corners[1][1], color=(0, 255, 0), thickness=2)
                img.draw_line(min_corners[1][0], min_corners[1][1], min_corners[2][0], min_corners[2][1], color=(0, 255, 0), thickness=2)
                img.draw_line(min_corners[2][0], min_corners[2][1], min_corners[3][0], min_corners[3][1], color=(0, 255, 0), thickness=2)
                img.draw_line(min_corners[3][0], min_corners[3][1], min_corners[0][0], min_corners[0][1], color=(0, 255, 0), thickness=2)

            diagonal1 = [min_corners[0], min_corners[2]]
            diagonal2 = [min_corners[1], min_corners[3]]
            center = get_line_intersection(diagonal1, diagonal2)
            center_x, center_y = int(center[0]), int(center[1])

            img.draw_circle(center_x, center_y, 2, color=(255, 255, 0), thickness=1)

            sorted_corners = sort_corners(min_corners, (center_x, center_y))
            recalc_circle = False

            if not use_prev or (not prev_has_rect and current_has_rect):
                circle_points, avg_radius = calculate_perspective_circle(
                    (center_x, center_y),
                    sorted_corners,
                    CIRCLE_RADIUS_CM
                )
                prev_circle_points = circle_points
                prev_avg_radius = avg_radius
                prev_center = (center_x, center_y)
                recalc_circle = True

            if prev_circle_points:
                threshold_val = max(5, int(prev_avg_radius / 8))

                if recalc_circle:
                    for i in range(len(prev_circle_points)):
                        start_point = prev_circle_points[i]
                        end_point = prev_circle_points[(i+1) % len(prev_circle_points)]
                        img.draw_line(start_point[0], start_point[1], end_point[0], end_point[1], color=(0, 255, 0), thickness=2)

                for i, point in enumerate(prev_circle_points):
                    x, y = point
                    if i == current_base_point_index:
                        img.draw_circle(x, y, 3, color=(255, 0, 0), thickness=1)
                    else:
                        img.draw_circle(x, y, 3, color=(255, 255, 0), thickness=1)

                current_point = prev_circle_points[current_base_point_index]
                cp_x, cp_y = current_point

                dx = TARGET_POINT[0] - cp_x
                dy = TARGET_POINT[1] - cp_y

                abs_dx = abs(dx)
                abs_dy = abs(dy)

                if abs_dx < threshold_val and abs_dy < threshold_val:
                    base_point_counter += 1
                else:
                    base_point_counter = 0

                if base_point_counter >= 2:
                    current_base_point_index = (current_base_point_index + 1) % 16
                    base_point_counter = 0
                    #print("基准点切换到:", current_base_point_index)

                dx_center = TARGET_POINT[0] - center_x
                dy_center = TARGET_POINT[1] - center_y
                dx_base = TARGET_POINT[0] - cp_x
                dy_base = TARGET_POINT[1] - cp_y

                if dx_center < 0:
                    sign_dx_center = 1
                    dx_center_val = min(255, int(abs(dx_center)))
                else:
                    sign_dx_center = 0
                    dx_center_val = min(255, int(abs(dx_center)))

                if dy_center < 0:
                    sign_dy_center = 1
                    dy_center_val = min(255, int(abs(dy_center)))
                else:
                    sign_dy_center = 0
                    dy_center_val = min(255, int(abs(dy_center)))

                if dx_base < 0:
                    sign_dx_base = 1
                    dx_base_val = min(255, int(abs(dx_base)))
                else:
                    sign_dx_base = 0
                    dx_base_val = min(255, int(abs(dx_base)))

                if dy_base < 0:
                    sign_dy_base = 1
                    dy_base_val = min(255, int(abs(dy_base)))
                else:
                    sign_dy_base = 0
                    dy_base_val = min(255, int(abs(dy_base)))

        if min_corners is None:
            prev_min_corners = None
            prev_center = None
            prev_circle_points = None

        prev_has_rect = current_has_rect

        # 显示状态信息 - 使用最简单的draw_string
        try:
            img.draw_string(10, 10, "fps: " + str(clock.fps()), color=(255, 0, 0))
            status_str = "Tracking" if flag_detected else "Lost"
            status_color = (0, 255, 0) if flag_detected else (255, 0, 0)
            img.draw_string(10, 25, "status: " + status_str, color=status_color)
            if min_corners is not None and not use_prev:
                img.draw_string(10, 40, "valid ratio: " + str(min_black_ratio), color=(255, 255, 255))
                img.draw_string(10, 55, "base point: " + str(current_base_point_index), color=(255, 0, 0))
        except:
            pass

        # PID控制 - 目标将dx_center和dy_center控制为0
        if min_corners is not None and flag_detected:
            ctl_x = pid_x.step(dx_center, 0)
            ctl_y = pid_y.step(dy_center, 0)
            print(dy_center)
        else:
            ctl_x = pid_x.step(TARGET_POINT[0], TARGET_POINT[0])
            ctl_y = pid_y.step(TARGET_POINT[1], TARGET_POINT[1])

        x_motor_speed(ctl_x)
        y_motor_speed(ctl_y)



        # 发送串口数据
        sending_data(
            flag_byte,
            sign_dx_center,
            dx_center_val,
            sign_dy_center,
            dy_center_val,
            base_index,
            sign_dx_base,
            dx_base_val,
            sign_dy_base,
            dy_base_val
        )

        img.compressed_for_ide()
        Display.show_image(img, x=(800-320)//2, y=(480-240)//2)
        time.sleep_ms(10)

finally:
    if isinstance(sensor, Sensor):
        sensor.stop()
    Display.deinit()
    os.exitpoint(os.EXITPOINT_ENABLE_SLEEP)
    time.sleep_ms(100)
    MediaManager.deinit()
