"""
Rectangle detection with PID-stabilized motor control at 200Hz.
Uses RK3588 sysfs PWM + GPIO for stepper motors (LubanCat 4 / 鲁班猫4).

Architecture:
  Main thread (camera FPS):  capture → detect rectangle → update shared ox/oy
  PID  thread (200Hz):       read ox/oy → PID step → motor_x/y.set_speed()
"""
import cv2
import numpy as np
import time
import threading
from collections import deque
from typing import List, Dict, Optional, Any, Union

from pwm_control import PWMMotor

# ============================================================
# 硬件配置 — 根据实际接线修改
# ============================================================

# X 轴步进电机 (脉冲 = PWM, 方向 = GPIO)
PWM_CHIP_X = 3          # /sys/class/pwm/pwmchipX
PWM_CHANNEL_X = 0       # pwmX 通道
DIR_GPIO_X = 102         # 方向脚 GPIO 编号

# Y 轴步进电机
PWM_CHIP_Y = 4
PWM_CHANNEL_Y = 0
DIR_GPIO_Y = 111

STEPS_PER_REV = 3200    # 步进电机每转脉冲数

# ============================================================
# PID 参数 (移植自 shibie.py, 控制频率 200Hz → dt=0.005)
# ============================================================

PID_FREQ = 200          # 控制频率 Hz
PID_DT = 1.0 / PID_FREQ

# PID 增益 (可通过键盘实时调节)
PID_X_KP = 1.7
PID_X_KI = 0.4
PID_X_KD = 0.10
PID_Y_KP = 1.4
PID_Y_KI = 0.4
PID_Y_KD = 0.1

PID_MAX = 3000
PID_MIN = -3000

# 增益调节步长
PID_KP_STEP = 0.01
PID_KI_STEP = 0.005
PID_KD_STEP = 0.001


# ============================================================
# PID 控制器 (移植自 shibie.py)
# ============================================================

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


# ============================================================
# 200Hz PID 控制线程
# ============================================================

class PIDControlLoop:
    """Runs PID + motor output at a fixed frequency in a background thread."""

    def __init__(self, motor_x: PWMMotor, motor_y: PWMMotor):
        self.motor_x = motor_x
        self.motor_y = motor_y

        self.pid_x = PID(PID_X_KP, PID_X_KI, PID_X_KD, PID_MAX, PID_MIN, PID_DT)
        self.pid_y = PID(PID_Y_KP, PID_Y_KI, PID_Y_KD, PID_MAX, PID_MIN, PID_DT)

        self._lock = threading.Lock()
        self._ox = 0.0
        self._oy = 0.0
        self._has_target = False
        self._running = False
        self._thread = None

    def update_target(self, ox: float, oy: float, has_target: bool):
        """Called from main thread each frame with latest offsets."""
        with self._lock:
            self._ox = ox
            self._oy = oy
            self._has_target = has_target

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self.motor_x.stop()
        self.motor_y.stop()

    def adjust_gain(self, axis: str, gain: str, delta: float):
        """Thread-safe PID gain adjustment from main thread."""
        pid = self.pid_x if axis == 'x' else self.pid_y
        if gain == 'kp':
            pid.Kp = max(0, round(pid.Kp + delta, 4))
        elif gain == 'ki':
            pid.Ki = max(0, round(pid.Ki + delta, 4))
        elif gain == 'kd':
            pid.Kd = max(0, round(pid.Kd + delta, 4))

    def reset(self, axis: str = None):
        """Clear integrator and first-sample flag."""
        for pid, label in [(self.pid_x, 'x'), (self.pid_y, 'y')]:
            if axis is None or axis == label:
                pid.clear()

    def _loop(self):
        period = PID_DT
        next_cycle = time.perf_counter()

        while self._running:
            with self._lock:
                ox = self._ox
                oy = self._oy
                has_target = self._has_target

            if has_target:
                ctl_x = self.pid_x.step(ox, 0)
                ctl_y = self.pid_y.step(oy, 0)
            else:
                # 无目标时将当前值同时作为 value 和 target, 误差=0, PID 保持
                ctl_x = self.pid_x.step(0, 0)
                ctl_y = self.pid_y.step(0, 0)

            self.motor_x.set_speed(ctl_x)
            self.motor_y.set_speed(ctl_y)

            # 精确 200Hz 定时
            next_cycle += period
            delay = next_cycle - time.perf_counter()
            if delay > 0.001:
                time.sleep(delay - 0.0005)
            while time.perf_counter() < next_cycle:
                pass


# ============================================================
# 矩形检测器 (保留原有逻辑, 新增 PID 控制)
# ============================================================

class RectangleDetector:
    def __init__(self, source: Union[int, str] = 0, width: int = 640, height: int = 480,
                 is_video_file: bool = False, enable_pid: bool = True):
        if isinstance(source, int):
            self.camera = cv2.VideoCapture(source)
        elif isinstance(source, str):
            self.camera = cv2.VideoCapture(source)
        else:
            self.camera = cv2.VideoCapture(str(source))

        if not self.camera.isOpened():
            raise OSError(f"Could not open source: {source}")

        if not is_video_file and isinstance(source, int):
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.camera.set(cv2.CAP_PROP_FPS, 60)

        actual_width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.frame_center_x = actual_width // 2
        self.frame_center_y = actual_height // 2 + 20

        self.min_area = 2000
        self.max_area = 200000
        self.canny_low_threshold = 50
        self.canny_high_threshold = 150
        self.blur_kernel_size = 5
        self.track_largest = True

        self.smoothing_window = 5
        self.position_history = deque(maxlen=self.smoothing_window)
        self.size_history = deque(maxlen=self.smoothing_window)
        self.detection_threshold = 3
        self.detection_counter = 0
        self.last_valid_rect = None
        self.max_disappear_frames = 10
        self.disappear_counter = 0
        self.smoothed_rect = None

        # PID 控制
        self._enable_pid = enable_pid
        self._pid_loop = None
        if enable_pid:
            motor_x = PWMMotor(PWM_CHIP_X, PWM_CHANNEL_X, DIR_GPIO_X, STEPS_PER_REV)
            motor_y = PWMMotor(PWM_CHIP_Y, PWM_CHANNEL_Y, DIR_GPIO_Y, STEPS_PER_REV)
            self._pid_loop = PIDControlLoop(motor_x, motor_y)
            self._motor_x = motor_x
            self._motor_y = motor_y

        print(f"视频源已初始化: {actual_width}x{actual_height}")
        print(f"画面中心: ({self.frame_center_x}, {self.frame_center_y})")
        if enable_pid:
            print(f"PID 控制: {PID_FREQ}Hz, dt={PID_DT:.4f}s")

    def calculate_iou(self, rect1, rect2):
        if rect1 is None or rect2 is None:
            return 0
        x1 = max(rect1['x'], rect2['x'])
        y1 = max(rect1['y'], rect2['y'])
        x2 = min(rect1['x'] + rect1['width'], rect2['x'] + rect2['width'])
        y2 = min(rect1['y'] + rect1['height'], rect2['y'] + rect2['height'])
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = rect1['width'] * rect1['height']
        area2 = rect2['width'] * rect2['height']
        union = area1 + area2 - intersection
        return intersection / union if union > 0 else 0

    def detect_rectangle(self, frame: np.ndarray) -> tuple:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (self.blur_kernel_size, self.blur_kernel_size), 0)
        _, binary = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY_INV)

        kernel = np.ones((5, 5), np.uint8)
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=3)

        contours, hierarchy = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        edges = cv2.Canny(blurred, self.canny_low_threshold, self.canny_high_threshold)

        rectangles: List[Dict[str, Any]] = []
        if hierarchy is None or len(contours) == 0:
            return rectangles, edges

        h = hierarchy[0]

        def line_intersection(p1, p2, p3, p4):
            x1, y1 = p1
            x2, y2 = p2
            x3, y3 = p3
            x4, y4 = p4
            denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            if abs(denom) < 1e-6:
                return None
            px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
            py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
            return int(px), int(py)

        def build_rect(contour, idx):
            rect_area = cv2.contourArea(contour)
            x, y, w_box, h_box = cv2.boundingRect(contour)
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

            if len(approx) == 4:
                pts = approx.reshape(4, 2)
                center = line_intersection(pts[0], pts[2], pts[1], pts[3])
                if center is None:
                    center = (x + w_box // 2, y + h_box // 2)
            else:
                center = (x + w_box // 2, y + h_box // 2)

            cx, cy = center
            return {
                'center_x': cx, 'center_y': cy,
                'offset_x': cx - self.frame_center_x,
                'offset_y': cy - self.frame_center_y,
                'x': x, 'y': y,
                'width': w_box, 'height': h_box,
                'area': rect_area,
                'contour': approx,
                'contour_idx': idx,
            }

        for i, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            if area < self.min_area or area > self.max_area:
                continue

            child_idx = int(h[i][2])
            if child_idx == -1:
                continue

            child_cnt = contours[child_idx]
            child_area = cv2.contourArea(child_cnt)
            if child_area < self.min_area * 0.3:
                continue

            outer_rect = build_rect(cnt, i)
            inner_rect = build_rect(child_cnt, child_idx)

            area_ratio = inner_rect['area'] / outer_rect['area']
            if area_ratio < 0.2 or area_ratio > 0.95:
                continue
            if inner_rect['x'] <= outer_rect['x'] or inner_rect['y'] <= outer_rect['y']:
                continue
            if (inner_rect['x'] + inner_rect['width'] >= outer_rect['x'] + outer_rect['width']
                    or inner_rect['y'] + inner_rect['height'] >= outer_rect['y'] + outer_rect['height']):
                continue

            outer_rect['inner_rect'] = inner_rect
            rectangles.append(outer_rect)

        if self.track_largest and len(rectangles) > 1:
            rectangles.sort(key=lambda r: r['area'], reverse=True)

        return rectangles, edges

    def smooth_detection(self, rectangles):
        if len(rectangles) == 0:
            self.disappear_counter += 1
            if self.last_valid_rect is not None and self.disappear_counter <= self.max_disappear_frames:
                return [self.last_valid_rect]
            else:
                self.position_history.clear()
                self.size_history.clear()
                self.detection_counter = 0
                return []
        else:
            self.disappear_counter = 0
            current_rect = rectangles[0]

            if self.last_valid_rect is not None:
                iou = self.calculate_iou(current_rect, self.last_valid_rect)
                if iou < 0.3:
                    self.detection_counter = 0

            self.detection_counter = min(self.detection_counter + 1, self.detection_threshold)
            self.position_history.append((current_rect['center_x'], current_rect['center_y']))
            self.size_history.append((current_rect['width'], current_rect['height']))

            if self.detection_counter >= self.detection_threshold:
                if len(self.position_history) > 0:
                    avg_center_x = int(np.mean([p[0] for p in self.position_history]))
                    avg_center_y = int(np.mean([p[1] for p in self.position_history]))
                    avg_width = int(np.mean([s[0] for s in self.size_history]))
                    avg_height = int(np.mean([s[1] for s in self.size_history]))

                    smoothed = current_rect.copy()
                    smoothed['center_x'] = avg_center_x
                    smoothed['center_y'] = avg_center_y
                    smoothed['offset_x'] = avg_center_x - self.frame_center_x
                    smoothed['offset_y'] = avg_center_y - self.frame_center_y
                    smoothed['width'] = avg_width
                    smoothed['height'] = avg_height
                    smoothed['x'] = avg_center_x - avg_width // 2
                    smoothed['y'] = avg_center_y - avg_height // 2

                    if current_rect.get('inner_rect') is not None:
                        inner = current_rect['inner_rect']
                        dx = inner['center_x'] - current_rect['center_x']
                        dy = inner['center_y'] - current_rect['center_y']
                        w_ratio = inner['width'] / current_rect['width'] if current_rect['width'] > 0 else 0
                        h_ratio = inner['height'] / current_rect['height'] if current_rect['height'] > 0 else 0
                        smoothed_inner = inner.copy()
                        smoothed_inner['center_x'] = avg_center_x + int(dx)
                        smoothed_inner['center_y'] = avg_center_y + int(dy)
                        smoothed_inner['width'] = int(avg_width * w_ratio)
                        smoothed_inner['height'] = int(avg_height * h_ratio)
                        smoothed_inner['x'] = smoothed_inner['center_x'] - smoothed_inner['width'] // 2
                        smoothed_inner['y'] = smoothed_inner['center_y'] - smoothed_inner['height'] // 2
                        smoothed_inner['offset_x'] = smoothed_inner['center_x'] - self.frame_center_x
                        smoothed_inner['offset_y'] = smoothed_inner['center_y'] - self.frame_center_y
                        smoothed['inner_rect'] = smoothed_inner

                    self.smoothed_rect = smoothed
                    self.last_valid_rect = smoothed
                    return [smoothed]

            if self.last_valid_rect is not None:
                return [self.last_valid_rect]
            return []

    def draw_result(self, frame: np.ndarray, rectangles: List[Dict[str, Any]],
                    target_index: int = 0, paused: bool = False) -> np.ndarray:
        result_frame = frame.copy()

        cv2.line(result_frame, (self.frame_center_x, 0),
                 (self.frame_center_x, result_frame.shape[0]), (255, 255, 255), 1)
        cv2.line(result_frame, (0, self.frame_center_y),
                 (result_frame.shape[1], self.frame_center_y), (255, 255, 255), 1)
        cv2.circle(result_frame, (self.frame_center_x, self.frame_center_y), 5, (255, 255, 255), -1)

        if len(rectangles) > 0:
            target_rect = rectangles[target_index]

            if self.smoothed_rect is not None:
                sr = self.smoothed_rect
                cv2.rectangle(result_frame,
                              (sr['x'], sr['y']),
                              (sr['x'] + sr['width'], sr['y'] + sr['height']),
                              (0, 255, 0), 2)
                cv2.putText(result_frame, "Outer",
                            (sr['x'], sr['y'] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                if sr.get('inner_rect') is not None:
                    ir = sr['inner_rect']
                    cv2.rectangle(result_frame,
                                  (ir['x'], ir['y']),
                                  (ir['x'] + ir['width'], ir['y'] + ir['height']),
                                  (255, 0, 0), 2)
                    cv2.putText(result_frame, "Inner",
                                (ir['x'], ir['y'] - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                cv2.circle(result_frame, (sr['center_x'], sr['center_y']),
                           6, (0, 255, 0), -1)

                # 显示 ox/oy (PID 输入)
                ox = sr['offset_x']
                oy = sr['offset_y']
                cv2.putText(result_frame, f"ox: {ox:+d}  oy: {oy:+d}",
                            (10, result_frame.shape[0] - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            cv2.drawContours(result_frame, [target_rect['contour']], -1, (0, 0, 255), 1)
            if target_rect.get('inner_rect') is not None:
                cv2.drawContours(result_frame, [target_rect['inner_rect']['contour']], -1,
                                 (0, 0, 255), 1)

        status_text = f"Detections: {self.detection_counter}/{self.detection_threshold}"
        cv2.putText(result_frame, status_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        if self.disappear_counter > 0:
            disappear_text = f"Lost: {self.disappear_counter}/{self.max_disappear_frames}"
            cv2.putText(result_frame, disappear_text, (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)

        if paused:
            cv2.putText(result_frame, "PAUSED", (result_frame.shape[1] // 2 - 80, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # PID 状态显示 (右上角)
        if self._pid_loop is not None:
            px = self._pid_loop.pid_x
            py = self._pid_loop.pid_y
            lines = [
                f"PID X: Kp={px.Kp:.3f} Ki={px.Ki:.3f} Kd={px.Kd:.3f}",
                f"PID Y: Kp={py.Kp:.3f} Ki={py.Ki:.3f} Kd={py.Kd:.3f}",
                f"X err={px.error:+.1f} int={px.ierror:+.1f} out={px.ctl_value:+.1f}",
                f"Y err={py.error:+.1f} int={py.ierror:+.1f} out={py.ctl_value:+.1f}",
            ]
            x0 = result_frame.shape[1] - 320
            for i, line in enumerate(lines):
                cv2.putText(result_frame, line, (x0, 20 + i * 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        return result_frame

    def run(self, display: bool = True) -> None:
        fps_counter = 0
        fps_time = time.time()
        fps = 0
        paused = False
        last_frame = None
        last_edges = None
        last_rectangles: List[Dict[str, Any]] = []

        # 启动 PID 控制线程
        if self._pid_loop is not None:
            self._pid_loop.start()
            print(f"PID 控制线程已启动 ({PID_FREQ}Hz)")

        print("开始检测")
        print("按键控制:")
        print("  'q' - 退出")
        print("  '空格' - 暂停/继续")
        print("  '+' - 增大最小面积")
        print("  '-' - 减小最小面积")
        print("  'w'/'s' - 平滑窗口 +/-")
        print("  'a'/'z' - Kp_x +/-")
        print("  'd'/'c' - Ki_x +/-")
        print("  'f'/'v' - Kd_x +/-")
        print("  'g'/'b' - Kp_y +/-")
        print("  'h'/'n' - Ki_y +/-")
        print("  'j'/'m' - Kd_y +/-")
        print("  'r' - 清零 PID 积分")

        try:
            while True:
                if not paused:
                    ret, frame = self.camera.read()
                    if not ret:
                        print("视频结束或无法读取画面")
                        break

                    rectangles, edges = self.detect_rectangle(frame)
                    smoothed_rectangles = self.smooth_detection(rectangles)

                    last_frame = frame.copy()
                    last_edges = edges.copy() if edges is not None else None
                    last_rectangles = smoothed_rectangles

                    # 更新 PID 目标
                    if len(smoothed_rectangles) > 0:
                        target_rect = smoothed_rectangles[0]
                        ox = target_rect['offset_x']
                        oy = -target_rect['offset_y']  # 图像 y 轴翻转
                        if self._pid_loop is not None:
                            self._pid_loop.update_target(ox, oy, True)
                    else:
                        if self._pid_loop is not None:
                            self._pid_loop.update_target(0, 0, False)

                if display and last_frame is not None:
                    if not paused:
                        fps_counter += 1
                        if time.time() - fps_time >= 1:
                            fps = fps_counter
                            fps_counter = 0
                            fps_time = time.time()

                    result_frame = self.draw_result(last_frame, last_rectangles, paused=paused)
                    current_edges = last_edges

                    cv2.putText(result_frame, f"FPS: {fps}", (10, result_frame.shape[0] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

                    cv2.imshow("Rectangle Detection", result_frame)
                    if current_edges is not None:
                        cv2.imshow("Edges", current_edges)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord(' '):
                    paused = not paused
                    if self._pid_loop is not None:
                        if paused:
                            self._pid_loop.update_target(0, 0, False)
                    print("暂停" if paused else "继续")
                elif key == ord('+') or key == ord('='):
                    self.min_area += 500
                    print(f"最小面积: {self.min_area}")
                elif key == ord('-') or key == ord('_'):
                    self.min_area = max(100, self.min_area - 500)
                    print(f"最小面积: {self.min_area}")
                elif key == ord('w'):
                    self.smoothing_window += 1
                    self.position_history = deque(maxlen=self.smoothing_window)
                    self.size_history = deque(maxlen=self.smoothing_window)
                    print(f"平滑窗口: {self.smoothing_window}")
                elif key == ord('s'):
                    self.smoothing_window = max(1, self.smoothing_window - 1)
                    self.position_history = deque(maxlen=self.smoothing_window)
                    self.size_history = deque(maxlen=self.smoothing_window)
                    print(f"平滑窗口: {self.smoothing_window}")
                # ---- PID 增益调节 ----
                elif self._pid_loop is not None and key == ord('a'):
                    self._pid_loop.adjust_gain('x', 'kp', +PID_KP_STEP)
                    self._print_pid()
                elif self._pid_loop is not None and key == ord('z'):
                    self._pid_loop.adjust_gain('x', 'kp', -PID_KP_STEP)
                    self._print_pid()
                elif self._pid_loop is not None and key == ord('d'):
                    self._pid_loop.adjust_gain('x', 'ki', +PID_KI_STEP)
                    self._print_pid()
                elif self._pid_loop is not None and key == ord('c'):
                    self._pid_loop.adjust_gain('x', 'ki', -PID_KI_STEP)
                    self._print_pid()
                elif self._pid_loop is not None and key == ord('f'):
                    self._pid_loop.adjust_gain('x', 'kd', +PID_KD_STEP)
                    self._print_pid()
                elif self._pid_loop is not None and key == ord('v'):
                    self._pid_loop.adjust_gain('x', 'kd', -PID_KD_STEP)
                    self._print_pid()
                elif self._pid_loop is not None and key == ord('g'):
                    self._pid_loop.adjust_gain('y', 'kp', +PID_KP_STEP)
                    self._print_pid()
                elif self._pid_loop is not None and key == ord('b'):
                    self._pid_loop.adjust_gain('y', 'kp', -PID_KP_STEP)
                    self._print_pid()
                elif self._pid_loop is not None and key == ord('h'):
                    self._pid_loop.adjust_gain('y', 'ki', +PID_KI_STEP)
                    self._print_pid()
                elif self._pid_loop is not None and key == ord('n'):
                    self._pid_loop.adjust_gain('y', 'ki', -PID_KI_STEP)
                    self._print_pid()
                elif self._pid_loop is not None and key == ord('j'):
                    self._pid_loop.adjust_gain('y', 'kd', +PID_KD_STEP)
                    self._print_pid()
                elif self._pid_loop is not None and key == ord('m'):
                    self._pid_loop.adjust_gain('y', 'kd', -PID_KD_STEP)
                    self._print_pid()
                elif self._pid_loop is not None and key == ord('r'):
                    self._pid_loop.reset()
                    print("PID 积分已清零")

        finally:
            if self._pid_loop is not None:
                print("正在停止 PID 控制线程...")
                self._pid_loop.stop()
            self.camera.release()
            cv2.destroyAllWindows()

    def cleanup(self):
        """手动清理硬件资源。"""
        if hasattr(self, '_motor_x'):
            self._motor_x.cleanup()
        if hasattr(self, '_motor_y'):
            self._motor_y.cleanup()

    def _print_pid(self):
        """打印当前 PID 增益到控制台。"""
        if self._pid_loop is None:
            return
        px = self._pid_loop.pid_x
        py = self._pid_loop.pid_y
        print(f"PID X: Kp={px.Kp:.4f} Ki={px.Ki:.4f} Kd={px.Kd:.4f}  "
              f"Y: Kp={py.Kp:.4f} Ki={py.Ki:.4f} Kd={py.Kd:.4f}")


def main() -> None:
    detector = None
    try:
<<<<<<< HEAD
        detector = RectangleDetector(source=11, enable_pid=True)
=======
        # ========== 配置区 ==========
        #选择1：使用摄像头（取消注释下面这行，注释视频文件那行）
        #detector = RectangleDetector(source=0)

        # 选择2：使用视频文件（修改为你的视频路径）
        video_path = "test.mp4"  # 👈 改成你的视频文件路径
        detector = RectangleDetector(source=video_path, is_video_file=True)
        # ============================

>>>>>>> 290efde1abbb2a8e44a954d0fdc6102fe74220f0
        detector.run(display=True)
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if detector is not None:
            detector.cleanup()


if __name__ == "__main__":
    main()
