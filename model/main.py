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
from typing import List, Dict, Any, Union

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

# 误差钳位: 转弯时画面偏移可达几百像素, 限制输入避免 PID 输出过大导致高速转动
PID_ERROR_CLAMP = 120

# 增益调节步长
PID_KP_STEP = 0.01
PID_KI_STEP = 0.005
PID_KD_STEP = 0.001


# ============================================================
# PID 控制器 (移植自 shibie.py)
# ============================================================

class PID:
    def __init__(self, kp, ki, kd, ctl_max, ctl_min, dt, error_clamp=None):
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
        self.error_clamp = error_clamp
        self.first = True

    def clear(self):
        self.last_value = 0.0
        self.ierror = 0.0
        self.first = True

    def set_limit(self, max_val, min_val):
        self.ctl_max = max_val
        self.ctl_min = min_val

    def set_error_clamp(self, clamp):
        self.error_clamp = clamp

    def step(self, value, target):
        if self.first:
            self.last_value = value
            self.first = False
        self.now_value = value
        self.target_value = target
        raw_error = self.target_value - self.now_value
        self.error = raw_error
        if self.error_clamp is not None:
            self.error = max(-self.error_clamp, min(self.error_clamp, raw_error))
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

        self.pid_x = PID(PID_X_KP, PID_X_KI, PID_X_KD, PID_MAX, PID_MIN, PID_DT, PID_ERROR_CLAMP)
        self.pid_y = PID(PID_Y_KP, PID_Y_KI, PID_Y_KD, PID_MAX, PID_MIN, PID_DT, PID_ERROR_CLAMP)

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

    def adjust_error_clamp(self, delta: int):
        """Thread-safe error_clamp adjustment from main thread."""
        self.pid_x.error_clamp = max(20, (self.pid_x.error_clamp or PID_ERROR_CLAMP) + delta)
        self.pid_y.error_clamp = max(20, (self.pid_y.error_clamp or PID_ERROR_CLAMP) + delta)

    def reset(self, axis: str = None):
        """Clear integrator and first-sample flag."""
        for pid, label in [(self.pid_x, 'x'), (self.pid_y, 'y')]:
            if axis is None or axis == label:
                pid.clear()

    def _loop(self):
        period = PID_DT
        next_cycle = time.perf_counter()
        _was_lost = False

        while self._running:
            with self._lock:
                ox = self._ox
                oy = self._oy
                has_target = self._has_target

            if has_target:
                _was_lost = False
                ctl_x = self.pid_x.step(ox, 0)
                ctl_y = self.pid_y.step(oy, 0)
            else:
                # 目标丢失: 清零 PID 防止 D 项尖峰, 电机立即停止
                if not _was_lost:
                    self.pid_x.clear()
                    self.pid_y.clear()
                    _was_lost = True
                ctl_x = 0.0
                ctl_y = 0.0

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
# Kalman 滤波器 — 恒速模型, 用于位置预测与防丢失
# ============================================================

class KalmanTracker:
    """6-state (cx, cy, vx, vy, w, h) constant-velocity Kalman filter for rectangle tracking."""

    def __init__(self,
                 process_noise: float = 0.05,
                 measurement_noise: float = 0.15,
                 velocity_decay: float = 0.85):
        self.kf = cv2.KalmanFilter(6, 4)
        # State: [cx, cy, vx, vy, w, h]
        # Measurement: [cx, cy, w, h]

        dt = 1.0  # frame-to-frame
        self.kf.transitionMatrix = np.array([
            [1, 0, dt, 0,  0, 0],
            [0, 1, 0,  dt, 0, 0],
            [0, 0, 1,  0,  0, 0],
            [0, 0, 0,  1,  0, 0],
            [0, 0, 0,  0,  1, 0],
            [0, 0, 0,  0,  0, 1],
        ], np.float32)

        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ], np.float32)

        self.kf.processNoiseCov = np.eye(6, dtype=np.float32) * process_noise
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * measurement_noise
        self.kf.errorCovPost = np.eye(6, dtype=np.float32)

        self.initialized = False
        self.missed_frames = 0
        self.velocity_decay = velocity_decay
        self._last_state = None

    def predict(self) -> np.ndarray:
        """Predict next state. During consecutive misses, decay velocity."""
        if self.missed_frames > 0 and self._last_state is not None:
            # 修改 statePost 的速度, 让衰减跨帧累积 (predict 会 statePre=T*statePost)
            decay = self.velocity_decay ** self.missed_frames
            self.kf.statePost[2, 0] = self._last_state[2, 0] * decay
            self.kf.statePost[3, 0] = self._last_state[3, 0] * decay
        return self.kf.predict()

    def correct(self, measurement: np.ndarray) -> np.ndarray:
        """Update with measurement."""
        self.initialized = True
        self.missed_frames = 0
        result = self.kf.correct(measurement)
        self._last_state = self.kf.statePost.copy()
        return result

    def init_state(self, cx: float, cy: float, w: float, h: float):
        """Initialize filter state on first detection."""
        self.kf.statePost = np.array([[cx], [cy], [0], [0], [w], [h]], np.float32)
        self.kf.statePre = self.kf.statePost.copy()
        self.kf.errorCovPost = np.eye(6, dtype=np.float32) * 0.5
        self._last_state = self.kf.statePost.copy()
        self.initialized = True
        self.missed_frames = 0

    def reset(self):
        """Reset filter to uninitialized state."""
        self.initialized = False
        self.missed_frames = 0
        self._last_state = None
        self.kf.errorCovPost = np.eye(6, dtype=np.float32)


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

        # Kalman 跟踪 & 防丢失
        self._kf = KalmanTracker(
            process_noise=0.05,       # 增大 → 对快速运动响应更快
            measurement_noise=0.15,   # 增大 → 平滑更强
            velocity_decay=0.80,      # 丢失时速度衰减率 (每帧乘0.8)
        )
        self.max_disappear_frames = 20    # 纯预测最多持续帧数 (30fps ≈ 0.67s)
        self.disappear_counter = 0
        self.last_valid_rect = None
        self.smoothed_rect = None
        self.detection_counter = 0        # 仅用于显示

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

    def _build_output_rect(self, source_rect: dict, cx: float, cy: float,
                           w: float, h: float, vx: float = 0, vy: float = 0,
                           predicted: bool = False) -> dict:
        """Build a result rect dict from Kalman state, preserving inner_rect offsets."""
        out = source_rect.copy()
        out['center_x'] = int(cx)
        out['center_y'] = int(cy)
        out['offset_x'] = int(cx) - self.frame_center_x
        out['offset_y'] = int(cy) - self.frame_center_y
        w_int, h_int = max(10, int(w)), max(10, int(h))
        out['width'] = w_int
        out['height'] = h_int
        out['x'] = int(cx) - w_int // 2
        out['y'] = int(cy) - h_int // 2
        out['velocity_x'] = vx
        out['velocity_y'] = vy
        out['predicted'] = predicted

        if source_rect.get('inner_rect') is not None:
            inner = source_rect['inner_rect']
            old_cx = source_rect['center_x']
            old_cy = source_rect['center_y']
            old_w = source_rect['width']
            old_h = source_rect['height']
            rel_dx = inner['center_x'] - old_cx
            rel_dy = inner['center_y'] - old_cy
            w_ratio = inner['width'] / old_w if old_w > 0 else 0
            h_ratio = inner['height'] / old_h if old_h > 0 else 0
            inr = inner.copy()
            inr['center_x'] = int(cx) + int(rel_dx)
            inr['center_y'] = int(cy) + int(rel_dy)
            inr['width'] = max(5, int(w_int * w_ratio))
            inr['height'] = max(5, int(h_int * h_ratio))
            inr['x'] = inr['center_x'] - inr['width'] // 2
            inr['y'] = inr['center_y'] - inr['height'] // 2
            inr['offset_x'] = inr['center_x'] - self.frame_center_x
            inr['offset_y'] = inr['center_y'] - self.frame_center_y
            out['inner_rect'] = inr
        return out

    def smooth_detection(self, rectangles):
        """Kalman-filtered tracking: 零延迟确认 + 速度预测防丢失."""

        # ---- 1. Kalman 预测 ----
        if self._kf.initialized:
            predicted = self._kf.predict()
            pred_cx = predicted[0, 0]
            pred_cy = predicted[1, 0]
            pred_vx = predicted[2, 0]
            pred_vy = predicted[3, 0]
            pred_w = predicted[4, 0]
            pred_h = predicted[5, 0]
        else:
            pred_cx = pred_cy = pred_vx = pred_vy = None
            pred_w = pred_h = None

        # ---- 2. 有检测 → 立即更新 Kalman, 无确认延迟 ----
        if len(rectangles) > 0:
            current_rect = rectangles[0]
            meas_cx = float(current_rect['center_x'])
            meas_cy = float(current_rect['center_y'])
            meas_w = float(current_rect['width'])
            meas_h = float(current_rect['height'])

            if not self._kf.initialized:
                self._kf.init_state(meas_cx, meas_cy, meas_w, meas_h)
                self.disappear_counter = 0
                self.detection_counter = 1
                result = self._build_output_rect(current_rect, meas_cx, meas_cy, meas_w, meas_h)
                self.smoothed_rect = result
                self.last_valid_rect = result
                return [result]

            # Kalman 更新
            measurement = np.array([[meas_cx], [meas_cy], [meas_w], [meas_h]], np.float32)
            corrected = self._kf.correct(measurement)

            kf_cx = corrected[0, 0]
            kf_cy = corrected[1, 0]
            kf_vx = corrected[2, 0]
            kf_vy = corrected[3, 0]
            kf_w = corrected[4, 0]
            kf_h = corrected[5, 0]

            result = self._build_output_rect(current_rect, kf_cx, kf_cy, kf_w, kf_h, kf_vx, kf_vy)
            self.disappear_counter = 0
            self.detection_counter = min(self.detection_counter + 1, 999)
            self.smoothed_rect = result
            self.last_valid_rect = result
            return [result]

        # ---- 3. 无检测 → Kalman 纯预测 (速度衰减) ----
        self.disappear_counter += 1
        self._kf.missed_frames = self.disappear_counter
        self.detection_counter = 0

        if self._kf.initialized and self.disappear_counter <= self.max_disappear_frames:
            # 使用节1已预测的值 (predict 已含速度衰减)
            if self.last_valid_rect is not None and pred_cx is not None:
                result = self._build_output_rect(
                    self.last_valid_rect, pred_cx, pred_cy, pred_w, pred_h,
                    predicted=True,
                )
                self.smoothed_rect = result
                return [result]

        # 超时 → 重置 Kalman, 上报丢失
        self._kf.reset()
        self.last_valid_rect = None
        self.smoothed_rect = None
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

                # 显示 ox/oy + 速度 (PID 输入)
                ox = sr['offset_x']
                oy = sr['offset_y']
                vx = sr.get('velocity_x', 0)
                vy = sr.get('velocity_y', 0)
                pred_flag = " [PRED]" if sr.get('predicted') else ""
                cv2.putText(result_frame,
                            f"ox: {ox:+d} oy: {oy:+d}  vx: {vx:+.0f} vy: {vy:+.0f}{pred_flag}",
                            (10, result_frame.shape[0] - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            cv2.drawContours(result_frame, [target_rect['contour']], -1, (0, 0, 255), 1)
            if target_rect.get('inner_rect') is not None:
                cv2.drawContours(result_frame, [target_rect['inner_rect']['contour']], -1,
                                 (0, 0, 255), 1)

        status_text = f"Detect: {self.detection_counter}  KF: {'on' if self._kf.initialized else 'off'}"
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
            ec = px.error_clamp if px.error_clamp is not None else 0
            lines = [
                f"PID X: Kp={px.Kp:.3f} Ki={px.Ki:.3f} Kd={px.Kd:.3f}",
                f"PID Y: Kp={py.Kp:.3f} Ki={py.Ki:.3f} Kd={py.Kd:.3f}",
                f"X err={px.error:+.1f} int={px.ierror:+.1f} out={px.ctl_value:+.1f}",
                f"Y err={py.error:+.1f} int={py.ierror:+.1f} out={py.ctl_value:+.1f}",
                f"Error clamp: {ec}",
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
        print("  '+'/'-' - 最小面积 +/-")
        print("  'w'/'s' - Kalman process_noise +/- (响应/平滑)")
        print("  't'/'y' - Kalman meas_noise +/- (平滑/响应)")
        print("  'u' - 重置 Kalman")
        print("  'a'/'z' - Kp_x +/-")
        print("  'd'/'c' - Ki_x +/-")
        print("  'f'/'v' - Kd_x +/-")
        print("  'g'/'b' - Kp_y +/-")
        print("  'h'/'n' - Ki_y +/-")
        print("  'j'/'m' - Kd_y +/-")
        print("  'r' - 清零 PID 积分")
        print("  'o'/'l' - 误差钳位 +/- (减小可抑制转弯高速)")

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

                    # 更新 PID 目标 (预测框不驱动电机, 防止偏离)
                    if len(smoothed_rectangles) > 0:
                        target_rect = smoothed_rectangles[0]
                        is_predicted = target_rect.get('predicted', False)
                        ox = target_rect['offset_x']
                        oy = -target_rect['offset_y']  # 图像 y 轴翻转
                        if self._pid_loop is not None:
                            # 仅真实检测驱动电机; 预测仅用于 Kalman 内部状态保持
                            self._pid_loop.update_target(ox, oy, not is_predicted)
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
                    self._kf.kf.processNoiseCov *= 1.3
                    pn = self._kf.kf.processNoiseCov[0, 0]
                    print(f"Kalman process_noise: {pn:.4f} (响应更快)")
                elif key == ord('s'):
                    self._kf.kf.processNoiseCov *= 0.7
                    pn = self._kf.kf.processNoiseCov[0, 0]
                    print(f"Kalman process_noise: {pn:.4f} (更平滑)")
                elif key == ord('t'):
                    self._kf.kf.measurementNoiseCov *= 1.3
                    mn = self._kf.kf.measurementNoiseCov[0, 0]
                    print(f"Kalman meas_noise: {mn:.4f} (更平滑)")
                elif key == ord('y'):
                    self._kf.kf.measurementNoiseCov *= 0.7
                    mn = self._kf.kf.measurementNoiseCov[0, 0]
                    print(f"Kalman meas_noise: {mn:.4f} (响应更快)")
                elif key == ord('u'):
                    self._kf.reset()
                    self.last_valid_rect = None
                    self.smoothed_rect = None
                    self.disappear_counter = 0
                    self.detection_counter = 0
                    print("Kalman 已重置")
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
                elif self._pid_loop is not None and key == ord('o'):
                    self._pid_loop.adjust_error_clamp(+10)
                    print(f"误差钳位: {self._pid_loop.pid_x.error_clamp}")
                elif self._pid_loop is not None and key == ord('l'):
                    self._pid_loop.adjust_error_clamp(-10)
                    print(f"误差钳位: {self._pid_loop.pid_x.error_clamp}")

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
        # ========== 配置区 ==========
        # 选择1：使用摄像头
        detector = RectangleDetector(source=11, enable_pid=True)

        # 选择2：使用视频文件（取消注释下面两行，注释上面那行）
        # video_path = "test.mp4"
        # detector = RectangleDetector(source=video_path, is_video_file=True, enable_pid=False)
        # ============================

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
