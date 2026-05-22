import cv2
import numpy as np
import time
from typing import List, Dict, Optional, Any, Union
from collections import deque


class RectangleDetector:
    def __init__(self, source: Union[int, str] = 0, width: int = 640, height: int = 480, is_video_file: bool = False):
        """
        source: 摄像头ID(整数) 或 视频文件路径(字符串)
        is_video_file: True表示source是视频文件路径
        """
        # 根据类型分别处理
        if isinstance(source, int):
            self.camera = cv2.VideoCapture(source)
        elif isinstance(source, str):
            self.camera = cv2.VideoCapture(source)
        else:
            self.camera = cv2.VideoCapture(str(source))

        if not self.camera.isOpened():
            raise OSError(f"Could not open source: {source}")

        # 如果不是视频文件，设置摄像头参数
        if not is_video_file and isinstance(source, int):
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.camera.set(cv2.CAP_PROP_FPS, 60)

        # 获取实际分辨率
        actual_width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 画面中心坐标
        self.frame_center_x = actual_width // 2
        self.frame_center_y = actual_height // 2

        # 检测参数设置
        self.min_area = 2000
        self.max_area = 200000
        self.canny_low_threshold = 50
        self.canny_high_threshold = 150
        self.blur_kernel_size = 5
        self.track_largest = True

        # 稳定性参数
        self.smoothing_window = 5  # 平滑窗口大小
        self.position_history = deque(maxlen=self.smoothing_window)  # 位置历史
        self.size_history = deque(maxlen=self.smoothing_window)  # 尺寸历史
        self.detection_threshold = 3  # 需要连续检测到多少次才认为有效
        self.detection_counter = 0  # 检测计数器
        self.last_valid_rect = None  # 上一个有效的矩形
        self.max_disappear_frames = 10  # 最大消失帧数
        self.disappear_counter = 0  # 消失计数器

        # 平滑后的矩形数据
        self.smoothed_rect = None

        print(f"视频源已初始化: {actual_width}x{actual_height}")
        print(f"画面中心点: ({self.frame_center_x}, {self.frame_center_y})")
        print(f"平滑窗口: {self.smoothing_window} 帧")

    def calculate_iou(self, rect1, rect2):
        """计算两个矩形的IoU（交并比）"""
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
        # 灰度 + 高斯模糊
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (self.blur_kernel_size, self.blur_kernel_size), 0)

        # 二值化提取黑色粗框区域（黑色 → 白，其余 → 黑）22, 39,80,255
        _, binary = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY_INV)

        # 形态学闭运算连接断裂
        kernel = np.ones((5, 5), np.uint8)
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=3)

        # 使用 RETR_TREE 获取层级关系，外框的 child 即内框
        contours, hierarchy = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # 产生 edges 用于可视化
        edges = cv2.Canny(blurred, self.canny_low_threshold, self.canny_high_threshold)

        rectangles: List[Dict[str, Any]] = []
        if hierarchy is None or len(contours) == 0:
            return rectangles, edges

        h = hierarchy[0]

        def build_rect(contour, idx):
            """将轮廓转为矩形字典"""
            rect_area = cv2.contourArea(contour)
            x, y, w_box, h_box = cv2.boundingRect(contour)
            cx = x + w_box // 2
            cy = y + h_box // 2
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            return {
                'center_x': cx,
                'center_y': cy,
                'offset_x': cx - self.frame_center_x,
                'offset_y': cy - self.frame_center_y,
                'x': x,
                'y': y,
                'width': w_box,
                'height': h_box,
                'area': rect_area,
                'contour': approx,
                'contour_idx': idx,
            }

        for i, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            if area < self.min_area or area > self.max_area:
                continue

            child_idx = int(h[i][2])
            # 必须有子轮廓（内框）才认为是粗框矩形
            if child_idx == -1:
                continue

            child_cnt = contours[child_idx]
            child_area = cv2.contourArea(child_cnt)
            if child_area < self.min_area * 0.3:
                continue

            outer_rect = build_rect(cnt, i)
            inner_rect = build_rect(child_cnt, child_idx)

            # 验证嵌套关系：内框在外框内部，且面积比合理
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
        """平滑检测结果，减少闪烁"""
        if len(rectangles) == 0:
            # 没有检测到矩形
            self.disappear_counter += 1

            if self.last_valid_rect is not None and self.disappear_counter <= self.max_disappear_frames:
                # 短时间内保持上一个有效矩形
                return [self.last_valid_rect]
            else:
                # 清空历史数据
                self.position_history.clear()
                self.size_history.clear()
                self.detection_counter = 0
                return []
        else:
            # 检测到矩形
            self.disappear_counter = 0
            current_rect = rectangles[0]

            # 如果有上一个有效矩形，检查是否为同一个目标
            if self.last_valid_rect is not None:
                iou = self.calculate_iou(current_rect, self.last_valid_rect)
                if iou < 0.3:  # IoU太低，可能是不同的目标
                    self.detection_counter = 0

            # 增加检测计数
            self.detection_counter = min(self.detection_counter + 1, self.detection_threshold)

            # 更新历史数据
            self.position_history.append((current_rect['center_x'], current_rect['center_y']))
            self.size_history.append((current_rect['width'], current_rect['height']))

            # 只有在连续检测到足够帧数后才输出平滑结果
            if self.detection_counter >= self.detection_threshold:
                # 计算平滑位置
                if len(self.position_history) > 0:
                    avg_center_x = int(np.mean([p[0] for p in self.position_history]))
                    avg_center_y = int(np.mean([p[1] for p in self.position_history]))
                    avg_width = int(np.mean([s[0] for s in self.size_history]))
                    avg_height = int(np.mean([s[1] for s in self.size_history]))

                    # 创建平滑后的外框
                    smoothed = current_rect.copy()
                    smoothed['center_x'] = avg_center_x
                    smoothed['center_y'] = avg_center_y
                    smoothed['offset_x'] = avg_center_x - self.frame_center_x
                    smoothed['offset_y'] = avg_center_y - self.frame_center_y
                    smoothed['width'] = avg_width
                    smoothed['height'] = avg_height
                    smoothed['x'] = avg_center_x - avg_width // 2
                    smoothed['y'] = avg_center_y - avg_height // 2

                    # 内框同步平滑（相对位置保持）
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

            # 还没达到检测阈值，返回上一个有效矩形或空
            if self.last_valid_rect is not None:
                return [self.last_valid_rect]
            return []

    def draw_result(self, frame: np.ndarray, rectangles: List[Dict[str, Any]], target_index: int = 0,
                    paused: bool = False) -> np.ndarray:
        result_frame = frame.copy()

        # 绘制中心十字线
        cv2.line(result_frame, (self.frame_center_x, 0),
                 (self.frame_center_x, result_frame.shape[0]), (255, 255, 255), 1)
        cv2.line(result_frame, (0, self.frame_center_y),
                 (result_frame.shape[1], self.frame_center_y), (255, 255, 255), 1)
        cv2.circle(result_frame, (self.frame_center_x, self.frame_center_y), 5, (255, 255, 255), -1)

        # 绘制目标矩形
        if len(rectangles) > 0:
            target_rect = rectangles[target_index]

            if self.smoothed_rect is not None:
                sr = self.smoothed_rect
                # 绘制平滑后的外框（绿色）
                cv2.rectangle(result_frame,
                              (sr['x'], sr['y']),
                              (sr['x'] + sr['width'], sr['y'] + sr['height']),
                              (0, 255, 0), 2)
                cv2.putText(result_frame, "Outer",
                            (sr['x'], sr['y'] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # 绘制平滑后的内框（蓝色）
                if sr.get('inner_rect') is not None:
                    ir = sr['inner_rect']
                    cv2.rectangle(result_frame,
                                  (ir['x'], ir['y']),
                                  (ir['x'] + ir['width'], ir['y'] + ir['height']),
                                  (255, 0, 0), 2)
                    cv2.putText(result_frame, "Inner",
                                (ir['x'], ir['y'] - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    # 内框偏移
                    cv2.putText(result_frame,
                                f"Inner offset: ({ir['offset_x']}, {ir['offset_y']})",
                                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                # 外框中心点
                cv2.circle(result_frame,
                           (sr['center_x'], sr['center_y']),
                           6, (0, 255, 0), -1)

                # 外框偏移
                cv2.putText(result_frame,
                            f"Outer offset: ({sr['offset_x']}, {sr['offset_y']})",
                            (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # 框线宽度估算
                if sr.get('inner_rect') is not None:
                    ir = sr['inner_rect']
                    border_w = (sr['width'] - ir['width']) // 2
                    border_h = (sr['height'] - ir['height']) // 2
                    cv2.putText(result_frame,
                                f"Border: {border_w}x{border_h}px",
                                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            # 原始检测轮廓（红色虚线参考）
            cv2.drawContours(result_frame, [target_rect['contour']], -1, (0, 0, 255), 1)
            if target_rect.get('inner_rect') is not None:
                cv2.drawContours(result_frame, [target_rect['inner_rect']['contour']], -1,
                                 (0, 0, 255), 1)

        # 显示状态信息
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

        return result_frame

    def run(self, display: bool = True) -> None:
        fps_counter = 0
        fps_time = time.time()
        fps = 0
        target_rect: Optional[Dict[str, Any]] = None
        paused = False
        last_frame = None
        last_edges = None
        last_rectangles: List[Dict[str, Any]] = []

        print("开始检测")
        print("按键控制:")
        print("  'q' - 退出")
        print("  '空格' - 暂停/继续")
        print("  '+' - 增大最小面积")
        print("  '-' - 减小最小面积")
        print("  'w' - 增大平滑窗口")
        print("  's' - 减小平滑窗口")

        while True:
            if not paused:
                ret, frame = self.camera.read()
                if not ret:
                    print("视频结束或无法读取画面")
                    break

                # 检测矩形
                rectangles, edges = self.detect_rectangle(frame)

                # 平滑处理
                smoothed_rectangles = self.smooth_detection(rectangles)

                # 保存当前帧和检测结果
                last_frame = frame.copy()
                last_edges = edges.copy() if edges is not None else None
                last_rectangles = smoothed_rectangles

                # 获取平滑后的矩形
                if len(smoothed_rectangles) > 0:
                    target_rect = smoothed_rectangles[0]
                else:
                    target_rect = None

            # 显示部分（暂停时仍然显示最后一帧的检测结果）
            if display and last_frame is not None:
                # FPS计算（只在非暂停时更新）
                if not paused:
                    fps_counter += 1
                    if time.time() - fps_time >= 1:
                        fps = fps_counter
                        fps_counter = 0
                        fps_time = time.time()

                # 绘制结果（使用保存的检测结果）
                result_frame = self.draw_result(last_frame, last_rectangles, paused=paused)
                current_edges = last_edges

                # 显示FPS
                cv2.putText(result_frame, f"FPS: {fps}", (10, result_frame.shape[0] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

                # 显示画面
                cv2.imshow("Rectangle Detection", result_frame)
                if current_edges is not None:
                    cv2.imshow("Edges", current_edges)

            # 键盘控制
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):  # 空格键暂停/继续
                paused = not paused
                print("⏸️ 暂停" if paused else "▶️ 继续")
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

        self.camera.release()
        cv2.destroyAllWindows()


def main() -> None:
    try:
        # ========== 配置区 ==========
         #选择1：使用摄像头（取消注释下面这行，注释视频文件那行）
        detector = RectangleDetector(source=0)

        # 选择2：使用视频文件（修改为你的视频路径）
        #video_path = "test.mp4"  # 👈 改成你的视频文件路径
        #detector = RectangleDetector(source=video_path, is_video_file=True)
        # ============================

        detector.run(display=True)
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()