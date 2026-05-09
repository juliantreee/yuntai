# bilibili搜索学不会电磁场看教程
# 离线运行代码 - 纯图像识别版

import time
import os
import sys

from media.sensor import *
from media.display import *
from media.media import *
from time import ticks_ms

sensor = None

try:
    print("camera_test")

    sensor = Sensor(width=320, height=320)
    sensor.reset()
    sensor.set_framesize(width=320, height=320)
    sensor.set_pixformat(Sensor.RGB565)

    Display.init(Display.ST7701, width=800, height=480, to_ide=False)
    MediaManager.init()
    sensor.run()

    # 等sensor完全启动
    time.sleep_ms(500)

    clock = time.clock()

    # 用于矩形定位的目标坐标
    target_x = 0
    target_y = 0
    target_locked = False

    while True:
        clock.tick()
        os.exitpoint()
        img = sensor.snapshot(chn=CAM_CHN_ID_0)

        # ====== 矩形识别 ======
        img_rect = img.to_grayscale()
        img_rect = img_rect.binary([(82, 212)])
        rects = img_rect.find_rects(threshold=3000)

        rect_count = 0
        if not rects == None:
            rect_count = len(rects)
            for rect in rects:
                corner = rect.corners()

                # 绘制矩形边框
                img.draw_line(corner[0][0], corner[0][1], corner[1][0], corner[1][1], color=(0, 255, 0), thickness=3)
                img.draw_line(corner[2][0], corner[2][1], corner[1][0], corner[1][1], color=(0, 255, 0), thickness=3)
                img.draw_line(corner[2][0], corner[2][1], corner[3][0], corner[3][1], color=(0, 255, 0), thickness=3)
                img.draw_line(corner[0][0], corner[0][1], corner[3][0], corner[3][1], color=(0, 255, 0), thickness=3)

                # 计算中心点
                center_x = (corner[0][0] + corner[1][0] + corner[2][0] + corner[3][0]) // 4
                center_y = (corner[0][1] + corner[1][1] + corner[2][1] + corner[3][1]) // 4

                # 绘制中心点
                img.draw_circle(center_x, center_y, 5, color=(255, 0, 0), thickness=2)
                img.draw_string_advanced(center_x + 10, center_y, 20, f"({center_x},{center_y})", color=(255, 255, 0))

                # 如果识别到2个矩形，锁定第一个的坐标
                if rect_count >= 2 and not target_locked:
                    target_x = center_x
                    target_y = center_y
                    target_locked = True
                    print("锁定目标坐标: ({}, {})".format(target_x, target_y))

        # ====== 色块识别 ======
        blobs = img.find_blobs([(41, 57, 31, 83, 13, 71)], False,
                               (0, 0, 320, 320), x_stride=10, y_stride=10,
                               pixels_threshold=1500, margin=True)

        for blob in blobs:
            img.draw_rectangle(blob.x(), blob.y(), blob.w(), blob.h(), color=(0, 255, 0), thickness=4, fill=False)

            blob_cx = blob.x() + blob.w() // 2
            blob_cy = blob.y() + blob.h() // 2

            # 绘制色块中心
            img.draw_circle(blob_cx, blob_cy, 4, color=(255, 255, 0), thickness=2)
            img.draw_string_advanced(blob_cx + 10, blob_cy, 20, f"({blob_cx},{blob_cy})", color=(0, 255, 255))

        # ====== 如果锁定了目标，绘制目标标记 ======
        if target_locked:
            img.draw_cross(target_x, target_y, color=(255, 0, 0), size=10, thickness=2)
            img.draw_string_advanced(target_x + 15, target_y - 10, 25, "TARGET", color=(255, 0, 0))

        # 显示状态信息
        img.draw_string_advanced(5, 5, 30, "fps: {:.1f}".format(clock.fps()), color=(255, 0, 0))
        img.draw_string_advanced(5, 35, 25, "rects: {}".format(rect_count), color=(0, 255, 0))
        img.draw_string_advanced(5, 60, 25, "blobs: {}".format(len(blobs)), color=(0, 255, 255))

        status = "目标已锁定" if target_locked else "等待定位..."
        img.draw_string_advanced(5, 90, 25, status, color=(255, 255, 0))

        Display.show_image(img, x=(800-320)//2, y=(480-320)//2)

except KeyboardInterrupt as e:
    print("用户停止: ", e)
except BaseException as e:
    print(f"异常: {e}")
finally:
    if isinstance(sensor, Sensor):
        sensor.stop()
    Display.deinit()
    os.exitpoint(os.EXITPOINT_ENABLE_SLEEP)
    time.sleep_ms(100)
    MediaManager.deinit()
