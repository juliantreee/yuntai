# bilibili搜索学不会电磁场看教程
# 极限帧率版 + IDE显示

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

    Display.init(Display.ST7701, width=800, height=480, to_ide=True)
    MediaManager.init()
    sensor.run()

    time.sleep_ms(500)

    clock = time.clock()

    target_locked = False
    target_x = 0
    target_y = 0
    frame_count = 0

    while True:
        clock.tick()
        os.exitpoint()
        img = sensor.snapshot(chn=CAM_CHN_ID_0)

        # 每3帧做一次矩形检测
        if frame_count % 3 == 0:
            img_edges = img.to_grayscale().find_edges(image.EDGE_CANNY, threshold=(50, 80))
            rects = img_edges.find_rects(threshold=3000)

            if not rects == None:
                for rect in rects:
                    corner = rect.corners()

                    img.draw_line(corner[0][0], corner[0][1], corner[1][0], corner[1][1], color=(0, 255, 0), thickness=1)
                    img.draw_line(corner[1][0], corner[1][1], corner[2][0], corner[2][1], color=(0, 255, 0), thickness=1)
                    img.draw_line(corner[2][0], corner[2][1], corner[3][0], corner[3][1], color=(0, 255, 0), thickness=1)
                    img.draw_line(corner[3][0], corner[3][1], corner[0][0], corner[0][1], color=(0, 255, 0), thickness=1)

                    center_x = (corner[0][0] + corner[1][0] + corner[2][0] + corner[3][0]) >> 2
                    center_y = (corner[0][1] + corner[1][1] + corner[2][1] + corner[3][1]) >> 2

                    if len(rects) >= 2 and not target_locked:
                        target_x = center_x
                        target_y = center_y
                        target_locked = True

        # 色块检测：stride加大
        blobs = img.find_blobs([(41, 57, 31, 83, 13, 71)], False,
                               (0, 0, 320, 320), x_stride=30, y_stride=30,
                               pixels_threshold=2000, margin=True)

        # 只画第一个色块
        if blobs and frame_count % 3 == 0:
            img.draw_rectangle(blobs[0].x(), blobs[0].y(), blobs[0].w(), blobs[0].h(), color=(0, 255, 255), thickness=1, fill=False)

        # 每5帧显示FPS
        if frame_count % 5 == 0:
            img.draw_string_advanced(5, 5, 18, str(int(clock.fps())), color=(255, 0, 0))

        if target_locked and frame_count % 3 == 0:
            img.draw_cross(target_x, target_y, color=(255, 0, 0), size=6, thickness=1)

        Display.show_image(img, x=(800-320)//2, y=(480-320)//2)
        img.compress_for_ide()
        frame_count += 1

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
