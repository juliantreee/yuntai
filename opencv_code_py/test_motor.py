"""
Simple stepper motor test — runs motor forward/backward at a fixed speed.
"""
import time
import sys
from pwm_control import PWMMotor

# 硬件配置 (与 main.py 一致)
PWM_CHIP_X = 3
PWM_CHANNEL_X = 0
DIR_GPIO_X = 102

PWM_CHIP_Y = 4
PWM_CHANNEL_Y = 0
DIR_GPIO_Y = 111

STEPS_PER_REV = 3200

TEST_RPM = 60  # 测试转速, 可修改
TEST_DURATION = 3  # 每个方向转多少秒


def test_motor(name, pwm_chip, pwm_channel, dir_gpio):
    print(f"\n{'='*50}")
    print(f"测试 {name} 轴电机 (PWM chip{pwm_chip} ch{pwm_channel}, GPIO{dir_gpio})")
    print(f"转速: {TEST_RPM} RPM, 每方向 {TEST_DURATION}s")
    print(f"{'='*50}")

    motor = PWMMotor(pwm_chip, pwm_channel, dir_gpio, STEPS_PER_REV)
    print(f"PWM 路径: {motor._pwm_path}")

    try:
        # 正转
        print(f"\n>>> 正转 {TEST_RPM} RPM ...")
        motor.set_speed(TEST_RPM)
        time.sleep(TEST_DURATION)

        # 反转
        print(f">>> 反转 {TEST_RPM} RPM ...")
        motor.set_speed(-TEST_RPM)
        time.sleep(TEST_DURATION)

        # 停止
        print(">>> 停止")
        motor.stop()

    finally:
        motor.cleanup()
        print(f"{name} 轴测试完成\n")


if __name__ == "__main__":
    axis = sys.argv[1] if len(sys.argv) > 1 else "both"

    if axis in ("x", "both"):
        test_motor("X", PWM_CHIP_X, PWM_CHANNEL_X, DIR_GPIO_X)
    if axis in ("y", "both"):
        test_motor("Y", PWM_CHIP_Y, PWM_CHANNEL_Y, DIR_GPIO_Y)
