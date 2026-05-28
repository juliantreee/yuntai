"""
RK3588 sysfs PWM + GPIO stepper motor control.
Compatible with LubanCat 4 (鲁班猫4) 40-pin header.
"""
import os
import time


class PWMMotor:
    """Single stepper motor controlled by PWM step + DIR GPIO via Linux sysfs."""

    def __init__(self, pwm_chip: int, pwm_channel: int, dir_gpio: int,
                 steps_per_rev: int = 3200):
        self.pwm_chip = pwm_chip
        self.pwm_channel = pwm_channel
        self.dir_gpio = dir_gpio
        self.steps_per_rev = steps_per_rev
        self._pwm_path = f"/sys/class/pwm/pwmchip{pwm_chip}/pwm{pwm_channel}"
        self._enabled = False

        self._export_pwm()
        self._export_gpio()
        self._set_gpio_direction("out")

        # 同步硬件 enable 状态 (防止上次异常退出导致 PWM 仍处于 enable)
        self._sync_enable_state()

    # ---------- PWM sysfs helpers ----------

    def _export_pwm(self):
        if not os.path.exists(self._pwm_path):
            export_path = f"/sys/class/pwm/pwmchip{self.pwm_chip}/export"
            try:
                with open(export_path, "w") as f:
                    f.write(str(self.pwm_channel))
            except (OSError, PermissionError) as e:
                raise RuntimeError(
                    f"Cannot export PWM chip{self.pwm_chip} ch{self.pwm_channel}: {e}"
                )
            time.sleep(0.1)

    def _export_gpio(self):
        gpio_path = f"/sys/class/gpio/gpio{self.dir_gpio}"
        if not os.path.exists(gpio_path):
            try:
                with open("/sys/class/gpio/export", "w") as f:
                    f.write(str(self.dir_gpio))
            except (OSError, PermissionError) as e:
                raise RuntimeError(
                    f"Cannot export GPIO {self.dir_gpio}: {e}"
                )
            time.sleep(0.1)

    def _set_gpio_direction(self, direction: str):
        path = f"/sys/class/gpio/gpio{self.dir_gpio}/direction"
        with open(path, "w") as f:
            f.write(direction)

    def _set_gpio_value(self, value: int):
        path = f"/sys/class/gpio/gpio{self.dir_gpio}/value"
        with open(path, "w") as f:
            f.write(str(value))

    # ---------- Public API ----------

    def set_speed(self, rpm: float):
        """Set motor speed in RPM. Positive = forward (dir=1), negative = reverse (dir=0)."""
        freq = int(abs(rpm) * self.steps_per_rev / 60)

        # 速度太低或为零时直接停转，避免 period 超出硬件范围 (RK3588 PWM)
        min_freq = 100  # 最低 100 Hz, 对应约 1.9 RPM
        if freq < min_freq:
            self.stop()
            return

        if rpm > 0:
            self._set_gpio_value(1)
        elif rpm < 0:
            self._set_gpio_value(0)

        # 将 period 对齐到 100ns 边界，满足 RK3588 PWM 时钟精度要求
        period_ns = int(1e9 / freq)
        period_ns = max(period_ns, 100)
        period_ns = (period_ns // 25) * 25  # 对齐到 25ns (40MHz 时钟的整数倍)
        duty_ns = period_ns // 2

        # 某些 PWM 驱动要求修改 period 前先禁用，且 duty_cycle 须先清零
        self._disable()
        try:
            with open(f"{self._pwm_path}/duty_cycle", "w") as f:
                f.write("0")
        except OSError:
            pass

        with open(f"{self._pwm_path}/period", "w") as f:
            f.write(str(period_ns))
        with open(f"{self._pwm_path}/duty_cycle", "w") as f:
            f.write(str(duty_ns))
        with open(f"{self._pwm_path}/enable", "w") as f:
            f.write("1")
        self._enabled = True

    def _sync_enable_state(self):
        """Read actual PWM enable state from hardware (handles unclean shutdown)."""
        try:
            with open(f"{self._pwm_path}/enable", "r") as f:
                self._enabled = f.read().strip() == "1"
        except OSError:
            self._enabled = False

    def _disable(self):
        """Disable PWM output without updating _enabled flag for internal use."""
        try:
            with open(f"{self._pwm_path}/enable", "w") as f:
                f.write("0")
        except OSError:
            pass

    def stop(self):
        """Disable PWM output (motor free)."""
        self._disable()
        self._enabled = False

    def cleanup(self):
        """Stop PWM and unexport."""
        self.stop()
        try:
            with open(f"/sys/class/pwm/pwmchip{self.pwm_chip}/unexport", "w") as f:
                f.write(str(self.pwm_channel))
        except OSError:
            pass
        try:
            with open(f"/sys/class/gpio/unexport", "w") as f:
                f.write(str(self.dir_gpio))
        except OSError:
            pass
