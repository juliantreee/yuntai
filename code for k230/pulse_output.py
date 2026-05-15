import time
from machine import PWM, Pin

PULSE_COUNT = 80000
FREQUENCY = 64000  # 64kHz
DURATION = PULSE_COUNT / FREQUENCY  # 1.25s

# 1. 先将GPIO41设为高电平
pin41 = Pin(41, Pin.OUT)
pin41.value(1)
print("GPIO41 set to HIGH")

# 2. 在GPIO40上以64kHz发送80000个脉冲
pwm40 = PWM(Pin(40), freq=FREQUENCY, duty=50)
print(f"Sending {PULSE_COUNT} pulses at {FREQUENCY}Hz for {DURATION}s...")

time.sleep(DURATION)

# 3. 停止PWM，释放资源
pwm40.deinit()
print("Done")
