import serial
import struct

SERIAL_PORT = '/dev/ttyS0'
BAUD_RATE = 115200

def main():
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    print(f"已连接 {SERIAL_PORT} @ {BAUD_RATE} baud, 等待数据...")

    buf = bytearray()
    try:
        while True:
            buf.extend(ser.read(ser.in_waiting or 1))
            while len(buf) >= 4:
                ox, oy = struct.unpack('<hh', buf[:4])
                del buf[:4]
                print(f"ox={ox:6d}  oy={oy:6d}")
    except KeyboardInterrupt:
        print("\n退出")
    finally:
        ser.close()

if __name__ == '__main__':
    main()
