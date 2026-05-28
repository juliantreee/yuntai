import serial
import threading

SERIAL_PORT = '/dev/ttyS0'
BAUD_RATE = 115200

def reader(ser):
    while True:
        try:
            data = ser.readline()
            if data:
                print(data.decode(errors='replace'), end='', flush=True)
        except Exception as e:
            print(f"\n读取错误: {e}")
            break

def main():
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    print(f"已连接 {SERIAL_PORT} @ {BAUD_RATE} baud")
    print("输入文字发送，按 Ctrl+C 退出")

    t = threading.Thread(target=reader, args=(ser,), daemon=True)
    t.start()

    try:
        while True:
            line = input()
            ser.write((line + '\n').encode())
    except KeyboardInterrupt:
        print("\n退出")
    finally:
        ser.close()

if __name__ == '__main__':
    main()
