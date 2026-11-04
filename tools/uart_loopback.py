#!/usr/bin/env python3
"""Проверка UART loopback и отправки CMD_GYRO_CORRECTION (#75).

Режимы:
  self  — отправить один пакет #75 и проверить, что он вернулся байт-в-байт.
  app   — запустить myapp_modular с этим портом и посчитать реальные пакеты #75.

Требуется pyserial. Запуск от пользователя с доступом к порту (dialout) или sudo.

Примеры:
  python3 tools/uart_loopback.py --port /dev/ttyUSB0 --mode self
  python3 tools/uart_loopback.py --port /dev/ttyUSB0 --mode app --duration 5
"""

import argparse
import subprocess
import sys
import time

try:
    import serial
except ImportError:
    print("Нужен pyserial: sudo apt install python3-serial", file=sys.stderr)
    raise SystemExit(2)


def build_packet(x=100, y=-200, z=300, heading=0, imu=0):
    payload = bytes([imu]) + \
        int(x).to_bytes(2, 'little', signed=True) + \
        int(y).to_bytes(2, 'little', signed=True) + \
        int(z).to_bytes(2, 'little', signed=True) + \
        int(heading).to_bytes(2, 'little', signed=True)
    header = bytes([0x3E, 75, len(payload), (75 + len(payload)) & 0xFF])
    return header + payload + bytes([sum(payload) & 0xFF])


def parse_packets(buf):
    packets = []
    i = 0
    while i + 14 <= len(buf):
        if buf[i] == 0x3E and buf[i + 1] == 75 and buf[i + 2] == 9 and buf[i + 3] == 0x54:
            pkt = buf[i:i + 14]
            if (sum(pkt[4:13]) & 0xFF) == pkt[13]:
                packets.append(pkt)
                i += 14
                continue
        i += 1
    return packets


def mode_self(port, baud):
    pkt = build_packet()
    with serial.Serial(port, baud, timeout=1.0, write_timeout=1.0) as s:
        s.reset_input_buffer()
        time.sleep(0.1)
        s.write(pkt)
        s.flush()
        got = s.read(14)
    print("sent:", pkt.hex(' ').upper())
    print("got :", got.hex(' ').upper() if got else "NONE")
    ok = got == pkt
    print("RESULT:", "OK" if ok else "FAIL")
    return 0 if ok else 1


def mode_app(port, baud, app, duration, extra):
    pkt = build_packet()
    with serial.Serial(port, baud, timeout=0.2) as s:
        s.reset_input_buffer()
        cmd = [app, '--gyro-only', '--source', 'camera',
               '--gyro-port', port, '--gyro-enable', '--duration', str(duration)]
        cmd += extra
        print("run:", ' '.join(cmd), flush=True)
        proc = subprocess.Popen(cmd)
        buf = b''
        end = time.time() + duration + 2.0
        while time.time() < end:
            try:
                data = s.read(4096)
            except Exception:
                data = b''
            if data:
                buf += data
            if proc.poll() is not None and not data:
                break
        proc.wait(timeout=3)
    packets = parse_packets(buf)
    print(f"bytes={len(buf)} packets={len(packets)} "
          f"avg={len(packets) / max(duration, 0.1):.1f} Hz")
    for p in packets[:3]:
        print(" ", p.hex(' ').upper())
    return 0 if packets else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--port', default='/dev/ttyUSB0')
    ap.add_argument('--baud', type=int, default=115200)
    ap.add_argument('--mode', choices=['self', 'app'], default='self')
    ap.add_argument('--app', default='./build/myapp_modular')
    ap.add_argument('--duration', type=float, default=5.0)
    ap.add_argument('args', nargs=argparse.REMAINDER,
                    help='дополнительные аргументы myapp_modular после --')
    ns = ap.parse_args()

    extra = list(ns.args)
    if extra and extra[0] == '--':
        extra = extra[1:]

    if ns.mode == 'self':
        return mode_self(ns.port, ns.baud)
    return mode_app(ns.port, ns.baud, ns.app, ns.duration, extra)


if __name__ == '__main__':
    raise SystemExit(main())
