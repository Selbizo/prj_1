#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
iv009_isp_sensor.ko: 1920x1080 60 fps -> 120 fps (HTS 1036).

Меняет в готовом модуле ядра только два значения таблицы
setting_1920_1080_4lane_1440m_60fps:

    0x380c (HTS high): 0x08 -> 0x04
    0x380d (HTS low):  0x18 -> 0x0c

VTS (0x380e/0x380f = 0x0486 = 1158) не меняется.

    fps = SCLK / (HTS * VTS) = 144e6 / (1036 * 1158) = 120.0

Структура acam_reg_t в модуле (16 байт на запись):
    uint32 address, uint32 value, uint32 mask, uint8 len, 3 bytes padding

Использование:
    python3 patch_ko.py iv009_isp_sensor_ORIG_BPI.ko
    python3 patch_ko.py iv009_isp_sensor_ORIG_BPI.ko -o my_120fps.ko
    python3 patch_ko.py --check iv009_isp_sensor.ko

Скрипт не устанавливает модуль и не перезагружает систему.
"""

from __future__ import annotations

import argparse
import hashlib
import struct
import sys
from pathlib import Path

# 5 записей подряд: 0x380b, 0x380c, 0x380d, 0x380e, 0x380f
# Значения взяты из setting_1920_1080_4lane_1440m_60fps[].
OLD_SEQ = [
    (0x380B, 0x0038),
    (0x380C, 0x0008),
    (0x380D, 0x0018),
    (0x380E, 0x0004),
    (0x380F, 0x0086),
]
NEW_SEQ = [
    (0x380B, 0x0038),
    (0x380C, 0x0004),  # HTS = 1036
    (0x380D, 0x000C),
    (0x380E, 0x0004),  # VTS = 1158, без изменений
    (0x380F, 0x0086),
]

# Смещения value внутри записи и внутри блока из 5 записей.
VALUE_OFFSET = 4
ENTRY_SIZE = 16

# Известные контрольные суммы (для справки, не являются обязательными).
MD5_STOCK_BPI = "95db2e5fc780debd68b891a374d3c279"
MD5_REBUILT_120 = "a171c86b0ed4f3410d78fbdbd7b665f4"


def pack_entry(address: int, value: int) -> bytes:
    """Одна запись acam_reg_t: address, value, mask=0xff, len=1 + padding."""
    return struct.pack("<IIIB3x", address, value, 0xFF, 1)


def pack_seq(entries: list[tuple[int, int]]) -> bytes:
    return b"".join(pack_entry(address, value) for address, value in entries)


OLD_PATTERN = pack_seq(OLD_SEQ)
NEW_PATTERN = pack_seq(NEW_SEQ)


def md5(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def find_all(data: bytes, pattern: bytes) -> list[int]:
    offsets: list[int] = []
    start = 0
    while True:
        pos = data.find(pattern, start)
        if pos < 0:
            return offsets
        offsets.append(pos)
        start = pos + 1


def inspect(data: bytes) -> tuple[list[int], list[int]]:
    return find_all(data, OLD_PATTERN), find_all(data, NEW_PATTERN)


def verdict(old_hits: list[int], new_hits: list[int]) -> str:
    if len(old_hits) == 1 and len(new_hits) == 1:
        return "штатный модуль 60 fps (таблица 1080p60 ещё не пропатчена)"
    if len(old_hits) == 0 and len(new_hits) >= 1:
        return "модуль уже с 120 fps (таблица 1080p120 присутствует)"
    if len(old_hits) > 1:
        return f"неоднозначно: найдено несколько таблиц 1080p60 ({len(old_hits)})"
    return "неизвестный модуль: целевая таблица не найдена"


def print_report(path: Path, data: bytes, old_hits: list[int], new_hits: list[int]) -> None:
    print(f"Файл:   {path}")
    print(f"Размер: {len(data)} байт")
    print(f"MD5:    {md5(data)}")
    if md5(data) == MD5_STOCK_BPI:
        print("        (эталонный штатный BPI-модуль)")
    elif md5(data) == MD5_REBUILT_120:
        print("        (эталонный пересобранный модуль 120 fps)")
    print(f"Таблица 60 fps  (HTS 0x0818): {len(old_hits)} шт. {[hex(x) for x in old_hits]}")
    print(f"Таблица 120 fps (HTS 0x040c): {len(new_hits)} шт. {[hex(x) for x in new_hits]}")
    print(f"Вердикт: {verdict(old_hits, new_hits)}")


def check_elf(data: bytes, path: Path) -> None:
    if not data.startswith(b"\x7fELF"):
        raise SystemExit(f"Ошибка: {path} не является ELF-файлом (.ko).")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Патч iv009_isp_sensor.ko: 1080p60 -> 1080p120 (HTS 1036).",
    )
    parser.add_argument("input", type=Path, help="исходный .ko (штатный, 60 fps)")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="куда записать пропатченный .ko "
        "(по умолчанию <input>_1080p120_binpatch.ko)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="только проверить файл, ничего не записывать",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="перезаписать существующий output",
    )
    args = parser.parse_args()

    if not args.input.is_file():
        raise SystemExit(f"Ошибка: файл не найден: {args.input}")

    data = args.input.read_bytes()
    check_elf(data, args.input)
    old_hits, new_hits = inspect(data)

    if args.check:
        print_report(args.input, data, old_hits, new_hits)
        return 0

    if not old_hits:
        print_report(args.input, data, old_hits, new_hits)
        print("\nПатчить нечего: модуль уже содержит таблицу 120 fps.")
        return 0

    if len(old_hits) > 1:
        print_report(args.input, data, old_hits, new_hits)
        raise SystemExit(
            "\nОшибка: найдено несколько таблиц 1080p60. "
            "Скрипт не меняет файл, чтобы не задеть лишнее."
        )

    if args.output is None:
        out = args.input.with_name(args.input.stem + "_1080p120_binpatch.ko")
    else:
        out = args.output

    if out.resolve() == args.input.resolve():
        raise SystemExit("Ошибка: output совпадает с input. Укажите другой файл через -o.")

    if out.exists() and not args.force:
        raise SystemExit(f"Ошибка: {out} уже существует. Используйте --force или -o.")

    offset = old_hits[0]
    patched = bytearray(data)

    # Меняем только value у 0x380c и 0x380d; остальные байты не трогаем.
    struct.pack_into("<I", patched, offset + 1 * ENTRY_SIZE + VALUE_OFFSET, 0x0004)
    struct.pack_into("<I", patched, offset + 2 * ENTRY_SIZE + VALUE_OFFSET, 0x000C)

    result = bytes(patched)

    # Проверка: блок стал ровно NEW_PATTERN, старых таблиц не осталось.
    new_old_hits, new_new_hits = inspect(result)
    if result[offset : offset + len(NEW_PATTERN)] != NEW_PATTERN:
        raise SystemExit("Ошибка внутренней проверки: блок не совпал с ожидаемым.")
    if new_old_hits:
        raise SystemExit("Ошибка внутренней проверки: таблица 60 fps осталась.")
    if len(result) != len(data):
        raise SystemExit("Ошибка внутренней проверки: изменился размер файла.")

    out.write_bytes(result)
    out.chmod(args.input.stat().st_mode & 0o777)

    print_report(args.input, data, old_hits, new_hits)
    print(f"\nЗаписан: {out}")
    print(f"MD5:     {md5(result)}")
    print(f"Проверка: таблиц 60 fps — {len(new_old_hits)}, "
          f"таблиц 120 fps — {len(new_new_hits)}")
    print(
        "\nУстановка (нужна полная перезагрузка):\n"
        f"  SENS=/lib/modules/$(uname -r)/kernel/drivers/amlogic/"
        "isp_module/subdev/sensor\n"
        f"  sudo cp {out} \"$SENS/iv009_isp_sensor.ko\"\n"
        "  sudo reboot\n"
        "\nНе используйте rmmod/insmod: iv009_isp.ko не выгружается."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
