# OS08A10: 1080p60 → 1080p120 патч `iv009_isp_sensor.ko`

Короткий отчёт: что именно меняется в модуле и как воспроизвести результат.

## Файлы

| Файл | Назначение |
|---|---|
| `patch_ko.py` | Патчит готовый `.ko` (2 байта значений HTS), проверяет результат |
| `README.md` | Это пояснение |

## Что меняется

В таблице режима `setting_1920_1080_4lane_1440m_60fps[]` сенсорного модуля
уменьшен вдвое горизонтальный период строки **HTS**:

| Регистр | Было (60 fps) | Стало (120 fps) |
|---|---|---|
| `0x380c/0x380d` (HTS) | `0x0818` = 2072 | `0x040c` = 1036 |
| `0x380e/0x380f` (VTS) | `0x0486` = 1158 | без изменений |

```
fps = SCLK / (HTS · VTS) = 144·10⁶ / (1036 · 1158) = 120.0
```

VTS не меняется, новые режимы в `supported_modes[]` не добавляются.
Режим сенсора — 2×2 binning, поэтому HTS = 1036 меньше 1920 — это нормально.

В `.ko` запись таблицы занимает 16 байт:
`uint32 address`, `uint32 value`, `uint32 mask`, `uint8 len` + padding.
Скрипт находит уникальную последовательность из пяти записей
`0x380b…0x380f` и меняет только `value` у `0x380c` и `0x380d`.

## Использование

```bash
cd /home/pi/opencv_projects/prj_1/patch_scripts

# Пропатчить штатный модуль (60 fps)
python3 patch_ko.py \
  /home/pi/Desktop/empty_folder/OS08A10_65fps_2026-10-18/modules/iv009_isp_sensor_ORIG_BPI.ko \
  -o iv009_isp_sensor_1080p120_binpatch.ko

# Только проверить, что за модуль
python3 patch_ko.py --check iv009_isp_sensor_1080p120_binpatch.ko
```

Для эталонного `iv009_isp_sensor_ORIG_BPI.ko` результат имеет MD5
`629d1fab64a6e274e4854d60df31a3bc` (размер не меняется).

## Установка

```bash
SENS=/lib/modules/$(uname -r)/kernel/drivers/amlogic/isp_module/subdev/sensor
sudo cp iv009_isp_sensor_1080p120_binpatch.ko "$SENS/iv009_isp_sensor.ko"
sudo reboot
```

Нужна **полная перезагрузка**. `iv009_isp.ko` не выгружается (`refcnt=2`),
а `rmmod`/`insmod` без перезагрузки приводят к Oops.

## Проверка

```bash
md5sum /lib/modules/$(uname -r)/kernel/drivers/amlogic/isp_module/subdev/sensor/iv009_isp_sensor.ko
lsmod | grep iv009
```

Ожидаемые MD5:

| Модуль | MD5 |
|---|---|
| Штатный BPI, 60 fps | `95db2e5fc780debd68b891a374d3c279` |
| Пересобранный из исходников, 120 fps | `a171c86b0ed4f3410d78fbdbd7b665f4` |
| Бинарный патч скриптом (от штатного BPI) | `629d1fab64a6e274e4854d60df31a3bc` |

Быстрая проверка fps в приложении:

```bash
cd /home/pi/opencv_projects/prj_1
./build/myapp_modular --source camera --max-fps-raw --no-stab --no-gui --duration 10
```

Смотрите `capfps=` в строках `[PERF]`: ожидается ~100–120 (зависит от
температуры и фоновой нагрузки). `v4l2-ctl` для замера fps не использовать —
он показывает недостоверные значения.

## Откат на 60 fps

```bash
sudo cp /home/pi/Desktop/empty_folder/OS08A10_65fps_2026-10-18/modules/iv009_isp_sensor_ORIG_BPI.ko \
  /lib/modules/$(uname -r)/kernel/drivers/amlogic/isp_module/subdev/sensor/iv009_isp_sensor.ko
sudo reboot
```

## Важно

- На плате использовался модуль, **пересобранный из исходников Khadas** с правкой
  HTS (`a171c86b0ed4f3410d78fbdbd7b665f4`). Скрипт делает то же самое на уровне
  готового `.ko` — без тулчейна и пересборки ядра. MD5 результата отличается,
  потому что за основу берётся другой (штатный) бинарник.
- Бинарный вариант проверен побайтово (`modinfo`, ELF, наличие таблицы HTS);
  с перезагрузкой на живой системе проверялся пересобранный модуль `a171…`.
- `iv009_isp.ko` (патч recovery-loop, MD5 `eb5c71cc58bae1bd65cef2d9e9c97846`)
  к частоте кадров отношения не имеет и скриптом не затрагивается.
- `/dev/video0` монополизируется: перед запуском приложения закройте
  `lightdiag` и наоборот.
