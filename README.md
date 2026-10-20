# Видеостабилизатор 1920×1080 на Banana Pi CM4 (Amlogic A311D)

Многопоточный конвейер захвата, оценки смещения и стабилизации видео.
Рассчитан на сенсор **OmniVision OS08A10** (MIPI CSI-2, 4 lane, `/dev/video0`)
и режим **1920×1080 @ 120 fps**, который включается патченным модулем ядра
`iv009_isp_sensor.ko`.

Основные возможности:

- захват V4L2-камеры, IP-камеры, видеофайла или папки с кадрами;
- оценка смещения (`dx`, `dy`, `da`) по точкам;
- стабилизация через `warpAffine` (OpenCL/GPU Mali-G52 либо CPU);
- режим максимального fps с сырым захватом YUYV (`--max-fps-raw`);
- режим «только оценка смещения» без стабилизации и окна (`--estimate-only`);
- интерактивное окно с горячими клавишами и HUD.

---

## 1. Платформа и требования

| Компонент | Значение |
|---|---|
| Плата | Banana Pi CM4, Amlogic A311D |
| Ядро | `4.9.241-BPI-CM4` (arm64) |
| Сенсор | OS08A10, `/dev/video0` |
| OpenCV | 4.12.0, `/usr/local/opencv-4.12.0` |
| Компилятор | `/usr/bin/aarch64-linux-gnu-g++` |
| GPU | Mali-G52 (OpenCL) |

Запускать приложение нужно **от пользователя `pi`** (он в группе `video`).
`sudo` для запуска не требуется.

---

## 2. Сборка

Проект собирается через CMake. Бинарник — `build/myapp_modular`.

```bash
cd /home/pi/opencv_projects/prj_1
mkdir -p build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j6
```

Готовый бинарник:

```
/home/pi/opencv_projects/prj_1/build/myapp_modular
```

> **Важно:** собирайте именно `Release` (`-O3`). В уже существующем каталоге
> `build/` в кэше может стоять `CMAKE_BUILD_TYPE=Debug` — тогда конвейер будет
> заметно медленнее. Проверить и переконфигурировать:
>
> ```bash
> grep '^CMAKE_BUILD_TYPE' build/CMakeCache.txt
> cd build && cmake -DCMAKE_BUILD_TYPE=Release .. && make -j6
> ```

Если CMake не находит OpenCV, укажите путь явно:

```bash
cmake -DCMAKE_BUILD_TYPE=Release \
      -DOpenCV_DIR=/usr/local/opencv-4.12.0/lib/cmake/opencv4 ..
```

---

## 3. Модули ядра для 120 fps

### 3.1. Что именно нужно

За 120 fps отвечает **только сенсорный модуль** `iv009_isp_sensor.ko`
с патчем HTS. Остальные модули менять не требуется.

| Модуль | Установленный путь | MD5 | Роль |
|---|---|---|---|
| `iv009_isp_sensor.ko` | `/lib/modules/$(uname -r)/kernel/drivers/amlogic/isp_module/subdev/sensor/` | `a171c86b0ed4f3410d78fbdbd7b665f4` | **Обязателен для 120 fps.** Патч HTS 1036 |
| `iv009_isp.ko` | `/lib/modules/$(uname -r)/kernel/drivers/amlogic/isp_module/v4l2_dev/` | `eb5c71cc58bae1bd65cef2d9e9c97846` | Патч recovery-loop. **На fps не влияет** |
| `iv009_isp_lens.ko` | `/lib/modules/$(uname -r)/kernel/drivers/amlogic/isp_module/subdev/lens/` | `fa47b294fd547a3976dbff9863c213c8` | Штатный, не изменялся |
| `iv009_isp_iq.ko` | `/lib/modules/$(uname -r)/kernel/drivers/amlogic/isp_module/subdev/iq/` | `cc72ec5c393179d2cde121b7869c03ff` | Штатный, не изменялся |

Готовые файлы и эталоны для отката лежат в:

```
/home/pi/Desktop/empty_folder/OS08A10_65fps_2026-10-18/modules/
├── iv009_isp_sensor_1080p120.ko     # рабочий модуль 120 fps
├── iv009_isp_sensor_ORIG_BPI.ko     # эталон 60 fps (откат)
├── iv009_isp_PATCHED.ko             # ISP с патчем recovery-loop
└── iv009_isp_ORIG_BPI.ko            # оригинальный ISP (откат)
```

### 3.2. Установка

```bash
cd /home/pi/Desktop/empty_folder/OS08A10_65fps_2026-10-18
SENS=/lib/modules/$(uname -r)/kernel/drivers/amlogic/isp_module/subdev/sensor/iv009_isp_sensor.ko

# 1. Сенсорный модуль с HTS 1036 — это и есть включение 120 fps
sudo cp modules/iv009_isp_sensor_1080p120.ko "$SENS"

# 2. Обязательная полная перезагрузка
sudo reboot
```

Модуль `iv009_isp.ko` с патчем recovery-loop **не является рычагом fps**.
В текущей системе он уже установлен; отдельно копировать его для 120 fps не нужно.
Если хотите привести ISP к тому же состоянию:

```bash
ISP=/lib/modules/$(uname -r)/kernel/drivers/amlogic/isp_module/v4l2_dev/iv009_isp.ko
sudo cp modules/iv009_isp_PATCHED.ko "$ISP"
sudo reboot
```

> **Не пытайтесь перезагрузить модули «на живую».** `iv009_isp.ko` не выгружается
> (`refcnt=2`), а `rmmod`/`insmod` без перезагрузки приводят к Oops в
> `sbuf_fops_read [iv009_isp]` и помечают ядро как Tainted. Нужна только полная
> перезагрузка.

### 3.3. Проверка после перезагрузки

```bash
md5sum /lib/modules/$(uname -r)/kernel/drivers/amlogic/isp_module/subdev/sensor/iv009_isp_sensor.ko
# ожидается: a171c86b0ed4f3410d78fbdbd7b665f4

lsmod | grep iv009
# iv009_isp
# iv009_isp_sensor
# iv009_isp_lens
# iv009_isp_iq
```

Если MD5 совпадает, а `iv009_isp_sensor` загружен — сенсор работает в режиме
1080p120.

### 3.4. Откат на штатные 60 fps

```bash
cd /home/pi/Desktop/empty_folder/OS08A10_65fps_2026-10-18
SENS=/lib/modules/$(uname -r)/kernel/drivers/amlogic/isp_module/subdev/sensor/iv009_isp_sensor.ko

sudo cp modules/iv009_isp_sensor_ORIG_BPI.ko "$SENS"
sudo reboot
```

При необходимости вернуть и оригинальный ISP:

```bash
ISP=/lib/modules/$(uname -r)/kernel/drivers/amlogic/isp_module/v4l2_dev/iv009_isp.ko
sudo cp modules/iv009_isp_ORIG_BPI.ko "$ISP"
sudo reboot
```

### 3.5. В чём суть патча

В сенсорном модуле (`OV08a10_seq.h`, блок
`setting_1920_1080_4lane_1440m_60fps[]`) уменьшен вдвое горизонтальный период
строки **HTS**:

| Регистр | Было (60 fps) | Стало (120 fps) |
|---|---|---|
| `0x380c/0x380d` (HTS) | `0x0818` = 2072 | `0x040c` = 1036 |
| `0x380e/0x380f` (VTS) | `0x0486` = 1158 | без изменений |

```
fps = SCLK / (HTS · VTS) = 144·10⁶ / (1036 · 1158) = 120.0 fps
```

Используется режим 2×2 binning: полный массив 3840×2160 децимируется до
1920×1080, поэтому HTS = 1036 меньше 1920 — это нормально.

---

## 4. Запуск приложения

Все команды ниже выполняются из корня проекта:

```bash
cd /home/pi/opencv_projects/prj_1
```

### 4.1. Интерактивное меню

Если не указан `--source`, приложение показывает меню:

```bash
./build/myapp_modular
```

| Пункт | Источник |
|---|---|
| `0` | IP-камера (URL) |
| `1` | камера `/dev/video0` |
| `2` | папка `PXL_3` |
| `3` | папка `PXL_4K` |
| `4` | видеофайл |

### 4.2. Запуск с явным источником

```bash
# Камера
./build/myapp_modular --source camera
./build/myapp_modular --source /dev/video0

# IP-камера
./build/myapp_modular --source 'http://192.168.0.104:4747/video?500x500'
./build/myapp_modular --source 'rtsp://user:pass@host/stream'

# Папка с кадрами
./build/myapp_modular --source /home/pi/opencv_projects/videos/PXL_3/

# Видеофайл
./build/myapp_modular --source /home/pi/opencv_projects/videos/PXL_1.mp4
```

### 4.3. Режимы и типичные команды

| Задача | Команда | Ожидаемый результат |
|---|---|---|
| Только смещение `dx/dy/da`, максимум fps | `./build/myapp_modular --source camera --estimate-only --duration 30` | `[EST] fps` 100–121, end-to-end ~100–105 |
| Максимум fps без окна | `./build/myapp_modular --source camera --max-fps-raw --no-stab --no-gui --duration 30` | `capfps` 95–120 (в чистых замерах 104–120) |
| Максимум fps с окном | `./build/myapp_modular --source camera --max-fps-raw --no-stab` | `capfps` 95–110, окно 20 fps |
| Сырой захват + стабилизация | `./build/myapp_modular --source camera --max-fps-raw` | 45–50 fps |
| Обычный режим + стабилизация | `./build/myapp_modular --source camera` | ~25–30 fps |
| Обычный режим без стабилизации | `./build/myapp_modular --source camera --no-stab` | 44–53 fps |
| Меньше нагрузки от превью | `VS_DISPLAY_SCALE=0.25 ./build/myapp_modular --source camera --max-fps-raw --no-stab` | `capfps` выше, окно дешевле |

Пояснения:

- **`--max-fps-raw`** читает камеру сырым YUYV. Серый кадр для трекинга
  собирается из Y-плоскости, BGR для окна — одной конверсией `YUYV2BGR`.
  Без стабилизации конвертируется только ROI, а не весь кадр.
- В `--estimate-only` флаг `--max-fps-raw` **игнорируется** — там всегда
  используется самый быстрый путь захвата.
- В `--max-fps-raw` превью по умолчанию масштабируется до **0.5×** и
  ограничивается **20 fps**, чтобы окно не отбирало CPU у конвейера.
- В `--no-gui` HUD не рисуется и превью не масштабируется — только строки
  `[PERF]`/`[EST]` в консоль.
- `--no-stab` без `--max-fps-raw` всё равно идёт через полный BGR-путь,
  поэтому даёт лишь ~45–53 fps.

### 4.4. Опции командной строки

| Опция | Назначение |
|---|---|
| `--source <spec>` | `camera`, `/dev/videoN`, `http://…`, `rtsp://…`, папка или видеофайл |
| `--estimate-only` | только оценка смещения; без стабилизации, окна и HUD |
| `--duration <sec>` | работать заданное число секунд и корректно выйти |
| `--no-stab` | стартовать с выключенным контуром стабилизации |
| `--max-fps-raw` | сырой YUYV-захват + одна конверсия для окна |
| `--no-gui` | без окна, только консольный вывод |
| `-h`, `--help` | справка |

Остановка без `--duration` — `Ctrl-C`, либо `ESC`/`Q` в окне.

### 4.5. Переменные окружения

| Переменная | Действие |
|---|---|
| `VS_ESTIMATE_ONLY=1` | то же, что `--estimate-only` |
| `VS_MAX_FPS_RAW=1` | то же, что `--max-fps-raw` |
| `VS_NO_STAB=1` | то же, что `--no-stab` |
| `VS_NO_GUI=1` | то же, что `--no-gui` |
| `VS_FAST_RESIZE=1` | `INTER_LINEAR` вместо `INTER_AREA` при уменьшении кадра для трекинга |
| `VS_OCV_THREADS=N` | число потоков OpenCV (`0` — не менять) |
| `VS_PINNED=1\|2\|3` | режим pinned-буферов GPU (`1` — оба, `2` — только src, `3` — только dst) |
| `VS_WARP_CPU=1` | считать `warpAffine` на CPU, без upload/download на GPU |
| `VS_VERBOSE=1` | построчный лог кадров (влияет на производительность) |
| `VS_EST_VERBOSE=1` | печатать смещение на каждом кадре в `--estimate-only` |
| `VS_DISPLAY_SCALE=0.5` | масштаб превью (по умолчанию `0.5` в `--max-fps-raw`) |
| `VS_DISPLAY_FPS=20` | ограничение частоты превью (`0` — без ограничения) |
| `VS_CAM_WIDTH=1920` | запросить ширину камеры |
| `VS_CAM_HEIGHT=1080` | запросить высоту камеры |

---

## 5. Горячие клавиши

Работают в обычном режиме с окном.

| Клавиша | Действие |
|---|---|
| `ESC`, `Q` | выход |
| `Пробел` | пауза (продолжение — любая клавиша) |
| `F` | сохранить текущий кадр в `frame_XXXXXX.jpg` |
| `D` | включить/выключить режим отладки |
| `B` | включить/выключить контур стабилизации (пиковый fps) |
| `S` / `W` | увеличить/уменьшить область кадра (ROI) |
| `U` / `I` | увеличить/уменьшить коэффициент заполнения `D` |
| `J` / `K` | увеличить/уменьшить `NSR` |

---

## 6. Как читать вывод

### Обычный режим и `--max-fps-raw`

```
[PERF] fps=20.0 | capfps=110 | cap=9.3ms det+trk=3.7ms stab=0.1ms disp=7.5ms | ...
```

| Поле | Значение |
|---|---|
| `fps=` | частота отображения/окна |
| `capfps=` | частота захвата и прохождения конвейера |
| `cap=` | время чтения и подготовки кадра |
| `det+trk=` | детекция + трекинг |
| `stab=` | стабилизация (upload + warp + download) |
| `disp=` | отрисовка и `imshow` |
| `rawQ` / `procQ` / `buf` | заполнение очередей |
| `skip=` | пропущенные кадры |

В `--max-fps-raw` с окном `fps` намеренно ограничен ~20, а реальную скорость
показывает **`capfps`**. В `--no-gui` `fps` и `capfps` практически совпадают.

### `--estimate-only`

```
[EST] fps=115.0 | dx=-1.20 dy=+8.14 da=-0.00121 rad (-0.069 deg) | pts=9 | det+trk=3.68ms cap=8.93ms ... 
[EST] ИТОГО (end-to-end): 839 кадров за 8.02 с = 104.57 fps
```

- `dx`, `dy` — смещение в пикселях полного кадра;
- `da` — поворот в радианах и градусах;
- `pts` — число точек;
- итоговая строка — реальный end-to-end fps.

---

## 7. Производительность (замеры на этой плате)

| Режим | fps | Комментарий |
|---|---|---|
| `--estimate-only` | **100–121** | предел сенсора; end-to-end ~100–105 |
| `--max-fps-raw --no-stab --no-gui` | **95–120** (`capfps`) | окно и HUD отключены; чистые замеры 104–120 |
| `--max-fps-raw --no-stab` | **95–110** (`capfps`), окно 20 | превью 0.5× / 20 fps |
| `--max-fps-raw` + стабилизация | **45–50** | ограничено `warp` ~9–10 мс + upload/download ~7 мс |
| Обычный режим, `--no-stab` | **44–53** | полный BGR-путь дороже raw |
| Обычный режим + стабилизация | **~25–30** | полный конвейер |

Условия замеров: сенсор 120 fps, фоновая нагрузка минимальна.
Плата без охлаждения: под нагрузкой `ddr_thermal` 70–77 °C, поэтому возможен
дрейф результатов до ~20 %. Закрытые VS Code/браузер и холодная плата дают
результаты ближе к верхней границе.

---

## 8. Как проверить, что 120 fps действительно есть

### 8.1. Через приложение

```bash
./build/myapp_modular --source camera --max-fps-raw --no-stab --no-gui --duration 30
```

Смотрите на `capfps=` в строках `[PERF]`. На холодной плате ожидается 115–120,
на горячей или при фоновой нагрузке — 95–115.

Либо в режиме оценки смещения:

```bash
./build/myapp_modular --source camera --estimate-only --duration 30
```

В конце будет строка `[EST] ИТОГО (end-to-end): ... fps`.

### 8.2. Независимо от OpenCV

```bash
cd /home/pi/Desktop/empty_folder/OS08A10_65fps_2026-10-18
./tools/lightdiag_arm64 300 1920 1080
```

Утилита читает `/dev/video0` напрямую через V4L2 и печатает средний fps,
медианный интервал и число уникальных кадров. Ожидается ~119–120 fps
(на горячей плате — 115–117) и 100 % уникальных кадров.

> **`/dev/video0` монополизируется.** Перед `lightdiag` закройте
> `myapp_modular`, и наоборот.
>
> **`v4l2-ctl` показывает недостоверный fps.** Для проверки используйте
> `lightdiag_arm64` или счётчики приложения (`capfps` / `[EST]`).

---

## 9. Типичные проблемы

| Симптом | Причина и решение |
|---|---|
| `capfps` ~50–60 вместо 100+ | Установлен штатный сенсорный модуль. Проверьте MD5 `iv009_isp_sensor.ko`, скопируйте `iv009_isp_sensor_1080p120.ko` и **перезагрузитесь** |
| После `cp` ничего не изменилось | Модуль не перезагружен. Нужен полный `sudo reboot`, `rmmod` делать нельзя |
| fps ~1–2 | Запрошено промежуточное разрешение. Камера поддерживает только 1920×1080 и 3840×2160 |
| Камера занята / чёрный кадр | Другой процесс держит `/dev/video0` (например, `lightdiag`) |
| `--max-fps-raw` даёт ~45 fps | Включена стабилизация: `warp` и upload/download не успевают. Для 100+ fps нужен `--no-stab` |
| Видео в `--no-stab` обрезано по центру | Это нормально: показывается ROI 1344×756 |
| Сборка медленная | В `build/CMakeCache.txt` стоит `Debug`. Переконфигурируйте с `-DCMAKE_BUILD_TYPE=Release` |
| OpenCV не найден | `-DOpenCV_DIR=/usr/local/opencv-4.12.0/lib/cmake/opencv4` |
| `rmmod iv009_isp` падает с Oops | Так и должно быть (`refcnt=2`). Не выгружайте модули, только перезагрузка |
| Просадки fps при работе | Закройте VS Code/браузер, проверьте температуру (`ddr_thermal`); плата без охлаждения |

---

## 10. Структура проекта

```
prj_1/
├── CMakeLists.txt
├── README.md                  # этот файл
├── HANDOVER.md                # платформа, модули, сборка сенсорного модуля
├── REPORT_120FPS.md           # отчёт по производительности и режимам
├── src/
│   ├── main.cpp               # CLI, меню, запуск
│   ├── config/                # конфигурация и структуры кадров
│   ├── stabilizer/            # захват, трекинг, стабилизация, отображение
│   └── ...
├── build/
│   └── myapp_modular          # собранный бинарник
└── tools/                     # диагностика, замеры, финальные логи
```

Полезные документы:

- `HANDOVER.md` — как устроен режим 1080p120, установка/откат модулей;
- `REPORT_120FPS.md` — где теряется fps и что даёт каждый режим;
- `tools/README.md` — диагностические утилиты и команды замеров.
