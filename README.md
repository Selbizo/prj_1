# MIPI 1080p120: захват, оценка смещения и стабилизация

Проект для **Banana Pi CM4 (Amlogic A311D)** и сенсора **OmniVision OS08A10**
(MIPI CSI-2, `/dev/video0`). Рабочий режим — **1920×1080 @ 120 fps**, сырой
**YUYV** через V4L2.

Внутри: захват без GStreamer, быстрый путь `YUYV → GRAY`, оценка `dx/dy/da`
(GFTT + LK + RANSAC), опциональная стабилизация `warpAffine` (OpenCL/NEON) и
выдача `CMD_GYRO_CORRECTION (#75)`. Документ — для интеграции исходников в своё
устройство.

---

## 1. Быстрый старт

Нужны arm64, OpenCV 4.12+, CMake, C++17.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DOpenCV_DIR=/usr/local/opencv-4.12.0/lib/cmake/opencv4
cmake --build build -j$(nproc)
```

```bash
./build/myapp_modular --source camera              # окно + стабилизация
./build/myapp_modular --source camera --estimate-only --duration 10
```

Запуск от пользователя с доступом к `/dev/video0` (группа `video`), без `sudo`.

---

## 2. Режим 1080p120: модуль ядра

Частоту задаёт сенсорный модуль `iv009_isp_sensor.ko`, а не OpenCV. В рабочем
режиме изменены только регистры HTS:

| Регистр | 60 fps | 120 fps |
|---|---:|---:|
| `0x380c/0x380d` (HTS) | `0x0818` = 2072 | `0x040c` = 1036 |
| `0x380e/0x380f` (VTS) | `0x0486` = 1158 | без изменений |

`fps = SCLK / (HTS · VTS) = 144 МГц / (1036 · 1158) = 120.0`.

Готовый модуль — `backup_current_ko/iv009_isp_sensor.ko`
(MD5 `a171c86b0ed4f3410d78fbdbd7b665f4`).

```bash
SENS=/lib/modules/$(uname -r)/kernel/drivers/amlogic/isp_module/subdev/sensor
sudo cp backup_current_ko/iv009_isp_sensor.ko "$SENS/iv009_isp_sensor.ko"
sudo reboot

md5sum "$SENS/iv009_isp_sensor.ko"   # a171c86b0ed4f3410d78fbdbd7b665f4
```

Откат на штатные 60 fps: вернуть оригинальный модуль
(MD5 `95db2e5fc780debd68b891a374d3c279`) и полностью перезагрузиться.

Патч готового `.ko` без сборки ядра:

```bash
python3 patch_scripts/patch_ko.py stock_iv009_isp_sensor.ko -o patched.ko
python3 patch_scripts/patch_ko.py --check patched.ko
```

**Важно:** `iv009_isp.ko` не выгружается (`refcnt=2`). `rmmod`/`insmod` без
перезагрузки приводят к Oops. Нужен полный `reboot`.

---

## 3. Конвейер и точки интеграции

```
/dev/video0  (1920×1080 YUYV, 120 Гц)
   ▼
captureThread      read raw → resize Y-плоскости до 384×216
   ▼
rawFramesQueue
   ▼
detectionThread    GFTT + LK + RANSAC → dx, dy, da, trackingQuality
   ├── estimate-only: [EST] / [DRIFT] / #75 → выход наружу
   └── processedFramesQueue → stabilizationThread → displayThread
```

| Что | Файл |
|---|---|
| захват, формат, сырой YUYV, серый | `src/stabilizer/video_stabilizer.cpp`, `captureThread()` |
| `dx/dy/da` и качество | там же, `detectionAndTrackingThread()` |
| вывод результата | там же, лямбда `publish()` |
| дрейф | `src/gyro/drift_estimator.*` |
| команда #75, UART | `src/gyro/sbgc_protocol.*` |

Для максимума fps используйте `--estimate-only`: работают только захват и
трекинг, стабилизация и окно не создаются. Свой транспорт (UART, UDP, ROS)
встраивается в `publish()` вместо `[EST]`.

---

## 4. Открытие камеры и сырой YUYV

`cap.open()` без явного формата открывает узел в 640×480 RGB3, и драйвер
уходит в режим ~1–2 fps. Правильный порядок:

```cpp
cv::VideoCapture cap("/dev/video0", cv::CAP_V4L2);
cap.set(cv::CAP_PROP_FRAME_WIDTH, 1920);
cap.set(cv::CAP_PROP_FRAME_HEIGHT, 1080);
cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('Y','U','Y','V'));
cap.set(cv::CAP_PROP_CONVERT_RGB, 0);   // сырой YUYV
```

Размер переопределяется через `VS_CAM_WIDTH` / `VS_CAM_HEIGHT`. Рабочий режим —
1920×1080; промежуточные разрешения могут уводить драйвер в медленный режим.

Сырой кадр приходит плоским буфером `N×1 8UC1`, 2 байта на пиксель:

```cpp
if (buf.channels() == 1 && buf.total() % (width * 2) == 0)
    buf = buf.reshape(2, static_cast<int>(buf.total() / (width * 2)));

cvtColor(buf, gray, COLOR_YUV2GRAY_YUY2);  // трекинг
cvtColor(buf, bgr,  COLOR_YUV2BGR_YUY2);   // окно/стабилизация
```

Для YUYV начало и ширина ROI должны быть чётными.

---

## 5. Режимы

| Режим | fps | Комментарий |
|---|---:|---|
| `--estimate-only` | 100–118 | только захват + трекинг |
| `--max-fps-raw --no-stab --no-gui` | 100–120 (`capfps`) | максимум конвейера |
| `--max-fps-raw --no-stab` | 95–110 | превью 0.5× / 20 fps |
| обычный режим, `--no-stab` | 44–53 | полный BGR-путь |
| обычный режим + стабилизация | 25–50 | ограничение warp/upload/download |

Флаги: `--max-fps-raw`, `--no-stab`, `--no-gui`, `--duration <sec>`.
Числа зависят от температуры и фоновой нагрузки (плата без охлаждения, разброс
до ~20 %).

**Warp: GPU или CPU/NEON.** По умолчанию стабилизация считается на GPU (OpenCL).
`VS_WARP_CPU=1` переключает `warpAffine` на CPU: NEON-ядро (`gpu/warp_affine_neon.cpp`),
без upload/download кадров. Оно побитово совпадает с `cv::warpAffine` для
8UC1/8UC3 + `INTER_LINEAR` + `BORDER_CONSTANT`. Число потоков —
`VS_WARP_THREADS` (по умолчанию 4), `VS_WARP_NEON=0` отключает NEON (тогда
работает обычный `cv::warpAffine`). На этой плате в живом конвейере GPU-путь
обычно быстрее: CPU-warp конкурирует за ядра и память с захватом и трекингом.

---

## 6. Цветопередача

Формат **YUYV** подтверждён: сырой буфер 4 147 200 байт, ручная конверсия
`COLOR_YUV2BGR_YUY2` совпадает со штатным путём OpenCV (разница ~3 единицы BGR
при фиксированных AE/AWB). Интерпретация как UYVY даёт полосы.

По умолчанию приложение отключает AE/AWB (меньше бросков тока), но оставляет
случайную точку экспозиции/баланса белого. Если картинка тёмная или с оттенком:

```bash
./build/myapp_modular --source camera --cam-auto
```

или задайте контролы вручную:

```bash
v4l2-ctl -d /dev/video0 \
  -c white_balance_automatic=0 -c white_balance_temperature=5000 \
  -c gain_automatic=0 -c gain=100 -c auto_exposure=1
```

Диагностика: `tools/color_probe.cpp` снимает штатный BGR и ручной YUYV,
сохраняет PNG и печатает средние. Не ориентируйтесь на
`v4l2-ctl --get-fmt-video` вне стрима: драйвер может показывать `RGB4`/`META`
и игнорировать YUYV.

Если пурпурный/розовый оттенок остаётся и с `--cam-auto`, конвейер тут не
виноват: ISP test pattern (`v4l2-ctl -c isp_test_pattern=1`) даёт чистые
R/G/B, а сырой YUY2-кадр тоже пурпурный. В тёмном помещении это обычно
ИК-подсветка при отсутствии IR-cut фильтра — включите обычный свет, уберите
ИК-подсветку или поставьте IR-cut. Быстрая проверка: `-c saturation=0` —
если картинка становится нормальной серой, страдает только цвет (баланс
белого/ИК), а не каналы и не warp.

---

## 7. Дрейф и `CMD_GYRO_CORRECTION (#75)`

Оценка: `dx/dy/da` → углы через FOV → угловые скорости → ФНЧ → достоверность →
поправка #75. Медленный уход подвеса проходит, быстрые движения подавляются.
Медленное намеренное движение тоже попадёт в оценку.

Спецификация: **SimpleBGC32 Serial API Protocol Specification, стр. 61**
(<https://www.basecamelectronics.com/files/SimpleBGC_2_6_Serial_Protocol_Specification.pdf>).

Payload — 9 байт:

| Поле | Тип | Смысл |
|---|---|---|
| `IMU_TYPE` | 1u | 0 — main IMU, 1 — frame IMU |
| `GYRO_ZERO_CORR[3]` | 2s×3 | поправки X, Y, Z, little-endian |
| `GYRO_ZERO_HEADING_CORR` | 2s | поправка только по глобальной Z |

Единицы — **0.001 gyro sensor unit** (сырые LSB), диапазон `int16`. Для
16.4 LSB/(град/с) это `16400` единиц на 1 град/с — **гипотеза, требует
калибровки**.

Кадр protocol v1:

```
0x3E | CMD=75 | SIZE=9 | 0x54 | payload[9] | SUM(payload)&0xFF
3E 4B 09 54 00 00 00 00 00 00 00 00 00 00   # нулевая поправка
```

Protocol v2 (2.68b0+) — стартовый байт `0x24` и CRC16 (`0x8005`). Ответ —
`CMD_CONFIRM (#67)` с `CMD_ID=75`.

```bash
./build/myapp_modular --source camera --estimate-only --drift-report
./build/myapp_modular --source camera --estimate-only --gyro-correction
./build/myapp_modular --source camera --estimate-only --gyro-correction \
  --gyro-port /dev/ttyUSB0 --gyro-enable
```

По умолчанию — **dry-run** с печатью пакета; реальная отправка только с
`--gyro-enable`. Настройки: `--gyro-fov-h`, `--gyro-scale`, `--gyro-max`,
`--gyro-tau`, `--gyro-rate`, `--gyro-axis-*`, `--gyro-sign-*`.

Перед включением отправки откалибруйте:

1. **оси и знаки** — ступенчатая подача через `tools/sbgc_correction`, смотреть
   реакцию изображения;
2. **масштаб** — ступеньки 100/500/1000/2000 единиц, измерить `units/(deg/s)`;
3. **достоверность** — подобрать `VS_GYRO_MIN_CONF`, `VS_GYRO_DEADBAND`, `tau`.

---

## 8. Утилиты

В CMake не входят, собираются вручную. Камера монополизируется: перед запуском
закройте `myapp_modular`.

| Файл | Назначение |
|---|---|
| `probe_capture.cpp` | стоимость стадий захвата |
| `probe_gray_path.cpp` | побитовая проверка быстрого серого пути |
| `verify_fps.cpp` | fps и уникальность кадров |
| `bench_warp_neon.cpp` / `check_warp_neon.cpp` | бенчмарк и проверка NEON-варпа |
| `color_probe.cpp` | штатный BGR против ручного YUYV→BGR |
| `sbgc_correction.cpp` | ручная отправка #75 для калибровки |
| `make_pan_test.cpp` | справочный генератор видео (в приложении не используется) |

```bash
OC=/usr/local/opencv-4.12.0
g++ -O3 -std=gnu++17 -isystem $OC/include/opencv4 tools/color_probe.cpp \
  -o /tmp/color_probe -L$OC/lib -Wl,-rpath,$OC/lib \
  -lopencv_core -lopencv_imgproc -lopencv_videoio -lopencv_imgcodecs
/tmp/color_probe

g++ -O3 -std=gnu++17 -Isrc tools/sbgc_correction.cpp \
  src/gyro/sbgc_protocol.cpp -o /tmp/sbgc_correction
/tmp/sbgc_correction --self-test
/tmp/sbgc_correction --x 1000 --y 0 --z 0 --dry-run
```

---

## 9. Структура, грабли, чек-лист

```
src/
  main.cpp                 CLI, запуск
  config/                  конфигурация и структуры кадров
  stabilizer/              захват, трекинг, стабилизация, отображение
  gyro/                    протокол #75 и оценка дрейфа
  gpu/                     warpAffine: OpenCL и NEON
  filters/ sync/ utils/    Винер, очередь, вспомогательные функции
tools/                     диагностика и калибровка
backup_current_ko/         рабочий набор .ko и md5sums
patch_scripts/             бинарный патч сенсорного .ko
build/                     сборка (в git не входит)
```

Грабли:

- `/dev/video0` занимает только один процесс;
- `v4l2-ctl` показывает недостоверный fps;
- после замены `.ko` нужен полный `reboot`;
- `build/CMakeCache.txt` может содержать `Debug` — переконфигурируйте Release;
- промежуточные разрешения камеры могут дать ~1–2 fps;
- отключённые AE/AWB дают тёмную/цветную картинку;
- не включайте `--gyro-enable` до проверки осей, знаков и пределов; при потере
  трекинга поправка должна сбрасываться в 0.

Чек-лист внедрения:

1. установить модуль 120 fps, проверить MD5, перезагрузиться;
2. собрать Release и получить ≥100 fps в `--estimate-only`;
3. открыть камеру с явным `1920×1080 YUYV` и `CAP_PROP_CONVERT_RGB=0`;
4. для максимума fps оставить только захват + трекинг;
5. встроить свой транспорт в `publish()`;
6. проверить цвет при фиксированных AE/AWB или с `--cam-auto`;
7. для #75: сначала dry-run, затем калибровка, только потом `--gyro-enable`.
