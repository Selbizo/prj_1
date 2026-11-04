# Инструменты для замеров 120 fps MIPI

Отдельные диагностические программы. В CMake не входят — это замеры, а не
приложение. Все, кроме `make_pan_test`, держат `/dev/video0` монопольно, поэтому
перед запуском закройте `myapp_modular`.

Чистая версия проекта принимает только MIPI/V4L2-камеру: `--source camera` или
`--source /dev/videoN`.

```bash
OC=/usr/local/opencv-4.12.0
CXX=/usr/bin/aarch64-linux-gnu-g++
INC="-isystem $OC/include/opencv4"
LIB="-L$OC/lib -Wl,-rpath,$OC/lib -lopencv_core -lopencv_imgproc -lopencv_videoio"

# 1. Где теряется fps на пути захвата (до реализации режима estimate-only)
$CXX -O3 -std=gnu++17 $INC probe_capture.cpp -o /tmp/probe \
     $LIB -lopencv_video -lopencv_features2d -lopencv_calib3d
/tmp/probe

# 2. Побитовая проверка, что путь "resize YUYV LINEAR -> GRAY" равен эталону
#    "полный GRAY -> resize LINEAR" (варианты E / F / J)
$CXX -O3 -std=gnu++17 $INC probe_gray_path.cpp -o /tmp/probe_gray \
     $LIB -lopencv_imgcodecs
/tmp/probe_gray

# 3. Доказательство частоты кадров через тот же VideoCapture, что и в приложении.
#    Фаза A (пустой цикл) показывает, что «8 мс» — это ПЕРИОД СЕНСОРА, на котором
#    блокируется read(); фаза B (путь приложения) — что внутри busy-цикла кадр уже
#    готов и read() дешевле; фаза C — побайтовая проверка, что драйвер НЕ повторяет
#    один и тот же буфер (fps не завышен счётом дубликатов).
$CXX -O3 -std=gnu++17 $INC verify_fps.cpp -o /tmp/verify_fps $LIB
taskset -c 4-5 /tmp/verify_fps 12     # по 12 с на фазы A и B, 5 с на фазу C

# 4. Бенчмарк/проверка NEON-варпа
$CXX -O3 -std=gnu++17 $INC bench_warp_neon.cpp -o /tmp/bench_warp $LIB
/tmp/bench_warp

# 5. Диагностика цветового пути: штатный BGR vs ручной YUYV->BGR
$CXX -O3 -std=gnu++17 $INC color_probe.cpp -o /tmp/color_probe \
     $LIB -lopencv_imgcodecs
/tmp/color_probe

# 6. Ручная отправка CMD_GYRO_CORRECTION (#75) для калибровки
$CXX -O3 -std=gnu++17 -I../src sbgc_correction.cpp ../src/gyro/sbgc_protocol.cpp \
     -o /tmp/sbgc_correction
/tmp/sbgc_correction --self-test
/tmp/sbgc_correction --x 1000 --y 0 --z 0 --dry-run
```

## Сквозная проверка в самом приложении

В режиме estimate-only приложение печатает в конце строку
`[EST] ИТОГО (end-to-end): N кадров за T с = X fps` — полное число кадров,
прошедших через поток детектирования, делённое на фактическое время работы.

## Режим `--max-fps-raw` (окно + сырой захват)

Обычный режим с окном, но камера читается сырым YUYV: серый для трекинга
собирается из Y-плоскости, BGR для окна — одной конверсией `YUYV2BGR`.
При `--no-stab` конвертируется сразу ROI, а не весь кадр. Превью по умолчанию
0.5x / 20 fps — чтобы окно не отбирало CPU у конвейера.

```bash
# окно + максимальный fps (≈100–110 fps, превью 20 fps)
./build/myapp_modular --source camera --max-fps-raw --no-stab

# без окна — максимум конвейера (≈104–120 fps)
./build/myapp_modular --source camera --max-fps-raw --no-stab --no-gui --duration 30

# меньше превью — ещё дешевле для CPU
VS_DISPLAY_SCALE=0.25 ./build/myapp_modular --source camera --max-fps-raw --no-stab
```

`[PERF] capfps=N` — частота захвата (конвейера), `fps=` — частота окна.

## Примечания

- `make_pan_test.cpp` — справочный генератор синтетического видео с известным
  смещением. После перехода на MIPI-only источник приложение его не принимает;
  файл оставлен как материал прежней проверки точности трекинга.
- Сырые логи замеров удалены при очистке проекта. Итоги и выводы — в
  `../REPORT_120FPS.md`.
