# Инструменты для замеров

Отдельные программы, собираются вручную (в CMake не входят — это диагностика, а не
приложение). Все, кроме `make_pan_test`, держат `/dev/video0` монопольно, поэтому
перед запуском закрыть `myapp_modular` и `lightdiag`.

```bash
OC=/usr/local/opencv-4.12.0
CXX=/usr/bin/aarch64-linux-gnu-g++
INC="-isystem $OC/include/opencv4"
LIB="-L$OC/lib -Wl,-rpath,$OC/lib -lopencv_core -lopencv_imgproc -lopencv_videoio"

# 1. Где теряется fps на пути захвата (до реализации режима estimate-only)
$CXX -O3 -std=gnu++17 $INC probe_capture.cpp -o /tmp/probe \
     $LIB -lopencv_video -lopencv_features2d -lopencv_calib3d
/tmp/probe                      # сырой лог прошлого запуска: probe_capture_out.txt

# 2. Побитовая проверка, что путь "resize YUYV LINEAR -> GRAY" равен эталону
#    "полный GRAY -> resize LINEAR" (варианты E / F / J)
$CXX -O3 -std=gnu++17 $INC probe_gray_path.cpp -o /tmp/probe_gray \
     $LIB -lopencv_imgcodecs
/tmp/probe_gray

# 3. Синтетическое видео с ИЗВЕСТНЫМ смещением — для проверки dx/dy/da
$CXX -O3 -std=gnu++17 $INC make_pan_test.cpp -o /tmp/mkvideo $LIB
/tmp/mkvideo                    # создаёт /tmp/opencode/pan.mp4 (-2.0/-1.0 px за кадр)

# затем:
./build/myapp_modular --source /tmp/opencode/pan.mp4 --estimate-only --duration 10

# 4. Доказательство частоты кадров через тот же VideoCapture, что и в приложении.
#    Фаза A (пустой цикл) показывает, что «8 мс» — это ПЕРИОД СЕНСОРА, на котором
#    блокируется read(); фаза B (путь приложения) — что внутри busy-цикла кадр уже
#    готов и read() дешевле; фаза C — побайтовая проверка, что драйвер НЕ повторяет
#    один и тот же буфер (fps не завышен счётом дубликатов).
$CXX -O3 -std=gnu++17 $INC verify_fps.cpp -o /tmp/verify_fps $LIB
taskset -c 4-5 /tmp/verify_fps 12     # по 12 с на фазы A и B, 5 с на фазу C
```

## Сквозная проверка в самом приложении

В режиме estimate-only приложение печатает в конце строку
`[EST] ИТОГО (end-to-end): N кадров за T с = X fps` — это полное число кадров,
прошедших через поток детектирования, делённое на фактическое время работы.

## Логи замеров

- `probe_capture_out.txt` — изолированные стоимости стадий захвата (пути A/B/C).
- `probe_gray_path.cpp` — доказательство эквивалентности быстрого пути серого.
- `verify_fps.cpp` — независимое доказательство fps и отсутствия дублей кадров.
- `lightdiag_600.txt` — сырой V4L2 (без OpenCV), таймстампы драйвера: 119 fps.
- `ocv_threads_sweep.txt` — свип `VS_OCV_THREADS` (1/2/3/4/6) в estimate-only.
- `ab_estimate_only_results.txt` — перекрёстный A/B: estimate-only против
  `--no-stab --no-gui`, по 3 репа round-robin.

Итоги и выводы — в `../REPORT_120FPS.md` (§8) и `../PERF_PLAN.md` (§7).

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
Логи: `final_maxfps_nogui2.txt` (без окна), `final_maxfps_gui.txt` (окно),
`final_normal_nostab.txt`, `final_maxfps_stab.txt`, `final_estimate.txt`
(регрессии); скриншот окна — `maxfps_raw_window.jpg`.
