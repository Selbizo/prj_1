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
```

## Логи замеров

- `probe_capture_out.txt` — изолированные стоимости стадий захвата (пути A/B/C).
- `probe_gray_path.cpp` — доказательство эквивалентности быстрого пути серого.
- `ocv_threads_sweep.txt` — свип `VS_OCV_THREADS` (1/2/3/4/6) в estimate-only.
- `ab_estimate_only_results.txt` — перекрёстный A/B: estimate-only против
  `--no-stab --no-gui`, по 3 репа round-robin.

Итоги и выводы — в `../REPORT_120FPS.md` (§8) и `../PERF_PLAN.md` (§7).
