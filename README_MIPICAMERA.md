# Подключение MIPI камеры OS08A10 через OpenCV C++ на Banana Pi CM4

## Проблема с GStreamer по умолчанию

OpenCV по умолчанию использует GStreamer бэкенд для `VideoCapture`. На Banana Pi CM4 (Amlogic A311D) с камерой OS08A10 это вызывает ошибку:
```
[ WARN] global cap_gstreamer.cpp:1777 open OpenCV | GStreamer warning: Cannot query video position
[ WARN] global cap_gstreamer.cpp:2829 handleMessage OpenCV | GStreamer warning: Embedded video playback halted; module v4l2src0 reported: Internal data stream error.
```

**Причина:** GStreamer не может согласовать caps с драйвером сенсора — OS08A10 поддерживает только дискретные размеры 1920x1080 и 3840x2160, а OpenCV пытается запросить произвольное разрешение.

## ВАЖНО: обязательно задавать разрешение после open()

`cap.open("/dev/video0", CAP_V4L2)` **без** явного `cap.set(...)` открывает узел
в дефолтном режиме **640×480 RGB3**, и драйвер Amlogic уходит в медленный режим
(~1–2 fps). OS08A10 поддерживает только дискретные размеры, поэтому нужно явно
запросить 1920×1080 (режим 120 fps) или 3840×2160:

```cpp
cap.open("/dev/video0", cv::CAP_V4L2);
cap.set(cv::CAP_PROP_FRAME_WIDTH,  1920);
cap.set(cv::CAP_PROP_FRAME_HEIGHT, 1080);
cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('Y','U','Y','V'));
```

Без этих строк пример ниже заведётся, но на 640×480 в «медленном» режиме драйвера.

## Решение: использовать CAP_V4L2 напрямую

```cpp
#include <opencv2/opencv.hpp>
#include <iostream>
#include <cstdlib>   // для system()

int main() {
    std::string videoSource = "/dev/video0";  // MIPI камера OS08A10
    
    // Определяем тип источника: v4l2, IP-камера или видеофайл
    bool isV4L2Camera = (videoSource.find("/dev/video") != std::string::npos);
    
    cv::VideoCapture cap;
    
    if (isV4L2Camera) {
        // MIPI камера: используем CAP_V4L2 напрямую (не GStreamer).
        // Это обходит проблему согласования caps и предотвращает выключение платы.
        cap.open(videoSource, cv::CAP_V4L2);
        
        // Отключить автоконтролы сенсора — снижают current draw и предотвращают spikes тока:
        system("v4l2-ctl -d /dev/video0 -c gain_automatic=0 2>/dev/null");
        system("v4l2-ctl -d /dev/video0 -c white_balance_automatic=0 2>/dev/null");
        system("v4l2-ctl -d /dev/video0 -c auto_exposure=0 2>/dev/null");
        
        if (!cap.isOpened()) {
            std::cerr << "Не удалось открыть камеру: " << videoSource << std::endl;
            return -1;
        }
    } else {
        // IP-камера или видеофайл — используем стандартный бэкенд (GStreamer/FFMPEG)
        cap.open(videoSource);
        if (!cap.isOpened()) {
            std::cerr << "Не удалось открыть источник: " << videoSource << std::endl;
            return -1;
        }
    }
    
    // Захват кадров
    cv::Mat frame;
    while (true) {
        cap >> frame;
        if (frame.empty()) break;
        
        cv::imshow("Camera", frame);
        if (cv::waitKey(1) == 'q' || cv::waitKey(1) == 27) break;
    }
    
    return 0;
}
```

## Почему отключаем автоконтролы?

Автоконтролы сенсора (`gain_automatic`, `white_balance_automatic`) периодически увеличивают потребление тока, вызывая spikes которые триггерят выключение платы Banana Pi CM4. Отключение стабилизирует ток и предотвращает аварийные перезагрузки.

## Проверка доступных форматов

```bash
# Показать все поддерживаемые форматы
v4l2-ctl -d /dev/video0 --list-formats-ext

# Показать текущие автоконтролы
v4l2-ctl -d /dev/video0 -l | grep automatic

# Установить формат вручную (если нужно)
v4l2-ctl -d /dev/video0 --set-fmt-video=width=1920,height=1080,pixelformat=YUYV
```

## Типичные ошибки и решения

| Состояние | Причина | Решение |
|-----------|---------|---------|
| `Internal data stream error` | GStreamer не согласует caps с v4l2src | Использовать `cv::CAP_V4L2` |
| Разрешение 0x0, FPS 0 | Камера не открылась | Проверить `/dev/video0`, убедиться что нет других процессов |
| Плата выключается через несколько секунд | Current spike от автоконтролов | Отключить `gain_automatic`, `white_balance_automatic` |
| `Failed to open V4L2 camera` | Камера занята или не существует | Проверить `ls /dev/video*`, закрыть другие процессы |

## Сборка проекта

```bash
cd build && cmake .. && make -j$(nproc)
./myapp_test
```

## Примечания о сенсоре OS08A10

- MIPI-CSI, 4 lane, 8 Мп
- Дискретные режимы: только 1920x1080 и 3840x2160
- Поддерживаемые форматы: RGB4, RGB3, NV12, YUYV, UYVY, GREY, BYR2
- Драйвер показывает ~15fps для 1920x1080 (не 30fps как в официальной спецификации)
