# backup_current_ko — рабочая конфигурация модулей OS08A10

Снято с работающей платы: **20.10.2026**, ядро `4.9.241-BPI-CM4` (Banana Pi CM4, A311D).
Это точные копии модулей, которые были загружены в момент создания папки.

## Состав

| Файл | MD5 | Роль |
|---|---|---|
| `iv009_isp_sensor.ko` | `a171c86b0ed4f3410d78fbdbd7b665f4` | Сенсор OS08A10, **1080p120** (HTS 1036) |
| `iv009_isp.ko` | `eb5c71cc58bae1bd65cef2d9e9c97846` | ISP с патчем recovery-loop; на fps не влияет |
| `iv009_isp_lens.ko` | `fa47b294fd547a3976dbff9863c213c8` | Штатный, не изменялся |
| `iv009_isp_iq.ko` | `cc72ec5c393179d2cde121b7869c03ff` | Штатный, не изменялся |

Проверка целостности:

```bash
cd /home/pi/opencv_projects/prj_1/backup_current_ko
md5sum -c md5sums.txt
```

## Восстановление «как было»

```bash
cd /home/pi/opencv_projects/prj_1/backup_current_ko

BASE=/lib/modules/$(uname -r)/kernel/drivers/amlogic/isp_module

sudo cp iv009_isp_sensor.ko "$BASE/subdev/sensor/iv009_isp_sensor.ko"
sudo cp iv009_isp.ko        "$BASE/v4l2_dev/iv009_isp.ko"
sudo cp iv009_isp_lens.ko   "$BASE/subdev/lens/iv009_isp_lens.ko"
sudo cp iv009_isp_iq.ko     "$BASE/subdev/iq/iv009_isp_iq.ko"

sudo reboot
```

Обязательна **полная перезагрузка**: `iv009_isp.ko` не выгружается (`refcnt=2`),
`rmmod`/`insmod` на живой системе приводят к Oops.

## Откат на штатные 60 fps

Эталоны лежат в:

```text
/home/pi/Desktop/empty_folder/OS08A10_65fps_2026-10-18/modules/
├── iv009_isp_sensor_ORIG_BPI.ko   # 95db2e5fc780debd68b891a374d3c279
└── iv009_isp_ORIG_BPI.ko          # 96b59246c54975593ffd922f36d1cc4c
```

```bash
cd /home/pi/Desktop/empty_folder/OS08A10_65fps_2026-10-18
BASE=/lib/modules/$(uname -r)/kernel/drivers/amlogic/isp_module

sudo cp modules/iv009_isp_sensor_ORIG_BPI.ko "$BASE/subdev/sensor/iv009_isp_sensor.ko"
sudo cp modules/iv009_isp_ORIG_BPI.ko        "$BASE/v4l2_dev/iv009_isp.ko"
sudo reboot
```

## Примечания

- Это локальный бэкап, в git не добавлен.
- Для проверки 120 fps используйте `capfps` в приложении или `lightdiag_arm64`;
  `v4l2-ctl` показывает недостоверную частоту.
- `/dev/video0` монополизируется: перед диагностикой закройте `myapp_modular`.
