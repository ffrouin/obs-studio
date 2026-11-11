# OBS Vision Filter — Linux prototype

This is a Linux-only OBS Studio video filter prototype. It applies Sobel-style contour detection to the complete parent source/scene image using an OBS GPU effect shader.

## Prerequisites

- OBS Studio development package exposing `libobs.pc`
- CMake 3.16+
- C++17 compiler
- OpenCV 5 runtime/development libraries (`opencv_core`, `opencv_imgproc`, `opencv_objdetect`)

On this machine, `pkg-config --modversion libobs` reports `32.2.2`.

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc)"
```

## Install into a staging directory

```bash
cmake --install build --prefix "$HOME/obs-vision-filter-stage"
```

The staged files are:

```text
lib/obs-plugins/obs-vision-filter.so
share/obs/obs-plugins/obs-vision-filter/effects/contour.effect
share/obs/obs-plugins/obs-vision-filter/effects/face.effect
share/obs/obs-plugins/obs-vision-filter/models/face_detection_yunet_2026may.onnx
share/obs/obs-plugins/obs-vision-filter/locale/en-US.ini
```

Do not install over the system OBS installation from this prototype. Copy the staged plugin and data files into the matching OBS plugin/data locations only after reviewing the build.

The plugin registers two independent OBS filters:

- **Vision contour filter** — GPU contour detection and comic colorization.
- **Face detection filter** — asynchronous YuNet face detection with optional green face-detection rectangles and three configurable overlay slots for the biggest, second-biggest, and third-biggest faces. Each slot can select sunglasses, pirate hat, clown mask, or none. The rectangle display can be disabled in the filter properties. Sunglasses, the hat, and the clown mask are disabled by default. Overlay targets lock or switch after 600 ms of stability.

The XComposite resolver accepts the stable static X11 name `chromium-obs-studio` in either the window title or WM_CLASS resource/class fields, so tab-dependent Chromium titles do not need to remain fixed.
## Deployment script

The project includes `deploy.sh`. It deploys the current staged files to the system OBS installation at `/usr/lib/obs-plugins` and `/usr/share/obs/obs-plugins`. The script uses the user-owned `stage-sunglasses/` directory so rebuilding does not require root access.

Preview the actions without changing the system:

```bash
./deploy.sh --dry-run
```

Deploy the existing stage:

```bash
./deploy.sh
```

Rebuild, refresh the stage, then deploy:

```bash
./deploy.sh --build
```

The script refuses to deploy while OBS is running and backs up an existing deployment under `backups/` before replacing it. Remove the prototype with:

```bash
./deploy.sh --uninstall
```

