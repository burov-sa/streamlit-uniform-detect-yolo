**English** | [Русский](README.ru.md)

# Uniform Appearance Assessment Service

Streamlit application for automatic assessment of employee uniform compliance using YOLOv8 object detection. Detects uniform items and violations from images, video files, or a live camera feed, and produces a detailed compliance report.

## Features

- Three input modes: image, video, live camera.
- Uniform compliance check against a configurable uniform specification.
- Violation detection: missing items, excess items, and explicit violations (unbuttoned clothing, dirty uniform, etc.).
- Editable uniform sets directly from the sidebar; changes persisted to `config.yml`.
- Model management: select from YOLO models listed in `config.yml` or fall back to the default one.
- Duplicate suppression by IoU to avoid inflated counts.
- Real-time processing in camera mode via WebRTC.

## Requirements

- Python 3.10+
- `ffmpeg` (optional, for H.264 conversion of output video)

## Installation

```bash
git clone <repository-url>
cd <repository-name>
pip install -r requirements.txt
```

Place custom YOLO weights into `models/`. The default model `yolov8n.pt` is downloaded automatically by Ultralytics on first use.

## Usage

```bash
streamlit run main.py
```

### Sidebar controls

- **Input type** — `Изображение` (Image), `Видеозапись` (Video), `Камера` (Camera).
- **Uniform set** — selects the uniform specification to validate against.
- **Tabs** — view (📋), edit (📝), delete (❌) the selected uniform set. Changes are written to `config.yml`.
- **Detection threshold** — confidence threshold for YOLO detections (0–100 %, mapped to 0.0–1.0).
- **Model** — choose from `model_names` in `config.yml`, or the default.

### Image mode

1. Upload one or more files (`.jpg`, `.jpeg`, `.png`, `.bmp`, `.dib`).
2. Click **Оценить внешний вид**.
3. Annotated images are displayed in the `🗃` tab.
4. A compliance table is shown in the `📈` tab: each uniform component with `наличие` and `нарушения`.

### Video mode

1. Upload one or more files (`.mp4`, `.mov`, `.avi`).
2. Click **Оценить внешний вид**.
3. Processed videos are available for preview and download.
4. The compliance table uses the maximum simultaneous detections across all frames.

### Camera mode

1. Allow camera access in the browser.
2. The live stream is processed frame by frame; duplicates are removed.
3. The compliance table updates continuously as new maxima are observed.

## Configuration

### `config.yml`

```yaml
path: /data
train: images/train
val: images/train
names:
  0: бортпроводник(женский пол)
colors:
  0: [255, 0, 255]
model_names:
  - rivc-cnn-standart-0.3
model_files:
  - yolov8s90_1600_130623.pt
uniform:
  general_summer_outside_steward_woman:
    - бортпроводник(женский пол)
    - пилотка
uniform_consist:
  - бортпроводник(женский пол)
  - жакет
```

### `uniformmatch.csv`

Mapping between detected classes and uniform components.

- **Separator:** `;`
- **Encoding:** `utf-8-sig`
- **Rows:** detected class names (single items or combinations) plus `min` and `max` rows.
- **Columns:** individual uniform components.
- **Cells:**
  - `1` — the component is present in the combination / allowed,
  - `2` — the component is a violation when detected,
  - `0` / empty — not relevant.

The `min` and `max` rows define allowed counts per component. The classification sums contributions of every detected class into each component and compares the totals against `min`/`max`.

## Classification logic

1. Filter detections by the confidence threshold.
2. Remove duplicates: overlapping boxes of the same class (IoU above 50 % of the smaller box) are dropped.
3. For each detected class, look up its row in `uniformmatch.csv` and add the count to every component where the cell is `> 0`.
4. Sum contributions per component and compare with `min`/`max`:
   - `min <= count <= max` → component present, no violation.
   - otherwise → `несоответствие количества` (mismatch) or `отсутствует` (missing).
5. Classes marked as violations and unexpected items (not mapped to any component of the current uniform) are reported as separate `нарушение_N` rows.

## Project layout

```
.
├── main.py
├── config.yml
├── uniformmatch.csv
├── requirements.txt
├── headimage.jpg
├── log.ipynb
├── models/
└── temp/
```

## Model

The service uses Ultralytics YOLOv8:

- Default: `yolov8n.pt` (auto-downloaded).
- Custom: place `.pt` files in `models/` and list them under `model_names` / `model_files` in `config.yml`.

The model expects RGB input; the app converts BGR frames to RGB before inference.

## Dependencies

- `ultralytics` — YOLO models
- `streamlit`, `streamlit-webrtc` — UI and WebRTC streaming
- `opencv-python` — image/video IO
- `pandas` — tabular report
- `pyyaml` — configuration
- `numpy` — array operations

See `requirements.txt` for versions.

## Notes and limitations

- The classification logic is table-driven; adding a new violation type requires a new row in `uniformmatch.csv` with column `2` set for the corresponding component, or a standalone `(нарушение)` class name.
- Duplicate suppression uses IoU on the smaller box. For highly overlapping different classes, no suppression is applied.
- Camera mode keeps maxima across the whole session; reload the page to reset.
- Video output currently uses `avc1` codec with `mp4v` fallback. If neither is available, re-enable the `ffmpeg` conversion step in `main.py`.

## License

Internal use only. Contact the authors for licensing details.
