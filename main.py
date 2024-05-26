import os
import cv2
import yaml
import pandas as pd
import numpy as np
import streamlit as st
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer


CONFIG_FILE = "config.yml"
MATCH_FILE = "uniformmatch.csv"
TEMP_DIR = "./temp"
MODELS_DIR = "./models"
DEFAULT_COLOR_BGR = (0, 255, 0)


# --------------------------------------------------------------------------- #
# Утилиты
# --------------------------------------------------------------------------- #
def remove_duplicate(results_list, k_intersecion=0.5):
    """
    Удаление дубликатов: если два прямоугольника одного класса пересекаются
    более чем на k_intersecion от площади меньшего, второму присваивается class_id = -1.
    results_list: список [x1, y1, x2, y2, score, class_id]
    """
    for i in range(0, len(results_list) - 1):
        x1, y1, x2, y2, score, class_id = results_list[i]
        if int(class_id) == -1:
            continue
        square = abs(x1 - x2) * abs(y1 - y2)
        for j in range(i + 1, len(results_list)):
            x1_, y1_, x2_, y2_, score_, class_id_ = results_list[j]
            if int(class_id_) == -1:
                continue
            if int(class_id) == int(class_id_):
                square_ = abs(x1_ - x2_) * abs(y1_ - y2_)
                left = max(x1, x1_)
                top = max(y1, y1_)
                right = min(x2, x2_)
                bottom = min(y2, y2_)
                width = right - left
                height = bottom - top
                if width > 0 and height > 0:
                    intersection = width * height
                    if intersection > k_intersecion * min(square, square_):
                        results_list[j][5] = -1
    return results_list


@st.cache_resource()
def get_model(modeltype):
    """Загрузка модели YOLO по имени из config.yml (кэшируется)."""
    try:
        from ultralytics import YOLO
    except ImportError as e:
        st.error(f"Ultralytics не установлен: {e}")
        raise

    if model_names and modeltype in model_names:
        path = os.path.join(MODELS_DIR, model_files[model_names.index(modeltype)])
        model = YOLO(path)
        print(f"Модель '{modeltype}' успешно загружена")
    else:
        model = YOLO("yolov8n.pt")
    return model


def write_bytesio_to_file(filename, bytesio):
    """Запись BytesIO в файл."""
    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
    with open(filename, "wb") as outfile:
        outfile.write(bytesio.getbuffer())


def rm_temp():
    """Очистка временной директории."""
    os.makedirs(TEMP_DIR, exist_ok=True)
    for filename in os.listdir(TEMP_DIR):
        try:
            os.remove(os.path.join(TEMP_DIR, filename))
        except OSError:
            pass


def _get_bgr_color(class_id):
    """Цвет из config.yml (RGB) → BGR для OpenCV."""
    rgb = colors.get(class_id, (0, 255, 0)) if colors else (0, 255, 0)
    try:
        r, g, b = int(rgb[0]), int(rgb[1]), int(rgb[2])
        return (b, g, r)
    except Exception:
        return DEFAULT_COLOR_BGR


def _draw_detections(frame, detections, class_names_local):
    """Рисование рамок и подписей. detections: [x1,y1,x2,y2,score,class_id]."""
    for det in detections:
        x1, y1, x2, y2, score, class_id = det
        cid = int(class_id)
        if cid < 0 or cid >= len(class_names_local):
            continue
        color = _get_bgr_color(cid)
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 4)
        name = class_names_local[cid]
        label = f"{name}-{int(score * 100)}%"
        cv2.putText(frame, label, (int(x1), int(y1 - 10)),
                    cv2.FONT_HERSHEY_COMPLEX, 1.4, color, 3, cv2.LINE_AA)


# --------------------------------------------------------------------------- #
# Классификация
# --------------------------------------------------------------------------- #
def classification(results, rows):
    """
    Сравнение обнаруженных объектов с выбранным комплектом униформы.

    results: dict {class_name: count}
    rows:    list требуемых компонентов униформы, например
             ['блуза', 'жакет', 'юбка', 'туфли', ...]

    Возвращает DataFrame со столбцами 'наличие' и 'нарушения'.
    """
    try:
        df_match = pd.read_csv(
            MATCH_FILE, sep=";", index_col=0, encoding="utf-8-sig"
        ).fillna(0)
    except Exception as e:
        st.error(f"Не удалось прочитать {MATCH_FILE}: {e}")
        return pd.DataFrame("", index=rows, columns=["наличие", "нарушения"])

    uniform_rows = list(dict.fromkeys(rows))          # сохраняем порядок, убираем дубли
    df = pd.DataFrame("", index=uniform_rows, columns=["наличие", "нарушения"])

    component_counts = {row: 0 for row in uniform_rows}
    extra_items: list[str] = []

    # --- 1. Агрегируем вклады обнаруженных классов ---------------------------
    for classname, cnt in results.items():
        if cnt <= 0:
            continue

        if classname not in df_match.index:
            # Класс, которого нет в маппинге
            if "(нарушение)" in classname:
                extra_items.append(classname)
            continue

        row_data = df_match.loc[classname]
        mapped = False

        for col in df_match.columns:
            try:
                val = row_data[col]
            except KeyError:
                continue
            if val <= 0:
                continue

            if col in uniform_rows:
                component_counts[col] += cnt
                mapped = True
                if val == 2:
                    if classname not in df.loc[col, "нарушения"]:
                        df.loc[col, "нарушения"] += f"{classname}\n"
            else:
                # Класс соответствует компоненту, которого нет в текущем комплекте
                try:
                    min_val = df_match.loc["min", col]
                except KeyError:
                    min_val = 0
                if min_val > 0:
                    extra_items.append(classname)

        if "(нарушение)" in classname and not mapped:
            extra_items.append(classname)

    # --- 2. Проверка количества по каждому компоненту -----------------------
    for row in uniform_rows:
        if row not in df_match.columns:
            continue
        try:
            min_val = df_match.loc["min", row]
            max_val = df_match.loc["max", row]
        except KeyError:
            continue

        count = component_counts[row]
        if min_val <= count <= max_val:
            if count > 0:
                df.loc[row, "наличие"] = "+"
        else:
            if count == 0 and min_val > 0:
                df.loc[row, "нарушения"] += "отсутствует\n"
            else:
                df.loc[row, "нарушения"] += "несоответствие количества\n"

    # --- 3. Лишние/нарушающие предметы отдельными строками -------------------
    seen: set[str] = set()
    idx = 1
    for v in extra_items:
        if v in seen:
            continue
        seen.add(v)
        df.loc[f"нарушение_{idx}"] = ["", v]
        idx += 1

    return df


# --------------------------------------------------------------------------- #
# Сохранение конфигурации
# --------------------------------------------------------------------------- #
class IndentDumper(yaml.Dumper):
    def increase_indent(self, flow=False, indentless=False):
        return super().increase_indent(flow, False)


def save_config(**kwargs):
    """Сохранение секций в config.yml одним yaml.dump (без '---')."""
    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as fh:
            yaml.dump(
                kwargs, fh,
                sort_keys=False,
                allow_unicode=True,
                default_flow_style=False,
                Dumper=IndentDumper,
            )
        print("Файл config.yml успешно обновлён")
    except Exception as e:
        print(f"Ошибка записи config.yml: {e}")
        st.sidebar.error(f"Ошибка записи config.yml: {e}")


# --------------------------------------------------------------------------- #
# VideoTransformer
# --------------------------------------------------------------------------- #
def create_video_transformer(model, class_names_local, threshold, shared_results):
    """
    Фабрика: возвращает класс VideoTransformer, замкнутый на актуальные
    model / class_names / colors / threshold / shared_results.
    shared_results — список длиной len(class_names), куда пишутся максимумы
    по кадрам для последующей классификации.
    """

    class VideoTransformer(VideoTransformerBase):
        def __init__(self):
            self.model = model
            self.class_names = class_names_local
            self.threshold = threshold
            self.shared = shared_results

        def transform(self, frame):
            frame = frame.to_ndarray(format="bgr24")           # BGR
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)       # для модели
            results = self.model(rgb)[0]

            detections = remove_duplicate(results.boxes.data.tolist())
            frame_counts = [0] * len(self.class_names)
            valid_dets = []

            for det in detections:
                cid = int(det[5])
                score = float(det[4])
                if cid == -1 or cid >= len(self.class_names) or score <= self.threshold:
                    continue
                frame_counts[cid] += 1
                valid_dets.append(det)

            _draw_detections(frame, valid_dets, self.class_names)

            # сохраняем максимум по кадрам
            for i in range(len(self.shared)):
                if frame_counts[i] > self.shared[i]:
                    self.shared[i] = frame_counts[i]

            return frame  # BGR — streamlit_webrtc ожидает BGR

    return VideoTransformer


# --------------------------------------------------------------------------- #
# Основное приложение
# --------------------------------------------------------------------------- #
def main():
    st.set_page_config(page_title="Внешний вид")
    st.title("Сервис оценки внешнего вида сотрудников")
    rm_temp()

    global class_names, colors, model_names, model_files, uniform, uniform_consist
    global threshold, model

    # ---------- Загрузка конфигурации ----------
    try:
        with open(CONFIG_FILE, encoding="utf-8") as fh:
            read_data = yaml.load(fh, Loader=yaml.FullLoader)
        path_dataset = read_data.get("path", "/data")
        path_imtrain = read_data.get("train", "images/train")
        path_imval = read_data.get("val", "images/train")
        class_names = read_data.get("names", {}) or {}
        colors = read_data.get("colors", {}) or {}
        model_names = read_data.get("model_names", []) or []
        model_files = read_data.get("model_files", []) or []
        uniform = read_data.get("uniform", {}) or {}
        uniform_consist = read_data.get("uniform_consist", []) or []
        print("Файл конфигурации config.yml успешно загружен")
    except Exception as e:
        print(f"Ошибка загрузки config.yml: {e}")
        path_dataset, path_imtrain, path_imval = "/data", "images/train", "images/train"
        class_names, colors = {}, {}
        model_names, model_files = [], []
        uniform, uniform_consist = {}, []

    if not uniform:
        st.error("В config.yml не задан ни один комплект одежды.")
        return

    # ---------- Боковая панель ----------
    if os.path.exists("headimage.jpg"):
        st.sidebar.image("headimage.jpg")

    detect_mode = st.sidebar.radio(
        "Тип входных данных",
        ("Изображение", "Видеозапись", "Камера"),
        index=0,
    )
    type_uniform = st.sidebar.selectbox(
        "Проверяемый комплект одежды", tuple(uniform.keys()), index=0
    )

    # ---------- Редактирование комплекта ----------
    tab_view, tab_edit, tab_delete = st.sidebar.tabs(["📋", "📝", "❌"])

    with tab_view:
        st.write("Текущий состав комплекта:")
        st.write(uniform[type_uniform])

    new_uniform = tab_edit.multiselect(
        "Состав комплекта", list(uniform_consist), default=uniform[type_uniform]
    )
    new_name = tab_edit.text_input("Наименование", value=type_uniform)

    if tab_edit.button("Сохранить"):
        if new_name and new_name != type_uniform:
            uniform[new_name] = new_uniform
            uniform.pop(type_uniform, None)
        else:
            uniform[type_uniform] = new_uniform
        save_config(
            path=path_dataset, train=path_imtrain, val=path_imval,
            names=class_names, colors=colors,
            model_names=model_names, model_files=model_files,
            uniform=uniform, uniform_consist=uniform_consist,
        )
        st.sidebar.success("Сохранено")
        st.rerun()

    if tab_delete.button("Удалить комплект одежды"):
        if len(uniform) <= 1:
            st.sidebar.error("Нельзя удалить единственный комплект")
        else:
            uniform.pop(type_uniform, None)
            save_config(
                path=path_dataset, train=path_imtrain, val=path_imval,
                names=class_names, colors=colors,
                model_names=model_names, model_files=model_files,
                uniform=uniform, uniform_consist=uniform_consist,
            )
            st.sidebar.success("Удалено")
            st.rerun()

    # ---------- Порог и модель ----------
    threshold = 0.01 * st.sidebar.slider(
        "Порог обнаружения", min_value=0.0, max_value=100.0, value=10.0
    )

    if not model_names:
        st.write("Алгоритмы распознавания не найдены. Используется yolov8n.pt")
        model_type = "default"
    else:
        model_type = st.sidebar.selectbox(
            "Алгоритм распознавания", tuple(model_names), index=0
        )

    model = get_model(model_type)
    if not class_names:
        class_names = model.names  # dict {id: name}

    uniform_items = uniform[type_uniform]

    # ---------- Камера ----------
    if detect_mode == "Камера":
        shared_results = [0] * len(class_names)
        VT = create_video_transformer(model, class_names, threshold, shared_results)
        webrtc_streamer(key="uniform_camera", video_transformer_factory=VT)

        if any(shared_results):
            results_dict = dict(zip(class_names.values(), shared_results))
            st.dataframe(
                classification(results=results_dict, rows=uniform_items),
                use_container_width=True,
            )

    # ---------- Видео ----------
    elif detect_mode == "Видеозапись":
        uploaded_files = st.file_uploader(
            "Выберите один или несколько видеофайлов",
            ["mp4", "mov", "avi"],
            accept_multiple_files=True,
        )
        if st.button("Оценить внешний вид"):
            if uploaded_files:
                rm_temp()
                tab_view2, tab_report = st.tabs(["🗃", "📈"])
                results_list = [0] * len(class_names)

                for uploaded_file in uploaded_files:
                    base, ext = os.path.splitext(uploaded_file.name)
                    temp_in = os.path.join(TEMP_DIR, uploaded_file.name)
                    temp_out = os.path.join(TEMP_DIR, f"{base}_out{ext}")

                    write_bytesio_to_file(temp_in, uploaded_file)

                    cap = cv2.VideoCapture(temp_in)
                    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
                    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
                    FPS = cap.get(cv2.CAP_PROP_FPS) or 25

                    fourcc = cv2.VideoWriter_fourcc(*"avc1")
                    out = cv2.VideoWriter(temp_out, fourcc, FPS, (W, H))
                    if not out.isOpened():
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        out = cv2.VideoWriter(temp_out, fourcc, FPS, (W, H))

                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break

                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        results = model(rgb)[0]
                        detections = remove_duplicate(results.boxes.data.tolist())

                        frame_counts = [0] * len(class_names)
                        valid = []
                        for det in detections:
                            cid = int(det[5])
                            score = float(det[4])
                            if cid == -1 or cid >= len(class_names) or score <= threshold:
                                continue
                            frame_counts[cid] += 1
                            valid.append(det)

                        _draw_detections(frame, valid, class_names)
                        out.write(frame)

                        for i in range(len(results_list)):
                            if frame_counts[i] > results_list[i]:
                                results_list[i] = frame_counts[i]

                    out.release()
                    cap.release()

                    tab_view2.video(temp_out, start_time=0)
                    with open(temp_out, "rb") as f:
                        tab_view2.download_button(
                            label="Загрузить",
                            data=f,
                            file_name=f"{base}_out{ext}",
                            mime="video/mp4",
                        )

                results_dict = dict(zip(class_names.values(), results_list))
                tab_report.dataframe(
                    classification(results=results_dict, rows=uniform_items),
                    use_container_width=True,
                )

    # ---------- Изображение ----------
    elif detect_mode == "Изображение":
        uploaded_files = st.file_uploader(
            "Выберите одно или несколько изображений",
            type=["jpg", "jpeg", "png", "bmp", "dib"],
            accept_multiple_files=True,
        )
        if st.button("Оценить внешний вид"):
            if uploaded_files:
                rm_temp()
                tab_view2, tab_report = st.tabs(["🗃", "📈"])
                results_list = [0] * len(class_names)

                for uploaded_file in uploaded_files:
                    base, ext = os.path.splitext(uploaded_file.name)
                    temp_in = os.path.join(TEMP_DIR, uploaded_file.name)
                    temp_out = os.path.join(TEMP_DIR, f"{base}_out{ext}")

                    with open(temp_in, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    img = cv2.imread(temp_in)
                    if img is None:
                        st.warning(f"Не удалось прочитать {uploaded_file.name}")
                        continue

                    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    results = model(rgb)[0]
                    detections = remove_duplicate(results.boxes.data.tolist())

                    valid = []
                    for det in detections:
                        cid = int(det[5])
                        score = float(det[4])
                        if cid == -1 or cid >= len(class_names) or score <= threshold:
                            continue
                        results_list[cid] += 1
                        valid.append(det)

                    _draw_detections(img, valid, class_names)
                    cv2.imwrite(temp_out, img)
                    tab_view2.image(temp_out)

                results_dict = dict(zip(class_names.values(), results_list))
                tab_report.dataframe(
                    classification(results=results_dict, rows=uniform_items),
                    use_container_width=True,
                )


if __name__ == "__main__":
    main()