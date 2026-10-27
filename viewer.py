"""
FUL Lab: a PyQt6 test bench for the FUL codec (v9: whole-image directional 9/7
wavelets, post-filter, learned priors, dependent quantisation, real-cost RDO, perceptual quantisation, context-modelled rANS,
denoising and grain).

Run:  python viewer.py [optional image or .ful path]

Open any image, tune the encoder settings, and compare the original with the
decoded result side by side. Every change re-encodes AND decodes, so what you
see on the right is a true round trip through the file format.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from PyQt6.QtCore import QBuffer, QIODevice, Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QAction, QImage, QKeySequence, QPixmap
from PyQt6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QFileDialog, QFormLayout, QGridLayout,
    QGroupBox,
    QHBoxLayout, QLabel, QMainWindow, QMessageBox, QPushButton, QScrollArea,
    QSpinBox, QSplitter, QVBoxLayout, QWidget,
)

import codec_v1
import codec_v2
import codec_v3
import codec_v4
import codec_v5
import codec_v6
import codec_v7
import codec_v8
import codec_v9
import ful
import formats
import metrics

FORMATS = formats.available_formats()


def _gpu_module():
    """Import the GPU decoder lazily so the lab still runs without PyOpenGL."""
    try:
        import gpu_decode
        return gpu_decode, None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"

IMAGE_FILTER = "Images (*.png *.jpg *.jpeg *.heic *.heif *.bmp *.gif *.tif *.tiff *.webp *.avif *.jxl *.jp2 *.j2k)"
LARGE_IMAGE = 6_000_000      # pixels; above this the all-formats comparison starts switched off
FUL_FILTER = "FUL images (*.ful)"
ZOOM_LEVELS = ["Fit", "25%", "50%", "100%", "200%", "400%", "800%"]


# --------------------------------------------------------------------------
# Conversions and small helpers
# --------------------------------------------------------------------------
def qimage_to_rgb(img: QImage) -> np.ndarray:
    img = img.convertToFormat(QImage.Format.Format_RGB888)
    w, h, bpl = img.width(), img.height(), img.bytesPerLine()
    ptr = img.constBits()
    ptr.setsize(img.sizeInBytes())
    rows = np.frombuffer(ptr, np.uint8).reshape(h, bpl)   # rows may be padded
    return rows[:, : w * 3].reshape(h, w, 3).copy()


def rgb_to_qimage(arr: np.ndarray) -> QImage:
    arr = np.ascontiguousarray(arr, dtype=np.uint8)
    h, w, _ = arr.shape
    return QImage(arr.tobytes(), w, h, 3 * w, QImage.Format.Format_RGB888).copy()


def qt_encoded_size(img: QImage, fmt: str, quality: int = -1) -> int:
    buf = QBuffer()
    buf.open(QIODevice.OpenModeFlag.WriteOnly)
    img.save(buf, fmt, quality)
    return buf.size()


def read_photo(path) -> np.ndarray:
    """Open a photo the way a phone or camera intends it to be seen: Pillow first
    (applies the EXIF rotation; HEIC/HEIF if pillow-heif is installed), then Qt,
    then the comparison codecs (AVIF, JPEG XL, JPEG 2000)."""
    path = Path(path)
    try:
        from PIL import Image, ImageOps
        try:
            import pillow_heif
            pillow_heif.register_heif_opener()
        except ImportError:
            pass
        with Image.open(path) as im:
            return np.ascontiguousarray(np.asarray(ImageOps.exif_transpose(im).convert("RGB")))
    except Exception:
        pass
    img = QImage(str(path))
    if not img.isNull():
        return qimage_to_rgb(img)
    data = path.read_bytes()
    for fmt in FORMATS:
        try:
            rgb = fmt.decode(data)
        except Exception:
            continue
        if rgb.ndim == 3 and rgb.dtype == np.uint8:
            return np.ascontiguousarray(rgb[..., :3])
    hint = " (for HEIC photos: pip install pillow-heif)" if path.suffix.lower() in (".heic", ".heif") else ""
    raise ValueError("Couldn't read this image format." + hint)


def human_bytes(n: float) -> str:
    if n < 1024:
        return f"{int(n)} B"
    if n < 1024 ** 2:
        return f"{n / 1024:.1f} KB"
    return f"{n / 1024 ** 2:.2f} MB"


# --------------------------------------------------------------------------
# Background worker so the UI stays responsive
# --------------------------------------------------------------------------
_codec_warmed = False


def warm_up_codec():
    """Compile the codec on a tiny image so timings measure the codec itself,
    not Numba's one-time compilation."""
    global _codec_warmed
    if not _codec_warmed:
        tiny = np.zeros((8, 8, 3), np.uint8)
        codec_v9.decode(codec_v9.encode(tiny, 50, True, denoise=1.0,
                                        grain=True, masking=True))
        _codec_warmed = True


class CodecWorker(QThread):
    done = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, rgb: np.ndarray, settings: dict, parent=None):
        super().__init__(parent)
        self.rgb = rgb
        self.settings = settings

    def run(self):
        try:
            warm_up_codec()
            t0 = time.perf_counter()
            settings = dict(self.settings)
            mode = settings.pop("denoise_mode")
            compare_all = settings.pop("compare_all")
            noise = codec_v9.estimate_noise_v2(self.rgb)
            strength = (codec_v9.auto_denoise_strength_v2(noise) if mode == "auto"
                        else float(mode))
            t0 = time.perf_counter()
            data = codec_v9.encode(self.rgb, denoise=strength, **settings)
            t1 = time.perf_counter()
            decoded = codec_v9.decode(data, grain=0)       # measured without grain
            t2 = time.perf_counter()
            header = codec_v9.read_header(data)
            has_grain = header["grain"]
            grained = codec_v9.decode(data, grain=1) if has_grain else decoded
            others = {}
            for fmt in FORMATS:
                if not compare_all and fmt.name != "JPEG":
                    continue
                found = formats.same_size(fmt, self.rgb, len(data))
                if found is None:
                    others[fmt.name] = None      # can't get this small
                    continue
                setting, fdata, pix = found
                others[fmt.name] = {"label": fmt.label(setting), "size": len(fdata),
                                    "decoded": pix, "ssim": metrics.ssim(self.rgb, pix),
                                    "psnr": metrics.psnr(self.rgb, pix)}
            self.done.emit({
                "data": data, "decoded": decoded, "grained": grained,
                "has_grain": has_grain, "wavelet": header["wavelet"],
                "levels": header["levels"], "directional": header["directional"],
                "dq": header["dq"], "filter": header["filter"], "noise": noise, "strength": strength,
                "settings": self.settings,
                "enc_ms": (t1 - t0) * 1000, "dec_ms": (t2 - t1) * 1000,
                "ssim": metrics.ssim(self.rgb, decoded),
                "psnr": metrics.psnr(self.rgb, decoded),
                "others": others,
            })
        except Exception as exc:  # report anything to the UI
            self.failed.emit(f"{type(exc).__name__}: {exc}")


# --------------------------------------------------------------------------
# Image pane with zoom
# --------------------------------------------------------------------------
class ImagePane(QWidget):
    def __init__(self, title: str):
        super().__init__()
        self.title = QLabel(title)
        self.title.setStyleSheet("font-weight: 600; padding: 4px 2px;")
        self.label = QLabel()
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.scroll = QScrollArea()
        self.scroll.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.scroll.setWidget(self.label)
        self.scroll.setStyleSheet("QScrollArea { background: #3a3a3a; }")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.title)
        layout.addWidget(self.scroll, 1)

        self._image: QImage | None = None
        self._zoom = "Fit"

    def set_title(self, text: str):
        self.title.setText(text)

    def set_image(self, img: QImage | None):
        self._image = img
        self._render()

    def set_zoom(self, zoom: str):
        self._zoom = zoom
        self._render()

    def _render(self):
        if self._image is None or self._image.isNull():
            self.label.clear()
            self.label.resize(1, 1)
            return
        pix = QPixmap.fromImage(self._image)
        if self._zoom == "Fit":
            vp = self.scroll.viewport().size()
            f = min(vp.width() / pix.width(), vp.height() / pix.height())
        else:
            f = int(self._zoom.rstrip("%")) / 100
        w, h = max(1, round(pix.width() * f)), max(1, round(pix.height() * f))
        # Nearest-neighbour when enlarging so you can inspect real pixels.
        mode = (Qt.TransformationMode.SmoothTransformation if f < 1
                else Qt.TransformationMode.FastTransformation)
        self.label.setPixmap(pix.scaled(w, h, Qt.AspectRatioMode.IgnoreAspectRatio, mode))
        self.label.resize(w, h)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._zoom == "Fit":
            self._render()


# --------------------------------------------------------------------------
# Main window
# --------------------------------------------------------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FUL Lab")
        self.resize(1400, 850)
        self.setAcceptDrops(True)

        self.source_rgb: np.ndarray | None = None
        self.source_qimg: QImage | None = None
        self.source_name = ""
        self.reference_sizes: dict[str, int] = {}
        self.result: dict | None = None
        self.decoded_qimg: QImage | None = None
        self.worker: CodecWorker | None = None
        self.pending = False
        self.first_run = True

        self.debounce = QTimer(self)
        self.debounce.setSingleShot(True)
        self.debounce.setInterval(250)
        self.debounce.timeout.connect(self.start_encode)

        self._build_ui()
        self._build_menu()

    # ----- UI construction -----
    def _build_ui(self):
        # Encoder settings
        self.quality = QSpinBox(minimum=1, maximum=100, value=75)
        self.perceptual = QCheckBox("Perceptual tuning")
        self.perceptual.setChecked(True)
        self.denoise = QComboBox()
        self.denoise.addItems(["Auto", "Off", "Light", "Standard", "Strong"])
        self.rdo = QCheckBox("Rate-distortion optimisation")
        self.rdo.setChecked(True)
        self.directional = QCheckBox("Directional lifting (honeycomb)")
        self.directional.setChecked(True)
        self.dq = QCheckBox("Dependent quantisation")
        self.dq.setChecked(True)
        self.post_filter = QCheckBox("Post-filter (edge-aware clean-up)")
        self.post_filter.setChecked(True)
        self.masking = QCheckBox("Texture masking (experimental)")
        hint = QLabel("Quality 100 is lossless (5/3 wavelet); lower qualities "
                      "use the 9/7. Auto denoise only cleans photos that need "
                      "it. Masking lowers SSIM in tests, so judge it by eye.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: gray; font-size: 11px;")
        enc_form = QFormLayout()
        enc_form.addRow("Quality", self.quality)
        enc_form.addRow(self.perceptual)
        enc_form.addRow("Denoise", self.denoise)
        enc_form.addRow(self.rdo)
        enc_form.addRow(self.directional)
        enc_form.addRow(self.dq)
        enc_form.addRow(self.post_filter)
        enc_form.addRow(self.masking)
        enc_form.addRow(hint)
        enc_box = QGroupBox("Encoder")
        enc_box.setLayout(enc_form)

        # View options
        self.right_mode = QComboBox()
        self.right_mode.addItems(["Decoded", "Difference"])
        for fmt in FORMATS:
            self.right_mode.addItems([f"{fmt.name} at same size", f"{fmt.name} difference"])
        self.diff_gain = QSpinBox(minimum=1, maximum=100, value=10, suffix="×")
        self.zoom = QComboBox()
        self.zoom.addItems(ZOOM_LEVELS)
        self.show_grain = QCheckBox("Regenerate grain")
        self.show_grain.setChecked(True)
        self.use_gpu = QCheckBox("Decode on GPU")
        self.flip = QPushButton("Hold to show original")
        view_form = QFormLayout()
        view_form.addRow("Right pane", self.right_mode)
        view_form.addRow("Difference gain", self.diff_gain)
        view_form.addRow("Zoom", self.zoom)
        view_form.addRow(self.show_grain)
        view_form.addRow(self.use_gpu)
        view_form.addRow(self.flip)
        view_box = QGroupBox("View")
        view_box.setLayout(view_form)

        # Results
        self.stats: dict[str, QLabel] = {}
        stats_form = QFormLayout()
        for key in ["Image", "Raw RGB", "FUL file", "Bits per pixel",
                    "Ratio vs raw", "PNG", "JPEG (q90)", "PSNR", "SSIM", "Max error", "Noise", "Wavelet",
                    "GPU decode", "GPU check",
                    "Encode time", "Decode time"]:
            lbl = QLabel("–")
            lbl.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            self.stats[key] = lbl
            stats_form.addRow(key, lbl)
        stats_box = QGroupBox("Results")
        stats_box.setLayout(stats_form)

        # FUL against other formats at the same file size
        vs_grid = QGridLayout()
        for col, name in enumerate(["Format", "Setting", "Size", "SSIM"]):
            head = QLabel(name)
            head.setStyleSheet("font-weight: 600;")
            vs_grid.addWidget(head, 0, col)
        self.vs: dict[tuple[str, int], QLabel] = {}
        self.vs_names = ["FUL"] + [f.name for f in FORMATS]
        for row, name in enumerate(self.vs_names, 1):
            vs_grid.addWidget(QLabel(name), row, 0)
            for col in (1, 2, 3):
                lbl = QLabel("–")
                self.vs[(name, col)] = lbl
                vs_grid.addWidget(lbl, row, col)
        self.compare_all = QCheckBox("Include WebP, AVIF, JPEG XL, JPEG 2000")
        self.compare_all.setChecked(True)
        self.compare_all.setToolTip("Each format is encoded several times to match the "
                                    "FUL file's size, which can be slow on large images.")
        self.vs_verdict = QLabel("")
        self.vs_verdict.setWordWrap(True)
        vs_layout = QVBoxLayout()
        vs_layout.addLayout(vs_grid)
        vs_layout.addWidget(self.compare_all)
        vs_layout.addWidget(self.vs_verdict)
        vs_box = QGroupBox("Against other formats at the same file size")
        vs_box.setLayout(vs_layout)

        sidebar = QWidget()
        sidebar.setFixedWidth(320)
        side_layout = QVBoxLayout(sidebar)
        side_layout.addWidget(enc_box)
        side_layout.addWidget(view_box)
        side_layout.addWidget(stats_box)
        side_layout.addWidget(vs_box)
        side_layout.addStretch(1)
        side_scroll = QScrollArea()
        side_scroll.setWidget(sidebar)
        side_scroll.setWidgetResizable(True)
        side_scroll.setFixedWidth(340)
        side_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        side_scroll.setFrameShape(QScrollArea.Shape.NoFrame)

        # Image panes
        self.left = ImagePane("Original")
        self.right = ImagePane("Decoded")
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.left)
        splitter.addWidget(self.right)

        central = QWidget()
        layout = QHBoxLayout(central)
        layout.addWidget(side_scroll)
        layout.addWidget(splitter, 1)
        self.setCentralWidget(central)

        # Keep both panes scrolled to the same spot
        for a, b in ((self.left.scroll, self.right.scroll),
                     (self.right.scroll, self.left.scroll)):
            a.horizontalScrollBar().valueChanged.connect(b.horizontalScrollBar().setValue)
            a.verticalScrollBar().valueChanged.connect(b.verticalScrollBar().setValue)

        # Signals
        self.quality.valueChanged.connect(self.schedule_encode)
        self.perceptual.toggled.connect(self.schedule_encode)
        self.masking.toggled.connect(self.schedule_encode)
        self.rdo.toggled.connect(self.schedule_encode)
        self.directional.toggled.connect(self.schedule_encode)
        self.dq.toggled.connect(self.schedule_encode)
        self.post_filter.toggled.connect(self.schedule_encode)
        self.denoise.currentIndexChanged.connect(self.schedule_encode)
        self.compare_all.toggled.connect(self.schedule_encode)
        self.show_grain.toggled.connect(self._on_grain_toggled)
        self.use_gpu.toggled.connect(self._on_gpu_toggled)
        self.right_mode.currentIndexChanged.connect(self.update_right_view)
        self.diff_gain.valueChanged.connect(self.update_right_view)
        self.zoom.currentTextChanged.connect(self._apply_zoom)
        self.flip.pressed.connect(self._show_original_on_right)
        self.flip.released.connect(self.update_right_view)

        self.statusBar().showMessage("Open an image, or drop one onto the window.")

    def _build_menu(self):
        from PyQt6.QtWidgets import QToolBar
        bar = QToolBar("Main")
        bar.setMovable(False)
        self.addToolBar(bar)
        for text, slot in (("Open image…", self.open_image_dialog),
                           ("Export .ful…", self.save_ful_dialog),
                           ("Export decoded PNG…", self.export_png_dialog)):
            act = QAction(text, self)
            act.triggered.connect(slot)
            bar.addAction(act)
        menu = self.menuBar().addMenu("&File")
        for text, shortcut, slot in [
            ("Open image…", QKeySequence.StandardKey.Open, self.open_image_dialog),
            ("Open FUL file…", QKeySequence("Ctrl+Shift+O"), self.open_ful_dialog),
            ("Export .ful…", QKeySequence.StandardKey.Save, self.save_ful_dialog),
            ("Export decoded PNG…", QKeySequence("Ctrl+Shift+E"), self.export_png_dialog),
            ("Load test pattern", None, self.load_test_pattern),
            ("Quit", QKeySequence.StandardKey.Quit, self.close),
        ]:
            action = QAction(text, self)
            if shortcut is not None:
                action.setShortcut(shortcut)
            action.triggered.connect(slot)
            menu.addAction(action)

    # ----- Loading and saving -----
    def open_image_dialog(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open image", "", IMAGE_FILTER)
        if path:
            self.open_path(path)

    def open_ful_dialog(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open FUL file", "", FUL_FILTER)
        if path:
            self.open_path(path)

    def open_path(self, path: str):
        p = Path(path)
        try:
            if p.suffix.lower() == ".ful":
                self.load_ful(p)
            else:
                self.load_image(p)
        except Exception as exc:
            QMessageBox.warning(self, "Couldn't open file", f"{p.name}\n\n{exc}")

    def load_image(self, path: Path):
        rgb = read_photo(path)
        self.source_path = path
        if rgb.shape[0] * rgb.shape[1] > LARGE_IMAGE and self.compare_all.isChecked():
            self.compare_all.blockSignals(True)
            self.compare_all.setChecked(False)
            self.compare_all.blockSignals(False)
            self.statusBar().showMessage(
                f"Large photo ({rgb.shape[1]} x {rgb.shape[0]}): comparison with other formats "
                "switched off to keep things quick. Tick it to compare anyway.", 8000)
        self.set_source(rgb, path.name)

    def load_ful(self, path: Path):
        data = path.read_bytes()
        info = ful.read_header(data)      # opens v1, v2 and v3 files
        rgb = ful.decode(data)
        if info["version"] >= 3:
            for widget, setter, value in (
                    (self.quality, self.quality.setValue, info["quality"]),
                    (self.perceptual, self.perceptual.setChecked, info["perceptual"])):
                widget.blockSignals(True)
                setter(value)
                widget.blockSignals(False)
        self.set_source(rgb, path.name)
        self.statusBar().showMessage(
            f"Decoded {path.name} (FUL v{info['version']}, {human_bytes(len(data))}). Its pixels are now "
            "the source, so the right pane shows a second-generation encode.")

    def load_test_pattern(self):
        self.set_source(codec_v1.make_test_image(), "Test pattern")

    def save_ful_dialog(self):
        if not self.result:
            QMessageBox.information(self, "Nothing to export", "Open an image first, then wait for it to encode.")
            return
        stem = Path(self.source_name).stem or "image"
        path, _ = QFileDialog.getSaveFileName(self, "Export .ful", f"{stem}.ful", FUL_FILTER)
        if not path:
            return
        data = self.result["data"]
        Path(path).write_bytes(data)
        lines = [f"Saved {Path(path).name}: {human_bytes(len(data))}",
                 f"Quality {self.quality.value()}, encoded in {self.result['enc_ms'] / 1000:.1f} s"]
        src = getattr(self, "source_path", None)
        if src is not None and Path(src).exists():
            orig = Path(src).stat().st_size
            lines.append(f"Original file {human_bytes(orig)}: the .ful is {len(data) / orig * 100:.0f}% of its size")
        jp = self.result.get("others", {}).get("JPEG")
        if jp:
            lines.append(f"A JPEG this size would be quality {jp['label'].split()[-1]} (SSIM {jp['ssim']:.4f} vs FUL {self.result['ssim']:.4f})")
        self.statusBar().showMessage(lines[0], 8000)
        QMessageBox.information(self, "Exported", "\n".join(lines))

    def export_png_dialog(self):
        """Save exactly what the .ful decodes to, as a PNG any app can open."""
        if not self.result:
            QMessageBox.information(self, "Nothing to export", "Open an image first, then wait for it to encode.")
            return
        stem = Path(self.source_name).stem or "image"
        path, _ = QFileDialog.getSaveFileName(self, "Export decoded PNG", f"{stem}_ful.png", "PNG images (*.png)")
        if not path:
            return
        rgb = self.result["grained"] if self.result.get("has_grain") and self.show_grain.isChecked() else self.result["decoded"]
        rgb_to_qimage(rgb).save(path, "PNG")
        self.statusBar().showMessage(f"Saved {Path(path).name}: what the .ful file looks like when decoded", 8000)

    def set_source(self, rgb: np.ndarray, name: str):
        self.source_rgb = rgb
        self.source_qimg = rgb_to_qimage(rgb)
        self.source_name = name
        self.result = None
        self.decoded_qimg = None
        self.reference_sizes = {
            "PNG": qt_encoded_size(self.source_qimg, "PNG"),
            "JPEG (q90)": len(FORMATS[0].encode(rgb, 90)),
        }
        self.left.set_title(f"Original: {name}")
        self.left.set_image(self.source_qimg)
        self.right.set_image(None)
        self.start_encode()

    # ----- Encoding -----
    def current_settings(self) -> dict:
        return {"quality": self.quality.value(),
                "perceptual": self.perceptual.isChecked(),
                "masking": self.masking.isChecked(),
                "rdo": self.rdo.isChecked(),
                "directional": self.directional.isChecked(),
                "dq": self.dq.isChecked(),
                "post_filter_on": self.post_filter.isChecked(),
                "compare_all": self.compare_all.isChecked(),
                "denoise_mode": {"Auto": "auto", "Off": 0.0, "Light": 0.5,
                                 "Standard": 1.0, "Strong": 1.5}[self.denoise.currentText()]}

    def schedule_encode(self):
        self.debounce.start()

    def start_encode(self):
        if self.source_rgb is None:
            return
        if self.worker is not None and self.worker.isRunning():
            self.pending = True
            return
        self.worker = CodecWorker(self.source_rgb, self.current_settings(), self)
        self.worker.done.connect(self.on_encoded)
        self.worker.failed.connect(self.on_failed)
        self.worker.finished.connect(self._on_worker_finished)
        msg = "Encoding…"
        if self.first_run and codec_v5.HAVE_NUMBA:
            msg = "Compiling the codec (first run only, a few seconds)…"
        self.statusBar().showMessage(msg)
        self.worker.start()

    def _on_worker_finished(self):
        sender = self.sender()
        if sender is self.worker:
            self.worker = None
        if sender is not None:
            sender.deleteLater()
        if self.pending:
            self.pending = False
            self.start_encode()

    def on_encoded(self, result: dict):
        worker = self.sender()
        if worker is not None and worker.rgb is not self.source_rgb:
            return  # result belongs to an image that has since been replaced
        self.first_run = False
        self.result = result
        self.decoded_qimg = rgb_to_qimage(result["decoded"])
        self.grained_qimg = (rgb_to_qimage(result["grained"]) if result["has_grain"]
                             else self.decoded_qimg)
        self.other_qimgs = {name: rgb_to_qimage(o["decoded"])
                            for name, o in result["others"].items() if o is not None}
        self.gpu_qimg = None
        self.update_stats()
        if self.use_gpu.isChecked():
            self._gpu_decode()
        else:
            self.update_right_view()
        self.statusBar().showMessage("Round trip complete.", 4000)

    def on_failed(self, message: str):
        self.statusBar().showMessage(f"Codec error: {message}")
        QMessageBox.critical(self, "Codec error", message)

    # ----- Display -----
    def _apply_zoom(self, zoom: str):
        self.left.set_zoom(zoom)
        self.right.set_zoom(zoom)

    def _show_original_on_right(self):
        if self.source_qimg is not None:
            self.right.set_title("Original (release to return)")
            self.right.set_image(self.source_qimg)

    def _on_grain_toggled(self):
        if self.use_gpu.isChecked() and self.result is not None:
            self._gpu_decode()
        else:
            self.update_right_view()

    def _on_gpu_toggled(self, on: bool):
        if on and self.result is not None:
            self._gpu_decode()
        else:
            self.gpu_qimg = None
            if not on:
                self.stats["GPU decode"].setText("–")
                self.stats["GPU check"].setText("–")
            self.update_right_view()

    def _gpu_decode(self):
        """Decode the current file on the GPU and check it against the CPU."""
        module, err = _gpu_module()
        decoder = module.get_decoder() if module else None
        if decoder is None:
            reason = err or module.last_error()
            self.stats["GPU decode"].setText("Unavailable")
            self.stats["GPU check"].setText("–")
            self.statusBar().showMessage(f"GPU decoding unavailable: {reason}")
            self.use_gpu.blockSignals(True)
            self.use_gpu.setChecked(False)
            self.use_gpu.blockSignals(False)
            self.update_right_view()
            return
        res = self.result
        grain_on = res["has_grain"] and self.show_grain.isChecked()
        try:
            rgb, t = decoder.decode(res["data"], grain=1.0 if grain_on else 0.0)
        except Exception as exc:
            self.stats["GPU decode"].setText("Failed")
            self.statusBar().showMessage(f"GPU decode failed: {type(exc).__name__}: {exc}")
            return
        expected = res["grained"] if grain_on else res["decoded"]
        differ = int(np.count_nonzero(np.any(rgb != expected, axis=2)))
        self.gpu_qimg = rgb_to_qimage(rgb)
        self.stats["GPU decode"].setText(f"{t['total_ms']:.0f} ms total, GPU {t['gpu_ms']:.0f} ms")
        self.stats["GPU decode"].setToolTip(
            f"CPU entropy decoding {t['cpu_ms']:.1f} ms, upload {t['upload_ms']:.1f} ms, "
            f"GPU {t['passes']} passes {t['gpu_ms']:.1f} ms, readback {t['readback_ms']:.1f} ms")
        self.stats["GPU check"].setText(
            "Identical to CPU decode" if differ == 0 else f"{differ:,} pixels differ")
        self.stats["GPU check"].setStyleSheet("" if differ == 0 else "color: #c62828;")
        self.statusBar().showMessage(
            f"Decoded on {decoder.renderer} in {t['passes']} shader passes.", 6000)
        self.update_right_view()

    def update_right_view(self):
        if self.result is None:
            return
        mode = self.right_mode.currentText()
        other_name = None
        if mode.endswith(" at same size"):
            other_name = mode[: -len(" at same size")]
        elif mode.endswith(" difference"):
            other_name = mode[: -len(" difference")]
        if other_name is not None:
            other = self.result["others"].get(other_name, "missing")
            if other == "missing":
                self.right.set_title(f"{other_name} is switched off "
                                     "(tick Include WebP, AVIF, JPEG XL, JPEG 2000)")
                self.right.set_image(None)
                return
            if other is None:
                self.right.set_title(f"{other_name} can't make a file this small")
                self.right.set_image(None)
                return
            oname = f"{other_name} at {other['label']}, {human_bytes(other['size'])}"
        if mode == "Decoded":
            grain_on = self.result["has_grain"] and self.show_grain.isChecked()
            extra = ", grain regenerated" if grain_on else ""
            if self.use_gpu.isChecked() and getattr(self, "gpu_qimg", None) is not None:
                self.right.set_title(f"Decoded on GPU, {human_bytes(len(self.result['data']))}{extra}")
                self.right.set_image(self.gpu_qimg)
            else:
                self.right.set_title(f"Decoded from FUL, {human_bytes(len(self.result['data']))}{extra}")
                self.right.set_image(self.grained_qimg if grain_on else self.decoded_qimg)
        elif mode.endswith(" at same size"):
            self.right.set_title(oname)
            self.right.set_image(self.other_qimgs[other_name])
        else:
            gain = self.diff_gain.value()
            pix = self.result["decoded"] if mode == "Difference" else other["decoded"]
            diff = np.abs(self.source_rgb.astype(np.int16) - pix.astype(np.int16)) * gain
            who = "FUL" if mode == "Difference" else oname
            self.right.set_title(f"{who} difference ×{gain} (black means identical)")
            self.right.set_image(rgb_to_qimage(np.clip(diff, 0, 255)))

    def update_stats(self):
        rgb, res = self.source_rgb, self.result
        h, w = rgb.shape[:2]
        raw, size = rgb.size, len(res["data"])
        p = res["psnr"]
        max_err = int(np.abs(rgb.astype(np.int16) - res["decoded"].astype(np.int16)).max())

        def vs(n: int) -> str:
            return f"{human_bytes(n)}  (FUL is {size / n * 100:.0f}%)"

        s = self.stats
        s["Image"].setText(f"{w} × {h}")
        s["Raw RGB"].setText(human_bytes(raw))
        s["FUL file"].setText(human_bytes(size))
        s["Bits per pixel"].setText(f"{size * 8 / (w * h):.2f}")
        s["Ratio vs raw"].setText(f"{raw / size:.2f}×")
        s["PNG"].setText(vs(self.reference_sizes["PNG"]))
        s["JPEG (q90)"].setText(vs(self.reference_sizes["JPEG (q90)"]))
        s["PSNR"].setText("Lossless" if p == float("inf") else f"{p:.2f} dB")
        s["SSIM"].setText("1 (identical)" if res["ssim"] >= 1 else f"{res['ssim']:.4f}")
        s["Max error"].setText(f"{max_err} of 255")
        s["Wavelet"].setText(f"{res['wavelet']}, {res['levels']} levels"
                             + (" (lossless)" if res["wavelet"] == "5/3" else "")
                             + (", directional" if res.get("directional") else "")
                             + (", DQ" if res.get("dq") else "")
                             + (", post-filter" if res.get("filter") else ""))
        strength = res["strength"]
        s["Noise"].setText(f"{res['noise']:.1f} levels, "
                           + ("not denoised" if strength == 0 else f"denoised at {strength:.2f}"))
        v = self.vs
        rows = {"FUL": {"label": f"quality {res['settings']['quality']}", "size": size,
                        "ssim": res["ssim"]}}
        rows.update(res["others"])
        scored = {}
        for name in self.vs_names:
            r = rows.get(name, "missing")
            for col in (1, 2, 3):
                v[(name, col)].setStyleSheet("")
            if r == "missing":
                for col, text in ((1, "off"), (2, ""), (3, "")):
                    v[(name, col)].setText(text)
                continue
            if r is None:
                for col, text in ((1, "can't go this small"), (2, ""), (3, "")):
                    v[(name, col)].setText(text)
                continue
            v[(name, 1)].setText(r["label"])
            v[(name, 2)].setText(human_bytes(r["size"]))
            v[(name, 3)].setText("1 (identical)" if r["ssim"] >= 1 else f"{r['ssim']:.4f}")
            scored[name] = r["ssim"]
        best = max(scored.values())
        for name, sc in scored.items():
            if best - sc < 0.001:                  # within a visible margin of the best
                v[(name, 3)].setStyleSheet("font-weight: 600; color: #2e7d32;")
        ful = scored["FUL"]
        ahead = [n for n in scored if n != "FUL" and scored[n] - ful > 0.001]
        level = [n for n in scored if n != "FUL" and abs(scored[n] - ful) <= 0.001]
        behind = [n for n in scored if n != "FUL" and ful - scored[n] > 0.001]
        join = lambda names: ", ".join(names[:-1]) + (" and " if len(names) > 1 else "") + names[-1]
        parts = []
        if behind:
            parts.append(f"ahead of {join(behind)}")
        if level:
            parts.append(f"level with {join(level)}")
        if ahead:
            parts.append(f"behind {join(ahead)}")
        if len(scored) == 1:
            note = "Nothing else can make a file this small."
        else:
            note = "At this size FUL is " + "; ".join(parts) + " (by SSIM; within 0.001 counts as level)."
        if res["strength"] > 0:
            note += (" FUL removed noise on purpose, which SSIM against the noisy "
                     "original counts as error, so judge by eye.")
        if res["has_grain"]:
            note += " SSIM excludes regenerated grain."
        self.vs_verdict.setText(note)
        s["Encode time"].setText(f"{res['enc_ms']:.1f} ms")
        s["Decode time"].setText(f"{res['dec_ms']:.1f} ms")

    # ----- Drag and drop -----
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if urls:
            self.open_path(urls[0].toLocalFile())


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    if len(sys.argv) > 1:
        win.open_path(sys.argv[1])
    else:
        win.load_test_pattern()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
