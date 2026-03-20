import ctypes
import os

import psutil
from i18n import t, LANGUAGES
from PyQt6.QtCore import QPoint, Qt, QTimer, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QCursor, QFont
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMenu,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizeGrip,
    QVBoxLayout,
    QWidget,
)

_GWL_EXSTYLE = -20
_WS_EX_TRANSPARENT = 0x20

DEFAULT_STYLE = {
    "preset": "default",
    "bg_color": "#000000",
    "bg_opacity": 240,
    "header_color": "#1a1a2e",
    "header_opacity": 230,
    "border_radius": 8,
    "original_font_family": "Microsoft YaHei",
    "translation_font_family": "Microsoft YaHei",
    "original_font_size": 11,
    "translation_font_size": 14,
    "original_color": "#cccccc",
    "translation_color": "#ffffff",
    "timestamp_color": "#888899",
    "window_opacity": 95,
}

_BASE = DEFAULT_STYLE

STYLE_PRESETS = {
    "default": dict(_BASE),
    "transparent": {
        **_BASE,
        "preset": "transparent",
        "bg_opacity": 120,
        "header_opacity": 120,
        "window_opacity": 70,
    },
    "compact": {
        **_BASE,
        "preset": "compact",
        "original_font_size": 9,
        "translation_font_size": 11,
    },
    "light": {
        **_BASE,
        "preset": "light",
        "bg_color": "#e8e8f0",
        "bg_opacity": 230,
        "header_color": "#c8c8d8",
        "header_opacity": 220,
        "original_color": "#333333",
        "translation_color": "#111111",
        "timestamp_color": "#666688",
    },
    "dracula": {
        **_BASE,
        "preset": "dracula",
        "bg_color": "#282a36",
        "bg_opacity": 235,
        "header_color": "#44475a",
        "header_opacity": 230,
        "original_color": "#f8f8f2",
        "translation_color": "#f8f8f2",
        "timestamp_color": "#6272a4",
    },
    "nord": {
        **_BASE,
        "preset": "nord",
        "bg_color": "#2e3440",
        "bg_opacity": 235,
        "header_color": "#3b4252",
        "header_opacity": 230,
        "original_color": "#d8dee9",
        "translation_color": "#eceff4",
        "timestamp_color": "#4c566a",
    },
    "monokai": {
        **_BASE,
        "preset": "monokai",
        "bg_color": "#272822",
        "bg_opacity": 235,
        "header_color": "#3e3d32",
        "header_opacity": 230,
        "original_color": "#f8f8f2",
        "translation_color": "#f8f8f2",
        "timestamp_color": "#75715e",
    },
    "solarized": {
        **_BASE,
        "preset": "solarized",
        "bg_color": "#002b36",
        "bg_opacity": 235,
        "header_color": "#073642",
        "header_opacity": 230,
        "original_color": "#839496",
        "translation_color": "#eee8d5",
        "timestamp_color": "#586e75",
    },
    "gruvbox": {
        **_BASE,
        "preset": "gruvbox",
        "bg_color": "#282828",
        "bg_opacity": 235,
        "header_color": "#3c3836",
        "header_opacity": 230,
        "original_color": "#ebdbb2",
        "translation_color": "#fbf1c7",
        "timestamp_color": "#928374",
    },
    "tokyo_night": {
        **_BASE,
        "preset": "tokyo_night",
        "bg_color": "#1a1b26",
        "bg_opacity": 235,
        "header_color": "#24283b",
        "header_opacity": 230,
        "original_color": "#a9b1d6",
        "translation_color": "#c0caf5",
        "timestamp_color": "#565f89",
    },
    "catppuccin": {
        **_BASE,
        "preset": "catppuccin",
        "bg_color": "#1e1e2e",
        "bg_opacity": 235,
        "header_color": "#313244",
        "header_opacity": 230,
        "original_color": "#cdd6f4",
        "translation_color": "#cdd6f4",
        "timestamp_color": "#6c7086",
    },
    "one_dark": {
        **_BASE,
        "preset": "one_dark",
        "bg_color": "#282c34",
        "bg_opacity": 235,
        "header_color": "#3e4452",
        "header_opacity": 230,
        "original_color": "#abb2bf",
        "translation_color": "#e5c07b",
        "timestamp_color": "#636d83",
    },
    "everforest": {
        **_BASE,
        "preset": "everforest",
        "bg_color": "#2d353b",
        "bg_opacity": 235,
        "header_color": "#343f44",
        "header_opacity": 230,
        "original_color": "#d3c6aa",
        "translation_color": "#d3c6aa",
        "timestamp_color": "#859289",
    },
    "kanagawa": {
        **_BASE,
        "preset": "kanagawa",
        "bg_color": "#1f1f28",
        "bg_opacity": 235,
        "header_color": "#2a2a37",
        "header_opacity": 230,
        "original_color": "#dcd7ba",
        "translation_color": "#dcd7ba",
        "timestamp_color": "#54546d",
    },
}


def _hex_to_rgba(hex_color: str, opacity: int) -> str:
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{opacity})"


class ChatMessage(QWidget):
    """Single chat message widget with original + async translation."""

    _current_style = DEFAULT_STYLE
    _compact_mode = False

    def __init__(
        self,
        msg_id: int,
        timestamp: str,
        original: str,
        source_lang: str,
        asr_ms: float,
        parent=None,
    ):
        super().__init__(parent)
        self.msg_id = msg_id
        self._original = original
        self._translated = ""
        self._timestamp = timestamp
        self._source_lang = source_lang
        self._asr_ms = asr_ms
        self._translate_ms = 0.0
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(8, 4, 8, 4)
        self._layout.setSpacing(2)

        s = self._current_style
        self._header_label = QLabel(self._build_header_html(s))
        self._header_label.setFont(
            QFont(s["original_font_family"], s["original_font_size"])
        )
        self._header_label.setTextFormat(Qt.TextFormat.RichText)
        self._header_label.setWordWrap(True)
        self._header_label.setStyleSheet("background: transparent;")
        self._layout.addWidget(self._header_label)

        self._trans_label = QLabel(
            f'<span style="color:#999; font-style:italic;">{t("translating")}</span>'
        )
        self._trans_label.setFont(
            QFont(s["translation_font_family"], s["translation_font_size"])
        )
        self._trans_label.setTextFormat(Qt.TextFormat.RichText)
        self._trans_label.setWordWrap(True)
        self._trans_label.setStyleSheet("background: transparent;")
        self._layout.addWidget(self._trans_label)

    def _build_header_html(self, s):
        if self._compact_mode:
            return (
                f'<span style="color:#6cf;">[{self._source_lang}]</span> '
                f'<span style="color:{s["original_color"]};">{_escape(self._original)}</span>'
            )
        return (
            f'<span style="color:{s["timestamp_color"]};">[{self._timestamp}]</span> '
            f'<span style="color:#6cf;">[{self._source_lang}]</span> '
            f'<span style="color:{s["original_color"]};">{_escape(self._original)}</span> '
            f'<span style="color:#8b8; font-size:9pt;">ASR {self._asr_ms:.0f}ms</span>'
        )

    def set_translation(self, translated: str, translate_ms: float):
        self._translated = translated or ""
        self._translate_ms = translate_ms
        s = self._current_style
        if translated:
            if self._compact_mode:
                self._trans_label.setText(
                    f'<span style="color:{s["translation_color"]};">&gt; {_escape(translated)}</span>'
                )
            else:
                self._trans_label.setText(
                    f'<span style="color:{s["translation_color"]};">&gt; {_escape(translated)}</span> '
                    f'<span style="color:#db8; font-size:9pt;">TL {translate_ms:.0f}ms</span>'
                )
        else:
            self._trans_label.setText(
                f'<span style="color:#aaa; font-style:italic;">&gt; {t("same_language")}</span>'
            )

    def apply_style(self, s: dict):
        self._header_label.setText(self._build_header_html(s))
        self._header_label.setFont(
            QFont(s["original_font_family"], s["original_font_size"])
        )
        self._trans_label.setFont(
            QFont(s["translation_font_family"], s["translation_font_size"])
        )
        if self._translated:
            if self._compact_mode:
                self._trans_label.setText(
                    f'<span style="color:{s["translation_color"]};">&gt; {_escape(self._translated)}</span>'
                )
            else:
                self._trans_label.setText(
                    f'<span style="color:{s["translation_color"]};">&gt; {_escape(self._translated)}</span> '
                    f'<span style="color:#db8; font-size:9pt;">TL {self._translate_ms:.0f}ms</span>'
                )

    def contextMenuEvent(self, event):
        menu = QMenu(self)
        menu.setStyleSheet("""
            QMenu { background: #2a2a3a; color: #ddd; border: 1px solid #555; }
            QMenu::item:selected { background: #444; }
        """)
        copy_orig = menu.addAction(t("copy_original"))
        copy_trans = menu.addAction(t("copy_translation"))
        copy_all = menu.addAction(t("copy_all"))
        menu.addSeparator()
        clear_list = menu.addAction(t("clear_list"))
        action = menu.exec(event.globalPos())
        if action == copy_orig:
            QApplication.clipboard().setText(self._original)
        elif action == copy_trans:
            QApplication.clipboard().setText(self._translated)
        elif action == copy_all:
            QApplication.clipboard().setText(f"{self._original}\n{self._translated}")
        elif action == clear_list:
            overlay = self.window()
            if hasattr(overlay, '_on_clear'):
                overlay._on_clear()


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


_BTN_CSS = """
    QPushButton {
        background: rgba(255,255,255,20);
        border: 1px solid rgba(255,255,255,40);
        border-radius: 3px;
        color: #aaa;
        font-size: 11px;
        padding: 0 6px;
    }
    QPushButton:hover {
        background: rgba(255,255,255,40);
        color: #ddd;
    }
"""

_BAR_CSS_TPL = """
    QProgressBar {{
        background: rgba(255,255,255,15);
        border: 1px solid rgba(255,255,255,30);
        border-radius: 3px;
        text-align: center;
        font-size: 8pt;
        color: #aaa;
    }}
    QProgressBar::chunk {{
        background: {color};
        border-radius: 2px;
    }}
"""


class MonitorBar(QWidget):
    """Compact system monitor displayed in the overlay."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("background: transparent;")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(2)

        row1 = QHBoxLayout()
        row1.setSpacing(6)

        # MIC bar (hidden when mic is disabled)
        self._mic_lbl = QLabel("MIC")
        self._mic_lbl.setFixedWidth(26)
        self._mic_lbl.setFont(QFont("Consolas", 8))
        self._mic_lbl.setStyleSheet("color: #888; background: transparent;")
        self._mic_lbl.setVisible(False)
        row1.addWidget(self._mic_lbl)

        self._mic_bar = QProgressBar()
        self._mic_bar.setRange(0, 100)
        self._mic_bar.setFixedHeight(14)
        self._mic_bar.setTextVisible(True)
        self._mic_bar.setFormat("%v%")
        self._mic_bar.setStyleSheet(_BAR_CSS_TPL.format(color="#c586c0"))
        self._mic_bar.setVisible(False)
        row1.addWidget(self._mic_bar)

        rms_lbl = QLabel("RMS:")
        rms_lbl.setFixedWidth(26)
        rms_lbl.setFont(QFont("Consolas", 8))
        rms_lbl.setStyleSheet("color: #888; background: transparent;")
        row1.addWidget(rms_lbl)

        self._rms_bar = QProgressBar()
        self._rms_bar.setRange(0, 100)
        self._rms_bar.setFixedHeight(14)
        self._rms_bar.setTextVisible(True)
        self._rms_bar.setFormat("%v%")
        self._rms_bar.setStyleSheet(_BAR_CSS_TPL.format(color="#4ec9b0"))
        row1.addWidget(self._rms_bar)

        vad_lbl = QLabel("VAD:")
        vad_lbl.setFixedWidth(26)
        vad_lbl.setFont(QFont("Consolas", 8))
        vad_lbl.setStyleSheet("color: #888; background: transparent;")
        row1.addWidget(vad_lbl)

        self._vad_bar = QProgressBar()
        self._vad_bar.setRange(0, 100)
        self._vad_bar.setFixedHeight(14)
        self._vad_bar.setTextVisible(True)
        self._vad_bar.setFormat("%v%")
        self._vad_bar.setStyleSheet(_BAR_CSS_TPL.format(color="#dcdcaa"))
        row1.addWidget(self._vad_bar)

        layout.addLayout(row1)

        self._stats_label = QLabel()
        self._stats_label.setFont(QFont("Consolas", 8))
        self._stats_label.setStyleSheet("color: #888; background: transparent;")
        self._stats_label.setTextFormat(Qt.TextFormat.RichText)
        self._stats_label.setWordWrap(True)
        layout.addWidget(self._stats_label)

        self._proc = psutil.Process(os.getpid())
        self._proc.cpu_percent(interval=None)  # Prime the counter
        self._cpu = 0
        self._ram_mb = 0.0
        self._gpu_text = "N/A"
        self._asr_device = ""
        self._asr_count = 0
        self._tl_count = 0
        self._prompt_tokens = 0
        self._completion_tokens = 0
        self._cost = 0.0

        self._sys_timer = QTimer(self)
        self._sys_timer.timeout.connect(self._update_system)
        self._sys_timer.start(1000)
        self._update_system()
        self._refresh_stats()

    def update_audio(self, rms: float, vad: float, mic_rms=None):
        self._rms_bar.setValue(min(100, int(rms * 500)))
        self._vad_bar.setValue(min(100, int(vad * 100)))
        mic_active = mic_rms is not None
        if self._mic_lbl.isVisible() != mic_active:
            self._mic_lbl.setVisible(mic_active)
            self._mic_bar.setVisible(mic_active)
        if mic_active:
            self._mic_bar.setValue(min(100, int(mic_rms * 500)))

    def update_asr_device(self, device: str):
        self._asr_device = device
        self._refresh_stats()

    def update_pipeline_stats(
        self, asr_count, tl_count, prompt_tokens, completion_tokens, cost=0.0
    ):
        self._asr_count = asr_count
        self._tl_count = tl_count
        self._prompt_tokens = prompt_tokens
        self._completion_tokens = completion_tokens
        self._cost = cost
        self._refresh_stats()

    def _update_system(self):
        try:
            self._cpu = int(self._proc.cpu_percent(interval=None) / os.cpu_count())
            self._ram_mb = self._proc.memory_info().rss / 1024 / 1024
        except Exception:
            pass
        try:
            import torch

            if torch.cuda.is_available():
                alloc = torch.cuda.memory_allocated() / 1024 / 1024
                self._gpu_text = f"{alloc:.0f}MB"
            else:
                self._gpu_text = "N/A"
        except Exception:
            self._gpu_text = "N/A"
        self._refresh_stats()

    def _refresh_stats(self):
        total = self._prompt_tokens + self._completion_tokens
        tokens_str = f"{total / 1000:.1f}k" if total >= 1000 else str(total)
        dev_str = ""
        if self._asr_device:
            dev_color = "#4ec9b0" if "cuda" in self._asr_device.lower() else "#dcdcaa"
            dev_str = (
                f'<span style="color:{dev_color};">{self._asr_device}</span> '
                f'<span style="color:#555;">|</span> '
            )
        cost_str = ""
        if self._cost > 0:
            from i18n import get_lang
            symbol = "¥" if get_lang() == "zh" else "$"
            cost_str = f' <span style="color:#fa5;">{symbol}{self._cost:.4f}</span>'
        self._stats_label.setText(
            f"{dev_str}"
            f'<span style="color:#6cf;">CPU</span> {self._cpu}% '
            f'<span style="color:#6cf;">RAM</span> {self._ram_mb:.0f}MB '
            f'<span style="color:#6cf;">GPU</span> {self._gpu_text} '
            f'<span style="color:#555;">|</span> '
            f'<span style="color:#8b8;">ASR</span> {self._asr_count} '
            f'<span style="color:#db8;">TL</span> {self._tl_count} '
            f'<span style="color:#c9c;">Tok</span> {tokens_str} '
            f'<span style="color:#666;">({self._prompt_tokens}\u2191{self._completion_tokens}\u2193)</span>'
            f'{cost_str}'
        )


class _DragArea(QWidget):
    """Small draggable area (title + grip)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setCursor(QCursor(Qt.CursorShape.SizeAllCursor))
        self._drag_pos = None

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_pos = (
                event.globalPosition().toPoint()
                - self.window().frameGeometry().topLeft()
            )

    def mouseMoveEvent(self, event):
        if self._drag_pos and event.buttons() & Qt.MouseButton.LeftButton:
            self.window().move(event.globalPosition().toPoint() - self._drag_pos)

    def mouseReleaseEvent(self, event):
        self._drag_pos = None


_COMBO_CSS = """
    QComboBox {
        background: rgba(255,255,255,20);
        border: 1px solid rgba(255,255,255,40);
        border-radius: 3px;
        color: #aaa;
        font-size: 11px;
        padding: 0 4px;
    }
    QComboBox:hover { background: rgba(255,255,255,40); color: #ddd; }
    QComboBox::drop-down { border: none; width: 14px; }
    QComboBox::down-arrow { image: none; border: none; }
    QComboBox QAbstractItemView {
        background: #2a2a3a; color: #ccc; selection-background-color: #444;
    }
"""

_SCROLL_CSS = """
    QScrollArea { border: none; background: transparent; }
    QScrollBar:vertical {
        width: 6px; background: transparent;
    }
    QScrollBar::handle:vertical {
        background: rgba(255,255,255,60); border-radius: 3px;
        min-height: 20px;
    }
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
        height: 0;
    }
"""


class RealtimeView(QWidget):
    """Dedicated view for real-time streaming transcription mode."""

    _max_committed = 30

    def __init__(self, parent=None):
        super().__init__(parent)
        self._style = DEFAULT_STYLE
        self._committed_labels = []  # list of (orig_label, tl_label)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # History scroll area
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._scroll.setStyleSheet(_SCROLL_CSS)

        self._history = QWidget()
        self._history.setStyleSheet("background: transparent;")
        self._history_layout = QVBoxLayout(self._history)
        self._history_layout.setContentsMargins(8, 2, 8, 2)
        self._history_layout.setSpacing(1)
        self._history_layout.addStretch()
        self._scroll.setWidget(self._history)
        layout.addWidget(self._scroll, 1)

        # Current partial area (fixed at bottom)
        partial_widget = QWidget()
        partial_widget.setStyleSheet("background: transparent;")
        partial_layout = QVBoxLayout(partial_widget)
        partial_layout.setContentsMargins(8, 4, 8, 4)
        partial_layout.setSpacing(2)

        self._partial_orig = QLabel("")
        self._partial_orig.setWordWrap(True)
        self._partial_orig.setTextFormat(Qt.TextFormat.RichText)
        self._partial_orig.setStyleSheet("background: transparent;")
        partial_layout.addWidget(self._partial_orig)

        self._partial_tl = QLabel("")
        self._partial_tl.setWordWrap(True)
        self._partial_tl.setTextFormat(Qt.TextFormat.RichText)
        self._partial_tl.setStyleSheet("background: transparent;")
        partial_layout.addWidget(self._partial_tl)

        layout.addWidget(partial_widget)
        self._apply_fonts()

    def _apply_fonts(self):
        s = self._style
        self._partial_orig.setFont(QFont(s["original_font_family"], s["original_font_size"]))
        self._partial_tl.setFont(QFont(s["translation_font_family"], s["translation_font_size"]))

    def set_partial(self, text: str):
        s = self._style
        if text:
            self._partial_orig.setText(
                f'<span style="color:{s["original_color"]};">{_escape(text)}</span>'
            )
        else:
            self._partial_orig.setText("")

    def set_partial_translation(self, text: str):
        s = self._style
        if text:
            self._partial_tl.setText(
                f'<span style="color:{s["translation_color"]};">&gt; {_escape(text)}</span>'
            )
        else:
            self._partial_tl.setText("")

    def commit(self, original: str, translation: str, source_lang: str):
        """Move a finalized pair to history."""
        s = self._style
        header = (
            f'<span style="color:#6cf;">[{_escape(source_lang)}]</span> '
            f'<span style="color:{s["original_color"]};">{_escape(original)}</span>'
        )
        orig_lbl = QLabel(header)
        orig_lbl.setWordWrap(True)
        orig_lbl.setTextFormat(Qt.TextFormat.RichText)
        orig_lbl.setFont(QFont(s["original_font_family"], s["original_font_size"]))
        orig_lbl.setStyleSheet("background: transparent;")
        self._history_layout.addWidget(orig_lbl)

        tl_lbl = None
        if translation:
            tl_lbl = QLabel(f'<span style="color:{s["translation_color"]};">&gt; {_escape(translation)}</span>')
            tl_lbl.setWordWrap(True)
            tl_lbl.setTextFormat(Qt.TextFormat.RichText)
            tl_lbl.setFont(QFont(s["translation_font_family"], s["translation_font_size"]))
            tl_lbl.setStyleSheet("background: transparent;")
            self._history_layout.addWidget(tl_lbl)

        self._committed_labels.append((orig_lbl, tl_lbl))

        # Trim oldest
        while len(self._committed_labels) > self._max_committed:
            old_orig, old_tl = self._committed_labels.pop(0)
            self._history_layout.removeWidget(old_orig)
            old_orig.deleteLater()
            if old_tl:
                self._history_layout.removeWidget(old_tl)
                old_tl.deleteLater()

        # Clear partial
        self._partial_orig.setText("")
        self._partial_tl.setText("")
        QTimer.singleShot(50, self._scroll_to_bottom)

    def update_committed_translation(self, index: int, translation: str):
        """Update the translation of a committed item by index (from end)."""
        if 0 <= index < len(self._committed_labels):
            s = self._style
            _, tl_lbl = self._committed_labels[index]
            if tl_lbl:
                tl_lbl.setText(f'<span style="color:{s["translation_color"]};">&gt; {_escape(translation)}</span>')

    def clear(self):
        for orig_lbl, tl_lbl in self._committed_labels:
            self._history_layout.removeWidget(orig_lbl)
            orig_lbl.deleteLater()
            if tl_lbl:
                self._history_layout.removeWidget(tl_lbl)
                tl_lbl.deleteLater()
        self._committed_labels.clear()
        self._partial_orig.setText("")
        self._partial_tl.setText("")

    def apply_style(self, s: dict):
        self._style = s
        self._apply_fonts()
        # Rebuild history labels
        for orig_lbl, tl_lbl in self._committed_labels:
            orig_lbl.setFont(QFont(s["original_font_family"], s["original_font_size"]))
            if tl_lbl:
                tl_lbl.setFont(QFont(s["translation_font_family"], s["translation_font_size"]))
        # Update partial colors
        if self._partial_orig.text():
            self._partial_orig.setFont(QFont(s["original_font_family"], s["original_font_size"]))
        if self._partial_tl.text():
            self._partial_tl.setFont(QFont(s["translation_font_family"], s["translation_font_size"]))

    def _scroll_to_bottom(self):
        sb = self._scroll.verticalScrollBar()
        sb.setValue(sb.maximum())


_CHECK_CSS = (
    "QCheckBox { color: #888; background: transparent; spacing: 3px; }"
    "QCheckBox::indicator { width: 12px; height: 12px; }"
)


class DragHandle(QWidget):
    """Top bar: row1=title+buttons, row2=checkboxes+combos."""

    settings_clicked = pyqtSignal()
    subtitle_clicked = pyqtSignal()
    click_through_toggled = pyqtSignal(bool)
    topmost_toggled = pyqtSignal(bool)
    auto_scroll_toggled = pyqtSignal(bool)
    taskbar_toggled = pyqtSignal(bool)
    target_language_changed = pyqtSignal(str)
    model_changed = pyqtSignal(int)
    start_clicked = pyqtSignal()
    stop_clicked = pyqtSignal()
    clear_clicked = pyqtSignal()
    hide_clicked = pyqtSignal()
    quit_clicked = pyqtSignal()
    mode_changed = pyqtSignal(str)  # "full" or "compact"
    realtime_toggled = pyqtSignal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._mode = "full"
        self.setFixedHeight(62)
        self.setStyleSheet("background: rgba(60, 60, 80, 200); border-radius: 4px;")

        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 2, 8, 2)
        outer.setSpacing(2)

        # Row 1: drag title + action buttons
        row1 = QHBoxLayout()
        row1.setContentsMargins(0, 0, 0, 0)
        row1.setSpacing(3)

        drag = _DragArea()
        drag.setStyleSheet("background: transparent;")
        drag_layout = QHBoxLayout(drag)
        drag_layout.setContentsMargins(0, 0, 4, 0)
        drag_layout.setSpacing(6)

        title = QLabel("\u2630 LiveTrans")
        title.setFont(QFont("Consolas", 9, QFont.Weight.Bold))
        title.setStyleSheet("color: #aaa; background: transparent;")
        drag_layout.addWidget(title)
        drag_layout.addStretch()
        row1.addWidget(drag, 1)

        def _btn(text, tip=None):
            b = QPushButton(text)
            b.setFixedHeight(20)
            b.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
            b.setFont(QFont("Consolas", 8))
            b.setStyleSheet(_BTN_CSS)
            if tip:
                b.setToolTip(tip)
            return b

        hide_btn = _btn(t("hide"))
        hide_btn.clicked.connect(self.hide_clicked.emit)
        row1.addWidget(hide_btn)

        self._subtitle_btn = _btn(t("subtitle"))
        self._subtitle_btn.clicked.connect(self.subtitle_clicked.emit)
        row1.addWidget(self._subtitle_btn)

        self._rt_btn = _btn(t("btn_realtime"))
        self._rt_btn.clicked.connect(self._on_rt_toggle)
        self._rt_checked = False
        row1.addWidget(self._rt_btn)

        self._running = False
        self._start_stop_btn = _btn(t("paused"))
        self._start_stop_btn.clicked.connect(self._on_start_stop)
        row1.addWidget(self._start_stop_btn)

        self._clear_btn = _btn(t("clear"))
        self._clear_btn.clicked.connect(self.clear_clicked.emit)
        row1.addWidget(self._clear_btn)

        # Mode toggle button
        self._mode_btn = _btn(t("mode_full"))
        self._mode_btn.clicked.connect(self._toggle_mode)
        row1.addWidget(self._mode_btn)

        settings_btn = _btn(t("settings"))
        settings_btn.clicked.connect(self.settings_clicked.emit)
        row1.addWidget(settings_btn)

        quit_btn = _btn(t("quit"))
        quit_btn.setStyleSheet(
            _BTN_CSS.replace("rgba(255,255,255,20)", "rgba(200,60,60,40)").replace(
                "rgba(255,255,255,40)", "rgba(200,60,60,80)"
            )
        )
        quit_btn.clicked.connect(self.quit_clicked.emit)
        row1.addWidget(quit_btn)

        outer.addLayout(row1)

        # Row 2 area: checkboxes (row 2a) + model/lang combos (row 2b)
        self._row2_widget = QWidget()
        self._row2_widget.setStyleSheet("background: transparent;")
        row2_outer = QVBoxLayout(self._row2_widget)
        row2_outer.setContentsMargins(0, 0, 0, 0)
        row2_outer.setSpacing(2)

        # Row 2a: checkboxes
        row2a = QHBoxLayout()
        row2a.setContentsMargins(0, 0, 0, 0)
        row2a.setSpacing(6)

        self._ct_check = QCheckBox(t("click_through"))
        self._ct_check.setFont(QFont("Consolas", 8))
        self._ct_check.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self._ct_check.setStyleSheet(_CHECK_CSS)
        self._ct_check.toggled.connect(self.click_through_toggled.emit)
        row2a.addWidget(self._ct_check)

        self._topmost_check = QCheckBox(t("top_most"))
        self._topmost_check.setFont(QFont("Consolas", 8))
        self._topmost_check.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self._topmost_check.setStyleSheet(_CHECK_CSS)
        self._topmost_check.setChecked(True)
        self._topmost_check.toggled.connect(self.topmost_toggled.emit)
        row2a.addWidget(self._topmost_check)

        self._auto_scroll = QCheckBox(t("auto_scroll"))
        self._auto_scroll.setFont(QFont("Consolas", 8))
        self._auto_scroll.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self._auto_scroll.setStyleSheet(_CHECK_CSS)
        self._auto_scroll.setChecked(True)
        self._auto_scroll.toggled.connect(self.auto_scroll_toggled.emit)
        row2a.addWidget(self._auto_scroll)

        self._taskbar_check = QCheckBox(t("taskbar"))
        self._taskbar_check.setFont(QFont("Consolas", 8))
        self._taskbar_check.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self._taskbar_check.setStyleSheet(_CHECK_CSS)
        self._taskbar_check.setChecked(False)
        self._taskbar_check.toggled.connect(self.taskbar_toggled.emit)
        row2a.addWidget(self._taskbar_check)

        row2a.addStretch()
        row2_outer.addLayout(row2a)

        # Row 2b: model + target language combos
        row2b = QHBoxLayout()
        row2b.setContentsMargins(0, 0, 0, 0)
        row2b.setSpacing(6)

        model_lbl = QLabel(t("model_label"))
        model_lbl.setFont(QFont("Consolas", 8))
        model_lbl.setStyleSheet("color: #888; background: transparent;")
        row2b.addWidget(model_lbl)

        self._model_combo = QComboBox()
        self._model_combo.setFixedHeight(18)
        self._model_combo.setMinimumWidth(140)
        self._model_combo.setFont(QFont("Consolas", 8))
        self._model_combo.setStyleSheet(_COMBO_CSS)
        self._model_combo.currentIndexChanged.connect(self.model_changed.emit)
        row2b.addWidget(self._model_combo)

        row2b.addStretch()

        tgt_lbl = QLabel(t("target_label"))
        tgt_lbl.setFont(QFont("Consolas", 8))
        tgt_lbl.setStyleSheet("color: #888; background: transparent;")
        row2b.addWidget(tgt_lbl)

        self._target_lang = QComboBox()
        self._target_lang.setFixedHeight(18)
        self._target_lang.setMinimumWidth(60)
        self._target_lang.setFont(QFont("Consolas", 8))
        self._target_lang.setStyleSheet(_COMBO_CSS)
        for code, native in LANGUAGES:
            if code == "auto":
                continue
            self._target_lang.addItem(f"{code} - {native}", code)
        self._target_lang.currentIndexChanged.connect(
            lambda idx: self.target_language_changed.emit(
                self._target_lang.currentData() or "zh"
            )
        )
        row2b.addWidget(self._target_lang)

        row2_outer.addLayout(row2b)

        outer.addWidget(self._row2_widget)

    def _on_start_stop(self):
        if self._running:
            self.stop_clicked.emit()
        else:
            self.start_clicked.emit()

    _PAUSED_CSS = _BTN_CSS.replace(
        "rgba(255,255,255,20)", "rgba(220,180,60,50)"
    ).replace("color: #aaa", "color: #ddb")

    def set_target_language(self, lang: str):
        idx = self._target_lang.findData(lang)
        if idx >= 0:
            self._target_lang.setCurrentIndex(idx)

    def set_models(self, models: list, active_index: int = 0):
        self._model_combo.blockSignals(True)
        self._model_combo.clear()
        for m in models:
            self._model_combo.addItem(m.get("name", m.get("model", "?")))
        if 0 <= active_index < self._model_combo.count():
            self._model_combo.setCurrentIndex(active_index)
        self._model_combo.blockSignals(False)

    @property
    def auto_scroll(self) -> bool:
        return self._auto_scroll.isChecked()

    def set_running(self, running: bool):
        self._running = running
        if running:
            self._start_stop_btn.setText(t("running"))
            self._start_stop_btn.setStyleSheet(_BTN_CSS)
        else:
            self._start_stop_btn.setText(t("paused"))
            self._start_stop_btn.setStyleSheet(self._PAUSED_CSS)

    def _toggle_mode(self):
        new_mode = "compact" if self._mode == "full" else "full"
        self._apply_mode(new_mode)
        self.mode_changed.emit(new_mode)

    def _apply_mode(self, mode: str):
        self._mode = mode
        compact = mode == "compact"
        self._row2_widget.setVisible(not compact)
        self._clear_btn.setVisible(not compact)
        self._subtitle_btn.setVisible(not compact)
        self._rt_btn.setVisible(not compact)
        self._mode_btn.setText(t("mode_compact") if compact else t("mode_full"))
        self.setFixedHeight(24 if compact else 62)

    def set_mode(self, mode: str):
        if mode != self._mode:
            self._apply_mode(mode)

    def _on_rt_toggle(self):
        self._rt_checked = not self._rt_checked
        self._update_rt_btn_style()
        self.realtime_toggled.emit(self._rt_checked)

    def _update_rt_btn_style(self):
        if self._rt_checked:
            self._rt_btn.setStyleSheet(
                _BTN_CSS.replace("rgba(255,255,255,20)", "rgba(80,180,80,40)").replace(
                    "rgba(255,255,255,40)", "rgba(80,180,80,80)"
                )
            )
        else:
            self._rt_btn.setStyleSheet(_BTN_CSS)

    def set_realtime_checked(self, checked: bool):
        if self._rt_checked != checked:
            self._rt_checked = checked
            self._update_rt_btn_style()

    def set_subtitle_checked(self, checked: bool):
        self._subtitle_btn.setStyleSheet(
            _BTN_CSS.replace("rgba(255,255,255,20)", "rgba(80,180,80,40)").replace(
                "rgba(255,255,255,40)", "rgba(80,180,80,80)"
            ) if checked else _BTN_CSS
        )


class SubtitleOverlay(QWidget):
    """Chat-style overlay window for displaying live transcription."""

    add_message_signal = pyqtSignal(int, str, str, str, float)
    update_translation_signal = pyqtSignal(int, str, float)
    clear_signal = pyqtSignal()
    # Monitor signals (thread-safe)
    update_monitor_signal = pyqtSignal(float, float, object)
    update_stats_signal = pyqtSignal(int, int, int, int, float)
    update_asr_device_signal = pyqtSignal(str)
    # RT mode signals (thread-safe)
    update_rt_partial_signal = pyqtSignal(str)
    update_rt_partial_tl_signal = pyqtSignal(str)
    commit_rt_signal = pyqtSignal(str, str, str)  # (original, translation, lang)

    settings_requested = pyqtSignal()
    target_language_changed = pyqtSignal(str)
    model_switch_requested = pyqtSignal(int)
    start_requested = pyqtSignal()
    stop_requested = pyqtSignal()
    hide_requested = pyqtSignal()
    quit_requested = pyqtSignal()
    subtitle_toggled = pyqtSignal()
    mode_changed = pyqtSignal(str)  # "full" or "compact"
    realtime_toggled = pyqtSignal(bool)

    def __init__(self, config):
        super().__init__()
        self._config = config
        self._messages = {}
        self._max_messages = 50
        self._click_through = False
        self._realtime_active = False
        self._setup_ui()

        self.add_message_signal.connect(self._on_add_message)
        self.update_translation_signal.connect(self._on_update_translation)
        self.clear_signal.connect(self._on_clear)
        self.update_monitor_signal.connect(self._on_update_monitor)
        self.update_stats_signal.connect(self._on_update_stats)
        self.update_asr_device_signal.connect(self._on_update_asr_device)
        self.update_rt_partial_signal.connect(self._on_update_rt_partial)
        self.update_rt_partial_tl_signal.connect(self._on_update_rt_partial_tl)
        self.commit_rt_signal.connect(self._on_commit_rt)

    def _setup_ui(self):
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setWindowTitle("LiveTrans")
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)

        screen = QApplication.primaryScreen()
        geo = screen.geometry()
        width = 620
        height = 500
        x = geo.width() - width - 20
        y = geo.height() - height - 60
        self.setGeometry(x, y, width, height)
        self.setMinimumSize(480, 280)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        self._container = QWidget()
        self._container.setStyleSheet(
            "background-color: rgba(15, 15, 25, 200); border-radius: 8px;"
        )

        container_layout = QVBoxLayout(self._container)
        container_layout.setContentsMargins(4, 4, 4, 4)
        container_layout.setSpacing(0)

        # Drag handle
        self._handle = DragHandle()
        self._handle.settings_clicked.connect(self.settings_requested.emit)
        self._handle.subtitle_clicked.connect(self.subtitle_toggled.emit)
        self._handle.click_through_toggled.connect(self._set_click_through)
        self._handle.topmost_toggled.connect(self._set_topmost)
        self._handle.taskbar_toggled.connect(self._set_taskbar)
        self._handle.target_language_changed.connect(self.target_language_changed.emit)
        self._handle.model_changed.connect(self.model_switch_requested.emit)
        self._handle.start_clicked.connect(self.start_requested.emit)
        self._handle.stop_clicked.connect(self.stop_requested.emit)
        self._handle.hide_clicked.connect(self.hide_requested.emit)
        self._handle.clear_clicked.connect(self._on_clear)
        self._handle.quit_clicked.connect(self.quit_requested.emit)
        self._handle.mode_changed.connect(self._on_mode_changed)
        self._handle.realtime_toggled.connect(self.realtime_toggled.emit)
        container_layout.addWidget(self._handle)

        # Monitor bar (collapsible)
        self._monitor = MonitorBar()
        container_layout.addWidget(self._monitor)

        # Scroll area
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._scroll.setStyleSheet("""
            QScrollArea { border: none; background: transparent; }
            QScrollBar:vertical {
                width: 6px; background: transparent;
            }
            QScrollBar::handle:vertical {
                background: rgba(255,255,255,60); border-radius: 3px;
                min-height: 20px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0;
            }
        """)

        self._msg_container = QWidget()
        self._msg_container.setStyleSheet("background: transparent;")
        self._msg_layout = QVBoxLayout(self._msg_container)
        self._msg_layout.setContentsMargins(0, 0, 0, 0)
        self._msg_layout.setSpacing(2)
        self._msg_layout.addStretch()

        self._scroll.setWidget(self._msg_container)
        container_layout.addWidget(self._scroll)

        # Real-time view (initially hidden)
        self._rt_view = RealtimeView()
        self._rt_view.setVisible(False)
        container_layout.addWidget(self._rt_view)

        grip_row = QHBoxLayout()
        grip_row.addStretch()
        self._grip = QSizeGrip(self)
        self._grip.setFixedSize(16, 16)
        self._grip.setStyleSheet("background: transparent;")
        grip_row.addWidget(self._grip)
        container_layout.addLayout(grip_row)

        main_layout.addWidget(self._container)

        self._ct_timer = QTimer(self)
        self._ct_timer.timeout.connect(self._check_click_through)
        self._ct_timer.start(50)

    def set_running(self, running: bool):
        self._handle.set_running(running)

    def _set_topmost(self, enabled: bool):
        flags = self.windowFlags()
        if enabled:
            flags |= Qt.WindowType.WindowStaysOnTopHint
        else:
            flags &= ~Qt.WindowType.WindowStaysOnTopHint
        self.setWindowFlags(flags)
        self.show()

    def _set_taskbar(self, enabled: bool):
        flags = self.windowFlags()
        if enabled:
            flags &= ~Qt.WindowType.Tool
        else:
            flags |= Qt.WindowType.Tool
        self.setWindowFlags(flags)
        self.show()

    def _set_click_through(self, enabled: bool):
        self._click_through = enabled
        if not enabled:
            hwnd = int(self.winId())
            style = ctypes.windll.user32.GetWindowLongW(hwnd, _GWL_EXSTYLE)
            if style & _WS_EX_TRANSPARENT:
                ctypes.windll.user32.SetWindowLongW(
                    hwnd, _GWL_EXSTYLE, style & ~_WS_EX_TRANSPARENT
                )

    def _check_click_through(self):
        if not self._click_through:
            return
        cursor = QCursor.pos()
        local = self.mapFromGlobal(cursor)
        hwnd = int(self.winId())
        style = ctypes.windll.user32.GetWindowLongW(hwnd, _GWL_EXSTYLE)

        content_widget = self._rt_view if self._realtime_active else self._scroll
        scroll_top = content_widget.mapTo(self, QPoint(0, 0)).y()
        in_header = 0 <= local.x() <= self.width() and 0 <= local.y() < scroll_top

        if in_header:
            if style & _WS_EX_TRANSPARENT:
                ctypes.windll.user32.SetWindowLongW(
                    hwnd, _GWL_EXSTYLE, style & ~_WS_EX_TRANSPARENT
                )
        else:
            if not (style & _WS_EX_TRANSPARENT):
                ctypes.windll.user32.SetWindowLongW(
                    hwnd, _GWL_EXSTYLE, style | _WS_EX_TRANSPARENT
                )

    def _on_mode_changed(self, mode: str):
        compact = mode == "compact"
        self._monitor.setVisible(not compact)
        ChatMessage._compact_mode = compact
        s = ChatMessage._current_style
        for msg in self._messages.values():
            msg.apply_style(s)
        self.mode_changed.emit(mode)

    def set_mode(self, mode: str):
        self._handle.set_mode(mode)

    def set_subtitle_checked(self, checked: bool):
        self._handle.set_subtitle_checked(checked)

    @pyqtSlot(float, float, object)
    def _on_update_monitor(self, rms: float, vad_conf: float, mic_rms):
        self._monitor.update_audio(rms, vad_conf, mic_rms)

    @pyqtSlot(int, int, int, int, float)
    def _on_update_stats(self, asr_count, tl_count, prompt_tokens, completion_tokens, cost):
        self._monitor.update_pipeline_stats(
            asr_count, tl_count, prompt_tokens, completion_tokens, cost
        )

    @pyqtSlot(str)
    def _on_update_asr_device(self, device: str):
        self._monitor.update_asr_device(device)

    @pyqtSlot(int, str, str, str, float)
    def _on_add_message(self, msg_id, timestamp, original, source_lang, asr_ms):
        msg = ChatMessage(msg_id, timestamp, original, source_lang, asr_ms)
        self._messages[msg_id] = msg
        self._msg_layout.addWidget(msg)

        if len(self._messages) > self._max_messages:
            oldest_id = min(self._messages.keys())
            old_msg = self._messages.pop(oldest_id)
            self._msg_layout.removeWidget(old_msg)
            old_msg.deleteLater()

        QTimer.singleShot(50, self._scroll_to_bottom)

    @pyqtSlot(int, str, float)
    def _on_update_translation(self, msg_id, translated, translate_ms):
        msg = self._messages.get(msg_id)
        if msg:
            msg.set_translation(translated, translate_ms)
            QTimer.singleShot(50, self._scroll_to_bottom)

    @pyqtSlot()
    def _on_clear(self):
        for msg in self._messages.values():
            self._msg_layout.removeWidget(msg)
            msg.deleteLater()
        self._messages.clear()
        self._rt_view.clear()

    # RT mode slots
    @pyqtSlot(str)
    def _on_update_rt_partial(self, text: str):
        self._rt_view.set_partial(text)

    @pyqtSlot(str)
    def _on_update_rt_partial_tl(self, text: str):
        self._rt_view.set_partial_translation(text)

    @pyqtSlot(str, str, str)
    def _on_commit_rt(self, original: str, translation: str, source_lang: str):
        self._rt_view.commit(original, translation, source_lang)

    def _scroll_to_bottom(self):
        if not self._handle.auto_scroll:
            return
        sb = self._scroll.verticalScrollBar()
        sb.setValue(sb.maximum())

    def apply_style(self, style: dict):
        s = {**DEFAULT_STYLE, **style}
        # Migrate old single font_family to split fields
        if "font_family" in s and "original_font_family" not in style:
            s["original_font_family"] = s["font_family"]
            s["translation_font_family"] = s["font_family"]
        # Container background
        bg_rgba = _hex_to_rgba(s["bg_color"], s["bg_opacity"])
        self._container.setStyleSheet(
            f"background-color: {bg_rgba}; border-radius: {s['border_radius']}px;"
        )
        # Header background
        hdr_rgba = _hex_to_rgba(s["header_color"], s["header_opacity"])
        self._handle.setStyleSheet(f"background: {hdr_rgba}; border-radius: 4px;")
        # Window opacity
        self.setWindowOpacity(s["window_opacity"] / 100.0)
        # Update all existing messages
        ChatMessage._current_style = s
        for msg in self._messages.values():
            msg.apply_style(s)
        # Update RT view style
        self._rt_view.apply_style(s)

    def set_realtime_mode(self, enabled: bool):
        """Switch between normal chat view and real-time streaming view."""
        self._realtime_active = enabled
        self._scroll.setVisible(not enabled)
        self._rt_view.setVisible(enabled)
        self._handle.set_realtime_checked(enabled)

    # Thread-safe public API
    def add_message(self, msg_id, timestamp, original, source_lang, asr_ms):
        self.add_message_signal.emit(msg_id, timestamp, original, source_lang, asr_ms)

    def update_translation(self, msg_id, translated, translate_ms):
        self.update_translation_signal.emit(msg_id, translated, translate_ms)

    def update_monitor(self, rms, vad_conf, mic_rms=None):
        self.update_monitor_signal.emit(rms, vad_conf, mic_rms)

    def update_stats(self, asr_count, tl_count, prompt_tokens, completion_tokens, cost=0.0):
        self.update_stats_signal.emit(
            asr_count, tl_count, prompt_tokens, completion_tokens, cost
        )

    def update_asr_device(self, device: str):
        self.update_asr_device_signal.emit(device)

    # RT mode thread-safe API
    def update_rt_partial(self, text: str):
        self.update_rt_partial_signal.emit(text)

    def update_rt_partial_tl(self, text: str):
        self.update_rt_partial_tl_signal.emit(text)

    def commit_rt(self, original: str, translation: str, source_lang: str):
        self.commit_rt_signal.emit(original, translation, source_lang)

    def set_target_language(self, lang: str):
        self._handle.set_target_language(lang)

    def set_models(self, models: list, active_index: int = 0):
        self._handle.set_models(models, active_index)

    def clear(self):
        self.clear_signal.emit()
