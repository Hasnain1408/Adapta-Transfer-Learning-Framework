from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QSplitter,
    QTabWidget,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from presentation.qt.learning_materials_content import (
    CLASSIFIERS_CONTENT,
    METRICS_CONTENT,
    PREPROCESSING_CONTENT,
    TL_METHODS_CONTENT,
)
from presentation.qt.style import build_style_sheet
from presentation.qt.title_bar import install_title_bar


class LearningMaterialsDialog(QDialog):
    """Dialog surfacing comprehensive learning materials for Transfer Learning methods,
    classifiers, evaluation metrics, and feature preprocessing techniques.
    """

    def __init__(self, palette: dict[str, str], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._palette = palette
        self.setWindowTitle("Learning Materials & Domain Adaptation Guide")
        self.resize(1024, 680)
        self.setMinimumSize(850, 550)
        self.setStyleSheet(build_style_sheet(palette))

        self._build_ui()

    def set_palette(self, palette: dict[str, str]) -> None:
        self._palette = palette
        self.setStyleSheet(build_style_sheet(palette))
        self._update_all_content()

    def select_topic(self, category_index: int, item_index: int = 0) -> None:
        """Utility to programmatically open a specific tab and topic."""
        if 0 <= category_index < self._tabs.count():
            self._tabs.setCurrentIndex(category_index)
            list_widget = self._category_lists[category_index]
            if 0 <= item_index < list_widget.count():
                list_widget.setCurrentRow(item_index)

    def _build_ui(self) -> None:
        body = QWidget()
        layout = QVBoxLayout(body)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(12)

        # Header
        header = QWidget()
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(12)

        title_box = QVBoxLayout()
        title_box.setSpacing(2)
        title = QLabel("📖 Learning Materials")
        title.setObjectName("appTitle")
        subtitle = QLabel("Comprehensive reference for transfer learning methods, classifiers, metrics, and data scaling")
        subtitle.setObjectName("appSubtitle")
        title_box.addWidget(title)
        title_box.addWidget(subtitle)

        header_layout.addLayout(title_box)
        header_layout.addStretch(1)

        layout.addWidget(header)

        # Main Tab Widget
        self._tabs = QTabWidget()
        self._tabs.setDocumentMode(True)

        self._category_lists: list[QListWidget] = []
        self._category_browsers: list[QTextBrowser] = []

        categories = [
            ("Transfer Learning Methods", TL_METHODS_CONTENT),
            ("Classifiers", CLASSIFIERS_CONTENT),
            ("Evaluation Metrics", METRICS_CONTENT),
            ("Normalization & Scaling", PREPROCESSING_CONTENT),
        ]

        for tab_name, items in categories:
            tab_widget, list_widget, browser = self._build_category_tab(items)
            self._tabs.addTab(tab_widget, tab_name)
            self._category_lists.append(list_widget)
            self._category_browsers.append(browser)

        layout.addWidget(self._tabs, 1)

        install_title_bar(self, "Learning Materials & Domain Adaptation Guide", body)

    def _build_category_tab(
        self, items: list[tuple[str, str]]
    ) -> tuple[QWidget, QListWidget, QTextBrowser]:
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(12)

        splitter = QSplitter(Qt.Horizontal)

        # Left panel: Sidebar List
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(8)

        search_input = QLineEdit()
        search_input.setPlaceholderText("Filter topics…")
        search_input.setClearButtonEnabled(True)

        list_widget = QListWidget()
        list_widget.setFixedWidth(240)

        for title, _ in items:
            list_widget.addItem(title)

        left_layout.addWidget(search_input)
        left_layout.addWidget(list_widget, 1)

        # Right panel: Content Browser
        browser = QTextBrowser()
        browser.setOpenExternalLinks(True)

        splitter.addWidget(left_widget)
        splitter.addWidget(browser)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        layout.addWidget(splitter)

        # Connect signals
        def on_selection_changed():
            row = list_widget.currentRow()
            if 0 <= row < len(items):
                html = self._render_article_html(items[row][0], items[row][1])
                browser.setHtml(html)

        list_widget.currentRowChanged.connect(on_selection_changed)

        def filter_items(text: str):
            text_lower = text.lower().strip()
            for i in range(list_widget.count()):
                item = list_widget.item(i)
                item.setHidden(text_lower not in item.text().lower())

        search_input.textChanged.connect(filter_items)

        if items:
            list_widget.setCurrentRow(0)

        return container, list_widget, browser

    def _update_all_content(self) -> None:
        categories_data = [
            TL_METHODS_CONTENT,
            CLASSIFIERS_CONTENT,
            METRICS_CONTENT,
            PREPROCESSING_CONTENT,
        ]
        for idx, items in enumerate(categories_data):
            list_widget = self._category_lists[idx]
            browser = self._category_browsers[idx]
            row = list_widget.currentRow()
            if 0 <= row < len(items):
                html = self._render_article_html(items[row][0], items[row][1])
                browser.setHtml(html)

    def _render_article_html(self, title: str, body_content: str) -> str:
        p = self._palette
        return f"""
        <html>
        <head>
        <style>
            body {{
                font-family: 'Segoe UI', 'Inter', Helvetica, Arial, sans-serif;
                background-color: {p['surface']};
                color: {p['text']};
                margin: 16px;
                line-height: 1.6;
                font-size: 13px;
            }}
            h1 {{
                color: {p['accent']};
                font-size: 20px;
                margin-top: 0;
                margin-bottom: 8px;
                border-bottom: 2px solid {p['border']};
                padding-bottom: 6px;
            }}
            h2 {{
                color: {p['text']};
                font-size: 15px;
                margin-top: 18px;
                margin-bottom: 6px;
            }}
            h3 {{
                color: {p['text_secondary']};
                font-size: 13px;
                margin-top: 12px;
                margin-bottom: 4px;
            }}
            p {{
                margin-top: 4px;
                margin-bottom: 12px;
                color: {p['text']};
            }}
            .badge {{
                display: inline-block;
                background-color: {p['accent_soft']};
                color: {p['accent']};
                font-weight: bold;
                font-size: 11px;
                padding: 3px 8px;
                border-radius: 4px;
                margin-bottom: 12px;
            }}
            .card {{
                background-color: {p['surface_alt']};
                border: 1px solid {p['border_strong']};
                border-radius: 8px;
                padding: 12px 14px;
                margin: 10px 0;
            }}
            .formula {{
                font-family: 'Consolas', 'Courier New', monospace;
                background-color: {p['surface_sunken']};
                color: {p['metric_teal']};
                padding: 8px 12px;
                border-left: 3px solid {p['accent']};
                border-radius: 4px;
                font-size: 13px;
                margin: 8px 0;
            }}
            ul {{
                margin-top: 4px;
                margin-bottom: 12px;
                padding-left: 20px;
            }}
            li {{
                margin-bottom: 4px;
            }}
            strong {{
                color: {p['text']};
            }}
        </style>
        </head>
        <body>
            <h1>{title}</h1>
            {body_content}
        </body>
        </html>
        """

