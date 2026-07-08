STYLE_SHEET = """
QMainWindow {
    background-color: #0b1220;
}
QWidget#centralWidget {
    background-color: #0b1220;
}
QLabel {
    color: #94a3b8;
    font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
    font-size: 13px;
    font-weight: 500;
}
QLabel#appTitle {
    color: #f8fafc;
    font-size: 20px;
    font-weight: 700;
    letter-spacing: 0.3px;
}
QLabel#appSubtitle {
    color: #64748b;
    font-size: 12px;
    font-weight: 400;
}
QLabel#fieldLabel {
    color: #cbd5e1;
    font-size: 12px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.4px;
}
QGroupBox {
    background-color: #131c2e;
    border: 1px solid #1e293b;
    border-radius: 10px;
    margin-top: 14px;
    padding: 18px 16px 16px 16px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 14px;
    top: -4px;
    padding: 0 8px;
    color: #38bdf8;
    font-size: 13px;
    font-weight: 700;
    background-color: #0b1220;
}
QLineEdit {
    background-color: #0d1527;
    border: 1.5px solid #263348;
    border-radius: 6px;
    color: #f8fafc;
    padding: 9px 12px;
    font-size: 13px;
    font-family: 'Segoe UI', sans-serif;
    selection-background-color: #2563eb;
}
QLineEdit:hover {
    border: 1.5px solid #334155;
}
QLineEdit:focus {
    border: 1.5px solid #3b82f6;
    background-color: #0a1220;
}
QLineEdit:disabled {
    color: #475569;
    background-color: #0a0f1a;
}
QComboBox {
    background-color: #0d1527;
    border: 1.5px solid #263348;
    border-radius: 6px;
    color: #f8fafc;
    padding: 9px 12px;
    font-size: 13px;
    font-family: 'Segoe UI', sans-serif;
}
QComboBox:hover {
    border: 1.5px solid #334155;
}
QComboBox:focus {
    border: 1.5px solid #3b82f6;
}
QComboBox::drop-down {
    border: none;
    width: 28px;
}
QComboBox QAbstractItemView {
    background-color: #131c2e;
    border: 1px solid #263348;
    border-radius: 6px;
    selection-background-color: #2563eb;
    selection-color: #ffffff;
    color: #f8fafc;
    padding: 4px;
}
QPushButton {
    background-color: #1e293b;
    border: 1px solid #334155;
    border-radius: 6px;
    color: #e2e8f0;
    padding: 9px 18px;
    font-size: 13px;
    font-weight: 600;
    font-family: 'Segoe UI', sans-serif;
}
QPushButton:hover {
    background-color: #29374d;
    border: 1px solid #3b4a63;
}
QPushButton:pressed {
    background-color: #1a2333;
}
QPushButton#runButton {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #4f46e5, stop:1 #0891b2);
    border: none;
    color: #ffffff;
    font-size: 14px;
    font-weight: 700;
    padding: 11px 26px;
    border-radius: 7px;
}
QPushButton#runButton:hover {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #4338ca, stop:1 #0e7490);
}
QPushButton#runButton:pressed {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #3730a3, stop:1 #155e75);
}
QPushButton#runButton:disabled {
    background: #1e293b;
    color: #475569;
}
QPushButton#browseButton {
    background-color: #182131;
    border: 1px solid #263348;
    font-weight: 500;
    padding: 9px 14px;
}
QPushButton#browseButton:hover {
    background-color: #202b40;
    border: 1px solid #334155;
}
QPushButton#browseButton:pressed {
    background-color: #131c2e;
}
QFrame#divider {
    background-color: #1e293b;
    max-height: 1px;
    min-height: 1px;
    border: none;
}
QFrame#statusCard {
    background-color: #131c2e;
    border: 1px solid #1e293b;
    border-radius: 8px;
}
QTableWidget {
    background-color: #0d1527;
    border: 1px solid #1e293b;
    border-radius: 8px;
    gridline-color: #1c2941;
    color: #cbd5e1;
    font-size: 12px;
    font-family: 'Segoe UI', sans-serif;
    alternate-background-color: #111b2d;
}
QTableWidget::item {
    border-bottom: 1px solid #1c2941;
    padding: 7px;
}
QTableWidget::item:selected {
    background-color: #1e3a8a;
    color: #f8fafc;
}
QHeaderView::section {
    background-color: #0f172a;
    color: #38bdf8;
    padding: 10px 8px;
    font-weight: 700;
    font-size: 11.5px;
    letter-spacing: 0.3px;
    border: none;
    border-bottom: 2px solid #263348;
    border-right: 1px solid #1c2941;
}
QScrollBar:vertical {
    background: #0d1527;
    width: 11px;
    margin: 0;
    border-radius: 5px;
}
QScrollBar::handle:vertical {
    background: #334155;
    border-radius: 5px;
    min-height: 24px;
}
QScrollBar::handle:vertical:hover {
    background: #475569;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0;
}
QScrollBar:horizontal {
    background: #0d1527;
    height: 11px;
    margin: 0;
    border-radius: 5px;
}
QScrollBar::handle:horizontal {
    background: #334155;
    border-radius: 5px;
    min-width: 24px;
}
QScrollBar::handle:horizontal:hover {
    background: #475569;
}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
    width: 0;
}
"""
