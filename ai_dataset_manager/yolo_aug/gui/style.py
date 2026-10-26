STYLE = """
QWidget { color: #1f2937; }
QMainWindow, QStackedWidget > QWidget { background: #f7f8fa; }
#side { background: #111827; }
#logo { color: #ffffff; font-size: 18pt; font-weight: 700; padding: 18px 14px 10px 16px; }
#ver { color: #6b7280; padding: 8px 16px; }
#sideSection { color: #60a5fa; font-weight: 600; font-size: 10pt; padding: 0 16px 10px 16px; }
#steps { background: transparent; border: none; color: #d1d5db; font-size: 10.5pt; outline: 0; }
#steps::item { padding: 11px 14px; border-radius: 6px; margin: 2px 8px; }
#steps::item:selected { background: #2563eb; color: #ffffff; }
#steps::item:hover:!selected { background: #1f2937; }
#pageTitle { font-size: 17pt; font-weight: 700; color: #111827; }
#pageSub { color: #6b7280; font-size: 10pt; }
#sectionTitle { font-weight: 600; font-size: 11pt; }
QGroupBox { background: #ffffff; border: 1px solid #e5e7eb; border-radius: 8px; margin-top: 14px;
            padding: 12px 10px 8px 10px; font-weight: 600; }
QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 4px; color: #374151; }
QGroupBox QLabel, QGroupBox QCheckBox { font-weight: normal; }
QCheckBox#opCheck { font-weight: 600; min-width: 190px; padding: 3px 0; }
#opDesc { color: #4b5563; padding: 3px 0; }
#tag { color: #b45309; font-size: 9pt; padding: 3px 0; }
#note { color: #4b5563; }
#banner { background: #eff6ff; border: 1px solid #bfdbfe; border-radius: 6px; padding: 8px 10px; color: #1e3a8a; }
#summary { background: #ffffff; border: 1px solid #e5e7eb; border-radius: 8px; padding: 10px 12px; }
#big { font-size: 12pt; font-weight: 700; }
#caption { color: #4b5563; font-size: 9pt; }
#imageView { background: #1f2937; color: #9ca3af; border-radius: 6px; }
QPushButton { background: #ffffff; border: 1px solid #d1d5db; border-radius: 6px; padding: 6px 14px; }
QPushButton:hover { background: #f3f4f6; }
QPushButton:disabled { color: #9ca3af; background: #f3f4f6; }
QPushButton#primary { background: #2563eb; color: #ffffff; border: 1px solid #1d4ed8; font-weight: 600; }
QPushButton#primary:hover { background: #1d4ed8; }
QPushButton#primary:disabled { background: #93c5fd; border-color: #93c5fd; }
QLineEdit, QComboBox { background: #ffffff; border: 1px solid #d1d5db;
            border-radius: 5px; padding: 4px 6px; min-height: 20px; }
QLineEdit:focus, QComboBox:focus { border-color: #2563eb; }
QAbstractSpinBox { min-height: 22px; padding: 2px 4px; }
QTextBrowser, QPlainTextEdit, QTableWidget { background: #ffffff; border: 1px solid #e5e7eb; border-radius: 6px; }
#log { font-family: Consolas, 'DejaVu Sans Mono', monospace; font-size: 9pt; }
QProgressBar { border: 1px solid #d1d5db; border-radius: 5px; text-align: center; background: #ffffff; height: 20px; }
QProgressBar::chunk { background: #10b981; border-radius: 4px; }
QScrollArea { background: transparent; }
QScrollArea > QWidget > QWidget { background: transparent; }
QHeaderView::section { background: #f3f4f6; border: none; padding: 4px; }
#mainTabs::pane { border: none; }
#mainTabs > QTabBar::tab { padding: 8px 22px; font-weight: 600; background: #e5e7eb; border: none; margin-right: 2px; }
#mainTabs > QTabBar::tab:selected { background: #111827; color: #ffffff; }
#genaiTabs::pane { border: 1px solid #e5e7eb; background: #f7f8fa; }
#genaiTabs > QTabBar::tab { padding: 6px 16px; }
QStatusBar { background: #ffffff; border-top: 1px solid #e5e7eb; }
"""
