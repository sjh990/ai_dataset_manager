"""시설물 라벨링 툴 실행 파일.  사용법: python main.py"""
import sys

from PySide6.QtWidgets import QApplication

from labeler import config as C
from labeler.ui.common import STYLE
from labeler.ui.main_window import MainWindow


def main():
    C.move_old_datasets_folder()       # 예전 projects 폴더 → datasets
    app = QApplication(sys.argv)
    app.setStyleSheet(STYLE)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
