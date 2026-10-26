from __future__ import annotations

import time

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QFormLayout, QGroupBox, QHBoxLayout,
                               QHeaderView, QLabel, QMessageBox, QPushButton, QSplitter, QTableWidget,
                               QTableWidgetItem, QVBoxLayout, QWidget)

from ..assist import core as A
from ..assist import data as D
from .genai_tab import JobThread, RunBar, _dspin, _ispin, _scroll, open_folder

TASK_TEXT = {"detect": "Boxes (Detect / OBB)", "segment": "Polygons (Segment / OBB)",
             "pose": "Keypoints (Pose)", "classify": "Image class (Classify)"}
ENGINE_TEXT = {"rfdetr": "RF-DETR  (Apache-2.0, recommended)",
               "ultralytics": "Ultralytics YOLO26  (AGPL-3.0, internal use only)",
               "examples": "Examples only - OWLv2, no training",
               "vitpose": "ViTPose fine-tune  (Apache-2.0, keypoints on existing boxes)",
               "clip": "CLIP linear probe  (MIT, seconds on CPU)"}
SCOPE_TEXT = {"unlabeled": "Images without any label", "unconfirmed": "Images not fully confirmed",
              "all": "All images (keeps confirmed labels)"}


class AssistPage(QWidget):
    def __init__(self, win, tab):
        super().__init__()
        self.win, self.tab = win, tab
        self._th = None
        root = QHBoxLayout(self)
        split = QSplitter(Qt.Horizontal)
        left = QWidget()
        L = QVBoxLayout(left)
        L.setContentsMargins(0, 0, 6, 0)

        g0 = QGroupBox("Confirmed training data in this project")
        g0l = QVBoxLayout(g0)
        self.lb_data = QLabel("-")
        self.lb_data.setWordWrap(True)
        self.lb_data.setTextFormat(Qt.RichText)
        g0l.addWidget(self.lb_data)
        n = QLabel("Only images whose labels are all confirmed (solid lines) are used. "
                   "Images that still contain dashed AI labels are skipped. Data never leaves this project.")
        n.setWordWrap(True)
        n.setObjectName("note")
        g0l.addWidget(n)
        L.addWidget(g0)

        g1 = QGroupBox("1. Train a project model")
        f = QFormLayout(g1)
        self.cb_task = QComboBox()
        for k, t in TASK_TEXT.items():
            self.cb_task.addItem(t, k)
        self.cb_task.currentIndexChanged.connect(self._task_changed)
        self.cb_engine = QComboBox()
        self.cb_engine.currentIndexChanged.connect(self._engine_changed)
        self.cb_size = QComboBox()
        for k, t in (("nano", "Nano (fast, 8GB OK)"), ("small", "Small"), ("medium", "Medium (RF-DETR only)")):
            self.cb_size.addItem(t, k)
        self.sp_epochs = _ispin(50, 1, 1000, 5)
        self.sp_batch = _ispin(4, 1, 64)
        self.sp_imgsz = _ispin(640, 320, 1600, 32)
        self.sp_val = _dspin(0.15, 0.05, 0.5, 0.05)
        self.sp_ex = _ispin(3, 1, 10)
        self.cb_dev = QComboBox()
        for k, t in (("auto", "auto"), ("cuda", "GPU (cuda)"), ("cpu", "CPU")):
            self.cb_dev.addItem(t, k)
        self.rows = {}
        for key, label, w in (("task", "Assist for", self.cb_task), ("engine", "Engine", self.cb_engine),
                              ("size", "Model size", self.cb_size), ("epochs", "Epochs", self.sp_epochs),
                              ("batch", "Batch", self.sp_batch), ("imgsz", "Image size", self.sp_imgsz),
                              ("val", "Validation ratio", self.sp_val), ("ex", "Examples per class", self.sp_ex),
                              ("dev", "Device", self.cb_dev)):
            lab = QLabel(label)
            f.addRow(lab, w)
            self.rows[key] = (lab, w)
        self.lb_engine = QLabel("")
        self.lb_engine.setWordWrap(True)
        self.lb_engine.setObjectName("note")
        f.addRow(self.lb_engine)
        self.b_train = QPushButton("Train")
        self.b_train.setObjectName("primary")
        self.b_train.clicked.connect(self.train)
        f.addRow(self.b_train)
        L.addWidget(g1)

        g2 = QGroupBox("2. Pre-label with a project model")
        f2 = QFormLayout(g2)
        self.cb_scope = QComboBox()
        for k, t in SCOPE_TEXT.items():
            self.cb_scope.addItem(t, k)
        self.sp_thr = _dspin(0.35, 0.01, 0.99, 0.05)
        self.chk_replace = QCheckBox("Replace previous AI (dashed) labels")
        self.chk_replace.setChecked(True)
        f2.addRow("Target", self.cb_scope)
        f2.addRow("Confidence threshold", self.sp_thr)
        f2.addRow(self.chk_replace)
        row = QHBoxLayout()
        self.b_pred = QPushButton("Run on selected model")
        self.b_pred.setObjectName("primary")
        self.b_pred.clicked.connect(self.predict)
        self.b_review = QPushButton("Review: least confident first")
        self.b_review.clicked.connect(self.review)
        row.addWidget(self.b_pred)
        row.addWidget(self.b_review)
        f2.addRow(row)
        n2 = QLabel("Predictions are added as dashed AI labels. Fix them in Labeling and press confirm; "
                    "then train again - each round needs fewer corrections.")
        n2.setWordWrap(True)
        n2.setObjectName("note")
        f2.addRow(n2)
        L.addWidget(g2)
        L.addStretch(1)
        split.addWidget(_scroll(left))

        right = QWidget()
        R = QVBoxLayout(right)
        R.setContentsMargins(6, 0, 0, 0)
        g3 = QGroupBox("Project models")
        g3l = QVBoxLayout(g3)
        self.tbl = QTableWidget(0, 6)
        self.tbl.setHorizontalHeaderLabels(["Task", "Engine", "Images", "Metric", "Created", "Time"])
        self.tbl.verticalHeader().setVisible(False)
        self.tbl.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.tbl.setSelectionMode(QAbstractItemView.SingleSelection)
        self.tbl.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.tbl.horizontalHeader().setSectionResizeMode(4, QHeaderView.Stretch)
        g3l.addWidget(self.tbl)
        r2 = QHBoxLayout()
        for t, fn in (("Refresh", self.refresh), ("Open folder", self._open), ("Delete", self._delete)):
            b = QPushButton(t)
            b.clicked.connect(fn)
            r2.addWidget(b)
        r2.addStretch(1)
        g3l.addLayout(r2)
        R.addWidget(g3, 2)
        self.run = RunBar("")
        self.run.btn_run.hide()
        self.run.btn_cancel.clicked.connect(lambda: self._th and self._th.cancel.set())
        self.run.btn_open.clicked.connect(self._open)
        R.addWidget(self.run, 3)
        split.addWidget(right)
        split.setSizes([480, 700])
        root.addWidget(split)
        tab.project_changed.connect(self.refresh)
        self._task_changed()

    def _task_changed(self):
        task = self.cb_task.currentData()
        cur = self.cb_engine.currentData()
        self.cb_engine.blockSignals(True)
        self.cb_engine.clear()
        for e in A.engines_for(task):
            self.cb_engine.addItem(ENGINE_TEXT[e], e)
        i = self.cb_engine.findData(cur)
        self.cb_engine.setCurrentIndex(max(0, i))
        self.cb_engine.blockSignals(False)
        self._engine_changed()

    def _engine_changed(self):
        e = self.cb_engine.currentData()
        vis = {"size": e in ("rfdetr", "ultralytics"), "epochs": e in ("rfdetr", "ultralytics", "vitpose"),
               "batch": e in ("rfdetr", "ultralytics", "vitpose"), "imgsz": e == "ultralytics",
               "val": e in ("rfdetr", "ultralytics", "vitpose", "clip"), "ex": e == "examples",
               "dev": e != "examples"}
        for k, on in vis.items():
            for w in self.rows[k]:
                w.setVisible(on)
        notes = {
            "rfdetr": "Fine-tunes RF-DETR on your confirmed labels. Needs base weights (Models / GPU: RF-DETR). "
                      "About 20+ confirmed images per class recommended. Batch 2-4 fits 8GB GPUs.",
            "ultralytics": "Optional engine. AGPL-3.0: fine for internal labeling, do not ship it in a delivered "
                           "product. Needs 'pip install -r requirements-ultralytics.txt'.",
            "examples": "No training: uses your labeled objects as visual examples (OWLv2 image-guided search). "
                        "Useful with only a few labels.",
            "vitpose": "Trains keypoints for your project schema (any number of points). Predicts keypoints for "
                       "pose-class boxes that do not have keypoints yet - run box assist first for new images.",
            "clip": "Learns your image classes from confirmed image-class labels."}
        self.lb_engine.setText(notes.get(e, ""))
        if e == "vitpose":
            self.sp_epochs.setValue(max(self.sp_epochs.value(), 30))

    def showEvent(self, e):
        super().showEvent(e)
        self.refresh()

    def refresh(self):
        p = self.tab.project
        self.tbl.setRowCount(0)
        if p is None:
            self.lb_data.setText("-")
            return
        s = D.summary(p)

        def fmt(d):
            pc = ", ".join(f"{k} {v}" for k, v in sorted(d.get("per_class", {}).items()))
            return f"{d['images']} images" + (f" · {pc}" if pc else "")

        self.lb_data.setText(
            f"<b>Boxes / polygons</b>: {fmt(s['detect'])} (polygons {s['segment'].get('polygons', 0)})<br>"
            f"<b>Keypoints</b>: {fmt(s['pose'])}<br>"
            f"<b>Image classes</b>: {fmt(s['classify'])}<br>"
            f"Unlabeled images: {s['unlabeled']} / {s['total']}")
        models = A.list_models(p)
        self.tbl.setRowCount(len(models))
        for r, m in enumerate(models):
            met = m.get("metrics", {})
            if "val_pck" in met:
                mt = f"PCK {met['val_pck']}"
            elif "val_acc" in met:
                mt = f"acc {met['val_acc']}"
            elif "exemplars" in met:
                mt = f"{met['exemplars']} examples"
            else:
                mt = f"train {met.get('n_train', '-')} / val {met.get('n_val', '-')}"
            vals = [m["task"], m["engine"], str(m.get("n_images", "")), mt, m.get("created", ""),
                    f"{m.get('seconds', 0):.0f}s"]
            for c, v in enumerate(vals):
                it = QTableWidgetItem(v)
                it.setData(Qt.UserRole, m)
                self.tbl.setItem(r, c, it)
        if models:
            self.tbl.selectRow(0)

    def _selected(self):
        r = self.tbl.currentRow()
        if r < 0 or self.tbl.item(r, 0) is None:
            return None
        return self.tbl.item(r, 0).data(Qt.UserRole)

    def _open(self):
        m = self._selected()
        if m:
            open_folder(m["dir"])
        elif self.tab.project is not None:
            open_folder(A.assist_root(self.tab.project))

    def _delete(self):
        m = self._selected()
        if not m:
            return
        if QMessageBox.question(self, "Delete", f"Delete model {m['name']}?") == QMessageBox.Yes:
            A.delete_model(m["dir"])
            self.refresh()
            self.tab.label._refresh_assist_models()

    def _busy(self, on: bool):
        self.b_train.setEnabled(not on)
        self.b_pred.setEnabled(not on)
        self.run.running(on)

    def train(self):
        p = self.tab.project
        if p is None:
            return
        self.tab.label.save_current()
        cfg = A.AssistConfig(task=self.cb_task.currentData(), engine=self.cb_engine.currentData(),
                             size=self.cb_size.currentData(), epochs=self.sp_epochs.value(),
                             batch=self.sp_batch.value(), imgsz=self.sp_imgsz.value(), val_ratio=self.sp_val.value(),
                             device=self.cb_dev.currentData(), models_dir=self.tab.models.models_dir(),
                             exemplars_per_class=self.sp_ex.value())
        if cfg.engine == "ultralytics":
            from ..assist.engines import ultralytics_available
            ok, why = ultralytics_available()
            if not ok:
                QMessageBox.warning(self, "Ultralytics", why)
                return
        if cfg.engine == "rfdetr":
            from ..assist.engines import rfdetr_available
            ok, why = rfdetr_available()
            if not ok:
                QMessageBox.warning(self, "RF-DETR", why)
                return
        self.run.log.clear()
        self._busy(True)

        def job(progress=None, log=print, cancel=None):
            return A.train_model(p, cfg, progress, log, cancel)

        self._th = JobThread(job)
        self._th.progress.connect(lambda d, t, m: (self.run.bar.setMaximum(max(t, 1)), self.run.bar.setValue(d)))
        self._th.log.connect(self.run.log.appendPlainText)
        self._th.done.connect(self._trained)
        self._th.failed.connect(lambda m: (self._busy(False), self.run.log.appendPlainText(m)))
        self._th.start()

    def _trained(self, meta):
        self._busy(False)
        self.refresh()
        self.tab.label._refresh_assist_models()
        for r in range(self.tbl.rowCount()):
            if self.tbl.item(r, 0).data(Qt.UserRole)["name"] == meta["name"]:
                self.tbl.selectRow(r)

    def predict(self):
        p = self.tab.project
        m = self._selected()
        if p is None or not m:
            QMessageBox.information(self, "AI assist", "Train or select a project model first.")
            return
        self.tab.label.save_current()
        imgs = A.target_images(p, self.cb_scope.currentData())
        if not imgs:
            QMessageBox.information(self, "AI assist", "No target images.")
            return
        thr, rep = self.sp_thr.value(), self.chk_replace.isChecked()
        md, dev = self.tab.models.models_dir(), self.cb_dev.currentData()
        self.run.log.clear()
        self._busy(True)
        t0 = time.time()

        def job(progress=None, log=print, cancel=None):
            return A.predict_images(p, m, imgs, thr, rep, md, dev, progress, log, cancel)

        self._th = JobThread(job)
        self._th.progress.connect(lambda d, t, msg: (self.run.bar.setMaximum(max(t, 1)), self.run.bar.setValue(d)))
        self._th.log.connect(self.run.log.appendPlainText)
        self._th.done.connect(lambda r: (self._busy(False), self.run.log.appendPlainText(
            f"done in {time.time() - t0:.0f}s"), self.tab.set_project(p)))
        self._th.failed.connect(lambda msg: (self._busy(False), self.run.log.appendPlainText(msg)))
        self._th.start()

    def review(self):
        lp = self.tab.label
        lp.cb_filter.setCurrentIndex(5)
        lp.cur = None
        lp.refresh()
        self.tab.setCurrentWidget(lp)
