from __future__ import annotations


from PySide6.QtWidgets import (QCheckBox, QComboBox, QFileDialog, QGridLayout, QGroupBox, QHBoxLayout, QLabel,
                               QLineEdit, QMessageBox, QPushButton)

from ..genai.video import VIDEO_EXTS, FrameConfig, estimate_count, extract_many, list_videos, probe
from .genai_tab import JobThread, _dspin, _ispin
from .widgets import FuncThread


class VideoBox(QGroupBox):
    def __init__(self, page):
        super().__init__("Extract frames from video  (video -> images into this project)")
        self.page = page
        self._th = None
        self._pth = None
        self._infos: list = []
        g = QGridLayout(self)
        self.ed_src = QLineEdit()
        self.ed_src.setPlaceholderText("Video file or folder (mp4, mov, avi, mkv, ts ...; subfolders included)")
        self.ed_src.editingFinished.connect(self.scan)
        b_file = QPushButton("File...")
        b_file.clicked.connect(self._pick_file)
        b_dir = QPushButton("Folder...")
        b_dir.clicked.connect(self._pick_dir)
        row = QHBoxLayout()
        row.addWidget(self.ed_src, 1)
        row.addWidget(b_file)
        row.addWidget(b_dir)
        g.addLayout(row, 0, 0, 1, 8)

        self.cb_mode = QComboBox()
        self.cb_mode.addItem("Every N seconds", "seconds")
        self.cb_mode.addItem("Every N frames", "frames")
        self.cb_mode.currentIndexChanged.connect(self._mode_changed)
        self.sp_int = _dspin(1.0, 0.04, 3600.0, 0.5, 2)
        self.sp_start = _dspin(0.0, 0.0, 86400.0, 1.0, 1)
        self.sp_end = _dspin(0.0, 0.0, 86400.0, 1.0, 1)
        self.sp_end.setSpecialValueText("end")
        self.sp_max = _ispin(0, 0, 100000, 10)
        self.sp_max.setSpecialValueText("no limit")
        self.sp_long = _ispin(0, 0, 8192, 64)
        self.sp_long.setSpecialValueText("original")
        self.sp_sim = _dspin(3.0, 0.0, 50.0, 0.5, 1)
        self.sp_sim.setSpecialValueText("off")
        self.sp_sim.setToolTip("Skip a frame when its mean pixel difference (0-255, 64x64 gray) to the last saved "
                               "frame is below this value. Removes near-duplicates from hovering drones.")
        self.sp_blur = _dspin(0.0, 0.0, 2000.0, 10.0, 0)
        self.sp_blur.setSpecialValueText("off")
        self.sp_blur.setToolTip("Skip frames whose sharpness (Laplacian variance) is below this value. "
                                "Typical blurry frames: < 50~100.")
        self.cb_fmt = QComboBox()
        self.cb_fmt.addItem("jpg", "jpg")
        self.cb_fmt.addItem("png", "png")
        self.chk_prefix = QCheckBox("Use video name as prefix")
        self.chk_prefix.setChecked(True)
        for w in (self.cb_mode, self.sp_int, self.sp_start, self.sp_end, self.sp_max, self.sp_sim, self.sp_blur):
            sig = getattr(w, "valueChanged", None) or w.currentIndexChanged
            sig.connect(self._estimate)
        items = [("Sampling", self.cb_mode), ("Interval", self.sp_int), ("Start (s)", self.sp_start),
                 ("End (s)", self.sp_end), ("Max / video", self.sp_max), ("Resize long side", self.sp_long),
                 ("Skip similar", self.sp_sim), ("Skip blur", self.sp_blur), ("Format", self.cb_fmt)]
        for i, (t, w) in enumerate(items):
            r, c = 1 + i // 4, (i % 4) * 2
            g.addWidget(QLabel(t), r, c)
            g.addWidget(w, r, c + 1)
        g.addWidget(self.chk_prefix, 3, 2, 1, 2)
        self.lb = QLabel("")
        self.lb.setObjectName("note")
        self.lb.setWordWrap(True)
        g.addWidget(self.lb, 4, 0, 1, 6)
        self.b_run = QPushButton("Extract to project")
        self.b_run.setObjectName("primary")
        self.b_run.clicked.connect(self.start)
        g.addWidget(self.b_run, 4, 6, 1, 2)

    def _pick_file(self):
        flt = "Video (" + " ".join(f"*{e}" for e in sorted(VIDEO_EXTS)) + ")"
        f, _ = QFileDialog.getOpenFileName(self, "Video file", self.ed_src.text(), flt)
        if f:
            self.ed_src.setText(f)
            self.scan()

    def _pick_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Video folder", self.ed_src.text())
        if d:
            self.ed_src.setText(d)
            self.scan()

    def _mode_changed(self):
        sec = self.cb_mode.currentData() == "seconds"
        self.sp_int.setDecimals(2 if sec else 0)
        self.sp_int.setMinimum(0.04 if sec else 1)
        self.sp_int.setSingleStep(0.5 if sec else 1)
        if not sec and self.sp_int.value() < 1:
            self.sp_int.setValue(1)

    def cfg(self) -> FrameConfig:
        return FrameConfig(mode=self.cb_mode.currentData(), interval=self.sp_int.value(),
                           start_sec=self.sp_start.value(), end_sec=self.sp_end.value(),
                           max_frames=self.sp_max.value(), long_side=self.sp_long.value(),
                           skip_similar=self.sp_sim.value(), skip_blur=self.sp_blur.value(),
                           image_format=self.cb_fmt.currentData(), prefix="")

    def scan(self):
        src = self.ed_src.text().strip()
        vids = list_videos(src) if src else []
        self._infos = []
        if not vids:
            self.lb.setText("No video found." if src else "")
            return
        self.lb.setText(f"Reading {len(vids)} video(s)...")

        def job():
            return [(v, probe(v)) for v in vids[:500]]

        self._pth = FuncThread(job)
        self._pth.done.connect(self._scanned)
        self._pth.failed.connect(lambda m: self.lb.setText(m.splitlines()[0]))
        self._pth.start()

    def _scanned(self, infos):
        self._infos = infos
        self._estimate()

    def _estimate(self):
        if not self._infos:
            return
        cfg = self.cfg()
        ok = [(v, i) for v, i in self._infos if i["ok"]]
        bad = len(self._infos) - len(ok)
        secs = sum(i["seconds"] for _, i in ok)
        est = sum(estimate_count(i, cfg) for _, i in ok)
        txt = f"{len(ok)} video(s), total {secs / 60:.1f} min"
        if len(ok) == 1:
            i = ok[0][1]
            txt += f" ({i['width']}x{i['height']}, {i['fps']:.2f} fps, {i['frames']} frames)"
        txt += f" -> about {est} images before skip filters"
        if bad:
            txt += f" · {bad} file(s) cannot be opened"
        self.lb.setText(txt)

    def start(self):
        page = self.page
        p = page.tab.project
        if p is None:
            QMessageBox.information(self, "Project", "Create or open a project first.")
            return
        vids = list_videos(self.ed_src.text().strip())
        if not vids:
            QMessageBox.warning(self, "Video", "Select a video file or a folder with videos.")
            return
        page.save()
        cfg = self.cfg()
        use_prefix = self.chk_prefix.isChecked()
        out = p.work_images
        self.b_run.setEnabled(False)
        page.run.log.clear()
        page.run.running(True)

        def job(progress=None, log=print, cancel=None):
            if not use_prefix:
                cfg.prefix = "frame"
            rep = extract_many(vids, out, cfg, progress, log, cancel)
            srcs = set(p.data.get("sources", []))
            srcs |= {str(v) for v in vids}
            p.data["sources"] = sorted(srcs)
            p.save()
            return rep

        self._th = JobThread(job)
        page._th = self._th
        self._th.progress.connect(lambda d, t, m: (page.run.bar.setMaximum(t), page.run.bar.setValue(d)))
        self._th.log.connect(page.run.log.appendPlainText)
        self._th.done.connect(lambda r: self._done(r, p))
        self._th.failed.connect(lambda m: (self.b_run.setEnabled(True), page.run.running(False),
                                           page.run.log.appendPlainText(m)))
        self._th.start()

    def _done(self, rep, p):
        self.b_run.setEnabled(True)
        self.page.run.running(False)
        self.page._stats()
        self.page.tab.set_project(p)
        self.lb.setText(f"Saved {rep['saved']} images from {rep['videos']} video(s) into the project. "
                        "Label them in step 3.")
