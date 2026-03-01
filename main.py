# TTS-Only Bot – Full Featured GUI (Dark Theme)
# Whisper / Vosk / Silero STT, pyttsx3/espeak/sam TTS, SoX FX, config, export, bypass

import sys, os, json, subprocess, tempfile
import pyttsx3, sounddevice as sd
from PyQt5 import QtCore, QtGui, QtWidgets
from speech_thread import SpeechThread, speak_once

CONFIG_FILE = 'tts_bot_config.json'

# Optional dark theme
try:
    import qdarktheme
    qdarktheme.setup_theme('dark')
except:
    pass


# GUI ---------------------------------------------------------------------------
class App(QtWidgets.QWidget):
    def __init__(self):
        super().__init__(); self.setWindowTitle('TTS-Only Bot'); self.resize(800,600)
        self.cfg=self._load_config(); self.thread=None
        self._build_ui(); self._populate_model_paths(); self._populate_voices(); self._connect_signals()
        self._refresh_devices()
        self.loop_chk.setChecked(self.cfg.get('listen_self', False))
        self.stt_cb.setCurrentText(self.cfg.get('stt_engine', 'Whisper'))
        if self.cfg.get('bypass'):
            self.bypass_btn.setText('Bypass ON')

    def _load_config(self):
        if os.path.exists(CONFIG_FILE):
            try: return json.load(open(CONFIG_FILE))
            except: pass
        return {
            'in_dev':None,'out_dev':None,'listen_self':False,
            'stt_engine':'Whisper','model_path':'','stt_gain':1.0,
            'tts_engine':'pyttsx3','tts_voice':'','tts_vol':100,
            'words_chunk':5,'chunk_ms':500,'pitch':0,'tempo':1,'filter':'none',
            'bypass':False,'typing_only':False
        }

    def _save_config(self):
        json.dump(self.cfg, open(CONFIG_FILE,'w'))

    def _build_ui(self):
        g=QtWidgets.QGridLayout(self)
        r=0
        # I/O
        g.addWidget(QtWidgets.QLabel('Input device:'),r,0); self.in_cb=QtWidgets.QComboBox(); g.addWidget(self.in_cb,r,1,1,3); r+=1
        g.addWidget(QtWidgets.QLabel('Output device:'),r,0); self.out_cb=QtWidgets.QComboBox(); g.addWidget(self.out_cb,r,1,1,3); r+=1
        self.loop_chk=QtWidgets.QCheckBox('Listen to self (loopback)'); g.addWidget(self.loop_chk,r,0,1,2); r+=1
        # STT/TTS
        g.addWidget(QtWidgets.QLabel('STT Engine:'),r,0); self.stt_cb=QtWidgets.QComboBox();
        self.stt_cb.addItems(['Whisper','Fast Whisper','Vosk','Silero']);
        g.addWidget(self.stt_cb,r,1);
        g.addWidget(QtWidgets.QLabel('Model path:'),r,2); self.model_cb=QtWidgets.QComboBox(); self.model_cb.setEditable(True); g.addWidget(self.model_cb,r,3); r+=1
        g.addWidget(QtWidgets.QLabel('TTS Engine:'),r,0);
        self.tts_cb=QtWidgets.QComboBox();
        self.tts_cb.addItems(['pyttsx3','espeak','sam']);
        self.tts_cb.setCurrentText(self.cfg['tts_engine']);
        g.addWidget(self.tts_cb,r,1);
        g.addWidget(QtWidgets.QLabel('TTS Voice:'),r,2); self.voice_cb=QtWidgets.QComboBox(); g.addWidget(self.voice_cb,r,3); r+=1
        self.typing_chk = QtWidgets.QCheckBox('Typing only (TTS only)');
        self.typing_chk.setChecked(self.cfg.get('typing_only', False));
        g.addWidget(self.typing_chk,r,0,1,2); r+=1
        # Sliders
        def add_slider(label, attr, row):
            g.addWidget(QtWidgets.QLabel(label), row, 0)
            sl = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            if attr in ('stt_gain', 'tempo'):
                sl.factor = 100
                sl.setRange(0, 200)
                sl.setValue(int(self.cfg[attr] * sl.factor))
            else:
                sl.factor = 1
                max_val = 100 if attr.endswith('vol') else 2000 if attr == 'pitch' else 1000
                sl.setRange(0, max_val)
                sl.setValue(int(self.cfg[attr]))
            g.addWidget(sl, row, 1, 1, 3)
            return sl
        self.gain_sl   = add_slider('STT Gain%',   'stt_gain',  r); r+=1
        self.vol_sl    = add_slider('TTS Vol%',    'tts_vol',   r); r+=1
        self.words_sl  = add_slider('Words Threshold','words_chunk',r); r+=1
        self.chunk_sl  = add_slider('Chunk ms',    'chunk_ms',  r); r+=1
        self.pitch_sl  = add_slider('Pitch cents', 'pitch',     r); r+=1
        self.tempo_sl  = add_slider('Tempo %',     'tempo',     r); r+=1
        self.filter_cb = QtWidgets.QComboBox(); self.filter_cb.addItems(['none','lowpass','highpass'])
        g.addWidget(QtWidgets.QLabel('Filter:'),r,0); g.addWidget(self.filter_cb,r,1,1,3); r+=1
        # level meter
        self.level_pb = QtWidgets.QProgressBar(); self.level_pb.setRange(0,100)
        g.addWidget(QtWidgets.QLabel('Input Level:'),r,0); g.addWidget(self.level_pb,r,1,1,3); r+=1
        # Controls
        self.start_btn = QtWidgets.QPushButton('Start')
        self.stop_btn  = QtWidgets.QPushButton('Stop')
        self.stop_btn.setEnabled(False)
        self.bypass_btn= QtWidgets.QPushButton('Toggle Bypass (Ctrl+B)')
        g.addWidget(self.start_btn,r,0,1,1); g.addWidget(self.stop_btn,r,1,1,1); g.addWidget(self.bypass_btn,r,2,1,2); r+=1
        # Log and export
        self.log_te    = QtWidgets.QPlainTextEdit(); self.log_te.setReadOnly(True)
        g.addWidget(self.log_te,r,0,3,4); r+=3
        self.export_btn= QtWidgets.QPushButton('Export Transcript')
        g.addWidget(self.export_btn,r,0,1,3)
        self.tts_input = QtWidgets.QLineEdit(); self.speak_btn = QtWidgets.QPushButton('Speak')
        g.addWidget(self.tts_input,r,1,1,2); g.addWidget(self.speak_btn,r,3,1,1); r+=1

    def _search_models(self):
        """Return a list of possible STT model directories."""
        bases = [
            '.', 'models', 'model', os.path.expanduser('~/models'), '/usr/share/vosk'
        ]
        found = []
        for base in bases:
            if not os.path.isdir(base):
                continue
            for root, dirs, _ in os.walk(base):
                depth = root[len(base):].count(os.sep)
                if depth > 2:
                    dirs[:] = []
                    continue
                if (
                    'model' in os.path.basename(root).lower()
                    or 'vosk' in root.lower()
                    or 'whisper' in root.lower()
                ):
                    found.append(os.path.abspath(root))
                for d in list(dirs):
                    if d.startswith('.') or d.startswith('__'):
                        dirs.remove(d)
        return sorted(set(found))

    def _populate_model_paths(self):
        self.model_cb.clear()
        paths = self._search_models()
        self.model_cb.addItems(paths)
        if not self.cfg.get('model_path') and paths:
            self.cfg['model_path'] = paths[0]
        elif self.cfg.get('model_path') and self.cfg['model_path'] not in paths:
            self.model_cb.addItem(self.cfg['model_path'])
        self.model_cb.setCurrentText(self.cfg.get('model_path', ''))
        self._save_config()

    def _populate_voices(self):
        self.voice_cb.clear()
        eng = self.cfg['tts_engine']
        if eng == 'pyttsx3':
            if 'COMTYPES_CACHE' not in os.environ:
                os.environ['COMTYPES_CACHE'] = os.path.join(tempfile.gettempdir(), 'comtypes_cache')
            os.makedirs(os.environ['COMTYPES_CACHE'], exist_ok=True)
            try:
                t = pyttsx3.init('sapi5')
            except Exception:
                self.voice_cb.addItem('Default', None)
                self.cfg['tts_voice'] = None
                return
            default_id = t.getProperty('voice')
            for v in t.getProperty('voices'):
                self.voice_cb.addItem(v.name, v.id)
            target = self.cfg.get('tts_voice') or default_id
            idx = self.voice_cb.findData(target)
            if idx < 0:
                idx = self.voice_cb.findData(default_id)
            self.voice_cb.setCurrentIndex(max(0, idx))
            self.cfg['tts_voice'] = self.voice_cb.currentData()
        elif eng == 'espeak':
            try:
                out = subprocess.check_output(['espeak', '--voices'], encoding='utf8', errors='ignore')
                for line in out.splitlines()[1:]:
                    parts = line.split()
                    if len(parts) >= 4:
                        self.voice_cb.addItem(parts[3])
            except Exception:
                pass
            idx = self.voice_cb.findText(self.cfg.get('tts_voice',''))
            self.voice_cb.setCurrentIndex(max(0, idx))
            self.cfg['tts_voice'] = self.voice_cb.currentText()

    def _connect_signals(self):
        QtWidgets.QShortcut(QtGui.QKeySequence('Ctrl+B'), self).activated.connect(self._toggle_bypass)
        self.start_btn.clicked.connect(self._start)
        self.stop_btn.clicked.connect(self._stop)
        self.export_btn.clicked.connect(self._export)
        self.speak_btn.clicked.connect(self._speak_manual)
        # track config changes
        widgets = [ (self.loop_chk,'listen_self'), (self.stt_cb,'stt_engine'), (self.model_cb,'model_path'),
                    (self.tts_cb,'tts_engine'),(self.voice_cb,'tts_voice'),(self.gain_sl,'stt_gain'),
                    (self.vol_sl,'tts_vol'),(self.words_sl,'words_chunk'),(self.chunk_sl,'chunk_ms'),
                    (self.pitch_sl,'pitch'),(self.tempo_sl,'tempo'),(self.filter_cb,'filter'),
                    (self.in_cb,'in_dev'),(self.out_cb,'out_dev') ]
        for w,key in widgets:
            sig = w.currentIndexChanged if isinstance(w,QtWidgets.QComboBox) else w.valueChanged if isinstance(w,QtWidgets.QSlider) else w.stateChanged if isinstance(w,QtWidgets.QCheckBox) else w.editingFinished
            sig.connect(lambda _,k=key,w=w: self._update_cfg(k,w))
        self.log_te.clear()

    def _update_cfg(self,key, widget=None):
        if widget is None:
            return
        if isinstance(widget, QtWidgets.QCheckBox):
            val = widget.isChecked()
        elif isinstance(widget, QtWidgets.QSlider):
            val = widget.value() / getattr(widget, 'factor', 1)
        elif isinstance(widget, QtWidgets.QLineEdit):
            val = widget.text()
        elif isinstance(widget, QtWidgets.QComboBox):
            val = widget.currentData() if widget.currentData() is not None else widget.currentText()
        else:
            val = None
        self.cfg[key] = val
        if key == 'tts_engine':
            self._populate_voices()
        elif key == 'stt_engine':
            self._populate_model_paths()
        self._save_config()

    def _refresh_devices(self):
        devs = sd.query_devices()
        self.in_cb.blockSignals(True); self.out_cb.blockSignals(True)
        self.in_cb.clear(); self.out_cb.clear()
        self.in_cb.addItem('Default', None)
        self.out_cb.addItem('Default', None)
        for i,d in enumerate(devs):
            if d['max_input_channels']>0:
                self.in_cb.addItem(f"{i}: {d['name']}", i)
            if d['max_output_channels']>0:
                self.out_cb.addItem(f"{i}: {d['name']}", i)
        self.in_cb.setCurrentIndex(max(0,self.in_cb.findData(self.cfg.get('in_dev'))))
        self.out_cb.setCurrentIndex(max(0,self.out_cb.findData(self.cfg.get('out_dev'))))
        self.in_cb.blockSignals(False); self.out_cb.blockSignals(False)

    def _toggle_bypass(self):
        self.cfg['bypass'] = not self.cfg.get('bypass', False)
        self.bypass_btn.setText('Bypass ON' if self.cfg['bypass'] else 'Toggle Bypass (Ctrl+B)')
        self._save_config()

    def _start(self):
        if self.thread is not None:
            return
        self.thread = SpeechThread(self.cfg.copy())
        self.thread.new_text.connect(self._on_new_text)
        self.thread.level.connect(self.level_pb.setValue)
        self.thread.latency.connect(self._on_latency)
        self.thread.start()
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

    def _stop(self):
        if not self.thread:
            return
        self.thread.running = False
        self.thread.wait()
        self.thread = None
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def _export(self):
        fn, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Save Transcript', 'transcript.txt', 'Text Files (*.txt)')
        if fn:
            open(fn,'w',encoding='utf8').write(self.log_te.toPlainText())

    def _speak_manual(self):
        txt = self.tts_input.text().strip()
        if txt:
            if self.thread:
                self.thread._speak(txt)
            else:
                speak_once(self.cfg.copy(), txt)

    def _on_new_text(self, text):
        self._append_log(text)

    def _on_latency(self, ms):
        self._append_log(f"[latency {ms:.0f} ms]")

    def _append_log(self, txt):
        self.log_te.appendPlainText(txt)

    def closeEvent(self, e):
        self._stop()
        self._save_config()
        super().closeEvent(e)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = App()
    window.show()
    sys.exit(app.exec_())


