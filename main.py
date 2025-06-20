# TTS-Only Bot – Full Featured GUI (Dark Theme)
# Whisper / Vosk / Silero STT, pyttsx3/espeak/sam TTS, SoX FX, config, export, bypass

import sys, os, json, queue, subprocess, tempfile, shutil, time, traceback
import numpy as np, sounddevice as sd, soundfile as sf, pyttsx3, torch
from PyQt5 import QtCore, QtGui, QtWidgets

# Optional dark theme
try:
    import qdarktheme
    qdarktheme.setup_theme('auto')
except:
    pass

# Optional STT backends
try:
    from faster_whisper import WhisperModel
except:
    WhisperModel = None
try:
    from vosk import Model as VoskModel, KaldiRecognizer
except:
    VoskModel = KaldiRecognizer = None

CONFIG_FILE = 'tts_bot_config.json'

def find_loopback():
    for i,d in enumerate(sd.query_devices()):
        if d['max_input_channels']>0 and 'loopback' in d['name'].lower():
            return i
    return None

class SpeechThread(QtCore.QThread):
    new_text = QtCore.pyqtSignal(str)
    level    = QtCore.pyqtSignal(int)
    latency  = QtCore.pyqtSignal(float)

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.running = True
        self.buffer = ''
        self.rate = 16000
        self._init_stt()
        self._init_tts()

    def _init_stt(self):
        stt = self.cfg['stt_engine']
        model_path = self.cfg['model_path']
        if stt=='Whisper' and WhisperModel:
            self.model = WhisperModel('base.en', compute_type='int8')
            self.mode = 'whisper'
        elif stt=='Vosk' and VoskModel:
            self.model = KaldiRecognizer(VoskModel(model_path), self.rate)
            self.model.SetWords(False)
            self.mode = 'vosk'
        else:
            m,dec,utils = torch.hub.load('snakers4/silero-models', 'silero_stt', language='en')
            m.to('cpu')
            self.model = m; self.decoder = dec; self.prepare = utils[-1]
            self.mode = 'silero'

    def _init_tts(self):
        self.sox_ok = shutil.which('sox') is not None
        eng = self.cfg['tts_engine']
        if eng == 'pyttsx3':
            try:
                self.tts = pyttsx3.init()
            except Exception:
                self.cfg['tts_engine'] = 'espeak'
                return self._init_tts()
            self.tts.setProperty('rate', 180)
            self.tts.setProperty('volume', self.cfg['tts_vol'] / 100)
            if self.cfg['tts_voice']:
                self.tts.setProperty('voice', self.cfg['tts_voice'])
        elif shutil.which(eng) is None:
            self.cfg['tts_engine'] = 'pyttsx3'
            self._init_tts()
        
    def run(self):
        q = queue.Queue()
        block = int(self.rate * self.cfg['chunk_ms']/1000)
        def cb(indata,fr,ti,st): q.put(indata.copy())
        inputs = [self.cfg['in_dev']]
        if self.cfg['listen_self']:
            lb = find_loopback()
            if lb not in inputs: inputs.append(lb)
        streams = [sd.InputStream(device=d,samplerate=self.rate,channels=1,blocksize=block,callback=cb) for d in inputs]
        out_stream = None
        if self.cfg['out_dev'] is not None:
            out_stream = sd.OutputStream(device=self.cfg['out_dev'],samplerate=self.rate,channels=1)
            out_stream.start()
        for s in streams: s.start()
        try:
            while self.running:
                audio = np.squeeze(q.get()) * self.cfg['stt_gain']
                self.level.emit(int(np.abs(audio).max()*100))
                if self.cfg['bypass'] and out_stream:
                    out_stream.write(audio); continue
                t0=time.time(); txt=''
                if self.mode=='whisper':
                    segs,_ = self.model.transcribe(audio, language='en', beam_size=1)
                    txt = ' '.join(s.text for s in segs)
                elif self.mode=='vosk':
                    if self.model.AcceptWaveform(audio.tobytes()):
                        txt = json.loads(self.model.Result()).get('text','')
                else:
                    batch = self.prepare([torch.from_numpy(audio).float()],[self.rate])
                    txt = self.decoder(self.model(batch)[0].cpu())
                self.latency.emit((time.time()-t0)*1000)
                if not txt: continue
                words = txt.split()
                if len(words)>=self.cfg['words_chunk']:
                    out=txt; self.buffer=''
                else:
                    self.buffer += ' '+txt
                    if len(self.buffer.split())<self.cfg['words_chunk']:
                        continue
                    out=self.buffer.strip(); self.buffer=''
                self.new_text.emit(out)
                self._speak(out)
        except Exception as e:
            self.new_text.emit(f"[Error] {e}")
        finally:
            for s in streams: s.stop(); s.close()
            if out_stream: out_stream.stop(); out_stream.close()

    def _speak(self,text):
        eng=self.cfg['tts_engine']
        if eng=='pyttsx3':
            self.tts.say(text); self.tts.runAndWait(); return
        fd,wav = tempfile.mkstemp('.wav'); os.close(fd)
        subprocess.run([eng,'-w',wav,text],check=True)
        out=wav
        if self.sox_ok and (self.cfg['pitch'] or self.cfg['tempo']!=1 or self.cfg['filter']!='none'):
            fx=wav.replace('.wav','_fx.wav'); cmd=['sox',wav,fx]
            if self.cfg['pitch']: cmd+=['pitch',str(self.cfg['pitch'])]
            if self.cfg['tempo']!=1: cmd+=['tempo',str(self.cfg['tempo'])]
            if self.cfg['filter']!='none': cmd+=[self.cfg['filter'],'3000']
            subprocess.run(cmd,check=False); out=fx
        data,fs=sf.read(out,dtype='float32'); sd.play(data,fs); sd.wait()
        for f in (wav,out):
            try: os.unlink(f)
            except: pass

# GUI ---------------------------------------------------------------------------
class App(QtWidgets.QWidget):
    def __init__(self):
        super().__init__(); self.setWindowTitle('TTS-Only Bot'); self.resize(800,600)
        self.cfg=self._load_config(); self.thread=None
        self._build_ui(); self._populate_voices(); self._connect_signals()
        self._refresh_devices()

    def _load_config(self):
        if os.path.exists(CONFIG_FILE):
            try: return json.load(open(CONFIG_FILE))
            except: pass
        return {
            'in_dev':None,'out_dev':None,'listen_self':False,
            'stt_engine':'Whisper','model_path':'model','stt_gain':1.0,
            'tts_engine':'espeak','tts_voice':'','tts_vol':100,
            'words_chunk':5,'chunk_ms':500,'pitch':0,'tempo':1,'filter':'none',
            'bypass':False
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
        g.addWidget(QtWidgets.QLabel('STT Engine:'),r,0); self.stt_cb=QtWidgets.QComboBox(); self.stt_cb.addItems(['Whisper','Vosk','Silero']); g.addWidget(self.stt_cb,r,1);
        g.addWidget(QtWidgets.QLabel('Model path:'),r,2); self.model_le=QtWidgets.QLineEdit(self.cfg['model_path']); g.addWidget(self.model_le,r,3); r+=1
        g.addWidget(QtWidgets.QLabel('TTS Engine:'),r,0);
        self.tts_cb=QtWidgets.QComboBox();
        self.tts_cb.addItems(['pyttsx3','espeak','sam']);
        self.tts_cb.setCurrentText(self.cfg['tts_engine']);
        g.addWidget(self.tts_cb,r,1);
        g.addWidget(QtWidgets.QLabel('TTS Voice:'),r,2); self.voice_cb=QtWidgets.QComboBox(); g.addWidget(self.voice_cb,r,3); r+=1
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
        self.bypass_btn= QtWidgets.QPushButton('Toggle Bypass (Ctrl+B)')
        g.addWidget(self.start_btn,r,0,1,1); g.addWidget(self.stop_btn,r,1,1,1); g.addWidget(self.bypass_btn,r,2,1,2); r+=1
        # Log and export
        self.log_te    = QtWidgets.QPlainTextEdit(); self.log_te.setReadOnly(True)
        g.addWidget(self.log_te,r,0,3,4); r+=3
        self.export_btn= QtWidgets.QPushButton('Export Transcript')
        g.addWidget(self.export_btn,r,0,1,3)
        self.tts_input = QtWidgets.QLineEdit(); self.speak_btn = QtWidgets.QPushButton('Speak')
        g.addWidget(self.tts_input,r,1,1,2); g.addWidget(self.speak_btn,r,3,1,1); r+=1

    def _populate_voices(self):
        try:
            t = pyttsx3.init()
        except Exception:
            self.cfg['tts_engine'] = 'espeak'
            self.tts_cb.setCurrentText('espeak')
            return
        for v in t.getProperty('voices'):
            self.voice_cb.addItem(v.name, v.id)
        idx = self.voice_cb.findData(self.cfg['tts_voice'])
        self.voice_cb.setCurrentIndex(max(0, idx))

    def _connect_signals(self):
        QtWidgets.QShortcut(QtGui.QKeySequence('Ctrl+B'), self).activated.connect(self._toggle_bypass)
        self.start_btn.clicked.connect(self._start)
        self.stop_btn.clicked.connect(self._stop)
        self.export_btn.clicked.connect(self._export)
        self.speak_btn.clicked.connect(self._speak_manual)
        # track config changes
        widgets = [ (self.loop_chk,'listen_self'), (self.stt_cb,'stt_engine'), (self.model_le,'model_path'),
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
            self.thread._speak(txt) if self.thread else SpeechThread(self.cfg)._speak(txt)

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


