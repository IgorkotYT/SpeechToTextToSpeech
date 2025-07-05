import sys, os, json, queue, subprocess, tempfile, shutil, time, traceback
import numpy as np, sounddevice as sd, soundfile as sf, pyttsx3, torch
import os
import tempfile
import zipfile
from PyQt5 import QtCore

try:
    from faster_whisper import WhisperModel
except Exception:
    WhisperModel = None
try:
    from vosk import Model as VoskModel, KaldiRecognizer
except Exception:
    VoskModel = KaldiRecognizer = None


def find_loopback():
    for i, d in enumerate(sd.query_devices()):
        if d['max_input_channels'] > 0 and 'loopback' in d['name'].lower():
            return i
    return None


class SpeechThread(QtCore.QThread):
    new_text = QtCore.pyqtSignal(str)
    level = QtCore.pyqtSignal(int)
    latency = QtCore.pyqtSignal(float)

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.running = True
        self.buffer = ''
        self.rate = 16000
        self._init_stt()
        self._init_tts()

    def _init_stt(self):
        """
        Initialize the chosen STT engine, with automatic path detection:
          - WhisperModel: defaults to 'tiny.en' if no path given
          - VoskModel: unzips .zip, descends into single-subfolder directories
          - Silero: fallback if neither Whisper nor Vosk loads
        """
        stt = self.cfg.get('stt_engine', 'Whisper').lower()
        model_p = self.cfg.get('model_path', '').strip()

        # --- WHISPER (via faster-whisper) ---
        if stt == 'whisper' and WhisperModel:
            model_name = model_p or 'tiny.en'
            self.model = WhisperModel(model_name, compute_type='int8')
            self.mode = 'whisper'
            return

        # --- VOSK (local model directory or zip) ---
        if stt == 'vosk' and VoskModel:
            if model_p.lower().endswith('.zip') and os.path.isfile(model_p):
                tmp = tempfile.mkdtemp(prefix='vosk_')
                with zipfile.ZipFile(model_p, 'r') as z:
                    z.extractall(tmp)
                model_p = tmp

            if os.path.isdir(model_p):
                subs = [os.path.join(model_p, d) for d in os.listdir(model_p)
                        if os.path.isdir(os.path.join(model_p, d))]
                if len(subs) == 1:
                    model_p = subs[0]

            if os.path.isdir(model_p):
                self.model = KaldiRecognizer(VoskModel(model_p), self.rate)
                self.model.SetWords(False)
                self.mode = 'vosk'
                return
            else:
                print(f"[WARN] Vosk model not found at: {model_p}")

        # --- SILERO (last-resort fallback) ---
        try:
            m, dec, utils = torch.hub.load(
                'snakers4/silero-models',
                'silero_stt',
                language=self.cfg.get('silero_lang', 'en')
            )
            m.to('cpu')
            self.model = m
            self.decoder = dec
            self.prepare = utils[-1]
            self.mode = 'silero'
            return
        except Exception as e:
            print(f"[ERROR] Silero load failed: {e}")

        raise RuntimeError("No valid STT engine could be initialized.")

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
        block = int(self.rate * self.cfg['chunk_ms'] / 1000)

        def cb(indata, fr, ti, st):
            q.put(indata.copy())

        inputs = [self.cfg['in_dev']]
        if self.cfg['listen_self']:
            lb = find_loopback()
            if lb not in inputs:
                inputs.append(lb)
        streams = [sd.InputStream(device=d, samplerate=self.rate, channels=1, blocksize=block, callback=cb) for d in inputs]
        out_stream = None
        if self.cfg['out_dev'] is not None:
            out_stream = sd.OutputStream(device=self.cfg['out_dev'], samplerate=self.rate, channels=1)
            out_stream.start()
        for s in streams:
            s.start()
        try:
            while self.running:
                audio = np.squeeze(q.get()) * self.cfg['stt_gain']
                self.level.emit(int(np.abs(audio).max() * 100))
                if self.cfg['bypass'] and out_stream:
                    out_stream.write(audio)
                    continue
                t0 = time.time()
                txt = ''
                if self.mode == 'whisper':
                    segs, _ = self.model.transcribe(audio, language='en', beam_size=1)
                    txt = ' '.join(s.text for s in segs)
                elif self.mode == 'vosk':
                    if self.model.AcceptWaveform(audio.tobytes()):
                        txt = json.loads(self.model.Result()).get('text', '')
                else:
                    batch = self.prepare([torch.from_numpy(audio).float()], [self.rate])
                    txt = self.decoder(self.model(batch)[0].cpu())
                self.latency.emit((time.time() - t0) * 1000)
                if not txt:
                    continue
                words = txt.split()
                if len(words) >= self.cfg['words_chunk']:
                    out = txt
                    self.buffer = ''
                else:
                    self.buffer += ' ' + txt
                    if len(self.buffer.split()) < self.cfg['words_chunk']:
                        continue
                    out = self.buffer.strip()
                    self.buffer = ''
                self.new_text.emit(out)
                self._speak(out)
        except Exception as e:
            self.new_text.emit(f"[Error] {e}")
        finally:
            for s in streams:
                s.stop(); s.close()
            if out_stream:
                out_stream.stop(); out_stream.close()

    def _speak(self, text):
        eng = self.cfg['tts_engine']
        if eng == 'pyttsx3':
            self.tts.say(text)
            self.tts.runAndWait()
            return
        fd, wav = tempfile.mkstemp('.wav'); os.close(fd)
        subprocess.run([eng, '-w', wav, text], check=True)
        out = wav
        if self.sox_ok and (self.cfg['pitch'] or self.cfg['tempo'] != 1 or self.cfg['filter'] != 'none'):
            fx = wav.replace('.wav', '_fx.wav'); cmd = ['sox', wav, fx]
            if self.cfg['pitch']:
                cmd += ['pitch', str(self.cfg['pitch'])]
            if self.cfg['tempo'] != 1:
                cmd += ['tempo', str(self.cfg['tempo'])]
            if self.cfg['filter'] != 'none':
                cmd += [self.cfg['filter'], '3000']
            subprocess.run(cmd, check=False); out = fx
        data, fs = sf.read(out, dtype='float32'); sd.play(data, fs); sd.wait()
        for f in (wav, out):
            try:
                os.unlink(f)
            except Exception:
                pass
