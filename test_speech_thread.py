import pytest
import os
import queue
import time
import numpy as np
from unittest.mock import patch, MagicMock, call
import tempfile
import json

from speech_thread import SpeechThread, speak_once, find_vosk_model

@pytest.fixture
def default_cfg():
    return {
        'in_dev':None,'out_dev':None,'listen_self':False,
        'stt_engine':'Whisper','model_path':'','stt_gain':1.0,
        'tts_engine':'espeak','tts_voice':'','tts_vol':100,
        'words_chunk':2,'chunk_ms':500,'pitch':0,'tempo':1,'filter':'none',
        'bypass':False,'typing_only':False
    }

@patch('speech_thread.openai_whisper.load_model')
@patch('speech_thread.pyttsx3.init')
def test_init_whisper_pyttsx3(mock_tts, mock_whisper, default_cfg):
    default_cfg['stt_engine'] = 'Whisper'
    default_cfg['tts_engine'] = 'pyttsx3'
    thread = SpeechThread(default_cfg)
    assert thread.mode == 'openai'
    mock_whisper.assert_called_once_with('base')
    mock_tts.assert_called_once()
    assert thread.cfg['tts_engine'] == 'pyttsx3'

@patch('speech_thread.KaldiRecognizer')
@patch('speech_thread.VoskModel')
@patch('speech_thread.find_vosk_model')
def test_init_vosk(mock_find, mock_vosk, mock_kaldi, default_cfg):
    default_cfg['stt_engine'] = 'Vosk'
    mock_find.return_value = '/dummy/vosk'
    thread = SpeechThread(default_cfg)
    assert thread.mode == 'vosk'
    mock_find.assert_called_once()
    mock_vosk.assert_called_once_with('/dummy/vosk')

@patch('speech_thread.torch.hub.load')
def test_init_silero(mock_hub_load, default_cfg):
    default_cfg['stt_engine'] = 'Silero'
    mock_model = MagicMock()
    mock_decoder = MagicMock()
    mock_utils = [MagicMock(), MagicMock(), MagicMock()] # utils is a list, prepare is utils[-1]
    mock_hub_load.return_value = (mock_model, mock_decoder, mock_utils)

    thread = SpeechThread(default_cfg)
    assert thread.mode == 'silero'
    mock_hub_load.assert_called_once()

@patch('speech_thread.sd.InputStream')
@patch('speech_thread.openai_whisper.load_model')
def test_thread_run_and_stop(mock_whisper, mock_input_stream, default_cfg):
    default_cfg['stt_engine'] = 'Whisper'
    thread = SpeechThread(default_cfg)

    # We will simulate the queue getting one empty item, then stop
    def mock_run_once():
        time.sleep(0.1)
        thread.running = False

    import threading
    stopper = threading.Thread(target=mock_run_once)
    stopper.start()

    thread.run()
    stopper.join()

    assert mock_input_stream.return_value.start.called
    assert mock_input_stream.return_value.stop.called

@patch('speech_thread.subprocess.run')
@patch('speech_thread.sf.read')
@patch('speech_thread.sd.play')
@patch('speech_thread.sd.wait')
@patch('speech_thread.openai_whisper.load_model')
def test_speak_once_espeak(mock_whisper, mock_wait, mock_play, mock_read, mock_run, default_cfg):
    default_cfg['tts_engine'] = 'espeak'
    mock_read.return_value = (np.array([0.0, 0.0]), 16000)

    speak_once(default_cfg, "hello world")

    mock_run.assert_called()
    assert "espeak" in mock_run.call_args_list[0][0][0]
    mock_play.assert_called_once()
    mock_wait.assert_called_once()

@patch('speech_thread.subprocess.run')
@patch('speech_thread.sf.read')
@patch('speech_thread.sd.play')
@patch('speech_thread.sd.wait')
@patch('speech_thread.openai_whisper.load_model')
def test_speak_once_sox_fx(mock_whisper, mock_wait, mock_play, mock_read, mock_run, default_cfg):
    default_cfg['tts_engine'] = 'espeak'
    default_cfg['pitch'] = 100
    mock_read.return_value = (np.array([0.0]), 16000)

    # Needs to pretend sox is installed for this test
    with patch('speech_thread.shutil.which', return_value='/usr/bin/sox'):
        speak_once(default_cfg, "test")

    assert mock_run.call_count == 2 # one for espeak, one for sox
    sox_call = mock_run.call_args_list[1][0][0]
    assert 'sox' in sox_call
    assert 'pitch' in sox_call
    assert '100' in sox_call

@patch('speech_thread.openai_whisper.load_model')
def test_find_vosk_model(mock_whisper):
    with tempfile.TemporaryDirectory() as temp_dir:
        vosk_dir = os.path.join(temp_dir, 'vosk-model')
        os.makedirs(vosk_dir)

        with patch('os.path.expanduser', return_value=temp_dir), \
             patch('speech_thread.os.listdir', return_value=['vosk-model']), \
             patch('os.path.isdir', return_value=True):

            result = find_vosk_model()
            # As the real path might be different depending on current dir
            assert result is not None
