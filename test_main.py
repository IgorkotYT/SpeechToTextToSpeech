import pytest
from PyQt5 import QtCore, QtWidgets
import sys
import os
import json
import tempfile
from unittest.mock import patch, MagicMock

# Force offscreen before importing App
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

app = QtWidgets.QApplication.instance()
if app is None:
    app = QtWidgets.QApplication(sys.argv)

import main
from main import App, CONFIG_FILE

@pytest.fixture
def temp_config_file():
    fd, path = tempfile.mkstemp(suffix='.json')
    os.close(fd)
    original_config = main.CONFIG_FILE
    main.CONFIG_FILE = path
    yield path
    main.CONFIG_FILE = original_config
    if os.path.exists(path):
        os.unlink(path)

@pytest.fixture
def mock_speech_thread():
    with patch('main.SpeechThread') as mock:
        yield mock

def test_app_init(qtbot, temp_config_file):
    window = App()
    qtbot.addWidget(window)
    assert window.windowTitle() == 'TTS-Only Bot'
    assert window.start_btn.isEnabled() is True
    assert window.stop_btn.isEnabled() is False
    window.close()

def test_app_load_config(qtbot, temp_config_file):
    cfg = {
        'in_dev':None,'out_dev':None,'listen_self':True,
        'stt_engine':'Vosk','model_path':'/some/path','stt_gain':1.5,
        'tts_engine':'espeak','tts_voice':'en','tts_vol':80,
        'words_chunk':10,'chunk_ms':1000,'pitch':100,'tempo':1.2,'filter':'lowpass',
        'bypass':True,'typing_only':True
    }
    with open(temp_config_file, 'w') as f:
        json.dump(cfg, f)

    with patch('sounddevice.query_devices', return_value=[]):
        window = App()
        qtbot.addWidget(window)
        assert window.cfg['listen_self'] is True
        assert window.cfg['stt_engine'] == 'Vosk'
        assert window.stt_cb.currentText() == 'Vosk'
        assert window.loop_chk.isChecked() is True
        assert window.bypass_btn.text() == 'Bypass ON'
        window.close()

def test_app_save_config_on_change(qtbot, temp_config_file):
    window = App()
    qtbot.addWidget(window)
    window.loop_chk.setChecked(True)

    with open(temp_config_file, 'r') as f:
        cfg = json.load(f)
        assert cfg['listen_self'] is True
    window.close()

def test_app_start_stop(qtbot, temp_config_file, mock_speech_thread):
    window = App()
    qtbot.addWidget(window)

    # Test Start
    qtbot.mouseClick(window.start_btn, QtCore.Qt.LeftButton)
    assert window.thread is not None
    mock_speech_thread.assert_called_once()
    assert window.start_btn.isEnabled() is False
    assert window.stop_btn.isEnabled() is True

    mock_thread_instance = mock_speech_thread.return_value
    mock_thread_instance.start.assert_called_once()

    # Test Stop
    qtbot.mouseClick(window.stop_btn, QtCore.Qt.LeftButton)
    assert mock_thread_instance.running is False
    mock_thread_instance.wait.assert_called_once()
    assert window.thread is None
    assert window.start_btn.isEnabled() is True
    assert window.stop_btn.isEnabled() is False
    window.close()

def test_app_toggle_bypass(qtbot, temp_config_file):
    window = App()
    qtbot.addWidget(window)
    assert window.cfg['bypass'] is False

    qtbot.mouseClick(window.bypass_btn, QtCore.Qt.LeftButton)
    assert window.cfg['bypass'] is True
    assert window.bypass_btn.text() == 'Bypass ON'

    qtbot.mouseClick(window.bypass_btn, QtCore.Qt.LeftButton)
    assert window.cfg['bypass'] is False
    assert window.bypass_btn.text() == 'Toggle Bypass (Ctrl+B)'
    window.close()

def test_app_speak_manual(qtbot, temp_config_file):
    with patch('main.speak_once') as mock_speak_once:
        window = App()
        qtbot.addWidget(window)
        window.tts_input.setText("hello world")
        qtbot.mouseClick(window.speak_btn, QtCore.Qt.LeftButton)
        mock_speak_once.assert_called_once_with(window.cfg.copy(), "hello world")
        window.close()

def test_app_speak_manual_with_thread(qtbot, temp_config_file, mock_speech_thread):
    window = App()
    qtbot.addWidget(window)
    qtbot.mouseClick(window.start_btn, QtCore.Qt.LeftButton)

    mock_thread_instance = mock_speech_thread.return_value
    window.tts_input.setText("hello thread")
    qtbot.mouseClick(window.speak_btn, QtCore.Qt.LeftButton)

    mock_thread_instance._speak.assert_called_once_with("hello thread")
    window.close()
