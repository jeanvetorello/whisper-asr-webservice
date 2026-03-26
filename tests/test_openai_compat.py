import json
from io import StringIO
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.openai_compat import create_openai_router


@pytest.fixture
def mock_asr():
    return MagicMock()


@pytest.fixture(autouse=True)
def mock_load_audio():
    with patch("app.openai_compat.load_audio", return_value=np.zeros(16000)):
        yield


@pytest.fixture
def client(mock_asr):
    app = FastAPI()
    app.include_router(create_openai_router(mock_asr))
    return TestClient(app), mock_asr


def _json_payload(text="Hello world.", language="en", segments=None):
    if segments is None:
        segments = [{"start": 0.0, "end": 2.5, "text": f" {text}"}]
    return StringIO(json.dumps({"text": text, "language": language, "segments": segments}))


def test_json_format_returns_text_only(client):
    tc, mock_asr = client
    mock_asr.transcribe.return_value = _json_payload()

    response = tc.post(
        "/v1/audio/transcriptions",
        data={"model": "whisper-1", "response_format": "json"},
        files={"file": ("audio.mp3", b"fake", "audio/mpeg")},
    )

    assert response.status_code == 200
    assert response.json() == {"text": "Hello world."}


def test_default_response_format_is_json(client):
    tc, mock_asr = client
    mock_asr.transcribe.return_value = _json_payload()

    response = tc.post(
        "/v1/audio/transcriptions",
        data={"model": "whisper-1"},
        files={"file": ("audio.mp3", b"fake", "audio/mpeg")},
    )

    assert response.status_code == 200
    assert "text" in response.json()


def test_verbose_json_format(client):
    tc, mock_asr = client
    mock_asr.transcribe.return_value = _json_payload()

    response = tc.post(
        "/v1/audio/transcriptions",
        data={"model": "whisper-1", "response_format": "verbose_json"},
        files={"file": ("audio.mp3", b"fake", "audio/mpeg")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["task"] == "transcribe"
    assert body["language"] == "en"
    assert body["duration"] == 2.5
    assert body["text"] == "Hello world."
    assert body["segments"] == [{"id": 0, "start": 0.0, "end": 2.5, "text": " Hello world."}]


def test_verbose_json_duration_zero_when_no_segments(client):
    tc, mock_asr = client
    mock_asr.transcribe.return_value = _json_payload(segments=[])

    response = tc.post(
        "/v1/audio/transcriptions",
        data={"model": "whisper-1", "response_format": "verbose_json"},
        files={"file": ("audio.mp3", b"fake", "audio/mpeg")},
    )

    assert response.json()["duration"] == 0.0


def test_text_format(client):
    tc, mock_asr = client
    mock_asr.transcribe.return_value = StringIO("Hello world.")

    response = tc.post(
        "/v1/audio/transcriptions",
        data={"model": "whisper-1", "response_format": "text"},
        files={"file": ("audio.mp3", b"fake", "audio/mpeg")},
    )

    assert response.status_code == 200
    assert response.text == "Hello world."


def test_srt_format(client):
    tc, mock_asr = client
    srt = "1\n00:00:00,000 --> 00:00:02,500\nHello world.\n"
    mock_asr.transcribe.return_value = StringIO(srt)

    response = tc.post(
        "/v1/audio/transcriptions",
        data={"model": "whisper-1", "response_format": "srt"},
        files={"file": ("audio.mp3", b"fake", "audio/mpeg")},
    )

    assert response.status_code == 200
    assert response.text == srt


def test_vtt_format(client):
    tc, mock_asr = client
    vtt = "WEBVTT\n\n00:00:00.000 --> 00:00:02.500\nHello world.\n"
    mock_asr.transcribe.return_value = StringIO(vtt)

    response = tc.post(
        "/v1/audio/transcriptions",
        data={"model": "whisper-1", "response_format": "vtt"},
        files={"file": ("audio.mp3", b"fake", "audio/mpeg")},
    )

    assert response.status_code == 200
    assert response.text == vtt


def test_model_field_is_ignored(client):
    tc, mock_asr = client
    mock_asr.transcribe.return_value = _json_payload()

    response = tc.post(
        "/v1/audio/transcriptions",
        data={"model": "gpt-4o-transcribe"},
        files={"file": ("audio.mp3", b"fake", "audio/mpeg")},
    )

    assert response.status_code == 200


def test_language_forwarded_to_transcribe(client):
    tc, mock_asr = client
    mock_asr.transcribe.return_value = _json_payload()

    tc.post(
        "/v1/audio/transcriptions",
        data={"model": "whisper-1", "language": "pt"},
        files={"file": ("audio.mp3", b"fake", "audio/mpeg")},
    )

    call_args = mock_asr.transcribe.call_args.args  # transcribe(audio, task, language, prompt, ...)
    assert call_args[2] == "pt"  # language is 3rd positional arg


def test_prompt_forwarded_as_initial_prompt(client):
    tc, mock_asr = client
    mock_asr.transcribe.return_value = _json_payload()

    tc.post(
        "/v1/audio/transcriptions",
        data={"model": "whisper-1", "prompt": "Some context."},
        files={"file": ("audio.mp3", b"fake", "audio/mpeg")},
    )

    args = mock_asr.transcribe.call_args[0]
    assert args[3] == "Some context."  # initial_prompt is 4th positional arg


def test_transcribe_called_with_json_output_for_json_format(client):
    tc, mock_asr = client
    mock_asr.transcribe.return_value = _json_payload()

    tc.post(
        "/v1/audio/transcriptions",
        data={"model": "whisper-1", "response_format": "json"},
        files={"file": ("audio.mp3", b"fake", "audio/mpeg")},
    )

    args = mock_asr.transcribe.call_args[0]
    assert args[7] == "json"  # output is 8th positional arg


def test_transcribe_called_with_txt_output_for_text_format(client):
    tc, mock_asr = client
    mock_asr.transcribe.return_value = StringIO("Hello.")

    tc.post(
        "/v1/audio/transcriptions",
        data={"model": "whisper-1", "response_format": "text"},
        files={"file": ("audio.mp3", b"fake", "audio/mpeg")},
    )

    args = mock_asr.transcribe.call_args[0]
    assert args[7] == "txt"  # output is 8th positional arg


def test_invalid_response_format_returns_400(client):
    tc, mock_asr = client

    response = tc.post(
        "/v1/audio/transcriptions",
        data={"model": "whisper-1", "response_format": "tsv"},
        files={"file": ("audio.mp3", b"fake", "audio/mpeg")},
    )

    assert response.status_code == 400


def test_vad_filter_forwarded(client):
    tc, mock_asr = client
    mock_asr.transcribe.return_value = _json_payload()

    tc.post(
        "/v1/audio/transcriptions",
        data={"model": "whisper-1", "vad_filter": "true"},
        files={"file": ("audio.mp3", b"fake", "audio/mpeg")},
    )

    args = mock_asr.transcribe.call_args[0]
    assert args[4] is True  # vad_filter is 5th positional arg


def test_word_timestamps_forwarded(client):
    tc, mock_asr = client
    mock_asr.transcribe.return_value = _json_payload()

    tc.post(
        "/v1/audio/transcriptions",
        data={"model": "whisper-1", "word_timestamps": "true"},
        files={"file": ("audio.mp3", b"fake", "audio/mpeg")},
    )

    args = mock_asr.transcribe.call_args[0]
    assert args[5] is True  # word_timestamps is 6th positional arg


def test_diarize_forwarded_in_options(client):
    tc, mock_asr = client
    mock_asr.transcribe.return_value = _json_payload()

    tc.post(
        "/v1/audio/transcriptions",
        data={"model": "whisper-1", "diarize": "true", "min_speakers": "2", "max_speakers": "4"},
        files={"file": ("audio.mp3", b"fake", "audio/mpeg")},
    )

    args = mock_asr.transcribe.call_args[0]
    assert args[6] == {"diarize": True, "min_speakers": 2, "max_speakers": 4}  # options is 7th positional arg
