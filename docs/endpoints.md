## Quick start

After running the docker image interactive Swagger API documentation is available at [localhost:9000/docs](http://localhost:9000/docs)

There are 3 endpoints available:

- [/asr](##Automatic-Speech-recognition-service-/asr) (Automatic Speech Recognition)
- [/v1/audio/transcriptions](##OpenAI-compatible-transcription-service-/v1/audio/transcriptions) (OpenAI-compatible transcription)
- [/detect-language](##Language-detection-service-/detect-language)

## Automatic speech recognition service /asr

- 2 task choices:
  - **transcribe**: (default) task, transcribes the uploaded file.
  - **translate**: will provide an English transcript no matter which language was spoken.
- Files are automatically converted with FFmpeg.
  - Full list of supported [audio](https://ffmpeg.org/general.html#Audio-Codecs) and [video](https://ffmpeg.org/general.html#Video-Codecs) formats.
- You can enable word level timestamps output by `word_timestamps` parameter
- You can Enable the voice activity detection (VAD) to filter out parts of the audio without speech by `vad_filter` parameter (only with `Faster Whisper` for now).

### Request URL Query Params

| Name            | Values                                        | Description                                                       |
| --------------- | --------------------------------------------- | ----------------------------------------------------------------- |
| audio_file      | File                                          | Audio or video file to transcribe                                 |
| output          | `text` (default), `json`, `vtt`, `srt`, `tsv` | Output format                                                     |
| task            | `transcribe`, `translate`                     | Task type - transcribe in source language or translate to English |
| language        | `en` (default is auto recognition)            | Source language code (see supported languages)                    |
| word_timestamps | false (default)                               | Enable word-level timestamps (Faster Whisper only)                |
| vad_filter      | false (default)                               | Enable voice activity detection filtering (Faster Whisper only)   |
| encode          | true (default)                                | Encode audio through FFmpeg before processing                     |
| diarize         | false (default)                               | Enable speaker diarization (WhisperX only)                        |
| min_speakers    | null (default)                                | Minimum number of speakers for diarization (WhisperX only)        |
| max_speakers    | null (default)                                | Maximum number of speakers for diarization (WhisperX only)        |

Example request with cURL

```bash
curl -X POST -H "content-type: multipart/form-data" -F "audio_file=@/path/to/file" 0.0.0.0:9000/asr?output=json
```

### Response (JSON)

- **text**: Contains the full transcript
- **segments**: Contains an entry per segment. Each entry provides `timestamps`, `transcript`, `token ids`, `word level timestamps` and other metadata
- **language**: Detected or provided language (as a language code)

### Response Formats

The API supports multiple output formats:

- **text**: Plain text transcript (default)
- **json**: Detailed JSON with segments, timestamps, and metadata
- **vtt**: WebVTT subtitle format
- **srt**: SubRip subtitle format
- **tsv**: Tab-separated values with timestamps

### Supported Languages

The service supports all languages supported by Whisper. Some common language codes:

- Turkish (tr)
- English (en)
- Spanish (es)
- French (fr)
- German (de)
- Italian (it)
- Portuguese (pt)
- And many more...

See the [Whisper documentation](https://github.com/openai/whisper#available-models-and-languages) for the full list of supported languages.

### Speaker Diarization

When using the WhisperX engine with diarization enabled (`diarize=true`), the output will include speaker labels for each segment. This requires:

1. WhisperX engine to be configured
2. Valid Hugging Face token set in HF_TOKEN
3. Sufficient memory for diarization models

You can optionally specify `min_speakers` and `max_speakers` if you know the expected number of speakers.

## OpenAI-compatible transcription service /v1/audio/transcriptions

This endpoint is compatible with OpenAI's `audio/transcriptions` API format and also accepts Whisper-specific extensions.

### Multipart Form Fields

| Name            | Values                                                 | Required | Description                                                                                         |
| --------------- | ------------------------------------------------------ | -------- | --------------------------------------------------------------------------------------------------- |
| file            | File                                                   | Yes      | Audio/video file to transcribe                                                                      |
| model           | String                                                 | Yes      | Accepted for OpenAI compatibility. Current service engine/model is defined by environment variables |
| language        | ISO language code (for example `en`, `pt`)             | No       | Source language hint. If omitted, language is auto-detected                                         |
| prompt          | String                                                 | No       | Optional prompt/context for transcription                                                           |
| response_format | `json` (default), `verbose_json`, `text`, `srt`, `vtt` | No       | Output format                                                                                       |
| temperature     | Float                                                  | No       | Accepted for compatibility, currently ignored                                                       |
| vad_filter      | `true`/`false` (default `false`)                       | No       | Whisper extension: enable VAD filtering                                                             |
| word_timestamps | `true`/`false` (default `false`)                       | No       | Whisper extension: return word-level timestamps                                                     |
| diarize         | `true`/`false` (default `false`)                       | No       | Whisper extension: enable speaker diarization                                                       |
| min_speakers    | Integer                                                | No       | Whisper extension: minimum speaker count hint                                                       |
| max_speakers    | Integer                                                | No       | Whisper extension: maximum speaker count hint                                                       |

### Example request (OpenAI-compatible)

```bash
curl -X POST "http://0.0.0.0:9000/v1/audio/transcriptions" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/audio.wav" \
  -F "model=whisper-1" \
  -F "language=pt" \
  -F "response_format=json"
```

### Response notes

- `json`: returns `{ "text": "..." }`
- `verbose_json`: returns detailed structure with `task`, `language`, `duration`, `text` and `segments`
- `text`, `srt`, `vtt`: return plain text subtitle/text output

## Language detection service /detect-language

Detects the language spoken in the uploaded file. Only processes first 30 seconds.

Returns a json with following fields:

- **detected_language**: Human readable language name (e.g. "english")
- **language_code**: ISO language code (e.g. "en")
- **confidence**: Confidence score between 0 and 1 indicating detection reliability

Example response:

```json
{
  "detected_language": "english",
  "language_code": "en",
  "confidence": 0.98
}
```
