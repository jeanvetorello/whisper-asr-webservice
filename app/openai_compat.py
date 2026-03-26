import json
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse

from app.utils import load_audio


def create_openai_router(asr_model) -> APIRouter:
    router = APIRouter(prefix="/v1", tags=["OpenAI Compatible"])

    @router.post("/audio/transcriptions")
    async def transcribe(
        file: UploadFile = File(...),  # noqa: B008
        model: str = Form(...),  # accepted but ignored — engine uses ASR_MODEL env
        language: Optional[str] = Form(None),
        prompt: Optional[str] = Form(None),
        response_format: Optional[str] = Form("json"),
        temperature: Optional[float] = Form(None),  # accepted but ignored
    ):
        if response_format not in {"json", "verbose_json", "text", "srt", "vtt"}:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid response_format '{response_format}'. Supported: json, verbose_json, text, srt, vtt",
            )
        audio = load_audio(file.file)

        if response_format in ("json", "verbose_json"):
            result_file = asr_model.transcribe(audio, "transcribe", language, prompt, False, False, {}, "json")
            result = json.loads(result_file.read())

            if response_format == "json":
                return JSONResponse({"text": result["text"]})

            segments = result.get("segments", [])
            duration = segments[-1]["end"] if segments else 0.0
            return JSONResponse(
                {
                    "task": "transcribe",
                    "language": result.get("language", language or ""),
                    "duration": duration,
                    "text": result["text"],
                    "segments": [
                        {"id": i, "start": s["start"], "end": s["end"], "text": s["text"]}
                        for i, s in enumerate(segments)
                    ],
                }
            )

        output_map = {"text": "txt", "srt": "srt", "vtt": "vtt"}
        output = output_map.get(response_format, "txt")
        result_file = asr_model.transcribe(audio, "transcribe", language, prompt, False, False, {}, output)
        return StreamingResponse(result_file, media_type="text/plain")

    return router
