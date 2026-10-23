"""Text to speech, local first.

Primary: Piper, a local neural TTS (open model, runs on CPU, works
offline). Fallback: Edge TTS when Piper or its voice file is
unavailable. Returns (path, ms) or (None, ms) when both fail, so
the pipeline degrades to text instead of breaking.
"""
import os
import tempfile
import time
import wave

_DIR = os.path.dirname(os.path.abspath(__file__))
VOICE_MODEL = os.environ.get(
    "VOICELOOP_PIPER_MODEL",
    os.path.join(_DIR, "..", "voices", "en_US-amy-medium.onnx"))
_voice = None


def _piper_speak(text, out_path):
    global _voice
    from piper import PiperVoice
    if _voice is None:
        _voice = PiperVoice.load(VOICE_MODEL,
                                 config_path=VOICE_MODEL + ".json")
    with wave.open(out_path, "wb") as wav_file:
        if hasattr(_voice, "synthesize_wav"):
            _voice.synthesize_wav(text, wav_file)
        else:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(_voice.config.sample_rate)
            _voice.synthesize(text, wav_file)


def _edge_speak(text, out_path):
    import asyncio

    import edge_tts

    async def _run():
        voice = os.environ.get("VOICELOOP_TTS_VOICE",
                               "en-US-AriaNeural")
        await edge_tts.Communicate(text, voice).save(out_path)

    asyncio.run(_run())


def speak(text, out_path=None):
    t0 = time.perf_counter()

    def _ms():
        return round((time.perf_counter() - t0) * 1000, 1)

    if out_path is None:
        fd, out_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
    for backend in (_piper_speak, _edge_speak):
        try:
            backend(text, out_path)
            if os.path.exists(out_path) and os.path.getsize(out_path) > 100:
                return out_path, _ms()
        except Exception:
            continue
    return None, _ms()
