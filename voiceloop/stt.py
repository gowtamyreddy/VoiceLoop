"""Speech to text with faster-whisper (base, int8, CPU).

Mirrors the EvidenceRAG approach: decode audio to a 16 kHz mono
float array with PyAV and hand the array to Whisper, avoiding
faster-whisper's file decode path.
"""
import time

import numpy as np


def _decode_to_array(audio_path):
    import av
    container = av.open(audio_path)
    stream = container.streams.audio[0]
    resampler = av.AudioResampler(format="flt", layout="mono",
                                  rate=16000)
    chunks = []
    for frame in container.decode(stream):
        for out in resampler.resample(frame):
            chunks.append(out.to_ndarray().reshape(-1))
    if chunks:
        return np.concatenate(chunks).astype(np.float32)
    return np.zeros(0, dtype=np.float32)


class WhisperSTT:
    def __init__(self, model_size="base"):
        self.model_size = model_size
        self._model = None

    def _load(self):
        if self._model is None:
            from faster_whisper import WhisperModel
            self._model = WhisperModel(self.model_size, device="cpu",
                                       compute_type="int8")
        return self._model

    def transcribe(self, audio_path):
        t0 = time.perf_counter()
        audio = _decode_to_array(audio_path)
        segments, _info = self._load().transcribe(audio, beam_size=1)
        text = " ".join(s.text.strip() for s in segments).strip()
        ms = round((time.perf_counter() - t0) * 1000, 1)
        return text, ms
