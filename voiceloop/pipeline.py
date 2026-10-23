"""The VoiceLoop turn: audio in, cited spoken answer out."""
REFUSAL_SPOKEN = ("The papers I have access to do not answer that "
                  "question, so I will not guess.")


class VoiceLoop:
    def __init__(self, stt, qa, tts_speak):
        self.stt = stt
        self.qa = qa
        self.tts_speak = tts_speak

    def process_text(self, question):
        result = self.qa.answer(question)
        spoken = (result["answer_text"] if result["status"] == "answered"
                  else REFUSAL_SPOKEN)
        audio_path, tts_ms = self.tts_speak(spoken[:600])
        total = round(result["latency_ms"] + tts_ms, 1)
        return {
            "transcript": question,
            "status": result["status"],
            "answer_text": result["answer_text"],
            "citations": result["citations"],
            "audio_path": audio_path,
            "latency_ms": {"stt": 0.0, "qa": result["latency_ms"],
                           "tts": tts_ms, "total": total},
        }

    def process_audio(self, audio_path):
        transcript, stt_ms = self.stt.transcribe(audio_path)
        out = self.process_text(transcript)
        out["latency_ms"]["stt"] = stt_ms
        out["latency_ms"]["total"] = round(
            stt_ms + out["latency_ms"]["qa"] + out["latency_ms"]["tts"], 1)
        return out
