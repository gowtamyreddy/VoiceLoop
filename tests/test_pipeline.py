import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from voiceloop.pipeline import VoiceLoop, REFUSAL_SPOKEN


class FakeSTT:
    def transcribe(self, path):
        return "What is RAPTOR?", 120.0


class FakeQA:
    def answer(self, question):
        if "swallow" in question:
            return {"status": "insufficient_evidence", "spans": [],
                    "citations": [], "answer_text": "",
                    "latency_ms": 300.0}
        return {"status": "answered", "spans": [{"text": "A tree."}],
                "citations": [{"doc_id": "2401.18059"}],
                "answer_text": "A tree.", "latency_ms": 300.0}


def fake_speak(text):
    return "/tmp/fake.mp3", 50.0


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.loop = VoiceLoop(FakeSTT(), FakeQA(), fake_speak)

    def test_audio_turn_books_all_stages(self):
        out = self.loop.process_audio("/tmp/q.wav")
        self.assertEqual(out["transcript"], "What is RAPTOR?")
        self.assertEqual(out["status"], "answered")
        self.assertEqual(out["latency_ms"]["stt"], 120.0)
        self.assertEqual(out["latency_ms"]["qa"], 300.0)
        self.assertEqual(out["latency_ms"]["tts"], 50.0)
        self.assertGreaterEqual(out["latency_ms"]["total"], 470.0)

    def test_refusal_speaks_refusal(self):
        out = self.loop.process_text("Tell me about a swallow")
        self.assertEqual(out["status"], "insufficient_evidence")
        self.assertEqual(out["audio_path"], "/tmp/fake.mp3")


if __name__ == "__main__":
    unittest.main()
