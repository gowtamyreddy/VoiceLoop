# VoiceLoop

Speak a question. Get a cited answer, spoken back.

VoiceLoop is a voice interface over EvidenceRAG. faster-whisper
transcribes the question locally, EvidenceRAG answers from 20+
research papers with document, page, and character citations (or
refuses, out loud, when the papers do not answer), and Edge TTS
speaks the answer. Every turn reports its latency per stage.

## Run it
Uses the EvidenceRAG venv (faster-whisper, FAISS, torch) plus
edge-tts:

    pip install edge-tts
    EVIDENCERAG_HOME=../evidencerag python app.py

Then open the local URL, allow the microphone, and ask about the
papers. Typing the question instead also works and skips the STT
stage.

## Status
V1: pipeline (STT, QA, TTS, latency ledger), Gradio UI, tests.
