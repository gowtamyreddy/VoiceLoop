# VoiceLoop: plan

## One line
Ask a question out loud. VoiceLoop transcribes it, answers from real
papers with citations, and speaks the answer back, timing every stage.

## Goal
Make Meganaa's voice AI experience visible: speech in, grounded
answer, speech out, with an honest latency budget per stage.

## The gap it fills
Most voice demos are a microphone wired to a chatbot: no sources, no
refusals, no latency accountability. VoiceLoop routes speech into
EvidenceRAG, so answers carry document, page, and character
citations, unanswerable questions are refused out loud, and every
stage reports what it cost in time.

## Pipeline
1. Speech to text: faster-whisper (base, int8, CPU), the same setup
   already proven inside EvidenceRAG.
2. Answer: EvidenceRAG live (hybrid FAISS + BM25 + RRF retrieval,
   cross-encoder rerank, extractive quoted answer with citations).
3. Text to speech: Piper, a local neural voice (free, offline,
   open model). Edge TTS is an optional fallback. Falls back to text
   only if synthesis is unavailable.
4. Latency ledger: stt_ms, qa_ms, tts_ms, total_ms on every turn.

## Latency budget (targets, CPU)
- STT for a short question: under 2,000 ms
- QA (retrieve, rerank, answer): under 3,000 ms
- TTS for a short answer: under 2,500 ms
- Full loop: under 8,000 ms

## Not in V1
- No wake word, no streaming partial transcripts, no phone calls.
  One clean turn at a time, measured honestly.

## Role split
- Boone: pipeline, UI, tests, demo.
- Meganaa: the questions she demos with, voice and pacing judgment,
  README story, interview explanation.
