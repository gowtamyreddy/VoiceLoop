"""VoiceLoop demo: ask out loud, get a cited answer spoken back."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gradio as gr

from voiceloop.stt import WhisperSTT
from voiceloop.qa import EvidenceRAGQA
from voiceloop.tts import speak
from voiceloop.pipeline import VoiceLoop

EVR_HOME = os.environ.get(
    "EVIDENCERAG_HOME",
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "..", "evidencerag"))
LOOP = VoiceLoop(WhisperSTT(), EvidenceRAGQA(EVR_HOME), speak)


def _fmt_answer(out):
    if out["status"] != "answered":
        return ("**No answer in the papers.** VoiceLoop refused rather "
                "than guess, and said so out loud.")
    parts = ["**Answer, in the paper's own words:**", "",
             "> " + out["answer_text"], ""]
    for cit in out["citations"][:3]:
        parts.append("<small>Source: arXiv %s, page %s, match score "
                     "%s</small>" % (cit["doc_id"], cit["page"],
                                     round(cit["score"], 3)))
    return "\n".join(parts)


def _latency_rows(out):
    lat = out["latency_ms"]
    return [["Speech to text", "%.0f ms" % lat["stt"]],
            ["Answer (retrieve, rerank, quote)", "%.0f ms" % lat["qa"]],
            ["Text to speech", "%.0f ms" % lat["tts"]],
            ["Total turn", "%.0f ms" % lat["total"]]]


def ask(audio_path, text):
    if audio_path:
        out = LOOP.process_audio(audio_path)
    elif text and text.strip():
        out = LOOP.process_text(text.strip())
    else:
        return "", "_Record a question or type one first._", None, []
    return (out["transcript"], _fmt_answer(out), out["audio_path"],
            _latency_rows(out))


HEADER = """# VoiceLoop: ask out loud, hear the evidence

Speak a question about retrieval-augmented generation. VoiceLoop
transcribes it locally with Whisper, answers from real research
papers with citations through EvidenceRAG, and speaks the answer
back. Every stage is timed, and if the papers cannot answer, it
refuses out loud instead of guessing."""

with gr.Blocks(title="VoiceLoop: Spoken Cited Answers",
               theme=gr.themes.Soft(primary_hue="indigo",
                                    neutral_hue="slate"),
               css=".footer {display: none !important}") as demo:
    gr.Markdown(HEADER)
    audio_in = gr.Audio(sources=["microphone", "upload"],
                        type="filepath", label="Your voice")
    text_in = gr.Textbox(label="Or type the question instead",
                         placeholder="How does RAPTOR organize "
                                     "retrieved text?")
    ask_btn = gr.Button("Ask", variant="primary")
    transcript = gr.Textbox(label="What VoiceLoop heard", interactive=False)
    answer_md = gr.Markdown("_Your answer will appear here._")
    audio_out = gr.Audio(label="Spoken answer", type="filepath")
    latency = gr.Dataframe(headers=["Stage", "Time"],
                           datatype=["str", "str"],
                           label="Latency ledger", interactive=False)
    gr.Markdown("<small>Local and free: Whisper and the answer "
                "pipeline run on CPU with open-weight models. Speech "
                "synthesis uses the free Edge TTS voice.</small>")
    ask_btn.click(ask, inputs=[audio_in, text_in],
                  outputs=[transcript, answer_md, audio_out, latency],
                  api_name="ask")

if __name__ == "__main__":
    try:  # warm the models once so the first visitor is not slow
        LOOP.stt._load()
        LOOP.qa.answer("What is retrieval augmented generation?")
    except Exception:
        pass
    demo.queue()
    demo.launch()
