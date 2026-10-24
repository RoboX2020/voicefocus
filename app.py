import os
import subprocess
import tempfile
import threading
import time

import numpy as np
import torch
import torchaudio
import gradio as gr

TARGET_SR = 16000
MAX_SECONDS = 90

_sep = None
_spk = None
_lock = threading.Lock()


def load_models():
    global _sep, _spk
    with _lock:
        if _sep is None or _spk is None:
            from speechbrain.inference.separation import SepformerSeparation
            from speechbrain.inference.speaker import EncoderClassifier
            tmp = tempfile.gettempdir()
            _sep = SepformerSeparation.from_hparams(
                source="speechbrain/sepformer-whamr16k",
                savedir=os.path.join(tmp, "voicefocus_sepformer"),
                run_opts={"device": "cpu"},
            )
            _spk = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir=os.path.join(tmp, "voicefocus_ecapa"),
                run_opts={"device": "cpu"},
            )
    return _sep, _spk


def load_mono(path, sr=TARGET_SR):
    try:
        out = tempfile.mktemp(suffix=".wav")
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-i", path, "-ac", "1", "-ar", str(sr), out],
            check=True, capture_output=True,
        )
        wav, _ = torchaudio.load(out)
        os.unlink(out)
    except Exception:
        wav, fs = torchaudio.load(path)
        wav = wav.mean(dim=0, keepdim=True)
        if fs != sr:
            wav = torchaudio.functional.resample(wav, fs, sr)
    return wav


def _save_wav(wav, fname):
    d = tempfile.mkdtemp()
    p = os.path.join(d, fname)
    peak = wav.abs().max()
    if peak > 0:
        wav = wav / peak * 0.9
    torchaudio.save(p, wav.detach().cpu().unsqueeze(0).float(), TARGET_SR)
    return p


def isolate_core(mix_path, ref_path, start_s, end_s):
    sep, spk = load_models()
    mix = load_mono(mix_path)
    total = mix.shape[1]
    dur = total / TARGET_SR
    if dur > MAX_SECONDS:
        raise gr.Error(
            f"This clip is {dur:.0f}s. The free demo handles up to {MAX_SECONDS}s - trim the recording and try again."
        )
    t0 = time.time()
    est = sep.separate_batch(mix)  # [1, T, 2]
    n_src = est.shape[-1]

    if ref_path:
        ref = load_mono(ref_path)
        if ref.shape[1] < TARGET_SR:
            raise gr.Error("The reference clip is too short - record at least 2-3 seconds of the voice to keep.")
        e_ref = spk.encode_batch(ref).flatten()
        sims = []
        for i in range(n_src):
            e_i = spk.encode_batch(est[..., i]).flatten()
            sims.append(torch.nn.functional.cosine_similarity(e_ref, e_i, dim=0).item())
        target = int(np.argmax(sims))
        match_txt = (
            f"Matched the reference to voice {target + 1} "
            f"(similarity {sims[target]:.2f} vs {sims[1 - target]:.2f} for the other)."
        )
    else:
        target = 0
        match_txt = "No reference clip - returned voice 1 and voice 2. Add a reference to pick a specific person."

    s = float(start_s or 0)
    e = float(end_s or 0)
    s_i = max(0, int(s * TARGET_SR))
    e_i = min(total, int(e * TARGET_SR)) if e > s else total
    if s_i >= e_i:
        raise gr.Error("That time range is empty - check the start and end seconds.")

    tgt = est[0, s_i:e_i, target]
    oth = est[0, s_i:e_i, 1 - target]
    out_t = _save_wav(tgt, "isolated_voice.wav")
    out_o = _save_wav(oth, "other_voice.wav")
    kept = (e_i - s_i) / TARGET_SR
    status = (
        f"{match_txt} Kept {kept:.1f}s of {dur:.1f}s. "
        f"Processed in {time.time() - t0:.0f}s. "
        "Both outputs are the full separated tracks - re-run with a different time range to focus elsewhere."
    )
    return out_t, out_o, status


def isolate(mix_path, ref_path, start_s, end_s):
    if not mix_path:
        raise gr.Error("Add the recording first.")
    return isolate_core(mix_path, ref_path, start_s, end_s)


theme = gr.themes.Soft(primary_hue="orange", neutral_hue="zinc")

with gr.Blocks(theme=theme, title="VoiceFocus") as demo:
    gr.Markdown(
        "# VoiceFocus\n"
        "Pull one voice out of a crowded recording. Add the recording, then a few seconds "
        "of just the voice you want to keep as a reference.\n\n"
        "Works best with two clear speakers on clips under 90 seconds. "
        "The first run downloads the AI models and can take a couple of minutes."
    )
    with gr.Row():
        mix_in = gr.Audio(
            label="1. The recording (two people talking)",
            sources=["upload", "microphone"],
            type="filepath",
        )
        ref_in = gr.Audio(
            label="2. Reference: a few seconds of the voice to keep",
            sources=["upload", "microphone"],
            type="filepath",
        )
    with gr.Accordion("Keep only part of it (optional)", open=False):
        with gr.Row():
            start_in = gr.Number(value=0, label="Start (seconds)")
            end_in = gr.Number(value=0, label="End (seconds, 0 = to the end)")
    go = gr.Button("Isolate the voice", variant="primary")
    status_out = gr.Markdown()
    tgt_out = gr.Audio(label="Isolated: the reference voice", type="filepath")
    oth_out = gr.Audio(label="The other voice", type="filepath")
    go.click(isolate, [mix_in, ref_in, start_in, end_in], [tgt_out, oth_out, status_out])
    gr.Markdown(
        "Built with pretrained SpeechBrain models (SepFormer separation + ECAPA speaker matching). "
        "It separates two voices and picks the one that sounds like your reference clip."
    )

if __name__ == "__main__":
    demo.launch()
