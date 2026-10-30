# ============================================================================
# VoiceFocus - the whole app lives in this one file.
#
# WHAT IT DOES, IN ONE SENTENCE:
#   You give it a recording of two people talking, plus a few seconds of just
#   the voice you want to keep, and it hands back two new audio files:
#   "isolated_voice.wav" (only the voice you picked) and "other_voice.wav"
#   (everyone else).
#
# HOW IT WORKS, BIG PICTURE:
#   1. An AI model called SepFormer splits the mixed recording into two
#      separate voice tracks. It does not know WHO is who - it just
#      untangles the sounds.
#   2. A second AI model called ECAPA listens to your short reference clip
#      and to each separated track, then picks the track that sounds like
#      the same person.
#   3. A small web page (built with the Gradio library) lets a person upload
#      the audio files and press one button to run all of this.
#
# THIS ONE FILE IS USED TWO WAYS:
#   - Locally: run `python app.py` and the web page opens on your computer.
#   - On the internet: modal_app.py copies this file onto Modal's servers
#     (a hosting service) so anyone can try the demo for free.
#
# IF SOMETHING BREAKS, THE USUAL SUSPECTS:
#   - "ffmpeg not found": install ffmpeg (this app uses it to convert audio).
#   - First run is very slow: the two AI models (hundreds of MB) download
#     once into your temp folder, then they are cached and it is fast.
#   - Out of memory: the models run on CPU here, so close heavy programs.
# ============================================================================

# built-in: file paths and deleting temp files
import os
# built-in: runs the ffmpeg program as a separate process
import subprocess
# built-in: creates throwaway files/folders the OS cleans up
import tempfile
# built-in: stops two visitors loading the AI models at the same time
import threading
# built-in: measures how long the AI processing took
import time

# arrays/math: used to find the highest similarity score
import numpy as np
# PyTorch: the AI framework both models run on
import torch
# audio loading, saving and resampling, built on PyTorch
import torchaudio
# Gradio: builds the web page and handles the uploads
import gradio as gr

# Every audio clip is converted to this sample rate: 16,000 measurements per
# second. Both AI models were trained on 16 kHz audio, so feeding them
# anything else (like the 44.1 kHz music apps use) would confuse them.
TARGET_SR = 16000

# The free demo refuses recordings longer than this many seconds. Reason:
# the demo runs on a free server, and one very long clip would keep the
# (shared) computer busy for minutes and block everyone else.
MAX_SECONDS = 90

# These two variables will hold the loaded AI models. They start as None
# ("nothing loaded yet") because loading takes a while; the app only loads
# them on the first request and then reuses them for every later request.
# will hold the SepFormer "voice splitter" model
_sep = None
# will hold the ECAPA "voice matcher" model
_spk = None

# A lock works like a bathroom key: only one piece of code can hold it at a
# time. If two visitors press the button at the same moment, they must not
# both start downloading the models at once - so load_models() takes this
# key first, and the second visitor simply waits their turn.
_lock = threading.Lock()


# Loads the two AI models (downloading them the first time) and returns them.
# "Lazy loading": nothing happens until the first person actually uses the
# app, so starting the program stays fast.
def load_models():
    global _sep, _spk
    with _lock:
        # If another visitor already loaded the models while we were waiting
        # for the lock, skip straight to reusing them.
        if _sep is None or _spk is None:
            from speechbrain.inference.separation import SepformerSeparation
            from speechbrain.inference.speaker import EncoderClassifier
            tmp = tempfile.gettempdir()
            # Model 1: SepFormer, trained to split a two-speaker mixture
            # into two separate voice tracks. Downloads into the temp folder
            # so a reboot cleans it up. Runs on CPU (no graphics card needed).
            _sep = SepformerSeparation.from_hparams(
                source="speechbrain/sepformer-whamr16k",
                savedir=os.path.join(tmp, "voicefocus_sepformer"),
                run_opts={"device": "cpu"},
            )
            # Model 2: ECAPA, trained to turn a voice clip into a numeric
            # "voice fingerprint". Two clips from the same person get similar
            # fingerprints, which is how we find the right voice.
            _spk = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir=os.path.join(tmp, "voicefocus_ecapa"),
                run_opts={"device": "cpu"},
            )
    return _sep, _spk


# Reads an audio file and returns it as one channel (mono) at 16 kHz, ready
# for the AI models. `path` is the uploaded file; `sr` is the target rate.
def load_mono(path, sr=TARGET_SR):
    try:
        # Plan A: ask ffmpeg (a separate program) to convert the file to a
        # mono 16 kHz WAV. This handles every format: mp3, m4a, webm, etc.
        out = tempfile.mktemp(suffix=".wav")
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-i", path, "-ac", "1", "-ar", str(sr), out],
            check=True, capture_output=True,
        )
        wav, _ = torchaudio.load(out)
        # delete the converted temp copy; we have it in memory
        os.unlink(out)
    except Exception:
        # Plan B: ffmpeg is missing or choked, so load the file directly and
        # fix it up ourselves: average the channels into one (mono), then
        # resample (stretch/squeeze) to 16 kHz if needed.
        wav, fs = torchaudio.load(path)
        wav = wav.mean(dim=0, keepdim=True)
        if fs != sr:
            wav = torchaudio.functional.resample(wav, fs, sr)
    return wav


# Saves a sound to a new temp .wav file and returns its path.
# Gradio needs a real file on disk to offer a play/download button.
def _save_wav(wav, fname):
    d = tempfile.mkdtemp()
    p = os.path.join(d, fname)
    # "Peak normalization": find the loudest moment and scale the whole clip
    # so that moment is at 90% volume. Without this, a too-loud clip would
    # distort (clip) when saved, and a too-quiet one would be hard to hear.
    peak = wav.abs().max()
    if peak > 0:
        wav = wav / peak * 0.9
    torchaudio.save(p, wav.detach().cpu().unsqueeze(0).float(), TARGET_SR)
    return p


# The main pipeline: everything the button press actually runs.
#   mix_path: the recording of two people talking (required)
#   ref_path: a few seconds of just the voice to keep (optional)
#   start_s / end_s: optional trim - only keep this part of the result
# Returns: (isolated voice file, the other voice file, status message)
def isolate_core(mix_path, ref_path, start_s, end_s):
    sep, spk = load_models()
    mix = load_mono(mix_path)
    # number of audio measurements in the whole clip
    total = mix.shape[1]
    # clip length in seconds
    dur = total / TARGET_SR
    if dur > MAX_SECONDS:
        # gr.Error shows the message on the web page instead of crashing.
        raise gr.Error(
            f"This clip is {dur:.0f}s. The free demo handles up to {MAX_SECONDS}s - trim the recording and try again."
        )
    t0 = time.time()
    # STEP 1: split the mixture into separate voice tracks.
    # The result has shape [1, T, 2]: one batch, T time steps, 2 voices.
    est = sep.separate_batch(mix)  # [1, T, 2]
    n_src = est.shape[-1]

    if ref_path:
        # STEP 2: work out WHICH of the separated voices is the one to keep.
        ref = load_mono(ref_path)
        if ref.shape[1] < TARGET_SR:
            raise gr.Error("The reference clip is too short - record at least 2-3 seconds of the voice to keep.")
        # Turn the reference clip into a voice fingerprint.
        e_ref = spk.encode_batch(ref).flatten()
        sims = []
        for i in range(n_src):
            # Fingerprint each separated voice, then compare it to the
            # reference. Cosine similarity gives a score near 1.0 for "same
            # person" and near 0 for "different person".
            e_i = spk.encode_batch(est[..., i]).flatten()
            sims.append(torch.nn.functional.cosine_similarity(e_ref, e_i, dim=0).item())
        # argmax = "which position has the highest score": that voice wins.
        target = int(np.argmax(sims))
        match_txt = (
            f"Matched the reference to voice {target + 1} "
            f"(similarity {sims[target]:.2f} vs {sims[1 - target]:.2f} for the other)."
        )
    else:
        # No reference clip given: just return both voices and let the
        # listener pick. Voice 1 is called "the voice" for now.
        target = 0
        match_txt = "No reference clip - returned voice 1 and voice 2. Add a reference to pick a specific person."

    # STEP 3: work out the trim range. The page sends seconds; the audio is
    # stored as sample numbers, so multiply by the sample rate to convert.
    s = float(start_s or 0)
    e = float(end_s or 0)
    s_i = max(0, int(s * TARGET_SR))
    e_i = min(total, int(e * TARGET_SR)) if e > s else total
    if s_i >= e_i:
        raise gr.Error("That time range is empty - check the start and end seconds.")

    # STEP 4: cut out the requested part of both separated voices.
    # the voice to keep
    tgt = est[0, s_i:e_i, target]
    # the other voice
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


# The function the web page button actually calls. It only checks that a
# recording was uploaded, then hands the real work to isolate_core().
def isolate(mix_path, ref_path, start_s, end_s):
    if not mix_path:
        raise gr.Error("Add the recording first.")
    return isolate_core(mix_path, ref_path, start_s, end_s)


# --------------------------------------------------------------------------
# WEB PAGE LAYOUT (everything below builds what the visitor sees)
# --------------------------------------------------------------------------

# The color theme of the page: orange highlights on a zinc-gray base.
theme = gr.themes.Soft(primary_hue="orange", neutral_hue="zinc")

# gr.Blocks is Gradio's page builder; everything indented under `with` is one
# piece of the page, top to bottom.
with gr.Blocks(theme=theme, title="VoiceFocus") as demo:
    # The instructions at the top of the page.
    gr.Markdown(
        "# VoiceFocus\n"
        "Pull one voice out of a crowded recording. Add the recording, then a few seconds "
        "of just the voice you want to keep as a reference.\n\n"
        "Works best with two clear speakers on clips under 90 seconds. "
        "The first run downloads the AI models and can take a couple of minutes."
    )
    # The two upload boxes, side by side in one row.
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
    # A folded-away section for the optional "keep only part of it" trimming.
    with gr.Accordion("Keep only part of it (optional)", open=False):
        with gr.Row():
            start_in = gr.Number(value=0, label="Start (seconds)")
            end_in = gr.Number(value=0, label="End (seconds, 0 = to the end)")
    # The big button, the status line, and the two result players.
    go = gr.Button("Isolate the voice", variant="primary")
    status_out = gr.Markdown()
    tgt_out = gr.Audio(label="Isolated: the reference voice", type="filepath")
    oth_out = gr.Audio(label="The other voice", type="filepath")
    # Wiring: when the button is clicked, run `isolate` with the 4 inputs and
    # put its 3 return values into the 3 output boxes, in this order.
    go.click(isolate, [mix_in, ref_in, start_in, end_in], [tgt_out, oth_out, status_out])
    gr.Markdown(
        "Built with pretrained SpeechBrain models (SepFormer separation + ECAPA speaker matching). "
        "It separates two voices and picks the one that sounds like your reference clip."
    )

# This only runs when the file is started directly (`python app.py`), not
# when modal_app.py imports it. It starts the web server and opens the page.
if __name__ == "__main__":
    demo.launch()
