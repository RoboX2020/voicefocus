# VoiceFocus

Isolate one voice from a two-person recording. Upload or record the mixed clip, give a few
seconds of the voice to keep as a reference, and get that voice isolated.

Live demo: https://prajval-2029--voicefocus-ui.us-east.modal.run
Portfolio: https://www.prajvalarora.com/projects/voicefocus

## How it works

Pretrained models only, no training:

1. SpeechBrain SepFormer (`speechbrain/sepformer-whamr16k`) separates the recording into
   two speaker tracks.
2. SpeechBrain ECAPA-TDNN (`speechbrain/spkrec-ecapa-voxceleb`) embeds the reference clip
   and each separated track, and the track with the highest cosine similarity is returned
   as the target speaker.

The web UI is Gradio; the app is deployed as a Gradio ASGI app on Modal's free tier, so it
works from any phone or laptop browser.

## Files

- `app.py` - Gradio UI and the separation + speaker-matching pipeline.
- `modal_app.py` - Modal deployment wrapper (builds the image, serves `app.py`).
- `test_run.py` - offline end-to-end check on the sample clips in `test/`; prints the
  speaker match scores.

## Run locally

```bash
pip install -r requirements.txt
python app.py
```

## Deploy on Modal

```bash
pip install modal
modal token new
modal deploy modal_app.py
```

## Limits

Built for two-speaker recordings; more voices or heavy overlap degrade the result, and a
noisy reference clip weakens the match. Speaker match scores are a ranking signal, not
proof of identity. The free host sleeps when idle, so the first visit can take about 40
seconds to wake. Audio is processed in memory for the request and not stored.
