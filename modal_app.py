# ============================================================================
# modal_app.py - puts VoiceFocus on the internet for free, using Modal.
#
# WHAT MODAL IS: a hosting service that runs your Python code on its own
# computers. The free tier is enough for this demo, which is why it was
# chosen. Deploy with:  modal deploy modal_app.py
#
# HOW THIS FILE FITS IN: the real app is app.py (same folder). This file
# only describes HOW to run app.py on Modal's servers:
#   1. Build a "container image" - a recipe for a fresh computer that has
#      Python, ffmpeg, the AI libraries, the two downloaded AI models, and a
#      copy of app.py.
#   2. Start a web server from that image that shows the same Gradio page.
#
# WHEN DEBUGGING: if the deployed site breaks but `python app.py` works
# locally, the difference is almost always something in the image recipe
# below (a missing system package or a different library version).
# ============================================================================

# the Modal service's Python library
import modal

app = modal.App("voicefocus")

# A one-line Python program (run once while building the image) that
# downloads both AI models ahead of time. Reason: without this, the first
# visitor would wait minutes while the models downloaded. Baking them into
# the image means every startup is fast.
MODEL_DL = (
    "from huggingface_hub import snapshot_download; "
    "snapshot_download('speechbrain/sepformer-whamr16k'); "
    "snapshot_download('speechbrain/spkrec-ecapa-voxceleb')"
)

# The image recipe, step by step (each line adds one layer):
image = (
    # start from a small Linux with Python 3.11
    modal.Image.debian_slim(python_version="3.11")
    # add the ffmpeg audio converter (app.py needs it)
    .apt_install("ffmpeg")
    .pip_install(
        "torch==2.7.1",
        "torchaudio==2.7.1",
        # CPU-only PyTorch: smaller and cheaper
        extra_index_url="https://download.pytorch.org/whl/cpu",
    )
    .pip_install("speechbrain==1.1.2", "gradio==5.50.0", "soundfile")
    # tell Hugging Face where to store the downloaded models
    .env({"HF_HUB_CACHE": "/models"})
    # run the pre-download described above
    .run_commands(f'python -c "{MODEL_DL}"')
    .add_local_file("/home/sandbox/voicefocus/app.py", "/root/voicefocus_app.py", copy=True)
    # ^ copies app.py from this computer into the image as voicefocus_app.py
)


# Settings for the web server that runs the app:
#   memory=4096      - 4 GB of RAM (the two AI models need room)
#   cpu=2            - 2 processor cores
#   timeout=900      - one run may take at most 15 minutes
#   max_containers=3 - at most 3 copies of the server running at once
#                      (keeps the free-tier bill at zero)
@app.function(image=image, memory=4096, cpu=2, timeout=900, max_containers=3)
# each copy can juggle up to 8 visitors at the same time
@modal.concurrent(max_inputs=8)
# this function returns a web app, not a single answer
@modal.asgi_app()
def ui():
    import sys

    # so Python can find voicefocus_app.py inside the image
    sys.path.insert(0, "/root")
    from fastapi import FastAPI
    from gradio import mount_gradio_app
    # this is the copy of app.py baked into the image
    import voicefocus_app as vf

    # Take the Gradio page (`vf.demo` from app.py) and attach it to a small
    # FastAPI web server at the root address ("/") so visitors can open it.
    return mount_gradio_app(FastAPI(), vf.demo, path="/")
