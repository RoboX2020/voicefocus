import modal

app = modal.App("voicefocus")

MODEL_DL = (
    "from huggingface_hub import snapshot_download; "
    "snapshot_download('speechbrain/sepformer-whamr16k'); "
    "snapshot_download('speechbrain/spkrec-ecapa-voxceleb')"
)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install(
        "torch==2.7.1",
        "torchaudio==2.7.1",
        extra_index_url="https://download.pytorch.org/whl/cpu",
    )
    .pip_install("speechbrain==1.1.2", "gradio==5.50.0", "soundfile")
    .env({"HF_HUB_CACHE": "/models"})
    .run_commands(f'python -c "{MODEL_DL}"')
    .add_local_file("/home/sandbox/voicefocus/app.py", "/root/voicefocus_app.py", copy=True)
)


@app.function(image=image, memory=4096, cpu=2, timeout=900, max_containers=3)
@modal.concurrent(max_inputs=8)
@modal.asgi_app()
def ui():
    import sys

    sys.path.insert(0, "/root")
    from fastapi import FastAPI
    from gradio import mount_gradio_app
    import voicefocus_app as vf

    return mount_gradio_app(FastAPI(), vf.demo, path="/")
