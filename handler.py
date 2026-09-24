"""RunPod serverless handler: Wan 2.2 TI2V-5B image-to-video via the official CLI.

Recreated 2026-09-23 from the RunPod serverless quickstart contract
(handler(job) / job["input"] / JSON-serializable return dict /
runpod.serverless.progress_update / runpod.serverless.start) and the
official Wan-Video/Wan2.2 generate.py CLI (--task ti2v-5B), which is the only
inference path the model authors support for this checkpoint. The earlier
diffusers-based handler was dropped: the TI2V-5B model page lists Diffusers
integration as unfinished, so generate.py is the path that actually runs.

Job input:
    {
      "image_url": "https://...",   # or "image_base64"
      "prompt": "slow cinematic push-in ...",
      "seconds": 5,                 # clip length (default 5)
      "steps": 30,                  # sampling steps (default 30)
      "guidance_scale": 5.0,        # default 5.0
      "seed": 42,                   # optional; -1 (default) = random
      "portrait": true              # default true -> 704*1280; false -> 1280*704
    }

Output:
    {
      "video_base64": "<mp4 bytes>",
      "actual": {"fps": 24.0, "frames": 121, "width": 704,
                 "height": 1280, "seconds": 5.04},
      "render_seconds": 812.3,
      "engine": "wan2.2-official-cli",
      "model": "Wan-AI/Wan2.2-TI2V-5B",
      "config": {...echo of effective settings...}
    }

Weights live on the attached network volume (/runpod-volume). HF_HOME is
redirected there too. The first job seeds the volume via snapshot_download
(one-time download); later jobs reuse it. Each job renders into its own
temporary directory, so concurrent jobs cannot collide and cleanup is
automatic even when encoding fails.
"""
import base64
import io
import logging
import os
import subprocess
import tempfile
import time

# Redirect BEFORE any huggingface import so downloads land on the volume.
os.environ.setdefault("HF_HOME", "/runpod-volume/hf-cache")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import cv2  # noqa: E402
import requests  # noqa: E402
import runpod  # noqa: E402
from PIL import Image  # noqa: E402

logging.basicConfig(level=logging.INFO, format="[handler] %(message)s")
log = logging.getLogger(__name__)

MODEL_ID = os.environ.get("WAN22_MODEL_ID", "Wan-AI/Wan2.2-TI2V-5B")
MODEL_DIR = os.environ.get("WAN22_MODEL_DIR", "/runpod-volume/wan22-ti2v-5b")
WAN22_DIR = os.environ.get("WAN22_DIR", "/opt/Wan2.2")
GENERATE_PY = os.path.join(WAN22_DIR, "generate.py")

# Used only to size frame_num (Wan needs 4n+1 frames). The real fps of the
# output file is probed afterwards and reported in the result.
ASSUMED_FPS = 24
# Backstop for one render; RunPod's execution timeout (3600s) is the outer one.
RENDER_TIMEOUT = int(os.environ.get("WAN22_RENDER_TIMEOUT", "3000"))

if not os.path.isfile(GENERATE_PY):
    raise RuntimeError(
        f"generate.py not found at {GENERATE_PY} - the image build did not "
        "clone the Wan2.2 repo correctly."
    )


def progress(job, message):
    """Best-effort progress update; never let it fail the job."""
    try:
        runpod.serverless.progress_update(job, message)
    except Exception as exc:  # noqa: BLE001
        log.warning("progress_update failed: %s", exc)


def ensure_weights(job):
    """Download weights to the network volume. Idempotent; skips existing."""
    from huggingface_hub import snapshot_download

    progress(job, "ensuring model weights on the volume ...")
    t0 = time.time()
    snapshot_download(repo_id=MODEL_ID, local_dir=MODEL_DIR)
    log.info("weights ready at %s (%.0fs)", MODEL_DIR, time.time() - t0)


def load_input_image(job_input, dest_path):
    """Fetch image_url or image_base64 and save it as a PNG for generate.py."""
    if job_input.get("image_base64"):
        raw = base64.b64decode(job_input["image_base64"])
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    else:
        url = job_input.get("image_url")
        if not url:
            raise ValueError("job input needs image_url or image_base64")
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        img = Image.open(io.BytesIO(r.content)).convert("RGB")
    img.save(dest_path, "PNG")
    log.info("input image: %dx%d -> %s", img.size[0], img.size[1], dest_path)
    return dest_path


def run_generate(cmd, job, timeout):
    """Run generate.py, streaming its log to the worker logs.

    Raises RuntimeError with the tail of the log on failure or timeout.
    """
    log.info("running: %s", " ".join(cmd))
    tail = []
    proc = subprocess.Popen(
        cmd,
        cwd=WAN22_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    deadline = time.time() + timeout
    last_ping = time.time()
    try:
        for line in proc.stdout:
            line = line.rstrip()
            if line:
                tail.append(line)
                del tail[:-200]
                log.info("[generate.py] %s", line)
            now = time.time()
            if now > deadline:
                proc.kill()
                raise RuntimeError(
                    f"generate.py timed out after {timeout}s; "
                    f"last log: {' | '.join(tail[-5:])}"
                )
            if now - last_ping > 180:
                progress(job, f"rendering ... {tail[-1][:100]}" if tail else "rendering ...")
                last_ping = now
        rc = proc.wait(timeout=max(1, int(deadline - time.time())))
    except Exception:
        proc.kill()
        raise
    if rc != 0:
        raise RuntimeError(
            f"generate.py exited with code {rc}; "
            f"last log: {' | '.join(tail[-15:])}"
        )
    return "\n".join(tail)


def probe_video(path):
    """Read real fps / frame count / dimensions off the rendered file."""
    cap = cv2.VideoCapture(path)
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if not fps or not frames:
        raise RuntimeError(f"could not probe rendered video at {path}")
    return {
        "fps": round(fps, 2),
        "frames": frames,
        "width": width,
        "height": height,
        "seconds": round(frames / fps, 2),
    }


def build_cmd(*, image_path, save_path, prompt, size, frames, steps,
              guide_scale, seed):
    # NOTE: --t5_cpu is a store_true flag in generate.py: it takes NO value.
    # Passing "--t5_cpu True" makes argparse die with "unrecognized arguments".
    # --offload_model uses str2bool, so it DOES take a string value.
    return [
        "python", GENERATE_PY,
        "--task", "ti2v-5B",
        "--ckpt_dir", MODEL_DIR,
        "--image", image_path,
        "--prompt", prompt,
        "--size", size,
        "--frame_num", str(frames),
        "--sample_steps", str(steps),
        "--sample_guide_scale", str(guide_scale),
        "--base_seed", str(seed),
        "--offload_model", "True",   # spill to CPU between forwards: 24GB-safe
        "--t5_cpu",                  # bare flag: keep the text encoder off GPU
        "--save_file", save_path,
    ]


def handler(job):
    t0 = time.time()
    inp = job["input"] or {}
    job_id = job.get("id", "?")
    log.info("job %s input keys: %s", job_id, sorted(inp.keys()))

    prompt = inp.get("prompt")
    if not prompt:
        raise ValueError("job input needs a 'prompt'")
    # The official CLI has no negative-prompt flag; ignore if supplied.
    if inp.get("negative_prompt"):
        log.info("note: negative_prompt is not supported by the official "
                 "CLI and will be ignored")

    seconds = int(inp.get("seconds", 5))
    steps = int(inp.get("steps", 30))
    guide_scale = float(inp.get("guidance_scale", 5.0))
    seed = int(inp.get("seed", -1))
    portrait = bool(inp.get("portrait", True))

    # Wan requires 4n+1 frames; snap down to the nearest valid count.
    frames = seconds * ASSUMED_FPS + 1
    while (frames - 1) % 4 != 0:
        frames -= 1

    # ti2v-5B only accepts 704*1280 / 1280*704 (SUPPORTED_SIZES in
    # wan/configs/__init__.py). Anything else dies with "Unsupport size".
    size = "704*1280" if portrait else "1280*704"
    size_fallback_used = False

    progress(job, "starting job")
    ensure_weights(job)

    with tempfile.TemporaryDirectory(prefix="wan22-") as workdir:
        image_path = os.path.join(workdir, "input.png")
        save_path = os.path.join(workdir, "out.mp4")
        load_input_image(inp, image_path)

        progress(job, f"rendering {seconds}s at {size} ({steps} steps) ...")
        cmd = build_cmd(
            image_path=image_path, save_path=save_path, prompt=prompt,
            size=size, frames=frames, steps=steps,
            guide_scale=guide_scale, seed=seed,
        )
        try:
            run_generate(cmd, job, RENDER_TIMEOUT)
        except RuntimeError as exc:
            # "704*1280" is valid per SUPPORTED_SIZES, but if this generate.py
            # build disagrees, retry once in landscape instead of failing.
            if "Unsupport size" in str(exc) and portrait:
                log.warning("portrait size rejected; retrying 1280*704")
                size = "1280*704"
                size_fallback_used = True
                cmd = build_cmd(
                    image_path=image_path, save_path=save_path,
                    prompt=prompt, size=size, frames=frames, steps=steps,
                    guide_scale=guide_scale, seed=seed,
                )
                run_generate(cmd, job, RENDER_TIMEOUT)
            else:
                raise

        actual = probe_video(save_path)
        log.info("rendered: %s", actual)
        with open(save_path, "rb") as f:
            video_b64 = base64.b64encode(f.read()).decode()

    render_seconds = round(time.time() - t0, 1)
    log.info("job %s done in %ss", job_id, render_seconds)
    return {
        "video_base64": video_b64,
        "actual": actual,
        "render_seconds": render_seconds,
        "engine": "wan2.2-official-cli",
        "model": MODEL_ID,
        "config": {
            "seconds": seconds,
            "size": size,
            "size_fallback_used": size_fallback_used,
            "frames": frames,
            "steps": steps,
            "guidance_scale": guide_scale,
            "seed": seed,
            "portrait": portrait,
        },
    }


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
