# RunPod serverless worker: Wan 2.2 TI2V-5B via the official Wan2.2 CLI.
# Repo-root layout: Dockerfile, handler.py, requirements.txt side by side.

# Pinned 2026-09-23: RunPod removed the old 2.4.0-py3.11-cuda12.4.1-devel tag
# scheme (their whole tag set rotated ~2026-09-23 midday). This tag is current
# on Docker Hub as of 2026-09-23: worker 1.3.3-rc.171, CUDA 12.8.1, torch 2.9.1.
FROM runpod/pytorch:1.3.3-rc.171-cu1281-torch291-ubuntu2404

RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# Official inference code (generate.py --task ti2v-5B). This is the only
# inference path the model authors support for the TI2V-5B checkpoint.
RUN git clone --depth 1 https://github.com/Wan-Video/Wan2.2.git /opt/Wan2.2
WORKDIR /opt/Wan2.2

# Install the repo's requirements EXCEPT the torch family (the base image
# already ships a working CUDA torch - installing theirs would downgrade it)
# and EXCEPT flash_attn (optional speedup; it has no compatible wheel here
# and its source build fails on this image - killed the 2026-09-23 build).
# Filter complete requirement lines so comma-separated version constraints
# such as "transformers>=4.49.0,<=4.51.3" remain intact. Keep this as a plain
# shell command because Runpod's Dockerfile frontend does not enable heredocs.
RUN sed -E '/^[[:space:]]*(torch|torchvision|torchaudio|flash_attn)([^A-Za-z0-9_.-]|$)/Id' \
    requirements.txt > /tmp/wan-reqs.txt \
    && test -s /tmp/wan-reqs.txt
# einops is imported by the Wan2.2 VAE/transformer modules but missing from
# upstream requirements.txt (2026-09-23: ModuleNotFoundError at
# wan/modules/vae2_1.py line 8 killed the first benchmark job on a green
# build). regex is imported by wan/modules/tokenizers.py; it resolved
# transitively, pinned here explicitly so it stays that way.
# decord + safetensors + peft: `import wan` eagerly pulls in speech2video and
# animate too, and those import decord/safetensors/peft at module level even
# though our ti2v-5B path never calls them. Verified 2026-09-23 by AST-walking
# the full import closure of `import wan` (56 files): these are the only
# third-party modules missing beyond upstream requirements. cosyvoice and
# torchaudio are function-level imports inside tts(), never executed here.
	RUN pip install --no-cache-dir -r /tmp/wan-reqs.txt einops regex decord peft safetensors librosa
# Worker-side deps (the handler itself only needs these; cv2 comes from the
# Wan2.2 requirements above).
COPY requirements.txt /worker-requirements.txt
# The base image's Debian-owned cryptography package has no pip RECORD, so pip
# cannot upgrade/uninstall it while resolving the Runpod SDK. Install a current
# wheel over it first, without asking pip to uninstall the system package.
RUN pip install --no-cache-dir --ignore-installed cryptography \
    && pip install --no-cache-dir -r /worker-requirements.txt

COPY handler.py /handler.py

# Weights live on the attached network volume (/runpod-volume), NOT in the
# image — keeps the image small and workers starting fast. The handler seeds
# the volume itself on the first job (HF_HOME is redirected to the volume).

CMD ["python", "-u", "/handler.py"]
