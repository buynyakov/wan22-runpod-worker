# RunPod serverless worker: Wan 2.2 TI2V-5B via the official Wan2.2 CLI.
# Repo-root layout: Dockerfile, handler.py, requirements.txt side by side.

# Pinned 2026-09-23: RunPod removed the old 2.4.0-py3.11-cuda12.4.1-devel tag
# scheme (their whole tag set rotated ~2026-09-23 midday). This tag is current
# on Docker Hub as of 2026-09-23: worker 1.3.3-rc.171, CUDA 12.8.1, torch 2.9.1.
FROM runpod/pytorch:1.3.3-rc.171-cu1281-torch291-ubuntu2404
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# Official inference code (generate.py --task ti2v-5B). This is the only
# inference path the model authors support for the TI2V-5B checkpoint.
RUN git clone --depth 1 https://github.com/Wan-Video/Wan2.2.git /opt/Wan2.2

RUN python3 -c "p='/opt/Wan2.2/wan/textimage2video.py'; s=open(p).read(); old='''            x0 = latents\n            if offload_model:\n                self.model.cpu()\n                torch.cuda.synchronize()\n                torch.cuda.empty_cache()\n            if self.rank == 0:\n                videos = self.vae.decode(x0)'''; new='''            x0 = latents\n            # Unconditional, deterministic DiT teardown before VAE decode\n            # (24GB serverless): bench 4/5 showed ~14.5 GiB still allocated\n            # during decode, i.e. the ~9.3 GiB bf16 DiT was still resident\n            # even though offload_model=True was passed. The DiT is never\n            # used after the sampling loop, so drop it outright. Each job\n            # runs in a fresh generate.py process, so there is no reuse cost.\n            del self.model\n            gc.collect()\n            torch.cuda.synchronize()\n            torch.cuda.empty_cache()\n            print('[wan] pre-decode GPU allocated: %.2f GiB' % (torch.cuda.memory_allocated() / 2**30), flush=True)\n            if self.rank == 0:\n                videos = self.vae.decode(x0)'''; assert s.count(old)==1, 'DiT teardown pattern not found'; open(p,'w').write(s.replace(old,new)); print('DiT teardown patch applied')"
RUN grep -q "first_chunk=True" /opt/Wan2.2/wan/modules/vae2_2.py || (echo "FATAL: vae2_2.py missing chunked decode" && exit 1)

# Wan2.2's DiT hard-asserts flash-attn, which has no wheel for torch 2.9/cu128
# (source build fails). Rewire to the repo's own attention() wrapper, which
# falls back to torch SDPA - numerically identical for the 5B model.
RUN grep -rl "from .*attention import flash_attention" /opt/Wan2.2/wan --include="*.py" \
    | xargs sed -i 's/import flash_attention$/import attention as flash_attention/'

# --convert_model_dtype only converts the DiT; the VAE defaults to fp32 and its
# full-video decode OOMs a 24GB card (bench 4: F.pad wanted 2.54 GiB, 1.07 free).
# bf16 VAE decode is a supported path (see the bfloat16 note in vae2_2.py Upsample).
RUN sed -i '/self\.vae = Wan2_2_VAE(/,/device=self\.device)/ { s/vae_pth=os\.path\.join(checkpoint_dir, config\.vae_checkpoint),/vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint), dtype=torch.bfloat16,/; s/device=self\.device)/device=self.device)\n        self.vae.model.to(torch.bfloat16)/; }' /opt/Wan2.2/wan/textimage2video.py

RUN python3 -c "p='/opt/Wan2.2/wan/textimage2video.py'; s=open(p).read(); old='''            if offload_model:\n                self.model.cpu()\n                torch.cuda.synchronize()\n                torch.cuda.empty_cache()\n\n            if self.rank == 0:\n                videos = self.vae.decode(x0)'''; new='''            # Unconditional, deterministic DiT teardown before VAE decode (i2v path):\n            # free the ~14.5 GiB DiT before the ~22.6 GiB VAE decode allocation.\n            # Each job runs in a fresh generate.py process, so there is no reuse cost.\n            del self.model\n            gc.collect()\n            torch.cuda.synchronize()\n            torch.cuda.empty_cache()\n            print('[wan] pre-decode GPU allocated: %.2f GiB' % (torch.cuda.memory_allocated() / 2**30), flush=True)\n            if self.rank == 0:\n                videos = self.vae.decode(x0)'''; assert s.count(old)==1, 'i2v DiT teardown pattern not found'; open(p,'w').write(s.replace(old,new)); print('i2v DiT teardown patch applied')"


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