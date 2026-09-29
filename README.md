# Krea2 Rebalance 2.0

Prompt controls for native Krea 2 in Forge. Restart Forge after updating.

## Start here

Enable the panel and begin with Balanced. Use Prompt strength to adjust the rebalance. The optional Avoid field accepts plain concepts such as `stripes, lettering, red background`. It is an experimental selective negative-attention path, not a guarantee that a concept disappears.

Avoid requires native Forge Krea 2, CFG 1 (including the hires pass), and no image/reference conditioning or conflicting model/attention hooks. Unsupported combinations produce a visible warning and metadata status; rebalance remains available. Avoid supports up to 128 tokens. It does not parse weighting syntax or prompt schedules. Leave it empty to disable it.

## Changes

- Reference and candidate text-fusion passes have independent input storage.
- Zero correction skips unnecessary candidate passes.
- Rebalance and legacy V-Flip share one final correction budget and energy limit.
- Invalid candidate tokens fall back to reference tokens; a nonfinite baseline stops generation with an explanation.
- Avoid encodes a separate concept span and negates only that span's attention values in text refiners and main transformer blocks. Encoder-layer attention remains untouched. All temporary hooks are removed after each model call, including exceptions.
- Optional LoRA loading now uses Forge's model patcher directly and reports how many patches it registered. The control is for diffusion-model adapters, including TextFusion. Put text-encoder LoRAs in Forge prompt tags instead.
- Matching prompt-managed LoRAs take priority and are reported as managed by Forge; no unsupported claim of successful patch application is made for that case.
- Unsupported files and ambiguous duplicate adapter names are rejected. Exact names and registered aliases are supported across configured LoRA directories.
- Optional single-entry text-fusion cache is scoped to the sampling pass. It compares actual input values and bypasses custom hooks, wrappers, masks, and unknown attention options. It costs memory and an equality check; benchmark before keeping it enabled.
- Repeated sampling hooks replace the extension's prior wrapper. Cleanup restores the incoming model patcher.
- Metadata records version, settings, adapter status, Avoid status, fusion-pass count, cache hits, and recovered candidate-token count.

## Simple controls

Gentle, Balanced, and Strong change rebalance settings only. Reset restores the panel's defaults, empties Avoid, and disables caching and legacy V-Flip. It leaves the master switch unchanged. Advanced retains existing power modes and legacy layer weights for compatibility; larger numbers are not guaranteed to improve results.

The original eleven script arguments stay in the same order. New trailing arguments are `avoid_text` (empty string by default) and `use_cache` (false by default).

## Validation and limits

The included `tests/test_krea2.py` runs 26 focused tests using Torch fixtures and the installed Gradio. Tests cover input ownership, correction bounds, cache invalidation, hook restoration, batch expansion, encoder token boundaries, LoRA registration status, sampling cleanup, and UI callbacks. `tests/gpu_check.py` checks FP16/BF16 tensor behavior on CUDA. These are not real-checkpoint image-quality benchmarks.

Real Krea 2 generation quality, selective-suppression effectiveness, adapter efficacy, and full-model speedups still require paired-seed image tests. Caching and Avoid are experimental. NAG and adaptive schedules are not implemented in this version.

## References

Selective attention-value negation is informed by the NegPiP approach; this extension uses its own scoped implementation and does not install or enable another extension.

- [Original NegPiP and Krea 2 support](https://github.com/hako-mikan/sd-webui-negpip)
- [Krea 2 official inference code](https://github.com/krea-ai/krea-2)

## Recovery

The updater creates a timestamped backup in the extension folder. Restore its script and README, then restart Forge, to return to the previous version. Version 2.0 intentionally fixes the numerical path, so old seeds/settings may generate different images.

## Optional Qwen moiré cleanup

The **Qwen moiré cleanup** accordion adds a final-image filter for noise patterning or moiré. It works independently of the Krea 2 controls and is disabled by default. Enable **Reduce output moiré**, then adjust **Cleanup strength** from 0 to 1; the default 0.35 blends a milder correction, while 1 applies the source kernel exactly.

The filter uses the referenced seven-tap kernel `[-1, 6, -15, 20, -15, 6, -1] / 64` with edge-clamped sampling and the source operation `I - Bx - By + Bxy`. RGB and RGBA outputs are supported; RGBA alpha and image metadata are preserved. Other modes are left unchanged.

Source: [Qwen Image 2.1 noise-patterning/moiré mild workaround](https://www.reddit.com/r/StableDiffusion/comments/1wlv3tf/qwen_image_21_noisepatterningmoire_mild_workaround/) and its [original Pastebin shader](https://pastebin.com/v7y1z0SH).

## Special Thanks

Special thanks to:

- [r/sdforall](https://www.reddit.com/r/sdforall/) — community discussion and testing
- [r/SECourses](https://www.reddit.com/r/SECourses/) — community discussion and testing
- [r/malcolmrey](https://www.reddit.com/r/malcolmrey/) — community discussion and testing
- [**Haoming02 / sd-webui-forge-classic (neo branch)**](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo) — the Forge Neo tree this extension targets
- [**Adeliox**](https://github.com/Adeliox) — original Klein Head Swap
- [Alissonerdx](https://huggingface.co/Alissonerdx/BFS-Best-Face-Swap) — BFS (Best Face Swap) workflow and LoRAs
- [PozzettiAndrea / ComfyUI-SAM3](https://github.com/PozzettiAndrea/ComfyUI-SAM3) and [Meta SAM3](https://github.com/facebookresearch/sam3) — segmentation workflow inspiration and optional upstream mask model
- [**ComfyUI**](https://github.com/comfyanonymous/ComfyUI) — reference for upstream sampler/scheduler coverage
- The Forge / AUTOMATIC1111 community — for the extension ecosystem this plugs into
- Project Invisible extensions — memory policy, GPU compatibility and extension philosophy

Thank you to the wider Forge, Diffusers, Qwen, DeGrid and open-source communities.
