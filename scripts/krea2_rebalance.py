import math
import os
import re
import html
import threading
from contextlib import contextmanager
from typing import Any

import gradio as gr
import numpy as np
import torch
from PIL import Image

from modules import scripts, shared
from modules.ui_components import InputAccordion


# ============================================================
# Krea 2 Constants
# ============================================================
KREA2_TAP_LAYERS = (2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35)
KREA2_TAP_DIM = 2560
KREA2_CHUNK_COUNT = 24
KREA2_CHUNK_DIM = 1280

ENHANCER_PROFILE_12 = (1.0, 1.0, 1.0, 1.0, 1.0, 1.3, 2.0, 4.0, 6.0, 1.5, 5.0, 1.3)
ENHANCER_CHUNK_PROFILE = ENHANCER_PROFILE_12 + ENHANCER_PROFILE_12
ENHANCER_GLOBAL_MULTIPLIER = 22.0
TXTFUSION_TOKEN_REL_CAP = 0.25  # keep the directional correction a trim, not a takeover
TXTFUSION_TOKEN_HARD_CAP = 0.5  # old presets/infotext cannot restore the unsafe 1.0-3.0 range

# Power modes: extra multiplicative headroom stacked on top of the Strength slider
ENHANCER_POWER_MODES = ["Standard", "High", "Extreme", "MAX"]
ENHANCER_POWER_MULTIPLIERS = {
    "Standard": 1.0,
    "High": 1.5,
    "Extreme": 2.25,
    "MAX": 3.0,
}

# Default Refusal Reduction LoRA name
DEFAULT_REFUSAL_LORA = "Krea2_TextFusion_Refusal_Reduction"
DEFAULT_REFUSAL_STRENGTH = 1.0

# NegPiP-style v-flip defaults
NEGPIP_WEIGHTS_DEFAULT = "0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2"
NEGPIP_TOKEN_CAP = 1.2

# Targeted refusal knobs (ported from AzKrea2GatedRebalance). Krea 2's safety
# filter lives in tapped projector layers 9/10 (primary) and 11 (secondary);
# layers 1-8 and 12 are style/anatomy priors and are left untouched in knob
# mode. Layer L (1-indexed) occupies chunk indices 2*(L-1) and 2*(L-1)+1 in
# the 24x1280 txtfusion stack.
KNOB_CHUNKS = {9: (16, 17), 10: (18, 19), 11: (20, 21)}
KNOB_DEFAULTS = {9: 0.4883, 10: 0.1094, 11: 0.0}  # FB2 deltas: -0.5117 / -0.8906


def _is_krea2_dm(dm: Any) -> bool:
    try:
        if not hasattr(dm, "txtfusion"):
            return False
        if not hasattr(dm, "txtmlp"):
            return False
        if not hasattr(dm, "blocks"):
            return False

        txtlayers = getattr(dm, "txtlayers", None)
        txtdim = getattr(dm, "txtdim", None)

        if txtlayers is not None and txtdim is not None:
            if int(txtlayers) == 12 and int(txtdim) == 2560:
                return True

        txtfusion = getattr(dm, "txtfusion", None)
        if txtfusion is not None:
            if hasattr(txtfusion, "projector") and hasattr(txtfusion, "layerwise_blocks"):
                return True

        return False
    except Exception:
        return False


def _bounded_float(value, default: float, lo: float, hi: float) -> float:
    try:
        v = float(value)
    except Exception:
        v = default
    if not math.isfinite(v):
        v = default
    return max(lo, min(hi, v))


def _chunk_gains(device: torch.device, dtype: torch.dtype, strength: float, power: float = 1.0) -> torch.Tensor:
    base = torch.tensor(ENHANCER_CHUNK_PROFILE, device=device, dtype=torch.float32)
    gains = 1.0 + float(strength) * float(power) * (base - 1.0)
    return gains.to(dtype=dtype)


def _parse_floats(text: Any) -> Any:
    """Parse a comma/semicolon-separated list of floats. Returns None on failure."""
    if not text:
        return None
    try:
        vals = [float(v) for v in str(text).replace(";", ",").split(",") if v.strip()]
    except Exception:
        return None
    return vals if vals and all(math.isfinite(v) and 0.0 <= v <= 3.0 for v in vals) else None


PATCH_LOCK = threading.RLock()
VERSION = "2.0.1"


def _rms(value):
    return value.square().mean(dim=-1, keepdim=True).sqrt()


def _limit_delta(delta, reference, cap):
    # Reject an invalid token as a unit; partial repair changes its direction.
    finite = torch.isfinite(delta).all(dim=-1, keepdim=True)
    delta = torch.where(finite, delta, torch.zeros_like(delta))
    radius = _rms(reference) * cap
    return delta * (radius / _rms(delta).clamp_min(1e-12)).clamp(max=1.0)


def _enhanced_txtfusion_forward(txtfusion, x, mask=None, transformer_options=None,
                                strength=1.0, power=1.0, token_cap=TXTFUSION_TOKEN_REL_CAP,
                                negpip=None, original_forward=None, stats=None,
                                knob_map=None):
    original = original_forward or txtfusion._krea2_rebalance_original_forward
    options = transformer_options or {}
    stats = stats if stats is not None else {}
    def run(value):
        # Native Forge blocks use in-place residuals. Every branch owns its input.
        stats["fusion_passes"] = stats.get("fusion_passes", 0) + 1
        return original(value.clone(), mask=mask, transformer_options=options)
    if x.ndim != 4 or tuple(x.shape[-2:]) != (12, 2560):
        return run(x)
    reference_out = run(x)
    legacy_active = bool(negpip and negpip.get("enabled") and negpip.get("neg_strength", 0) > 0)
    reference = reference_out.float()
    if not torch.isfinite(reference).all():
        raise RuntimeError("Krea 2 produced nonfinite baseline text features. Check model precision and adapters.")
    if strength <= 0 and not legacy_active:
        return reference_out
    delta = torch.zeros_like(reference)
    if strength > 0:
        b, seq, _, _ = x.shape
        gains = _chunk_gains(x.device, torch.float32, strength, power)
        scale = 1 + strength * power * (ENHANCER_GLOBAL_MULTIPLIER - 1)
        scaled = (x.float().reshape(b, seq, 24, 1280) * gains.view(1, 1, 24, 1) * scale).reshape_as(x)
        candidate = run(scaled.to(x.dtype)).float()
        stats["invalid_tokens"] = stats.get("invalid_tokens", 0) + (~torch.isfinite(candidate).all(dim=-1)).sum().detach()
        delta = _limit_delta(candidate - reference, reference, token_cap)
    if legacy_active:
        weights = torch.as_tensor(negpip["weights"], device=x.device, dtype=x.dtype).view(1, 1, 12, 1)
        flipped = run(x * -weights).float()
        stats["invalid_tokens"] = stats.get("invalid_tokens", 0) + (~torch.isfinite(flipped).all(dim=-1)).sum().detach()
        legacy_delta = (flipped - reference) * negpip["neg_strength"]
        delta = delta + _limit_delta(legacy_delta, reference, negpip["token_cap"])
    # One total correction budget, including the legacy branch.
    out = reference + _limit_delta(delta, reference, token_cap)
    # Projection onto the reference-energy ball cannot increase distance from
    # the reference, which itself lies in that ball.
    out = out * (_rms(reference) / _rms(out).clamp_min(1e-12)).clamp(max=1.0)
    return out.to(reference_out.dtype)


def _call_model_function(model_function, kwargs):
    return model_function(kwargs["input"], kwargs["timestep"], **kwargs["c"])


@contextmanager
def _negative_value_hooks(dm, start, end):
    """Negate only the appended concept tokens, never encoder-layer attention."""
    handles = []
    def flip(module, args, output):
        if output.ndim != 3 or output.shape[1] < end:
            raise RuntimeError("Avoid token alignment changed; stopping to prevent incorrect token edits.")
        return torch.cat((output[:, :start], -output[:, start:end], output[:, end:]), dim=1)
    try:
        targets = [block.attn.wv for block in dm.blocks]
        targets += [block.attn.wv for block in dm.txtfusion.refiner_blocks]
        for projection in targets:
            handles.append(projection.register_forward_hook(flip))
        yield
    finally:
        for handle in handles:
            handle.remove()


def _make_unet_wrapper(diffusion_model, strength, previous_wrapper=None, power=1.0,
                       token_cap=TXTFUSION_TOKEN_REL_CAP, negpip=None,
                       avoid_context=None, use_cache=False, stats=None):
    stats = stats if stats is not None else {}
    # A repeated sampling hook must replace our wrapper, not stack it.
    if getattr(previous_wrapper, "_krea2_owned", False):
        previous_wrapper = previous_wrapper._krea2_previous
    cache = {}
    def invoke(model_function, kwargs):
        if previous_wrapper is not None:
            return previous_wrapper(model_function, kwargs)
        return _call_model_function(model_function, kwargs)
    def wrapper(model_function, kwargs):
        if not _is_krea2_dm(diffusion_model):
            return invoke(model_function, kwargs)
        with PATCH_LOCK:
            fusion = diffusion_model.txtfusion
            original = fusion.forward
            had_instance_forward = "forward" in fusion.__dict__
            original_instance = fusion.__dict__.get("forward")
            # Cache only the native static fusion path without foreign wrappers.
            cache_allowed = use_cache and previous_wrapper is None and not had_instance_forward
            cache_allowed = cache_allowed and type(fusion).__module__ == "backend.nn.krea"
            cache_allowed = cache_allowed and not any(m._forward_hooks or m._forward_pre_hooks
                                                      for m in fusion.modules())
            def enhanced(x, mask=None, transformer_options=None):
                options = transformer_options or {}
                # Native text fusion ignores the sampler bookkeeping below. Custom
                # options (especially attention overrides) disable reuse.
                simple = set(options).issubset({"cond_or_uncond", "original_shape", "sigmas",
                                                "cond_mark", "cond_indices", "uncond_indices"})
                eligible = cache_allowed and simple and mask is None
                key = repr((options.get("cond_or_uncond"), options.get("original_shape")))
                if eligible and cache.get("key") == key:
                    old = cache["input"]
                    if old.device == x.device and old.dtype == x.dtype and old.shape == x.shape and torch.equal(old, x):
                        stats["cache_hits"] = stats.get("cache_hits", 0) + 1
                        return cache["output"].clone()
                result = _enhanced_txtfusion_forward(fusion, x, mask, options, strength, power,
                                                     token_cap, negpip, original, stats)
                if eligible:
                    cache.update(key=key, input=x.detach().clone(), output=result.detach().clone())
                else:
                    cache.clear()
                return result
            call_kwargs = dict(kwargs)
            call_kwargs["c"] = dict(kwargs["c"])
            context = call_kwargs["c"].get("c_crossattn")
            avoid = avoid_context is not None
            if avoid:
                if context is None or context.ndim != 4 or tuple(context.shape[-2:]) != (12, 2560):
                    raise RuntimeError("Avoid requires native Krea 2 conditioning.")
                start = context.shape[1]
                extra = avoid_context.to(device=context.device, dtype=context.dtype).expand(context.shape[0], -1, -1, -1)
                end = start + extra.shape[1]
                call_kwargs["c"]["c_crossattn"] = torch.cat((context, extra), dim=1)
            fusion.forward = enhanced
            try:
                if avoid:
                    with _negative_value_hooks(diffusion_model, start, end):
                        return invoke(model_function, call_kwargs)
                return invoke(model_function, call_kwargs)
            finally:
                if had_instance_forward:
                    fusion.forward = original_instance
                else:
                    del fusion.forward
    wrapper._krea2_owned = True
    wrapper._krea2_previous = previous_wrapper
    wrapper._krea2_stats = stats
    wrapper._krea2_cache = cache
    return wrapper


def _encode_avoid(p, text):
    from backend import memory_management
    from modules import devices
    engine = p.sd_model.text_processing_engine_qwen
    if "<|" in text or any(char in text for char in "[]()"):
        raise ValueError("Avoid accepts plain concepts, without prompt schedules, weighting, or template markers.")
    tokens = engine.tokenize([text])[0]
    starts = [i for i, token in enumerate(tokens) if token == engine.id_template]
    if len(starts) < 2:
        raise ValueError("Unsupported Krea 2 chat template.")
    start = starts[1] + 3
    if tokens[starts[1] + 1:start] != [872, 198]:
        raise ValueError("Unsupported Krea 2 user-token boundary.")
    end = next((i for i in range(start, len(tokens)) if tokens[i] == 151645), None)
    if end is None or not 0 < end - start <= 128:
        raise ValueError("Keep Avoid between 1 and 128 tokens.")
    memory_management.load_model_gpu(p.sd_model.forge_objects.clip.patcher)
    with torch.inference_mode(), devices.autocast():
        encoded = engine.process_tokens([tokens], [[1.0] * len(tokens)])
    if encoded.ndim != 4 or encoded.shape[1] != 12 or encoded.shape[2] != len(tokens):
        raise ValueError("Unsupported Krea 2 encoder output.")
    return encoded[:, :, start:end, :].permute(0, 2, 1, 3).detach().cpu().contiguous()


def _find_lora_file(lora_name):
    """Resolve exact names/aliases, reject ambiguous files and unsupported formats."""
    name = str(lora_name or "").strip()
    if not name:
        return None
    try:
        import networks
        registry = getattr(networks, "available_networks", {})
        aliases = getattr(networks, "available_network_aliases", {})
        selected = registry.get(name) or aliases.get(name)
    except ImportError:
        selected = None
    extensions = (".safetensors", ".pt", ".ckpt")
    stem = os.path.splitext(name)[0] if name.lower().endswith(extensions) else name
    roots = [getattr(shared.cmd_opts, "lora_dir", None)] + list(getattr(shared.cmd_opts, "lora_dirs", []) or [])
    if selected:
        roots.append(os.path.dirname(selected.filename))
        stem = os.path.splitext(os.path.basename(selected.filename))[0]
    matches = set()
    for root in dict.fromkeys(os.path.normcase(os.path.abspath(r)) for r in roots if r):
        if not os.path.isdir(root):
            continue
        for directory, _, files in os.walk(root):
            for filename in files:
                base, ext = os.path.splitext(filename)
                if ext.lower() in extensions and base.casefold() == stem.casefold():
                    matches.add(os.path.normcase(os.path.abspath(os.path.join(directory, filename))))
    if len(matches) > 1:
        raise ValueError("Multiple LoRAs share this filename. Give the adapter a unique filename and refresh Forge's LoRA list.")
    return next(iter(matches), None)


def _patch_count(unet):
    return sum(len(values) for values in unet.patches.values())


def _apply_adapter(p, unet, name, strength):
    if strength <= 0:
        return unet, "Off (zero strength)"
    import networks
    path = _find_lora_file(name)
    if not path:
        return unet, "Not found"
    # Respect a matching adapter that Forge already requested through the prompt.
    for loaded in networks.loaded_networks:
        filename = getattr(getattr(loaded, "network_on_disk", None), "filename", "")
        if filename and os.path.normcase(os.path.abspath(filename)) == path:
            return unet, "Managed by Forge prompt; inspect Forge's LoRA log for patch details"
    signature = (path, os.path.getmtime(path), os.path.getsize(path))
    if getattr(p, "_krea2_adapter_signature", None) != signature:
        p._krea2_adapter_data = networks.load_lora_state_dict(path)
        p._krea2_adapter_signature = signature
    before = _patch_count(unet)
    # This adapter control is intentionally for diffusion-model LoRAs: Krea's
    # TextFusion lives there. Text-encoder adapters belong in Forge's prompt.
    updated, _ = networks.load_lora_for_models(unet, None, p._krea2_adapter_data,
                                               strength, 0.0, filename=path, online_mode=False)
    count = _patch_count(updated) - before
    p.extra_generation_params["Krea2 Refusal LoRA Name"] = os.path.splitext(os.path.basename(path))[0]
    if count <= 0:
        return unet, "No model patches registered (incompatible or empty adapter)"
    return updated, f"Registered {count} model patches at {strength:.2f}"


MOIRE_NOTCH_KERNEL = np.asarray((-1, 6, -15, 20, -15, 6, -1), dtype=np.float32) / 64.0


def _convolve_axis_edge(values, kernel, axis):
    """Apply a 1-D kernel with edge-clamped sampling."""
    radius = len(kernel) // 2
    pads = [(0, 0)] * values.ndim
    pads[axis] = (radius, radius)
    padded = np.pad(values, pads, mode="edge")
    result = np.zeros_like(values, dtype=np.float32)
    for offset, weight in enumerate(kernel):
        slices = [slice(None)] * values.ndim
        slices[axis] = slice(offset, offset + values.shape[axis])
        result += float(weight) * padded[tuple(slices)]
    return result


def _apply_moire_filter(image, amount=0.35):
    """Blend the Qwen Image 2.1 Nyquist-notch workaround into an RGB/RGBA image."""
    if image is None or getattr(image, "mode", None) not in ("RGB", "RGBA"):
        return image
    amount = _bounded_float(amount, 0.35, 0.0, 1.0)
    if amount <= 0.0:
        return image

    info = dict(getattr(image, "info", {}) or {})
    pixels = np.asarray(image, dtype=np.uint8)
    rgb = pixels[..., :3].astype(np.float32) / 255.0
    blur_x = _convolve_axis_edge(rgb, MOIRE_NOTCH_KERNEL, axis=1)
    blur_y = _convolve_axis_edge(rgb, MOIRE_NOTCH_KERNEL, axis=0)
    blur_xy = _convolve_axis_edge(blur_x, MOIRE_NOTCH_KERNEL, axis=0)
    filtered = np.clip(rgb - blur_x - blur_y + blur_xy, 0.0, 1.0)
    output_rgb = np.rint(np.clip(rgb + amount * (filtered - rgb), 0.0, 1.0) * 255.0).astype(np.uint8)

    if image.mode == "RGBA":
        output = np.concatenate((output_rgb, pixels[..., 3:4]), axis=2)
    else:
        output = output_rgb
    result = Image.fromarray(output, mode=image.mode)
    result.info.update(info)
    return result


class Krea2RebalanceScript(scripts.ScriptBuiltinUI):
    sorting_priority = 18160

    def title(self):
        return "Krea2 Rebalance"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, *args, **kwargs):
        with InputAccordion(False, label=self.title(), elem_classes=["krea2-panel"]) as enable:
            if hasattr(enable, "accordion"):
                enable.accordion.elem_classes.append("krea2-panel")
            gr.HTML("<div class='krea2-hero'><span class='krea2-eyebrow'>PROMPT CONTROL</span>"
                    "<h3>More control. Less clutter.</h3>"
                    "<p>Choose a starting point, then fine-tune your strength.</p></div>")
            with gr.Row():
                gentle = gr.Button("Gentle", size="sm")
                balanced = gr.Button("Balanced", variant="primary", size="sm")
                strong = gr.Button("Strong", size="sm")
                reset = gr.Button("Reset", size="sm")
            strength = gr.Slider(0.0, 3.0, value=1.0, step=0.05,
                                 label="Prompt strength", info="Start at 1.0. Lower it if details distort.")
            avoid_text = gr.Textbox(interactive=True, label="Avoid (optional)", placeholder="e.g. stripes, lettering, red background",
                                    info="Experimental selective concept suppression. Plain text; CFG 1, no reference images.")
            summary = gr.Markdown("**Control level: 1.00×** · Correction limit: 0.25", elem_classes=["krea2-summary"])

            with gr.Accordion("Optional LoRA", open=False):
                enable_refusal = gr.Checkbox(label="Apply LoRA", value=True,
                                             info="Diffusion-model adapter, including TextFusion. Use Forge prompt tags for text-encoder LoRAs.")
                with gr.Column(visible=True) as lora_controls:
                    lora_name = gr.Textbox(label="LoRA name", value=DEFAULT_REFUSAL_LORA,
                                           info="Exact filename or registered alias. A matching Forge prompt adapter takes priority.")
                    refusal_strength = gr.Slider(0.0, 2.0, value=DEFAULT_REFUSAL_STRENGTH,
                                                  step=0.05, label="LoRA strength")
                    check_lora = gr.Button("Check LoRA", size="sm")
                    lora_status = gr.Markdown("Check that your adapter is available before generating.")

            with gr.Accordion("Advanced", open=False):
                use_cache = gr.Checkbox(interactive=True, value=False, label="Reuse unchanged text fusion (experimental)",
                                        info="One-entry cache per sampling pass. Automatically bypassed for custom hooks/options.")
                with gr.Row():
                    power_mode = gr.Dropdown(choices=ENHANCER_POWER_MODES, value="Standard",
                                             label="Power multiplier", info="Standard 1× · High 1.5× · Extreme 2.25× · MAX 3×")
                    token_cap = gr.Slider(0.05, TXTFUSION_TOKEN_HARD_CAP, value=TXTFUSION_TOKEN_REL_CAP, step=0.05,
                                          label="Correction limit", info="Lower this first if results wash out or overshoot.")
                enable_negpip = gr.Checkbox(label="Legacy global V-Flip", value=False,
                                            info="Experimental whole-prompt effect. Prefer the selective Avoid field. Adds processing time.")
                with gr.Column(visible=False) as negpip_controls:
                    negpip_weights = gr.Textbox(value=NEGPIP_WEIGHTS_DEFAULT,
                                                label="12 layer weights", info="Twelve numbers from 0 to 3, separated by commas.")
                    with gr.Row():
                        negpip_strength = gr.Slider(0.0, 3.0, value=1.0, step=0.05, label="V-Flip strength")
                        negpip_cap = gr.Slider(0.5, 3.0, value=NEGPIP_TOKEN_CAP, step=0.05, label="V-Flip correction limit")
            gr.Markdown("Krea 2 models only. Presets adjust prompt control; LoRA settings stay yours.", elem_classes=["krea2-footnote"])

            def describe(value, mode, cap):
                effective = _bounded_float(value, 1.0, 0, 3) * ENHANCER_POWER_MULTIPLIERS.get(mode, 1.0)
                return f"**Control level: {effective:.2f}×** · Correction limit: {float(cap):.2f}"

            def check_adapter(name):
                try:
                    found = _find_lora_file(name)
                except ValueError as exc:
                    return str(exc)
                if found:
                    return "Found: <code>" + html.escape(os.path.basename(found)) + "</code>. Available. Patch registration is checked during generation."
                return "LoRA not found. Check its filename and your configured LoRA folder."

            for button, values in ((gentle, (0.65, "Standard", 0.15)),
                                   (balanced, (1.0, "Standard", 0.25)),
                                   (strong, (1.2, "High", 0.35))):
                button.click(lambda v=values: (*v, describe(*v)), inputs=[],
                             outputs=[strength, power_mode, token_cap, summary], queue=False)
            reset.click(lambda: (1.0, "Standard", TXTFUSION_TOKEN_REL_CAP, True, DEFAULT_REFUSAL_LORA, 1.0,
                                  False, NEGPIP_WEIGHTS_DEFAULT, 1.0, NEGPIP_TOKEN_CAP,
                                  describe(1.0, "Standard", TXTFUSION_TOKEN_REL_CAP), gr.update(visible=True), gr.update(visible=False), "", False,
                                  "Check that your adapter is available before generating."), inputs=[],
                        outputs=[strength, power_mode, token_cap, enable_refusal, lora_name, refusal_strength,
                                 enable_negpip, negpip_weights, negpip_strength, negpip_cap, summary,
                                 lora_controls, negpip_controls, avoid_text, use_cache, lora_status], queue=False)
            for control in (strength, power_mode, token_cap):
                control.change(describe, inputs=[strength, power_mode, token_cap], outputs=[summary], queue=False)
            enable_refusal.change(lambda enabled: gr.update(visible=enabled), inputs=[enable_refusal], outputs=[lora_controls], queue=False)
            enable_negpip.change(lambda enabled: gr.update(visible=enabled), inputs=[enable_negpip], outputs=[negpip_controls], queue=False)
            check_lora.click(check_adapter, inputs=[lora_name], outputs=[lora_status], queue=False)
            lora_name.change(lambda: "Name changed. Check LoRA again.", inputs=[], outputs=[lora_status], queue=False)
        with gr.Accordion("Qwen moiré cleanup", open=False):
            moire_enabled = gr.Checkbox(value=False, label="Reduce output moiré",
                                        info="Optional final-image filter; works independently of Krea 2 controls.")
            moire_strength = gr.Slider(0.0, 1.0, value=0.35, step=0.05, label="Cleanup strength")

        for comp in (enable, enable_refusal, lora_name, strength, power_mode, refusal_strength, token_cap, enable_negpip, negpip_weights, negpip_strength, negpip_cap, avoid_text, use_cache, moire_enabled, moire_strength):
            comp.do_not_save_to_config = True

        self.infotext_fields = [
            (avoid_text, "Krea2 Avoid"),
            (use_cache, "Krea2 Cache"),
            (enable, "Krea2 Rebalance"),
            (strength, "Krea2 Rebalance Strength"),
            (power_mode, "Krea2 Rebalance Power Mode"),
            (enable_refusal, "Krea2 Refusal LoRA"),
            (lora_name, "Krea2 Refusal LoRA Name"),
            (refusal_strength, "Krea2 Refusal LoRA Strength"),
            (token_cap, "Krea2 Rebalance Adherence Cap"),
            (enable_negpip, "Krea2 NegPiP"),
            (negpip_weights, "Krea2 NegPiP Weights"),
            (negpip_strength, "Krea2 NegPiP Strength"),
            (negpip_cap, "Krea2 NegPiP Cap"),
            (moire_enabled, "Qwen Moire Cleanup"),
            (moire_strength, "Qwen Moire Cleanup Strength"),
        ]

        return [enable, enable_refusal, lora_name, strength, power_mode, refusal_strength, token_cap, enable_negpip, negpip_weights, negpip_strength, negpip_cap, avoid_text, use_cache, moire_enabled, moire_strength]

    def process(self, p, *args, **kwargs):
        # State belongs to the job, never the persistent Gradio Script instance.
        self._clear_state(p)

    @staticmethod
    def _clear_state(p):
        for name in ("_krea2_adapter_data", "_krea2_adapter_signature", "_krea2_avoid", "_krea2_avoid_key", "_krea2_base_unet", "_krea2_output_unet"):
            if hasattr(p, name):
                delattr(p, name)

    def process_before_every_sampling(self, p, enable, enable_refusal, lora_name, strength,
                                     power_mode="Standard", refusal_strength=1.0,
                                     token_cap=TXTFUSION_TOKEN_REL_CAP, enable_negpip=False,
                                     negpip_weights=NEGPIP_WEIGHTS_DEFAULT, negpip_strength=1.0,
                                     negpip_cap=NEGPIP_TOKEN_CAP, avoid_text="", use_cache=False,
                                     *args, **kwargs):
        if not enable:
            return
        unet = p.sd_model.forge_objects.unet
        if unet is getattr(p, "_krea2_output_unet", None):
            unet = p._krea2_base_unet
        p._krea2_base_unet = unet
        # Strip a prior installation before preparing this sampling pass.
        previous = unet.model_options.get("model_function_wrapper")
        if getattr(previous, "_krea2_owned", False):
            unet = unet.clone()
            if previous._krea2_previous is None:
                unet.model_options.pop("model_function_wrapper", None)
            else:
                unet.set_model_unet_function_wrapper(previous._krea2_previous)
        dm = unet.get_model_object("diffusion_model")
        if not _is_krea2_dm(dm):
            gr.Warning("Krea2 Rebalance skipped: this is not a supported Krea 2 model.")
            return
        strength = _bounded_float(strength, 1, 0, 3)
        requested_token_cap = _bounded_float(token_cap, TXTFUSION_TOKEN_REL_CAP, 0.05, 3)
        token_cap = min(requested_token_cap, TXTFUSION_TOKEN_HARD_CAP)
        if requested_token_cap > TXTFUSION_TOKEN_HARD_CAP:
            gr.Warning(f"Krea2 correction limit reduced from {requested_token_cap:.2f} to the safe maximum {token_cap:.2f}.")
        refusal_strength = _bounded_float(refusal_strength, 1, 0, 2)
        power_mode = power_mode if power_mode in ENHANCER_POWER_MULTIPLIERS else "Standard"
        meta = p.extra_generation_params
        status = "Off"
        if enable_refusal:
            try:
                unet, status = _apply_adapter(p, unet, lora_name, refusal_strength)
            except Exception as exc:
                status = f"Unavailable: {exc}"
            if not status.startswith(("Registered", "Managed", "Off")):
                gr.Warning("Krea2 LoRA: " + status)
        meta["Krea2 Adapter Status"] = status
        print("[Krea2 Rebalance] Adapter: " + status)
        negpip_cfg = None
        if enable_negpip and _bounded_float(negpip_strength, 1, 0, 3) > 0:
            weights = _parse_floats(negpip_weights)
            if weights is None or len(weights) != 12:
                gr.Warning("Krea2: invalid legacy weights; using twelve 0.2 weights.")
                weights = [0.2] * 12
            negpip_cfg = dict(enabled=True, weights=weights, neg_strength=_bounded_float(negpip_strength, 1, 0, 3),
                              token_cap=_bounded_float(negpip_cap, 1.2, 0.5, 3))
        avoid = None
        text = str(avoid_text or "").strip()
        avoid_status = "Off"
        if text:
            from backend.args import dynamic_args
            if not math.isclose(float(getattr(p, "hr_cfg", p.cfg_scale) if getattr(p, "is_hr_pass", False) else p.cfg_scale), 1.0):
                avoid_status = "Skipped: Avoid requires CFG 1"
            elif getattr(dynamic_args, "ref_latents", []) or getattr(p, "init_images", None):
                avoid_status = "Skipped: reference/image-conditioned generation is not supported by Avoid"
            elif unet.model_options.get("model_function_wrapper") is not None:
                avoid_status = "Skipped: another model wrapper is active"
            elif unet.model_options.get("transformer_options"):
                avoid_status = "Skipped: another attention configuration is active"
            elif any(m._forward_hooks or m._forward_pre_hooks or "forward" in m.__dict__ for m in dm.modules()):
                avoid_status = "Skipped: another model hook is active"
            elif type(dm).__module__ != "backend.nn.krea":
                avoid_status = "Skipped: Avoid requires native Forge Krea 2"
            else:
                try:
                    key = (id(p.sd_model), str(p.sd_model.forge_objects.clip.patcher.patches_uuid), text)
                    if getattr(p, "_krea2_avoid_key", None) != key:
                        p._krea2_avoid = _encode_avoid(p, text)
                        p._krea2_avoid_key = key
                    avoid = p._krea2_avoid
                    avoid_status = f"Active ({avoid.shape[1]} concept tokens)"
                except Exception as exc:
                    avoid_status = f"Skipped: {exc}"
            if avoid is None:
                gr.Warning("Krea2 " + avoid_status)
        meta["Krea2 Avoid Status"] = avoid_status
        if strength > 0 or negpip_cfg or avoid is not None:
            unet = unet.clone()
            stats = {}
            unet.set_model_unet_function_wrapper(_make_unet_wrapper(
                dm, strength, unet.model_options.get("model_function_wrapper"),
                ENHANCER_POWER_MULTIPLIERS[power_mode], token_cap, negpip_cfg, avoid, bool(use_cache), stats))
        p.sd_model.forge_objects.unet = unet
        p._krea2_output_unet = unet
        meta.update({"Krea2 Rebalance": True, "Krea2 Version": VERSION,
                     "Krea2 Rebalance Strength": strength, "Krea2 Rebalance Power Mode": power_mode,
                     "Krea2 Rebalance Adherence Cap": token_cap,
                     "Krea2 Rebalance Requested Adherence Cap": requested_token_cap,
                     "Krea2 Refusal LoRA": bool(enable_refusal),
                     "Krea2 Refusal LoRA Strength": refusal_strength, "Krea2 NegPiP": bool(negpip_cfg),
                     "Krea2 Avoid": text, "Krea2 Cache": bool(use_cache)})
        if negpip_cfg:
            meta.update({"Krea2 NegPiP Weights": ",".join(map(str, negpip_cfg["weights"])),
                         "Krea2 NegPiP Strength": negpip_cfg["neg_strength"], "Krea2 NegPiP Cap": negpip_cfg["token_cap"]})

    def post_sample(self, p, ps, *args):
        unet = p.sd_model.forge_objects.unet
        wrapper = unet.model_options.get("model_function_wrapper")
        if getattr(wrapper, "_krea2_owned", False):
            p.extra_generation_params["Krea2 Fusion Passes"] = wrapper._krea2_stats.get("fusion_passes", 0)
            p.extra_generation_params["Krea2 Cache Hits"] = wrapper._krea2_stats.get("cache_hits", 0)
            invalid = int(wrapper._krea2_stats.get("invalid_tokens", 0))
            p.extra_generation_params["Krea2 Recovered Tokens"] = invalid
            if invalid:
                gr.Warning(f"Krea2 recovered {invalid} nonfinite candidate tokens. Lower strength or check model precision.")
            wrapper._krea2_cache.clear()
            unet = unet.clone()
            if wrapper._krea2_previous is None:
                unet.model_options.pop("model_function_wrapper", None)
            else:
                unet.set_model_unet_function_wrapper(wrapper._krea2_previous)
            p.sd_model.forge_objects.unet = unet
        if p.sd_model.forge_objects.unet is getattr(p, "_krea2_output_unet", None) or getattr(wrapper, "_krea2_owned", False):
            if hasattr(p, "_krea2_base_unet"):
                p.sd_model.forge_objects.unet = p._krea2_base_unet
        self._clear_state(p)

    def postprocess(self, p, processed, *args):
        if hasattr(p, "_krea2_base_unet"):
            self.post_sample(p, None)
        self._clear_state(p)

    def postprocess_image(self, p, pp, *args):
        # The two controls are appended to preserve every existing script-argument index.
        if len(args) < 15 or not bool(args[13]):
            return
        amount = _bounded_float(args[14], 0.35, 0.0, 1.0)
        if amount <= 0.0:
            return
        filtered = _apply_moire_filter(pp.image, amount)
        if filtered is pp.image:
            return
        pp.image = filtered
        p.extra_generation_params["Qwen Moire Cleanup"] = True
        p.extra_generation_params["Qwen Moire Cleanup Strength"] = amount
