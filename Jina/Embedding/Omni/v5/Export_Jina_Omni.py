"""
Export_Jina_Omni.py — modular ONNX export + ORT runtime
=================================================================
jina-embeddings-v5-omni-nano: text + image + audio in ONE shared 768-dim space.

Plugin-LoRA design: every tower (Qwen3VL vision, Qwen2.5-Omni audio, EuroBERT/Llama text) is inlined
into ONE task-agnostic backbone; every per-task weight (text LoRA, vision merger, audio projector,
task tokens) is fed as a runtime input, not baked into N graph copies. An Omni_LoRA provider emits the
17 per-task tensors, run once per task at startup and cached as OrtValues. Only TEXT uses LoRA (r=32).

Task-agnostic graphs (exported ONCE):
    Omni_LoRA          task_index (fused) / ()         -> 17 tensors (8 LoRA + 6 merger + 2 proj + 1 tok)
    Omni_Embed         input_ids                       -> token embeddings (plain lookup)
    Omni_Vision        RAW image [1,3,H,W] + merger w  -> image features [n_img, 768]
    Omni_Audio         RAW waveform + feature_len + proj -> audio features [n_aud, 768]
    Omni_Combine_Image token_embeds + image_features   -> multimodal embeds (head ++ img ++ tail)
    Omni_Combine_Audio task_tokens + audio_features    -> multimodal embeds (start ++ aud ++ end)
    Omni_Main          multimodal embeds + input_ids + LoRA -> L2-normalized embedding

Combine = per-modality slice + concat (no scatter): text passes token_embeds through (no combine
graph), image splices features between head text and trailing EOS, audio brackets the features with
the audio_start/audio_end task tokens. Drops every Equal / NonZero / Where / ScatterND node.

Inlined preprocessing (runtime feeds raw media):
    VISION: a raw INPUT_IMAGE_SIZE image is resized in-graph to the FIXED target (factor*patch*merge)
            only when the two differ (QwenVL-style), then normalize -> patchify in-graph. The fixed
            target keeps the bilinear pos-embed + RoPE tables static; (x/255-mean)/std folds into the
            patch_embed Conv3d.
    AUDIO : Whisper log-mel via STFT_Process Conv1d STFT (center+reflect, n_fft=400, hop=160); mel
            sliced to feature_len (= R//hop) then split into 200-frame chunks by reshape.

Optimizations: fused QKV / gate_up, sum-based RMSNorm folded into the next linear, attention scale
folded into Q, flip-based RoPE, slice-only RoPE/mask buffers, int8 key-padding bias, gather-before-
norm pooling, per-modality slice+concat combine, OrtValue-only inter-graph chaining.

PARITY-CRITICAL (do not change):
    * Audio + merger use EXACT (erf) GELU, NOT tanh; tanh costs ~3e-3. Only the vision TOWER blocks
      are tanh. ORT optimization.enable_gelu_approximation MUST stay OFF (rewrites exact GELU to tanh).
    * STFT pad_mode MUST be "reflect" (Whisper center pad; "constant" corrupts the first mel frames).
    * Audio mel is sliced to feature_len so the token count matches; one extra silence token wrecks
      last-token pooling on short clips (cos ~0.64).
    * The key-padding mask is derived in-graph from input_ids (NOT the tokenizer); a static zero
      template silently attends to right-padding.

FUSE_LORA_INTO_EMBED: True -> one shared Omni_LoRA stacks all tasks (int32 task_index); False -> one
provider per task. PREVENT_F16_OVERFLOW prescales RMSNorm/LN inputs for fp16; default False is fp32
byte-identical (the scale cancels). Parity vs omni_embedding_standalone.py: ~1e-6.
"""

from __future__ import annotations

import gc
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import onnx
import onnxruntime
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parent))
from STFT_Process import STFT_Process                        # noqa: E402  (provided audio STFT module)
from transformers.audio_utils import mel_filter_bank         # noqa: E402  (CPU constant: mel filter bank)

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════
MODEL_DIR        = Path("/home/DakeQQ/Downloads/jina-embeddings-v5-omni-nano")   # model.safetensors + adapters/
OUTPUT_ROOT      = Path(__file__).resolve().parent / "Jina_Omni_ONNX"            # ONNX output root

DO_EXPORT        = True               # export the ONNX graphs (False: load existing)
RUN_DEMO         = True               # run the cross-modal similarity demo
# True: one shared Omni_LoRA stacks all tasks (int32 task_index); False: one provider per task.
FUSE_LORA_INTO_EMBED = True

# fp16 stability: prescale RMSNorm/LN inputs before squaring. False -> fp32 byte-identical (scale cancels).
PREVENT_F16_OVERFLOW = False
OVERFLOW_SCALE   = 0.01

OPSET            = 18                 # torch.onnx opset (dynamo=False)
MAX_SEQ_LEN      = 8192               # max input length; must be <= config.json

AUDIO_SR         = 16000
AUDIO_SECONDS    = 2.0
IMAGE_PLACEHOLDER = "<image>"

# Inlined IMAGE processor (QwenVL-style): a raw INPUT_IMAGE_SIZE image is resized in-graph to the FIXED
# internal target = factor * patch_size * merge_size (factor*32), which keeps the bilinear pos-embed /
# RoPE tables static. F.interpolate runs ONLY when INPUT_IMAGE_SIZE != the target (or DYNAMIC_IMAGE_SHAPE
# allows any H/W); a matching size traces to an identity (no Resize node) -> parity-exact.
IMAGE_HEIGHT_FACTOR = 16              # target_h = factor * patch_size * merge_size (16 -> 512)
IMAGE_WIDTH_FACTOR  = 16              # target_w = factor * patch_size * merge_size (16 -> 512)
INPUT_IMAGE_SIZE    = [512, 512]      # raw graph input image shape [H, W]; resized to the target when different
DYNAMIC_IMAGE_SHAPE = False           # False: static INPUT_IMAGE_SIZE; True: dynamic input H/W (always resize)

# Inlined AUDIO feature extractor (Whisper log-mel via STFT_Process), capped at AUDIO_MAX_SECONDS.
AUDIO_N_FFT         = 400             # Whisper n_fft
AUDIO_HOP           = 160             # Whisper hop_length
AUDIO_MAX_SECONDS   = 30.0            # truncation cap (Whisper chunk_length)

ORT_LOG          = False
ORT_FP16         = False              # FP16 ORT session opts (qdq accuracy level + fp16 cast optimizers)
ORT_Accelerate_Providers = []         # e.g. ['CUDAExecutionProvider'] or ['DmlExecutionProvider']; empty = CPU only
MAX_THREADS      = 0                  # 0 = let ORT decide
DEVICE_ID        = 0

# Execution provider configuration: pick device_type + provider_options from ORT_Accelerate_Providers.
if "CUDAExecutionProvider" in ORT_Accelerate_Providers:
    provider_options = [{"device_id": DEVICE_ID}]
    device_type = "cuda"
elif "DmlExecutionProvider" in ORT_Accelerate_Providers:
    provider_options = [{"device_id": DEVICE_ID}]
    device_type = "dml"
else:
    provider_options = None
    device_type = "cpu"

ORT_PROVIDERS = ORT_Accelerate_Providers if ORT_Accelerate_Providers else ["CPUExecutionProvider"]


# Repo configs loaded ONCE so every architecture constant is read from the model dir (adapts to any size).
def _load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


_CONFIG       = _load_json(MODEL_DIR / "config.json")               # architecture + special tokens + tasks
_PREPROCESSOR = _load_json(MODEL_DIR / "preprocessor_config.json")  # image_mean / image_std
_TOKENIZER    = _load_json(MODEL_DIR / "tokenizer.json")            # pad_id
_VCFG = _CONFIG["vision_config"]
_TCFG = _CONFIG["text_config"]
_ACFG = _CONFIG["audio_config"]


# ══════════════════════════════════════════════════════════════════════════════
# Architecture constants read from the repo configs (LN eps / vision RoPE base fall back to defaults).
# ══════════════════════════════════════════════════════════════════════════════
class Cfg:
    # vision (Qwen3VL) — config.json vision_config
    v_hidden = _VCFG["hidden_size"]
    v_depth = _VCFG["depth"]
    v_heads = _VCFG["num_heads"]
    v_head_dim = v_hidden // v_heads
    v_inter = _VCFG["intermediate_size"]
    v_patch = _VCFG["patch_size"]
    v_temporal_patch = _VCFG["temporal_patch_size"]
    v_in_ch = _VCFG["in_channels"]
    v_merge = _VCFG["spatial_merge_size"]
    v_num_pos = _VCFG["num_position_embeddings"]
    v_grid_side = int(v_num_pos ** 0.5)
    v_rope_theta = _VCFG.get("rope_theta", 10000.0)        # not in config.json (Qwen3VL vision default)
    v_ln_eps = _VCFG.get("layer_norm_eps", 1e-6)           # not in config.json

    # text (EuroBERT / Llama, bidirectional) — config.json text_config
    t_hidden = _TCFG["hidden_size"]
    t_layers = _TCFG["num_hidden_layers"]
    t_heads = _TCFG["num_attention_heads"]
    t_kv_heads = _TCFG["num_key_value_heads"]
    t_head_dim = _TCFG["head_dim"]
    t_inter = _TCFG["intermediate_size"]
    t_vocab = _TCFG["vocab_size"]
    t_rms_eps = _TCFG["rms_norm_eps"]
    t_rope_theta = (_TCFG.get("rope_parameters") or _TCFG).get("rope_theta", 1_000_000.0)

    # audio (Qwen2.5-Omni encoder) — config.json audio_config
    a_dmodel = _ACFG["d_model"]
    a_layers = _ACFG["encoder_layers"]
    a_heads = _ACFG["encoder_attention_heads"]
    a_head_dim = a_dmodel // a_heads
    a_ffn = _ACFG["encoder_ffn_dim"]
    a_mel = _ACFG["num_mel_bins"]
    a_max_pos = _ACFG["max_source_positions"]
    a_n_window = _ACFG["n_window"]
    a_ln_eps = _ACFG.get("layer_norm_eps", 1e-5)           # not in config.json

    # special tokens — config.json (+ tokenizer.json pad_id; text_config.pad_token_id is null)
    image_token_id = _CONFIG["image_token_index"]
    audio_token_id = _CONFIG["audio_token_id"]
    audio_start_token_id = _CONFIG["audio_start_token_id"]
    audio_end_token_id = _CONFIG["audio_end_token_id"]
    special_token_ids = _CONFIG["special_token_ids"]
    pad_token_id = (_TOKENIZER.get("padding") or {}).get("pad_id", 128004)   # valid keys = (input_ids != pad)

    matryoshka_dims = _CONFIG["matryoshka_dimensions"]


def task_key(task: str) -> str:
    """text-matching -> text_matching (safetensors key convention)."""
    return task.replace("-", "_")


def task_dir(task: str) -> Path:
    return OUTPUT_ROOT / task


# ══════════════════════════════════════════════════════════════════════════════
# CPU index helpers (ported from transformers; outputs feed the graphs as constants).
# ══════════════════════════════════════════════════════════════════════════════
def vision_position_ids(grid_thw: torch.Tensor, merge: int) -> torch.Tensor:
    pos = []
    for t, h, w in grid_thw.tolist():
        t, h, w = int(t), int(h), int(w)
        hpos = torch.arange(h).unsqueeze(1).expand(-1, w)
        hpos = hpos.reshape(h // merge, merge, w // merge, merge).transpose(1, 2).flatten()
        wpos = torch.arange(w).unsqueeze(0).expand(h, -1)
        wpos = wpos.reshape(h // merge, merge, w // merge, merge).transpose(1, 2).flatten()
        pos.append(torch.stack([hpos, wpos], dim=-1).repeat(t, 1))
    return torch.cat(pos, dim=0)


def vision_rope_tables(pos_ids: torch.Tensor):
    """cos = cat(cos, cos); sin = cat(-sin, sin) (sign baked for the flip trick)."""
    hd = Cfg.v_head_dim
    inv = 1.0 / (Cfg.v_rope_theta ** (torch.arange(0, hd // 2, 2).float() / (hd // 2)))   # [hd/4]
    rot = (pos_ids.unsqueeze(-1) * inv).flatten(1)                                        # [seq, hd/2]
    emb = torch.cat([rot, rot], dim=-1)                                                   # [seq, hd]
    cos = emb.cos()
    sin = emb.sin()
    sin_flip = torch.cat([-sin[:, : hd // 2], sin[:, hd // 2:]], dim=-1)
    return cos.contiguous(), sin_flip.contiguous()


def vision_bilinear_indices_and_weights(grid_thw, side: int, merge: int):
    idx_parts = [[] for _ in range(4)]
    w_parts = [[] for _ in range(4)]
    for t, h, w in grid_thw.tolist():
        t, h, w = int(t), int(h), int(w)
        h_grid = torch.linspace(0, side - 1, h)
        w_grid = torch.linspace(0, side - 1, w)
        h_floor = h_grid.int()
        w_floor = w_grid.int()
        h_ceil = (h_floor + 1).clamp(max=side - 1)
        w_ceil = (w_floor + 1).clamp(max=side - 1)
        h_frac = h_grid - h_floor
        w_frac = w_grid - w_floor
        h_fo = h_floor * side
        h_co = h_ceil * side
        corner_idx = [
            (h_fo[:, None] + w_floor[None, :]).flatten(),
            (h_fo[:, None] + w_ceil[None, :]).flatten(),
            (h_co[:, None] + w_floor[None, :]).flatten(),
            (h_co[:, None] + w_ceil[None, :]).flatten(),
        ]
        corner_w = [
            ((1 - h_frac)[:, None] * (1 - w_frac)[None, :]).flatten(),
            ((1 - h_frac)[:, None] * w_frac[None, :]).flatten(),
            (h_frac[:, None] * (1 - w_frac)[None, :]).flatten(),
            (h_frac[:, None] * w_frac[None, :]).flatten(),
        ]
        h_idx = torch.arange(h).view(h // merge, merge)
        w_idx = torch.arange(w).view(w // merge, merge)
        reorder = (h_idx[:, :, None, None] * w + w_idx[None, None, :, :]).transpose(1, 2).flatten().repeat(t)
        for i in range(4):
            idx_parts[i].append(corner_idx[i][reorder])
            w_parts[i].append(corner_w[i][reorder])
    bilinear_indices = torch.stack([torch.cat(p) for p in idx_parts])
    bilinear_weights = torch.stack([torch.cat(p) for p in w_parts])
    return bilinear_indices, bilinear_weights


def sinusoids(length: int, channels: int, max_timescale: float = 10000.0) -> torch.Tensor:
    log_inc = math.log(max_timescale) / (channels // 2 - 1)
    inv = torch.exp(-log_inc * torch.arange(channels // 2).float())
    scaled = torch.arange(length)[:, None].float() * inv[None, :]
    return torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=1)


# ══════════════════════════════════════════════════════════════════════════════
# Weight loading: base backbone loaded ONCE; every per-task tensor (LoRA, merger, projector, task
# tokens) is built separately and fed to the shared graphs as a runtime input.
# ══════════════════════════════════════════════════════════════════════════════
# 8 stacked LoRA tensors fed to the shared Omni_Main (block-diag fused).
LORA_INPUT_NAMES = [
    "lora_qkv_a", "lora_qkv_b",
    "lora_o_a", "lora_o_b",
    "lora_gate_up_a", "lora_gate_up_b",
    "lora_down_a", "lora_down_b",
]
# 6 per-task vision-merger tensors fed to the shared Omni_Vision.
MERGER_INPUT_NAMES = [
    "merger_norm_w", "merger_norm_b",
    "merger_fc1_w", "merger_fc1_b",
    "merger_fc2_w", "merger_fc2_b",
]
# 2 per-task audio-projector tensors fed to the shared Omni_Audio.
PROJECTOR_INPUT_NAMES = ["proj_w", "proj_b"]
# Per-task task-token table fed to Omni_Combine_Audio (audio_start/audio_end rows).
TASK_TOKEN_INPUT_NAME = "task_token_embeds"


def load_base_weights(model_dir: Path) -> dict:
    """Task-agnostic backbone weights (NO LoRA merge, NO per-task merger/projector/tokens).
    Text linears keep BASE weights; per-task LoRA is supplied by build_task_tensors() at runtime."""
    # stored bfloat16 -> upcast to float32 before any arithmetic
    sd = {k: v.float() for k, v in load_file(str(model_dir / "model.safetensors")).items()}

    W = {}

    # ---- vision tower (shared) ----
    W["vision_tower.patch_embed.weight"] = sd["vision_tower.patch_embed.proj.weight"]
    W["vision_tower.patch_embed.bias"] = sd["vision_tower.patch_embed.proj.bias"]
    W["vision_tower.pos_embed.weight"] = sd["vision_tower.pos_embed.weight"]
    for i in range(Cfg.v_depth):
        s = f"vision_tower.blocks.{i}."
        W[s + "norm1.weight"] = sd[s + "norm1.weight"]
        W[s + "norm1.bias"] = sd[s + "norm1.bias"]
        W[s + "norm2.weight"] = sd[s + "norm2.weight"]
        W[s + "norm2.bias"] = sd[s + "norm2.bias"]
        W[s + "qkv.weight"] = sd[s + "attn.qkv.weight"]
        W[s + "qkv.bias"] = sd[s + "attn.qkv.bias"]
        W[s + "proj.weight"] = sd[s + "attn.proj.weight"]
        W[s + "proj.bias"] = sd[s + "attn.proj.bias"]
        W[s + "linear_fc1.weight"] = sd[s + "mlp.linear_fc1.weight"]
        W[s + "linear_fc1.bias"] = sd[s + "mlp.linear_fc1.bias"]
        W[s + "linear_fc2.weight"] = sd[s + "mlp.linear_fc2.weight"]
        W[s + "linear_fc2.bias"] = sd[s + "mlp.linear_fc2.bias"]

    # ---- audio tower (shared); ln_post applied affine in-graph, NOT folded ----
    for p in ("conv1.weight", "conv1.bias", "conv2.weight", "conv2.bias",
              "ln_post.weight", "ln_post.bias"):
        W["audio_tower." + p] = sd["audio_tower." + p]
    for i in range(Cfg.a_layers):
        s = f"audio_tower.layers.{i}."
        W[s + "self_attn_layer_norm.weight"] = sd[s + "self_attn_layer_norm.weight"]
        W[s + "self_attn_layer_norm.bias"] = sd[s + "self_attn_layer_norm.bias"]
        W[s + "q_proj.weight"] = sd[s + "self_attn.q_proj.weight"]
        W[s + "q_proj.bias"] = sd[s + "self_attn.q_proj.bias"]
        W[s + "k_proj.weight"] = sd[s + "self_attn.k_proj.weight"]    # no bias
        W[s + "v_proj.weight"] = sd[s + "self_attn.v_proj.weight"]
        W[s + "v_proj.bias"] = sd[s + "self_attn.v_proj.bias"]
        W[s + "out_proj.weight"] = sd[s + "self_attn.out_proj.weight"]
        W[s + "out_proj.bias"] = sd[s + "self_attn.out_proj.bias"]
        W[s + "final_layer_norm.weight"] = sd[s + "final_layer_norm.weight"]
        W[s + "final_layer_norm.bias"] = sd[s + "final_layer_norm.bias"]
        W[s + "fc1.weight"] = sd[s + "fc1.weight"]
        W[s + "fc1.bias"] = sd[s + "fc1.bias"]
        W[s + "fc2.weight"] = sd[s + "fc2.weight"]
        W[s + "fc2.bias"] = sd[s + "fc2.bias"]

    # ---- text model: BASE embed + norm + BASE linears (no LoRA merge, no task tokens) ----
    W["text_model.embed_tokens.weight"] = sd["language_model.embed_tokens.weight"].clone()
    W["text_model.norm.weight"] = sd["language_model.norm.weight"]
    proj_map = {
        "q_proj": "self_attn.q_proj", "k_proj": "self_attn.k_proj",
        "v_proj": "self_attn.v_proj", "o_proj": "self_attn.o_proj",
        "gate_proj": "mlp.gate_proj", "up_proj": "mlp.up_proj", "down_proj": "mlp.down_proj",
    }
    for i in range(Cfg.t_layers):
        s = f"language_model.layers.{i}."
        d = f"text_model.layers.{i}."
        W[d + "input_layernorm.weight"] = sd[s + "input_layernorm.weight"]
        W[d + "post_attention_layernorm.weight"] = sd[s + "post_attention_layernorm.weight"]
        for dst, src in proj_map.items():
            W[d + dst + ".weight"] = sd[s + src + ".weight"].clone()

    return W


def build_task_tensors(model_dir: Path, task: str, sd: dict | None = None) -> dict:
    """Per-task runtime-input tensors (keyed by the *_INPUT_NAMES):
        * 8 block-diag-fused LoRA tensors for Omni_Main. The input_layernorm*sqrt(H) (and Q scale)
          that Omni_Main folds into the BASE qkv / gate_up is pre-multiplied into the matching lora_A
          (and the q-block of lora_qkv_b) here, so the delta matches base PEFT after the fold.
        * 6 raw vision-merger tensors for Omni_Vision (LayerNorm applied affine in-graph).
        * 2 raw audio-projector tensors for Omni_Audio (ln_post applied affine in-graph).
        * the task-token table for Omni_Combine_Audio.
    """
    tk = task_key(task)
    # reuse a preloaded fp32 state dict to avoid reloading model.safetensors per task
    if sd is None:
        sd = {k: v.float() for k, v in load_file(str(model_dir / "model.safetensors")).items()}
    lora = {k: v.float() for k, v in load_file(str(model_dir / "adapters" / task / "adapter_model.safetensors")).items()}

    H = Cfg.t_hidden
    norm_factor = H ** 0.5                 # sum-RMSNorm compensation folded into the next linear
    qscale = Cfg.t_head_dim ** -0.5        # attention scale folded into Q
    scaling = 32.0 / 32.0                  # lora_alpha / r == 1.0

    all_qkv_a, all_qkv_b = [], []
    all_o_a, all_o_b = [], []
    all_gate_up_a, all_gate_up_b = [], []
    all_down_a, all_down_b = [], []
    for i in range(Cfg.t_layers):
        s = f"language_model.layers.{i}."
        lp = f"base_model.model.language_model.layers.{i}."
        # fold layernorm*sqrt(H) into lora_A so the delta matches the base qkv / gate_up fold
        attn_in_scale = (sd[s + "input_layernorm.weight"] * norm_factor).unsqueeze(0)          # [1, H]
        mlp_in_scale = (sd[s + "post_attention_layernorm.weight"] * norm_factor).unsqueeze(0)  # [1, H]

        def la(mod):
            return lora[lp + mod + ".lora_A.weight"]

        def lb(mod):
            return lora[lp + mod + ".lora_B.weight"] * scaling

        # q/k/v share the attn input-norm scale on lora_A; q ALSO carries the folded Q scale on lora_B.
        q_a = la("self_attn.q_proj") * attn_in_scale
        k_a = la("self_attn.k_proj") * attn_in_scale
        v_a = la("self_attn.v_proj") * attn_in_scale
        q_b = lb("self_attn.q_proj") * qscale
        k_b = lb("self_attn.k_proj")
        v_b = lb("self_attn.v_proj")
        # block-diag B so each role only mixes its own rank-r subspace
        all_qkv_a.append(torch.cat([q_a, k_a, v_a], dim=0))
        all_qkv_b.append(torch.block_diag(q_b, k_b, v_b))
        # o_proj input is the attention output -> no input-norm fold on lora_A
        all_o_a.append(la("self_attn.o_proj"))
        all_o_b.append(lb("self_attn.o_proj"))
        # gate/up share the mlp input-norm scale on lora_A; block-diag B
        all_gate_up_a.append(torch.cat([la("mlp.gate_proj") * mlp_in_scale, la("mlp.up_proj") * mlp_in_scale], dim=0))
        all_gate_up_b.append(torch.block_diag(lb("mlp.gate_proj"), lb("mlp.up_proj")))
        # down_proj input is the SwiGLU output -> no input-norm fold on lora_A
        all_down_a.append(la("mlp.down_proj"))
        all_down_b.append(lb("mlp.down_proj"))

    T = {
        "lora_qkv_a": torch.stack(all_qkv_a).contiguous(),
        "lora_qkv_b": torch.stack(all_qkv_b).contiguous(),
        "lora_o_a": torch.stack(all_o_a).contiguous(),
        "lora_o_b": torch.stack(all_o_b).contiguous(),
        "lora_gate_up_a": torch.stack(all_gate_up_a).contiguous(),
        "lora_gate_up_b": torch.stack(all_gate_up_b).contiguous(),
        "lora_down_a": torch.stack(all_down_a).contiguous(),
        "lora_down_b": torch.stack(all_down_b).contiguous(),
    }

    # ---- vision merger (raw; affine LayerNorm done in-graph) ----
    T["merger_norm_w"] = sd[f"mergers.{tk}.norm.weight"].contiguous()
    T["merger_norm_b"] = sd[f"mergers.{tk}.norm.bias"].contiguous()
    T["merger_fc1_w"] = sd[f"mergers.{tk}.linear_fc1.weight"].contiguous()
    T["merger_fc1_b"] = sd[f"mergers.{tk}.linear_fc1.bias"].contiguous()
    T["merger_fc2_w"] = sd[f"mergers.{tk}.linear_fc2.weight"].contiguous()
    T["merger_fc2_b"] = sd[f"mergers.{tk}.linear_fc2.bias"].contiguous()

    # ---- audio projector (raw; ln_post done affine in-graph) ----
    T["proj_w"] = sd[f"audio_projectors.{tk}.weight"].contiguous()
    T["proj_b"] = sd[f"audio_projectors.{tk}.bias"].contiguous()

    # ---- task-token table; the audio_start/audio_end rows bracket the audio features in Omni_Combine_Audio ----
    T[TASK_TOKEN_INPUT_NAME] = sd[f"task_token_embeddings.{tk}"].contiguous()

    return T


# ══════════════════════════════════════════════════════════════════════════════
# Fusion helpers
# ══════════════════════════════════════════════════════════════════════════════
def fuse_ln_into_linear(ln_w, ln_b, lin_w, lin_b=None):
    """Fold LayerNorm affine into a following Linear: Linear(LN_affine(x)) =
    (lin_w*ln_w) @ x_norm + (lin_w @ ln_b + lin_b), x_norm = affine-free normalize.
    Handles the merger case where the Linear input is repeat_factor copies of the LN'd vector.
    """
    out_f, in_f = lin_w.shape
    if in_f != ln_w.shape[0]:
        rep = in_f // ln_w.shape[0]
        ln_w = ln_w.repeat(rep)
        ln_b = ln_b.repeat(rep)
    new_w = lin_w * ln_w.unsqueeze(0)
    base_b = lin_b if lin_b is not None else torch.zeros(out_f, dtype=lin_w.dtype)
    new_b = base_b + lin_w @ ln_b
    return new_w.contiguous(), new_b.contiguous()


def layernorm_no_affine(x, eps):
    """LayerNorm without the affine (weight/bias) — folded into the next linear."""
    if PREVENT_F16_OVERFLOW:
        # prescale (cancels through the normalize -> fp32 unchanged)
        x = x * OVERFLOW_SCALE
        eps = eps * (OVERFLOW_SCALE * OVERFLOW_SCALE)
    mu = x.mean(-1, keepdim=True)
    xc = x - mu
    var = (xc * xc).mean(-1, keepdim=True)
    return xc * torch.rsqrt(var + eps)


def sum_rms_norm(x, eps_sum):
    """Sum-based RMSNorm: x * rsqrt(sum(x^2) + H*eps). The per-channel weight*sqrt(H) is absorbed
    into the next linear, so this returns the raw direction only."""
    if PREVENT_F16_OVERFLOW:
        # prescale (cancels through the normalize -> fp32 unchanged)
        x = x * OVERFLOW_SCALE
        eps_sum = eps_sum * (OVERFLOW_SCALE * OVERFLOW_SCALE)
    return x * torch.rsqrt(x.square().sum(-1, keepdim=True) + eps_sum)


def flip_rotate_half(x):
    """rotate_half via reshape+flip (no runtime concat/negate): the sin buffer carries the sign flip,
    so this reproduces the standard cat(-x2, x1). Only the constant last dim is split (no Shape/Gather)."""
    return x.unflatten(-1, (2, -1)).flip(-2).flatten(-2)


# ══════════════════════════════════════════════════════════════════════════════
# 1) OMNI_EMBED — plain token-embedding lookup (task tokens consumed by Omni_Combine_Audio)
# ══════════════════════════════════════════════════════════════════════════════
class OmniEmbed(nn.Module):
    """Plain token-embedding lookup (a single Gather). The per-task task tokens are consumed by
    Omni_Combine_Audio, not here, so this graph never materializes the full [vocab, H] table."""

    def __init__(self, W):
        super().__init__()
        self.register_buffer("weight", W["text_model.embed_tokens.weight"].contiguous())

    def forward(self, input_ids):
        # input_ids is int32 -> a single int32 Gather (F.embedding accepts int32)
        return F.embedding(input_ids, self.weight)


# ══════════════════════════════════════════════════════════════════════════════
# 2) OMNI_VISION — Qwen3VL vision tower (shared) + per-task merger (weights as inputs)
# ══════════════════════════════════════════════════════════════════════════════
class OmniVision(nn.Module):
    """Single-image vision encoder with the image processor inlined: RAW RGB [1,3,H,W] (0-255) ->
    resize (fixed target) -> normalize (folded into patch_embed) -> patchify -> tower -> merger.
    The FIXED target makes the bilinear pos-embed + flip-RoPE tables static and the token count
    constant. Tower LayerNorms fold into the next linears; the Q·K scale folds into qkv. The per-task
    merger arrives as 6 forward inputs (LayerNorm applied affine in-graph).

    Inputs: image [1, 3, H, W] (0-255 float) + 6 merger weights.
    Output: image_features [num_merged, 768]  (num_merged = (target_h/32) * (target_w/32))
    """

    def __init__(self, W, target_h, target_w, dynamic_shape=True):
        super().__init__()
        hd = Cfg.v_head_dim
        scale = hd ** -0.25                       # folded into q AND k slices
        self.heads = Cfg.v_heads
        self.head_dim = hd
        self.target_h = target_h
        self.target_w = target_w
        self.dynamic_shape = dynamic_shape
        self.grid_h = target_h // Cfg.v_patch     # patches along H (e.g. 512/16 = 32)
        self.grid_w = target_w // Cfg.v_patch
        self.gh_m = self.grid_h // Cfg.v_merge    # merged-block grid (e.g. 16)
        self.gw_m = self.grid_w // Cfg.v_merge
        self.num_merged = self.gh_m * self.gw_m   # merged image tokens (e.g. 256)

        # Fold (x/255 - mean)/std normalization into the patch_embed Conv3d:
        #   conv(W, (x/255-mean)/std) = conv(W/(255*std), x) + (b - sum(W/(255*std) * 255*mean)).
        patch_w = W["vision_tower.patch_embed.weight"].clone()        # [768, 3, 2, 16, 16]
        patch_b = W["vision_tower.patch_embed.bias"].clone()
        mean = torch.tensor(_PREPROCESSOR["image_mean"], dtype=torch.float32).view(1, Cfg.v_in_ch, 1, 1, 1)
        std = torch.tensor(_PREPROCESSOR["image_std"], dtype=torch.float32).view(1, Cfg.v_in_ch, 1, 1, 1)
        patch_w = patch_w / (255.0 * std)
        patch_b = patch_b - (patch_w * (255.0 * mean)).sum(dim=[1, 2, 3, 4])
        self.patch_w = nn.Parameter(patch_w.contiguous(), requires_grad=False)
        self.patch_b = nn.Parameter(patch_b.contiguous(), requires_grad=False)

        # fixed grid -> bilinear pos-embed + flip-RoPE cos/sin are constant buffers (no runtime inputs)
        grid_thw = torch.tensor([[1, self.grid_h, self.grid_w]])
        bidx, bw = vision_bilinear_indices_and_weights(grid_thw, Cfg.v_grid_side, Cfg.v_merge)
        pos = (F.embedding(bidx.int(), W["vision_tower.pos_embed.weight"]) * bw.unsqueeze(-1)).sum(0)
        self.register_buffer("pos_embed", pos.contiguous())                       # [num_patches, 768]
        cos, sin = vision_rope_tables(vision_position_ids(grid_thw, Cfg.v_merge))
        self.register_buffer("rope_cos", cos.float().contiguous())               # [num_patches, head_dim]
        self.register_buffer("rope_sin", sin.float().contiguous())

        self.qkv_w, self.qkv_b = nn.ParameterList(), nn.ParameterList()
        self.proj_w, self.proj_b = nn.ParameterList(), nn.ParameterList()
        self.fc1_w, self.fc1_b = nn.ParameterList(), nn.ParameterList()
        self.fc2_w, self.fc2_b = nn.ParameterList(), nn.ParameterList()
        for i in range(Cfg.v_depth):
            s = f"vision_tower.blocks.{i}."
            # norm1 folded into qkv, norm2 folded into linear_fc1
            qkv_w, qkv_b = fuse_ln_into_linear(W[s + "norm1.weight"], W[s + "norm1.bias"], W[s + "qkv.weight"], W[s + "qkv.bias"])
            # fold the Q·K scale (head_dim^-0.25 each) into the q and k row-blocks of qkv (no runtime scale)
            H = Cfg.v_hidden
            qkv_w[0:H] *= scale
            qkv_b[0:H] *= scale
            qkv_w[H:2 * H] *= scale
            qkv_b[H:2 * H] *= scale
            fc1_w, fc1_b = fuse_ln_into_linear(W[s + "norm2.weight"], W[s + "norm2.bias"], W[s + "linear_fc1.weight"], W[s + "linear_fc1.bias"])
            self.qkv_w.append(nn.Parameter(qkv_w, requires_grad=False))
            self.qkv_b.append(nn.Parameter(qkv_b, requires_grad=False))
            self.proj_w.append(nn.Parameter(W[s + "proj.weight"], requires_grad=False))
            self.proj_b.append(nn.Parameter(W[s + "proj.bias"], requires_grad=False))
            self.fc1_w.append(nn.Parameter(fc1_w, requires_grad=False))
            self.fc1_b.append(nn.Parameter(fc1_b, requires_grad=False))
            self.fc2_w.append(nn.Parameter(W[s + "linear_fc2.weight"], requires_grad=False))
            self.fc2_b.append(nn.Parameter(W[s + "linear_fc2.bias"], requires_grad=False))

        # per-task merger weights arrive as forward inputs (LayerNorm applied affine in-graph, no fold)
        self.merge_in = Cfg.v_hidden * (Cfg.v_merge ** 2)   # 3072

    def forward(self, image, merger_norm_w, merger_norm_b, merger_fc1_w, merger_fc1_b,
                merger_fc2_w, merger_fc2_b):
        # image [1, 3, H, W] in 0-255 (any H,W): resize -> patchify -> patch-embed (normalize folded).
        img = image.float()
        # in-graph resize to the FIXED target (identity when input already == target)
        if self.dynamic_shape or img.shape[-2] != self.target_h or img.shape[-1] != self.target_w:
            img = F.interpolate(img, size=[self.target_h, self.target_w], mode="bilinear", align_corners=False)
        # spatial patchify into merge-block order (matches pos_embed / rope order):
        #   [1,3,H,W] -> [1,3, gh_m, m, p, gw_m, m, p] -> [gh_m*gw_m*m*m, 3, 1, p, p] -> dup temporal
        img = img.reshape(1, Cfg.v_in_ch, self.gh_m, Cfg.v_merge, Cfg.v_patch, self.gw_m, Cfg.v_merge, Cfg.v_patch)
        img = img.permute(0, 2, 5, 3, 6, 1, 4, 7).reshape(-1, Cfg.v_in_ch, 1, Cfg.v_patch, Cfg.v_patch)
        img = torch.cat([img, img], dim=2)        # temporal_patch_size=2 duplicate -> [n, 3, 2, 16, 16]

        # patch embed via Conv3d (normalization folded into patch_w / patch_b)
        x = F.conv3d(img, self.patch_w, self.patch_b, stride=[Cfg.v_temporal_patch, Cfg.v_patch, Cfg.v_patch]).view(-1, Cfg.v_hidden)

        # static learned position embedding (baked buffer)
        x = x + self.pos_embed

        # cos/sin stay [seq, head_dim] and broadcast over the [2, heads, seq, hd] stack
        for i in range(Cfg.v_depth):
            res = x
            hn = layernorm_no_affine(x, Cfg.v_ln_eps)
            # [3, heads, seq, hd] layout -> rotate Q&K together, no per-tensor transpose
            qkv = F.linear(hn, self.qkv_w[i], self.qkv_b[i]).view(-1, 3, self.heads, self.head_dim).permute(1, 2, 0, 3)
            # split (not integer index) keeps the stack axis -> q/k/v as plain slices, no Gather
            qk, v = qkv.split([2, 1], dim=0)            # qk:[2,heads,seq,hd]  v:[1,heads,seq,hd]
            qk = qk * self.rope_cos + flip_rotate_half(qk) * self.rope_sin
            q, k = qk.split([1, 1], dim=0)             # each [1, heads, seq, head_dim]
            attn = torch.matmul(q, k.transpose(2, 3))  # scale folded into qkv
            attn = torch.softmax(attn, dim=-1)
            out = torch.matmul(attn, v).transpose(1, 2).reshape(-1, Cfg.v_hidden)
            x = res + F.linear(out, self.proj_w[i], self.proj_b[i])

            res = x
            hn = layernorm_no_affine(x, Cfg.v_ln_eps)
            h = F.gelu(F.linear(hn, self.fc1_w[i], self.fc1_b[i]), approximate="tanh")
            x = res + F.linear(h, self.fc2_w[i], self.fc2_b[i])

        # per-task merger: affine LayerNorm in-graph (input weights), reshape to 3072, fc1/fc2.
        # EXACT (erf) GELU here (config projector_hidden_act="gelu"), NOT the tower's tanh GELU.
        mn = (layernorm_no_affine(x, Cfg.v_ln_eps) * merger_norm_w + merger_norm_b).reshape(-1, self.merge_in)
        m = F.gelu(F.linear(mn, merger_fc1_w, merger_fc1_b))
        return F.linear(m, merger_fc2_w, merger_fc2_b)


# ══════════════════════════════════════════════════════════════════════════════
# 3) OMNI_AUDIO — Qwen2.5-Omni audio encoder (shared) + per-task projector (weights as inputs)
# ══════════════════════════════════════════════════════════════════════════════
class OmniAudio(nn.Module):
    """Audio encoder with the Whisper feature extractor inlined via STFT_Process. RAW waveform
    [1,1,audio_len] runs in-graph: STFT (center+reflect, n_fft=400, hop=160) -> power -> mel ->
    log10 -> Whisper normalize -> slice to feature_len -> 200-frame chunks -> conv1/conv2 ->
    attention -> stride-2 pool -> ln_post -> projector.
    LayerNorms fold into q/k/v (k_proj gains a bias), final_layer_norm into fc1, attention scale into Q.
    Per-task projector arrives as forward inputs (ln_post affine in-graph); all chunk/mask/valid/pool
    indices derive in-graph, so the only audio inputs are waveform, feature_len and projector weights.

    Inputs:
        waveform    [1, 1, audio_len] int16   raw PCM (CPU zero-pads with an n_fft margin); forward()
                                              normalizes to float [-1, 1] via /32768
        feature_len [1] int64                 Whisper valid-frame count (R // hop); mel is sliced to
                                              this so the token count matches (extra frame -> +1 token
                                              -> cos ~0.64 on short clips)
        proj_w, proj_b                        per-task projector weight [768, 1280] + bias [768]
    Output: audio_features [num_pooled, 768]
    """

    def __init__(self, W, max_frames):
        super().__init__()
        hd = Cfg.a_head_dim
        qscale = hd ** -0.5                        # full attention scale folded into Q
        self.heads = Cfg.a_heads
        self.head_dim = hd

        # Provided STFT_Process Conv1d STFT (stft_B = real+imag), center + reflect, hann.
        # pad_mode MUST be "reflect" (Whisper center pad); "constant" corrupts the first ~2 mel frames.
        self.stft = STFT_Process(
            model_type="stft_B", n_fft=AUDIO_N_FFT, win_length=AUDIO_N_FFT, hop_len=AUDIO_HOP,
            max_frames=max_frames, window_type="hann", center_pad=True, pad_mode="reflect",
        )
        # Whisper log-mel filterbank (CPU constant), stored 3D [1, n_mel, n_freq] to batch-matmul the
        # [1, n_freq, T] power (no Gather). The int16->[-1,1] scale is linear through the STFT, so its
        # square (2**-30) folds into the filterbank -> forward just casts int16 -> float.
        pcm_scale_sq = (1.0 / 32768.0) ** 2
        mel_fb = mel_filter_bank(
            num_frequency_bins=AUDIO_N_FFT // 2 + 1, num_mel_filters=Cfg.a_mel,
            min_frequency=0.0, max_frequency=AUDIO_SR / 2.0, sampling_rate=AUDIO_SR,
            norm="slaney", mel_scale="slaney",
        )
        self.register_buffer("mel_filters", (torch.from_numpy(mel_fb).float() * pcm_scale_sq).t().unsqueeze(0).contiguous())

        self.conv1_w = nn.Parameter(W["audio_tower.conv1.weight"], requires_grad=False)
        self.conv1_b = nn.Parameter(W["audio_tower.conv1.bias"], requires_grad=False)
        self.conv2_w = nn.Parameter(W["audio_tower.conv2.weight"], requires_grad=False)
        self.conv2_b = nn.Parameter(W["audio_tower.conv2.bias"], requires_grad=False)
        # pos_emb pre-shaped [1, max_pos, dmodel] so the runtime add is a slice (no unsqueeze)
        self.register_buffer("pos_emb", sinusoids(Cfg.a_max_pos, Cfg.a_dmodel).unsqueeze(0).contiguous().half())

        # in-graph chunking: every chunk padded to chunk_size so the mel splits by a plain reshape
        # (no gather); per-chunk masks/lengths derive from feature_len.
        self.chunk_size = Cfg.a_n_window * 2                  # 200 mel frames per chunk
        self.chunk_minus = self.chunk_size - 1               # 199 (ceil-div helper)
        self.chunk_aftercnn = (self.chunk_size - 1) // 2 + 1  # 100 post-conv2 tokens per full chunk
        # full mel upper bound (n_fft reflect margin): ceil((max_samples + n_fft)/hop)*hop // hop + 1
        max_L_pad = ((int(AUDIO_MAX_SECONDS * AUDIO_SR) + AUDIO_N_FFT + AUDIO_HOP - 1) // AUDIO_HOP) * AUDIO_HOP
        max_feature_len = max_L_pad // AUDIO_HOP + 1
        max_chunks = (max_feature_len + self.chunk_minus) // self.chunk_size
        # chunk_starts_full[c] = c * chunk_size (sliced [:nc])
        self.register_buffer("chunk_starts_full",
                             (torch.arange(max_chunks + 1, dtype=torch.int64) * self.chunk_size).contiguous())
        # right-pad buffer (3D [1, n_mel, chunk_size]); pad_frames < chunk_size, so one column suffices
        self.register_buffer("mel_pad_zeros", torch.zeros(1, Cfg.a_mel, self.chunk_size, dtype=torch.int8))
        # static position ramps for the in-graph conv mask / attention key mask
        self.register_buffer("frame_pos", torch.arange(self.chunk_size, dtype=torch.int64))
        self.register_buffer("tok_pos", torch.arange(self.chunk_aftercnn, dtype=torch.int64))

        self.qkv_w, self.qkv_b = nn.ParameterList(), nn.ParameterList()
        self.o_w, self.o_b = nn.ParameterList(), nn.ParameterList()
        self.fc1_w, self.fc1_b = nn.ParameterList(), nn.ParameterList()
        self.fc2_w, self.fc2_b = nn.ParameterList(), nn.ParameterList()
        for i in range(Cfg.a_layers):
            s = f"audio_tower.layers.{i}."
            ln_w, ln_b = W[s + "self_attn_layer_norm.weight"], W[s + "self_attn_layer_norm.bias"]
            # self_attn_layer_norm folded into q/k/v (k_proj has no bias -> gains one from the fold)
            q_w, q_b = fuse_ln_into_linear(ln_w, ln_b, W[s + "q_proj.weight"], W[s + "q_proj.bias"])
            k_w, k_b = fuse_ln_into_linear(ln_w, ln_b, W[s + "k_proj.weight"], None)   # k gains a bias
            v_w, v_b = fuse_ln_into_linear(ln_w, ln_b, W[s + "v_proj.weight"], W[s + "v_proj.bias"])
            q_w, q_b = q_w * qscale, q_b * qscale                                       # fold head_dim^-0.5 into Q
            # fused QKV: one GEMM weight = cat([q, k, v]) (q carries the folded scale, k the folded bias)
            self.qkv_w.append(nn.Parameter(torch.cat([q_w, k_w, v_w], dim=0).contiguous(), requires_grad=False))
            self.qkv_b.append(nn.Parameter(torch.cat([q_b, k_b, v_b], dim=0).contiguous(), requires_grad=False))
            # final_layer_norm folded into fc1
            f1_w, f1_b = fuse_ln_into_linear(W[s + "final_layer_norm.weight"], W[s + "final_layer_norm.bias"], W[s + "fc1.weight"], W[s + "fc1.bias"])
            self.o_w.append(nn.Parameter(W[s + "out_proj.weight"], requires_grad=False))
            self.o_b.append(nn.Parameter(W[s + "out_proj.bias"], requires_grad=False))
            self.fc1_w.append(nn.Parameter(f1_w, requires_grad=False)); self.fc1_b.append(nn.Parameter(f1_b, requires_grad=False))
            self.fc2_w.append(nn.Parameter(W[s + "fc2.weight"], requires_grad=False)); self.fc2_b.append(nn.Parameter(W[s + "fc2.bias"], requires_grad=False))

        # ln_post kept as raw affine weights (applied in-graph before the per-task projector input)
        self.register_buffer("ln_post_w", W["audio_tower.ln_post.weight"].contiguous())
        self.register_buffer("ln_post_b", W["audio_tower.ln_post.bias"].contiguous())

    def forward(self, waveform, feature_len, proj_w, proj_b):
        # int16 -> float for the Conv1d STFT (the /32768 scale is folded into mel_filters).
        waveform = waveform.float()
        real, imag = self.stft(waveform)                 # each [1, n_freq, n_frames+1] (center STFT)
        power = real * real + imag * imag                # |stft|^2 -> [1, n_freq, T]
        # 3D mel_filters batch-matmuls the [1,n_freq,T] power (no Gather)
        mel = torch.matmul(self.mel_filters, power)      # [1,n_mel,n_freq]@[1,n_freq,T] -> [1,n_mel,T]
        mel = torch.clamp(mel, min=1e-10).log10()
        mel = torch.maximum(mel, mel.max() - 8.0)        # Whisper dynamic-range clamp (global max)
        mel = (mel + 4.0) * 0.25                          # Whisper (log_spec + 4) / 4

        # SLICE mel to feature_len (Whisper valid frames = R//hop) BEFORE chunking so the token count
        # matches exactly; one extra silence token shifts last-token pooling hard on short clips (cos ~0.64).
        mel = mel[..., :feature_len]                                      # [1, n_mel, feature_len]
        num_chunks = (feature_len + self.chunk_minus) // self.chunk_size    # ceil(feature_len / chunk_size)
        pad_frames = num_chunks * self.chunk_size - feature_len             # always < chunk_size
        mel = torch.cat([mel, self.mel_pad_zeros[..., :pad_frames].float()], dim=-1)  # [1, n_mel, num_chunks*chunk_size]
        chunks = mel.reshape(Cfg.a_mel, -1, self.chunk_size).permute(1, 0, 2)  # reshape drops batch-1 -> [nc, n_mel, chunk]

        # per-chunk real lengths -> conv mask + post-conv valid counts (only the last chunk is short)
        chunk_starts = self.chunk_starts_full[:num_chunks]                  # [nc]  (0, 200, 400, ...)
        raw_chunk_lens = torch.clamp(feature_len - chunk_starts, min=0, max=self.chunk_size)  # [nc]
        conv_mask = (self.frame_pos < raw_chunk_lens.unsqueeze(1)).unsqueeze(1).float()       # [nc,1,chunk]
        aftercnn_lens = (raw_chunk_lens - 1) // 2 + 1                       # [nc]  valid tokens after conv2

        # conv front-end: each chunk -> chunk_aftercnn tokens; padded frames zeroed by conv_mask.
        # EXACT (erf) GELU (audio "gelu", NOT tanh); tanh costs ~3-4e-3.
        e = F.gelu(F.conv1d(chunks, self.conv1_w, self.conv1_b, padding=1)) * conv_mask
        e = F.gelu(F.conv1d(e, self.conv2_w, self.conv2_b, stride=2, padding=1)).transpose(1, 2)  # [nc,t,1280]
        # pos_emb pre-shaped [1, max_pos, dmodel] -> slice the seq axis (no unsqueeze)
        e = e + self.pos_emb[:, : e.shape[1]].float()

        # per-chunk key mask: only the last chunk masks its invalid tail. -128 additive (fp16-safe,
        # exp(-128)~0); replaces the host-side block-diagonal attn_bias.
        key_mask = (self.tok_pos >= aftercnn_lens.unsqueeze(1)).view(-1, 1, 1, self.chunk_aftercnn).float() * -128.0

        h = e                                                              # [nc, t, 1280]  (block-diagonal per chunk)
        t = self.chunk_aftercnn
        for i in range(Cfg.a_layers):
            res = h
            hn = layernorm_no_affine(h, Cfg.a_ln_eps)
            # fused QKV (one GEMM) -> [3, nc, heads, t, hd]; q/k/v via one split (no Gather)
            qkv = F.linear(hn, self.qkv_w[i], self.qkv_b[i]).view(-1, t, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.split([1, 1, 1], dim=0)
            attn = torch.matmul(q, k.transpose(3, 4)) + key_mask           # scale folded into Q; per-chunk mask
            attn = torch.softmax(attn, dim=-1)
            out = torch.matmul(attn, v).transpose(2, 3).reshape(-1, t, Cfg.a_dmodel)
            h = res + F.linear(out, self.o_w[i], self.o_b[i])

            res = h
            hn = layernorm_no_affine(h, Cfg.a_ln_eps)
            ff = F.gelu(F.linear(hn, self.fc1_w[i], self.fc1_b[i]))   # EXACT (erf) GELU (audio "gelu")
            h = res + F.linear(ff, self.fc2_w[i], self.fc2_b[i])

        # flatten chunks into one token stream; only the last chunk is short, so valid tokens are
        # exactly [0, encoded_len) -- a plain slice (no gather).
        h = h.reshape(-1, Cfg.a_dmodel)
        encoded_len = aftercnn_lens.sum().reshape(1)                        # T total valid tokens
        h = h[:encoded_len]                                                # [T, 1280]

        # stride-2 average pool (== reference (h[pool] + h[pool+1]) / 2 over pool = 0,2,4,...)
        half = encoded_len // 2
        h = h[: 2 * half].reshape(-1, 2, Cfg.a_dmodel).mean(dim=1)          # [num_pooled, 1280]
        # per-task projector: affine ln_post in-graph (raw weights), then the input projector linear
        hn = layernorm_no_affine(h, Cfg.a_ln_eps) * self.ln_post_w + self.ln_post_b   # ln_post affine
        return F.linear(hn, proj_w, proj_b)                                # [num_pooled, 768]


# ══════════════════════════════════════════════════════════════════════════════
# 4) OMNI_COMBINE — per-modality slice + concat splice (no scatter)
# ══════════════════════════════════════════════════════════════════════════════
# Every real input is single-modality with a FIXED layout, so the multimodal embeds are assembled by
# slice + concat instead of a per-position scatter (drops every Equal / NonZero / Where / ScatterND):
#   TEXT  [prefix, ..., EOS]                    -> token_embeds passthrough (no combine graph)
#   IMAGE [head_text, <image>*n, EOS]           -> cat([head, image_features, tail])
#   AUDIO [audio_start, <audio>*n, audio_end]   -> cat([task_start, audio_features, task_end])
# Byte-exact vs scatter: image slots only hold image features, audio_start/audio_end are the only
# surviving task tokens, and text carries no specials.


class OmniCombineImage(nn.Module):
    """Image path: splice image features into the token-embedding sequence with slice + concat. The
    prompt tokenizes to [head_text, <image>*n, EOS], so the <image> block ends tail_len tokens before
    the end (tail = the appended EOS). head/tail bounds come from the seq length and feature count ->
    two Slices + one Concat (no position-level ops).

    Inputs: token_embeds [1, s, H], image_features [n_img, H].
    Output: multimodal_embeds [1, s, H]  (features occupy [s - tail_len - n_img, s - tail_len)).
    """

    def __init__(self, tail_len=1):
        super().__init__()
        self.tail_len = tail_len      # tokens trailing the <image> block (tokenizer appends one EOS)

    def forward(self, token_embeds, image_features):
        n = image_features.shape[0]
        head_end = token_embeds.shape[1] - self.tail_len - n               # first <image> position
        head = token_embeds[:, :head_end]                                  # [1, head_end, H] prefix text
        tail = token_embeds[:, token_embeds.shape[1] - self.tail_len:]     # [1, tail_len, H] (EOS)
        return torch.cat([head, image_features.unsqueeze(0), tail], dim=1)


class OmniCombineAudio(nn.Module):
    """Audio path: the sequence is [audio_start, <audio>*n, audio_end] (all special/media), so no base
    token_embeds survive -> built purely from the bracket task tokens and audio features:
    cat([task[start_row], audio_features, task[end_row]]) (rows are id - special_base).

    Inputs: task_token_embeds [num_special, H], audio_features [n_aud, H].
    Output: multimodal_embeds [1, n_aud + 2, H].
    """

    def __init__(self):
        super().__init__()
        base = min(Cfg.special_token_ids)
        self.start_row = Cfg.audio_start_token_id - base     # task-token row for audio_start
        self.end_row = Cfg.audio_end_token_id - base         # task-token row for audio_end

    def forward(self, task_token_embeds, audio_features):
        start_tok = task_token_embeds[self.start_row:self.start_row + 1]    # [1, H] audio_start
        end_tok = task_token_embeds[self.end_row:self.end_row + 1]          # [1, H] audio_end
        seq = torch.cat([start_tok, audio_features, end_tok], dim=0)        # [n_aud + 2, H]
        return seq.unsqueeze(0)                                             # [1, n_aud + 2, H]


# ══════════════════════════════════════════════════════════════════════════════
# 5) OMNI_MAIN — EuroBERT/Llama bidirectional encoder (shared) + plugin LoRA inputs
# ══════════════════════════════════════════════════════════════════════════════
class OmniMain(nn.Module):
    """Bidirectional text encoder (NO causal mask, NO QK-norm). Fused QKV / gate_up, sum-based RMSNorm
    with norm*sqrt(H) absorbed into the next linear, attention scale folded into Q, flip-based RoPE.
    Outputs one L2-normalized embedding (Matryoshka truncation is host-side).

    Plugin-LoRA: the per-task LoRA arrives as 8 block-diag-fused inputs; each layer adds
    lora_B @ (lora_A @ hn) to the matching base projection. build_task_tensors pre-folded the
    input-norm*sqrt(H) (and Q scale) into lora_A, so the delta matches base PEFT after the fold."""

    def __init__(self, W):
        super().__init__()
        H = Cfg.t_hidden
        hd = Cfg.t_head_dim
        self.heads = Cfg.t_heads
        self.kv_heads = Cfg.t_kv_heads
        self.head_dim = hd
        # eps_sum = hidden * eps, hoisted to one shared float32 buffer (no per-call constant rebuild)
        self.register_buffer("eps_sum", torch.tensor([H * Cfg.t_rms_eps], dtype=torch.float32))
        norm_factor = H ** 0.5
        qscale = hd ** -0.5

        self.qkv_w = nn.ParameterList()
        self.o_w = nn.ParameterList()
        self.gate_up_w = nn.ParameterList()
        self.down_w = nn.ParameterList()
        for i in range(Cfg.t_layers):
            d = f"text_model.layers.{i}."
            # absorb input-RMSNorm weight*sqrt(H) into the qkv linear
            in_scale = (W[d + "input_layernorm.weight"] * norm_factor).unsqueeze(0)
            q = (W[d + "q_proj.weight"] * in_scale) * qscale     # fold head_dim^-0.5 into Q
            k = W[d + "k_proj.weight"] * in_scale
            v = W[d + "v_proj.weight"] * in_scale
            # fused QKV: single GEMM weight = cat([q, k, v])
            self.qkv_w.append(nn.Parameter(torch.cat([q, k, v], dim=0).contiguous(), requires_grad=False))
            self.o_w.append(nn.Parameter(W[d + "o_proj.weight"].contiguous(), requires_grad=False))
            # absorb post-attention-RMSNorm weight*sqrt(H) into gate/up
            post_scale = (W[d + "post_attention_layernorm.weight"] * norm_factor).unsqueeze(0)
            gate = W[d + "gate_proj.weight"] * post_scale
            up = W[d + "up_proj.weight"] * post_scale
            # fused gate_up: single GEMM weight = cat([gate, up])
            self.gate_up_w.append(nn.Parameter(torch.cat([gate, up], dim=0).contiguous(), requires_grad=False))
            self.down_w.append(nn.Parameter(W[d + "down_proj.weight"].contiguous(), requires_grad=False))

        # final RMSNorm weight*sqrt(H) as one explicit multiply; stored [1, H] for the pooled [b, H] row
        self.register_buffer("final_norm_weight", (W["text_model.norm.weight"] * norm_factor).view(1, -1).contiguous())

        # precomputed flip-RoPE tables (cos = cat(cos,cos); sin = cat(-sin, sin))
        inv = 1.0 / (Cfg.t_rope_theta ** (torch.arange(0, hd, 2).float() / hd))  # [hd/2]
        pos = torch.arange(MAX_SEQ_LEN).float()
        freqs = torch.outer(pos, inv)                                            # [MAX, hd/2]
        emb = torch.cat([freqs, freqs], dim=-1)                                  # [MAX, hd]
        cos = emb.cos()
        sin = emb.sin()
        sin_flip = torch.cat([-sin[:, : hd // 2], sin[:, hd // 2:]], dim=-1)
        # cos/sin pre-shaped [1, 1, MAX, hd] so the runtime is a pure slice (no .view)
        self.register_buffer("rope_cos", cos.view(1, 1, MAX_SEQ_LEN, hd).contiguous().half())
        self.register_buffer("rope_sin", sin_flip.view(1, 1, MAX_SEQ_LEN, hd).contiguous().half())
        # Bidirectional KEY-PADDING bias (NO causal). Static int8 [1,1,1,MAX] -128 buffer; forward slices
        # it and gates with (input_ids == pad). Do NOT use a static zero template -- it silently attends
        # to right-padding (cos ~0.86-0.99).
        self.register_buffer("attention_bias_template",
                             torch.full((1, 1, 1, MAX_SEQ_LEN), -128, dtype=torch.int8), persistent=False)
        # truncate column indices (static int32 buffer); shorten its length to truncate in-graph
        self.register_buffer("cols", torch.arange(Cfg.t_hidden, dtype=torch.int32), persistent=False)
        self.mlp_split = Cfg.t_inter

    @staticmethod
    def _apply_lora(x, lora_a, lora_b):
        # delta = lora_b @ (lora_a @ x); lora_a carries the folded input-norm scale (x = affine-free dir)
        return F.linear(F.linear(x, lora_a), lora_b)

    def forward(self, inputs_embeds, input_ids,
                lora_qkv_a, lora_qkv_b, lora_o_a, lora_o_b,
                lora_gate_up_a, lora_gate_up_b, lora_down_a, lora_down_b):
        b, seq_len, _ = inputs_embeds.shape
        # rope_cos/sin pre-shaped [1,1,MAX,hd] -> slice the seq axis only (no .view)
        cos = self.rope_cos[:, :, :seq_len].float()
        sin = self.rope_sin[:, :, :seq_len].float()
        # bidirectional key-padding bias from input_ids (NOT a tokenizer mask): slice the static -128
        # template and gate by is_pad -> int8 {0 keep, -128 pad}, then one cast to float.
        is_pad = (input_ids == Cfg.pad_token_id).to(torch.int8)                        # [b, seq] (1 where pad)
        attn_bias = (self.attention_bias_template[:, :, :, :seq_len] * is_pad[:, None, None, :]).float()

        x = inputs_embeds
        for i in range(Cfg.t_layers):
            res = x
            hn = sum_rms_norm(x, self.eps_sum)
            # [b, 3*heads, s, hd] MHA layout -> rotate Q&K together; LoRA delta added to fused qkv first
            qkv = F.linear(hn, self.qkv_w[i]) + self._apply_lora(hn, lora_qkv_a[i], lora_qkv_b[i])
            qkv = qkv.view(b, seq_len, 3 * self.heads, self.head_dim).transpose(1, 2)
            qk, v = qkv.split([2 * self.heads, self.heads], dim=1)     # qk:[b,2*heads,s,hd]  v:[b,heads,s,hd]
            qk = qk * cos + flip_rotate_half(qk) * sin
            # one Split along the head axis -> q/k as [b,heads,s,hd] (no Gather)
            q, k = qk.split([self.heads, self.heads], dim=1)          # [b, heads, s, hd] each
            attn = torch.matmul(q, k.transpose(2, 3)) + attn_bias  # scale folded into Q; bidirectional + key-padding
            attn = torch.softmax(attn, dim=-1)
            out = torch.matmul(attn, v).transpose(1, 2).reshape(b, seq_len, -1)
            # o_proj base + LoRA delta (lora_o_a unscaled -- input is the attention output)
            x = res + F.linear(out, self.o_w[i]) + self._apply_lora(out, lora_o_a[i], lora_o_b[i])

            res = x
            hn = sum_rms_norm(x, self.eps_sum)
            # gate_up base + LoRA delta
            gate_up = F.linear(hn, self.gate_up_w[i]) + self._apply_lora(hn, lora_gate_up_a[i], lora_gate_up_b[i])
            gate, up = torch.split(gate_up, [self.mlp_split, self.mlp_split], dim=-1)
            mlp_hidden = F.silu(gate) * up
            # down_proj base + LoRA delta (lora_down_a unscaled -- input is the SwiGLU output)
            x = res + F.linear(mlp_hidden, self.down_w[i]) + self._apply_lora(mlp_hidden, lora_down_a[i], lora_down_b[i])

        # last-token pooling: last valid index = real-token count - 1, derived in-graph from is_pad.
        valid_count = (1 - is_pad).sum(dim=1, dtype=torch.int32)                       # [b] real-token count
        pool_idx = (valid_count - 1).view(-1, 1, 1).expand(-1, 1, Cfg.t_hidden)        # [b, 1, H] int32
        # gather-before-norm: gather the pooled token, then RMSNorm only that [b,1,H] row (row-wise norm
        # -> bit-identical to norming the whole sequence then gathering).
        pooled = sum_rms_norm(x.gather(1, pool_idx), self.eps_sum).squeeze(1) * self.final_norm_weight  # [b, H]

        # truncate before L2 normalize; len(cols) == H makes the index_select an identity (skipped in trace)
        trunc = pooled if self.cols.numel() == pooled.shape[-1] else pooled.index_select(1, self.cols)
        embeddings = trunc / trunc.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-7)
        return embeddings


# ══════════════════════════════════════════════════════════════════════════════
# 6) OMNI_LORA — per-task weight provider
# ══════════════════════════════════════════════════════════════════════════════
# Provider output order: 8 LoRA + 6 merger + 2 projector + 1 task-token table = 17 tensors
# (exactly the runtime inputs the shared Vision / Audio / Combine_Audio / Main graphs expect).
LORA_PROVIDER_OUTPUT_NAMES = (
    LORA_INPUT_NAMES + MERGER_INPUT_NAMES + PROJECTOR_INPUT_NAMES + [TASK_TOKEN_INPUT_NAME]
)


class OmniLoRA(nn.Module):
    """Single-task provider (SPLIT mode): stores this task's 17 tensors as constant buffers and emits
    them with no inputs. Run ONCE per task at startup; its OrtValues are cached for every encode()."""

    def __init__(self, task_tensors: dict):
        super().__init__()
        for name in LORA_PROVIDER_OUTPUT_NAMES:
            self.register_buffer(name, task_tensors[name].float().contiguous())

    def forward(self):
        return tuple(getattr(self, name) for name in LORA_PROVIDER_OUTPUT_NAMES)


class OmniLoRAFused(nn.Module):
    """All-task provider (FUSED mode): stacks EVERY task's 17 tensors on a leading task axis and selects
    one with an int32 `task_index`. Replaces the N per-task graphs; the selected slice is byte-identical
    to the split-mode provider (all tasks share the same shapes)."""

    def __init__(self, all_task_tensors):
        super().__init__()
        for name in LORA_PROVIDER_OUTPUT_NAMES:
            stacked = torch.stack([T[name].float() for T in all_task_tensors], dim=0).contiguous()
            self.register_buffer(name, stacked)

    def forward(self, task_index):
        # int32 task_index [1] -> Gather the selected task's row, then squeeze the task axis
        return tuple(getattr(self, name)[task_index].squeeze(0) for name in LORA_PROVIDER_OUTPUT_NAMES)


# ══════════════════════════════════════════════════════════════════════════════
# CPU preprocessing -> ONNX-input builders
# ══════════════════════════════════════════════════════════════════════════════
def prepare_vision_inputs(image) -> dict:
    """PIL.Image (or HxWx3 / HxW uint8 array) -> RAW graph input [1, 3, H, W] in 0-255. resize ->
    normalize -> patchify all run inside Omni_Vision, so the only CPU work here is the RGB/CHW layout."""
    arr = np.asarray(image.convert("RGB") if hasattr(image, "convert") else image)
    arr = arr.astype(np.float32)
    if arr.ndim == 2:                              # grayscale -> RGB
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.shape[-1] == 4:                         # drop alpha
        arr = arr[..., :3]
    chw = np.transpose(arr, (2, 0, 1))[None]       # [1, 3, H, W]
    return {"image": np.ascontiguousarray(chw, dtype=np.float32)}


def prepare_audio_inputs(waveform):
    """Build the RAW-waveform graph inputs for Omni_Audio (mel front-end + chunking run in-graph).
    Zero-pad with an n_fft reflection margin (boundary frames match Whisper's zero-pad) and pass
    feature_len (= R // hop) so the graph slices its mel to the exact token count. The graph input is
    int16 PCM (normalized in-graph via /32768): a float [-1, 1] waveform is quantized here.

    Returns ({"waveform": [1,1,L_pad] int16, "feature_len": [1] int64}, num_audio_tokens).
    """
    wav = np.asarray(waveform)
    if np.issubdtype(wav.dtype, np.floating):
        # float [-1, 1] -> int16 PCM (the graph divides by 32768 in forward() to recover [-1, 1]).
        wav = np.clip(np.rint(wav * 32768.0), -32768.0, 32767.0).astype(np.int16)
    wav = wav.astype(np.int16).reshape(-1)
    max_samples = int(AUDIO_MAX_SECONDS * AUDIO_SR)
    R = min(wav.shape[0], max_samples)                          # Whisper truncates to chunk_length

    # zero-pad to a multiple of hop with an n_fft zero tail so the last real frame's STFT window sees
    # zeros (matches Whisper's zero-pad).
    L_pad = ((R + AUDIO_N_FFT + AUDIO_HOP - 1) // AUDIO_HOP) * AUDIO_HOP
    wav_pad = np.zeros(L_pad, dtype=np.int16)
    wav_pad[:R] = wav[:R]

    # Whisper valid-frame count (verified n_valid = R // hop); the graph slices its mel to this.
    feature_len = R // AUDIO_HOP

    # audio token count from feature_len: per-chunk after-conv2 sum, then stride-2 pool.
    n_window2 = Cfg.a_n_window * 2                              # 200 mel frames per chunk
    num_chunks = max(1, (feature_len + n_window2 - 1) // n_window2)
    raw_chunk_lens = np.clip(feature_len - np.arange(num_chunks) * n_window2, 0, n_window2)
    encoded_len = int(((raw_chunk_lens - 1) // 2 + 1).sum())
    num_tokens = encoded_len // 2
    return {
        "waveform": np.ascontiguousarray(wav_pad.reshape(1, 1, L_pad), dtype=np.int16),
        "feature_len": np.array([feature_len], dtype=np.int64),
    }, num_tokens


# ══════════════════════════════════════════════════════════════════════════════
# EXPORT — task-agnostic shared graphs (exported ONCE) + per-task provider graphs.
# ══════════════════════════════════════════════════════════════════════════════
def shared_dir() -> Path:
    return OUTPUT_ROOT


def runtime_metadata_dict(tasks):
    """The model + preprocessing constants the ONNX inference reads back from the graph metadata, so it
    needs no config.json (just the tokenizer + onnxruntime). Single source of truth = this exporter."""
    return {
        "embed_dim": Cfg.t_hidden,                  # final embedding dim (Omni_Main output)
        "max_seq_len": MAX_SEQ_LEN,                 # text truncation length
        "audio_sr": AUDIO_SR,                       # Whisper sample rate
        "audio_n_fft": AUDIO_N_FFT,                 # Whisper n_fft
        "audio_hop": AUDIO_HOP,                     # Whisper hop_length
        "audio_max_seconds": AUDIO_MAX_SECONDS,     # truncation cap (chunk_length)
        "audio_n_window": Cfg.a_n_window,           # half-chunk frames (chunk = 2 * n_window)
        "audio_start_token_id": Cfg.audio_start_token_id,
        "audio_token_id": Cfg.audio_token_id,
        "audio_end_token_id": Cfg.audio_end_token_id,
        "image_height_factor": IMAGE_HEIGHT_FACTOR,  # target_h = factor * patch * merge
        "image_width_factor": IMAGE_WIDTH_FACTOR,
        "input_image_height": INPUT_IMAGE_SIZE[0],   # raw graph input image shape [H, W]
        "input_image_width": INPUT_IMAGE_SIZE[1],
        "image_placeholder": IMAGE_PLACEHOLDER,      # <image> token string
        "task_names": ",".join(tasks),               # task list (provider task_index order)
    }


def bake_metadata_props(onnx_path, metadata):
    """Bake runtime constants into an ONNX model's custom metadata_props (idempotent; existing keys are
    overwritten). Applied only to the tiny combine graphs (no external data), mirroring the Reranker."""
    model = onnx.load(str(onnx_path))
    existing = {prop.key: idx for idx, prop in enumerate(model.metadata_props)}
    for key, value in metadata.items():
        if key in existing:
            model.metadata_props[existing[key]].value = str(value)
        else:
            entry = model.metadata_props.add()
            entry.key = str(key)
            entry.value = str(value)
    onnx.save(model, str(onnx_path))


def export_shared_graphs():
    """Export the 6 task-agnostic graphs (Embed, Vision, Audio, Combine_Image, Combine_Audio, Main)
    ONCE. Every per-task tensor is a runtime input; the per-task Omni_LoRA provider (exported
    separately) supplies them. Text needs no combine graph (token_embeds passthrough)."""
    out_dir = shared_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[shared] loading base backbone weights (no LoRA merge, no per-task bake) ...")
    W = load_base_weights(MODEL_DIR)

    target_h = IMAGE_HEIGHT_FACTOR * Cfg.v_patch * Cfg.v_merge     # fixed vision input target (e.g. 512)
    target_w = IMAGE_WIDTH_FACTOR * Cfg.v_patch * Cfg.v_merge
    audio_max_frames = int(AUDIO_MAX_SECONDS * AUDIO_SR) // AUDIO_HOP + 1   # STFT_Process arg (unused for stft_B)
    embed = OmniEmbed(W).eval()
    vision = OmniVision(W, target_h, target_w, dynamic_shape=DYNAMIC_IMAGE_SHAPE).eval()
    audio = OmniAudio(W, audio_max_frames).eval()
    combine_image = OmniCombineImage().eval()
    combine_audio = OmniCombineAudio().eval()
    main = OmniMain(W).eval()
    gc.collect()

    with torch.inference_mode():
        # ---- Omni_Embed ---- (plain token lookup)
        print(f"[shared] export Omni_Embed ...")
        dummy_ids = torch.ones((1, 8), dtype=torch.int32)
        torch.onnx.export(
            embed, (dummy_ids,), str(out_dir / "Omni_Embed.onnx"),
            input_names=["input_ids"], output_names=["token_embeds"],
            dynamic_axes={"input_ids": {0: "batch", 1: "seq"}, "token_embeds": {0: "batch", 1: "seq"}},
            opset_version=OPSET, dynamo=False,
        )

        # ---- Omni_Vision ---- (RAW RGB image [1,3,H,W] 0-255 + per-task merger weights)
        # Dummy is the raw INPUT_IMAGE_SIZE; the in-graph resize to the fixed target runs only when it
        # differs. DYNAMIC_IMAGE_SHAPE keeps H/W dynamic (any size); otherwise the input is static.
        print(f"[shared] export Omni_Vision ...")
        d_img = torch.randint(0, 256, (1, Cfg.v_in_ch, INPUT_IMAGE_SIZE[0], INPUT_IMAGE_SIZE[1]), dtype=torch.float32)
        merge_in = Cfg.v_hidden * (Cfg.v_merge ** 2)
        d_merger = (
            torch.randn(Cfg.v_hidden), torch.randn(Cfg.v_hidden),                 # norm w/b
            torch.randn(merge_in, merge_in), torch.randn(merge_in),               # fc1 w/b
            torch.randn(Cfg.t_hidden, merge_in), torch.randn(Cfg.t_hidden),       # fc2 w/b
        )
        torch.onnx.export(
            vision, (d_img,) + d_merger, str(out_dir / "Omni_Vision.onnx"),
            input_names=["image"] + MERGER_INPUT_NAMES, output_names=["image_features"],
            dynamic_axes={"image": {2: "height", 3: "width"}} if DYNAMIC_IMAGE_SHAPE else None,
            opset_version=OPSET, dynamo=False,
        )

        # ---- Omni_Audio ---- (RAW int16 PCM [1,1,audio_len] + feature_len + per-task projector weights)
        print(f"[shared] export Omni_Audio ...")
        d_wave = 0.1 * np.sin(2 * np.pi * 440 * np.arange(int(AUDIO_SECONDS * AUDIO_SR)) / AUDIO_SR)
        d_aud, _ = prepare_audio_inputs(d_wave)
        d_proj = (torch.randn(Cfg.t_hidden, Cfg.a_dmodel), torch.randn(Cfg.t_hidden))
        aud_args = (torch.from_numpy(d_aud["waveform"]), torch.from_numpy(d_aud["feature_len"])) + d_proj
        torch.onnx.export(
            audio, aud_args, str(out_dir / "Omni_Audio.onnx"),
            input_names=["waveform", "feature_len"] + PROJECTOR_INPUT_NAMES,
            output_names=["audio_features"],
            dynamic_axes={"waveform": {2: "audio_len"}, "audio_features": {0: "num_tokens"}},
            opset_version=OPSET, dynamo=False,
        )

        # ---- Omni_Combine_Image ---- (slice + concat: head_text ++ image_features ++ tail EOS)
        print(f"[shared] export Omni_Combine_Image ...")
        ci_emb = torch.randn(1, 9, Cfg.t_hidden)        # [head(3), <image>(5), EOS(1)] dummy layout
        ci_img = torch.randn(5, Cfg.t_hidden)
        torch.onnx.export(
            combine_image, (ci_emb, ci_img), str(out_dir / "Omni_Combine_Image.onnx"),
            input_names=["token_embeds", "image_features"], output_names=["multimodal_embeds"],
            dynamic_axes={"token_embeds": {0: "batch", 1: "seq"}, "image_features": {0: "n_img"},
                          "multimodal_embeds": {0: "batch", 1: "seq"}},
            opset_version=OPSET, dynamo=False,
        )

        # ---- Omni_Combine_Audio ---- (concat: audio_start ++ audio_features ++ audio_end)
        print(f"[shared] export Omni_Combine_Audio ...")
        ca_tok = torch.randn(len(Cfg.special_token_ids), Cfg.t_hidden)
        ca_aud = torch.randn(50, Cfg.t_hidden)
        torch.onnx.export(
            combine_audio, (ca_tok, ca_aud), str(out_dir / "Omni_Combine_Audio.onnx"),
            input_names=[TASK_TOKEN_INPUT_NAME, "audio_features"], output_names=["multimodal_embeds"],
            dynamic_axes={"audio_features": {0: "n_aud"}, "multimodal_embeds": {0: "batch", 1: "seq"}},
            opset_version=OPSET, dynamo=False,
        )

        # ---- Omni_Main ---- (multimodal embeds + input_ids + 8 LoRA; key-padding bias and pool in-graph)
        print(f"[shared] export Omni_Main ...")
        m_emb = torch.randn(1, 12, Cfg.t_hidden)
        m_ids = torch.ones(1, 12, dtype=torch.int32)
        # dummy block-diag-fused LoRA tensors matching build_task_tensors shapes (r=32)
        r = 32
        L = Cfg.t_layers
        Hh = Cfg.t_hidden
        I = Cfg.t_inter
        m_lora = (
            torch.randn(L, 3 * r, Hh),               # lora_qkv_a  [L, 3r, H]
            torch.randn(L, 3 * Hh, 3 * r),           # lora_qkv_b  [L, 3H, 3r] (block-diag)
            torch.randn(L, r, Hh),                   # lora_o_a    [L, r, H]
            torch.randn(L, Hh, r),                   # lora_o_b    [L, H, r]
            torch.randn(L, 2 * r, Hh),               # lora_gate_up_a [L, 2r, H]
            torch.randn(L, 2 * I, 2 * r),            # lora_gate_up_b [L, 2I, 2r] (block-diag)
            torch.randn(L, r, I),                    # lora_down_a [L, r, I]
            torch.randn(L, Hh, r),                   # lora_down_b [L, H, r]
        )
        torch.onnx.export(
            main, (m_emb, m_ids) + m_lora, str(out_dir / "Omni_Main.onnx"),
            input_names=["inputs_embeds", "input_ids"] + LORA_INPUT_NAMES,
            output_names=["embeddings"],
            dynamic_axes={"inputs_embeds": {0: "batch", 1: "seq"}, "input_ids": {0: "batch", 1: "seq"},
                          "embeddings": {0: "batch", 1: "dim"}},
            opset_version=OPSET, dynamo=False,
        )

    del embed, vision, audio, combine_image, combine_audio, main, W
    gc.collect()
    sizes = {p.name: p.stat().st_size / 1024 / 1024 for p in out_dir.glob("Omni_*.onnx")}
    print(f"[shared] export done. ONNX sizes (MB): " + ", ".join(f"{k}={v:.1f}" for k, v in sizes.items()))


def export_task_provider(task: str, sd: dict | None = None):
    """Export the per-task Omni_LoRA provider graph (constant outputs: 8 LoRA + 6 merger + 2 proj + 1 tok)."""
    out_dir = task_dir(task)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[{task}] building per-task tensors (LoRA + merger + projector + task tokens) ...")
    T = build_task_tensors(MODEL_DIR, task, sd=sd)
    provider = OmniLoRA(T).eval()
    gc.collect()
    with torch.inference_mode():
        print(f"[{task}] export Omni_LoRA ...")
        torch.onnx.export(
            provider, (), str(out_dir / "Omni_LoRA.onnx"),
            input_names=[], output_names=list(LORA_PROVIDER_OUTPUT_NAMES),
            dynamic_axes={}, opset_version=OPSET, dynamo=False,
        )
    del provider, T
    gc.collect()
    size = (out_dir / "Omni_LoRA.onnx").stat().st_size / 1024 / 1024
    print(f"[{task}] export done. Omni_LoRA.onnx = {size:.1f} MB")


def export_fused_provider(tasks):
    """Export the single all-task Omni_LoRA provider graph (FUSED mode): every task's 17 tensors stacked
    on a task axis, selected by int32 `task_index`. Stored in shared/ since it serves all tasks."""
    out_dir = shared_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[fused] building per-task tensors for {len(tasks)} tasks "
          f"(LoRA + merger + projector + task tokens) ...")
    # load the full model state dict ONCE and reuse it for every task (only the adapter files differ)
    sd = {k: v.float() for k, v in load_file(str(MODEL_DIR / "model.safetensors")).items()}
    all_T = [build_task_tensors(MODEL_DIR, t, sd=sd) for t in tasks]
    provider = OmniLoRAFused(all_T).eval()
    gc.collect()
    with torch.inference_mode():
        print(f"[fused] export Omni_LoRA (task_index map = {dict(enumerate(tasks))}) ...")
        d_idx = torch.zeros(1, dtype=torch.int32)
        torch.onnx.export(
            provider, (d_idx,), str(out_dir / "Omni_LoRA.onnx"),
            input_names=["task_index"], output_names=list(LORA_PROVIDER_OUTPUT_NAMES),
            dynamic_axes={}, opset_version=OPSET, dynamo=False,
        )
    del provider, all_T
    gc.collect()
    size = (out_dir / "Omni_LoRA.onnx").stat().st_size / 1024 / 1024
    print(f"[fused] export done. shared/Omni_LoRA.onnx = {size:.1f} MB")


# ══════════════════════════════════════════════════════════════════════════════
# ORT SESSIONS & IOBINDING RUNTIME
# ══════════════════════════════════════════════════════════════════════════════
def make_session_opts():
    so = onnxruntime.SessionOptions()
    so.log_severity_level = 0 if ORT_LOG else 4
    so.log_verbosity_level = 4
    so.inter_op_num_threads = MAX_THREADS
    so.intra_op_num_threads = MAX_THREADS
    so.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
    so.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    # Session-option recipe (mirrors Inference_Jina_Embedding_ONNX.py): denormal-flush, spinning, QDQ
    # cleanup, device allocator for initializers, loop-level opt, cast-chain elim.
    for k, v in {
        "session.set_denormal_as_zero": "1",
        "session.intra_op.allow_spinning": "1",
        "session.inter_op.allow_spinning": "1",
        "session.enable_quant_qdq_cleanup": "1",
        "session.qdq_matmulnbits_accuracy_level": "2" if ORT_FP16 else "4",
        "session.use_device_allocator_for_initializers": "1",
        "session.graph_optimizations_loop_level": "2",
        # NOTE: do NOT add optimization.enable_gelu_approximation -- it rewrites the audio/merger EXACT
        # GELU to tanh and breaks audio parity by ~3e-3 (only the vision tower blocks are tanh).
        "optimization.minimal_build_optimizations": "",
        "optimization.enable_cast_chain_elimination": "1",
        "optimization.disable_specified_optimizers":
            "CastFloat16Transformer;FuseFp16InitializerToFp32NodeTransformer" if ORT_FP16 else "",
    }.items():
        so.add_session_config_entry(k, v)
    return so


# FP16 cast optimizers disabled at the InferenceSession level (mirrors the reference); None in fp32.
DISABLED_OPTIMIZERS = (
    ["CastFloat16Transformer", "FuseFp16InitializerToFp32NodeTransformer"] if ORT_FP16 else None
)


class OmniShared:
    """Holds the 5 task-agnostic ONNX sessions + IOBindings, loaded ONCE and reused by every task's
    OmniORT. FUSED mode also owns the shared all-task Omni_LoRA provider (selected by task_index);
    SPLIT mode loads a per-task provider in each OmniORT."""

    def __init__(self, tasks):
        d = shared_dir()
        so = make_session_opts()
        providers = ORT_PROVIDERS
        self.task_index_map = {t: i for i, t in enumerate(tasks)}
        self.run_options = onnxruntime.RunOptions()
        self.run_options.log_severity_level = 0 if ORT_LOG else 4
        self.run_options.log_verbosity_level = 4
        self.run_options.add_run_config_entry("disable_synchronize_execution_providers", "0")

        def sess(name):
            return onnxruntime.InferenceSession(
                str(d / name), sess_options=so, providers=providers,
                provider_options=provider_options, disabled_optimizers=DISABLED_OPTIMIZERS)

        self.s_embed = sess("Omni_Embed.onnx")
        self.s_vision = sess("Omni_Vision.onnx")
        self.s_audio = sess("Omni_Audio.onnx")
        self.s_combine_image = sess("Omni_Combine_Image.onnx")
        self.s_combine_audio = sess("Omni_Combine_Audio.onnx")
        self.s_main = sess("Omni_Main.onnx")
        self.b_embed = self.s_embed.io_binding()
        self.b_vision = self.s_vision.io_binding()
        self.b_audio = self.s_audio.io_binding()
        self.b_combine_image = self.s_combine_image.io_binding()
        self.b_combine_audio = self.s_combine_audio.io_binding()
        self.b_main = self.s_main.io_binding()
        # FUSED: one shared all-task provider session (selected by task_index). SPLIT: loaded per task.
        self.s_lora = sess("Omni_LoRA.onnx") if FUSE_LORA_INTO_EMBED else None


class OmniORT:
    """Runs the shared pipeline for ONE task entirely through IOBinding / OrtValue (only the final
    embedding is materialized). Each encode() is single-modality and takes one of three paths:
        TEXT  -> Embed -> Main                       (token_embeds passthrough; no combine graph)
        IMAGE -> Embed + Vision -> Combine_Image -> Main
        AUDIO -> Audio -> Combine_Audio -> Main      (no Embed; audio embeds = task tokens + features)
    The Omni_LoRA provider is run ONCE here and its 17 OrtValues are cached and bound on every encode()
    (FUSED: shared provider + task_index; SPLIT: this task's own provider)."""

    def __init__(self, task: str, shared: "OmniShared"):
        self.shared = shared
        self.run_options = shared.run_options
        # reuse stable-shape output buffers: (name, shape, dtype) -> zero-filled OrtValue
        self._shape_cache = {}

        # run the provider ONCE for this task and cache its 17 OrtValues, keyed by output name.
        if FUSE_LORA_INTO_EMBED:
            # FUSED: reuse the shared all-task provider, select this task by task_index.
            s_lora = shared.s_lora
            b_lora = s_lora.io_binding()
            out_names = [o.name for o in s_lora.get_outputs()]
            idx_ov = self._ov(np.array([shared.task_index_map[task]], dtype=np.int32))
            b_lora.bind_ortvalue_input("task_index", idx_ov)
            for name in out_names:
                b_lora.bind_output(name, device_type, DEVICE_ID)
            s_lora.run_with_iobinding(b_lora, self.run_options)
            self.task_ov = dict(zip(out_names, b_lora.get_outputs()))
            del b_lora
        else:
            # SPLIT: load and run this task's own constant-output provider graph.
            so = make_session_opts()
            s_lora = onnxruntime.InferenceSession(
                str(task_dir(task) / "Omni_LoRA.onnx"), sess_options=so, providers=ORT_PROVIDERS,
                provider_options=provider_options, disabled_optimizers=DISABLED_OPTIMIZERS)
            b_lora = s_lora.io_binding()
            out_names = [o.name for o in s_lora.get_outputs()]
            for name in out_names:
                b_lora.bind_output(name, device_type, DEVICE_ID)
            s_lora.run_with_iobinding(b_lora, self.run_options)
            self.task_ov = dict(zip(out_names, b_lora.get_outputs()))
            del s_lora, b_lora
        # pre-pack the per-graph input subsets
        self._merger_ov = {n: self.task_ov[n] for n in MERGER_INPUT_NAMES}
        self._proj_ov = {n: self.task_ov[n] for n in PROJECTOR_INPUT_NAMES}
        self._lora_ov = {n: self.task_ov[n] for n in LORA_INPUT_NAMES}
        self._task_tok_ov = self.task_ov[TASK_TOKEN_INPUT_NAME]

    @staticmethod
    def _ov(arr):
        return onnxruntime.OrtValue.ortvalue_from_numpy(np.ascontiguousarray(arr), device_type, DEVICE_ID)

    def _cached_out(self, name, shape, dtype):
        # reuse a pre-allocated OrtValue output buffer instead of letting ORT realloc every call
        key = (name, tuple(int(x) for x in shape), np.dtype(dtype).str)
        buf = self._shape_cache.get(key)
        if buf is None:
            buf = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros(shape, dtype=dtype), device_type, DEVICE_ID)
            self._shape_cache[key] = buf
        return buf

    def _run(self, session, binding, inputs_ov, out_names, out_bufs=None):
        binding.clear_binding_inputs()
        binding.clear_binding_outputs()
        for name, ov in inputs_ov.items():
            binding.bind_ortvalue_input(name, ov)
        if out_bufs is None:
            for name in out_names:
                binding.bind_output(name, device_type, DEVICE_ID)
            session.run_with_iobinding(binding, self.run_options)
            return binding.get_outputs()
        # bind reused output buffers by name (no per-call ORT allocation)
        for name, buf in zip(out_names, out_bufs):
            binding.bind_ortvalue_output(name, buf)
        session.run_with_iobinding(binding, self.run_options)
        return out_bufs

    def encode(self, input_ids, vision_inputs=None, audio_inputs=None):
        sh = self.shared
        ids = np.ascontiguousarray(input_ids.astype(np.int32))
        b = ids.shape[0]
        # one input_ids OrtValue per call, reused by Embed AND Main (which derives the bias + pool in-graph)
        ids_ov = self._ov(ids)

        if audio_inputs is not None:
            # AUDIO: cat([task_start, audio_features, task_end]) -- no Embed / token_embeds needed
            a_inputs = {k: self._ov(v) for k, v in audio_inputs.items()}
            a_inputs.update(self._proj_ov)
            aud_feat = self._run(sh.s_audio, sh.b_audio, a_inputs, ["audio_features"])[0]
            mm = self._run(sh.s_combine_audio, sh.b_combine_audio,
                           {TASK_TOKEN_INPUT_NAME: self._task_tok_ov, "audio_features": aud_feat},
                           ["multimodal_embeds"])[0]
        elif vision_inputs is not None:
            # IMAGE: splice image features into the token embeds -- cat([head, image_features, tail])
            token_embeds = self._run(sh.s_embed, sh.b_embed, {"input_ids": ids_ov}, ["token_embeds"])[0]
            v_inputs = {k: self._ov(v) for k, v in vision_inputs.items()}
            v_inputs.update(self._merger_ov)
            img_feat = self._run(sh.s_vision, sh.b_vision, v_inputs, ["image_features"])[0]
            mm = self._run(sh.s_combine_image, sh.b_combine_image,
                           {"token_embeds": token_embeds, "image_features": img_feat},
                           ["multimodal_embeds"])[0]
        else:
            # TEXT: token_embeds passthrough (no special/media tokens)
            mm = self._run(sh.s_embed, sh.b_embed, {"input_ids": ids_ov}, ["token_embeds"])[0]

        # reused cached buffer for the stable-shape Main output
        emb_buf = self._cached_out("embeddings", (b, Cfg.t_hidden), np.float32)
        # bind the multimodal embeds + input_ids + the 8 cached LoRA tensors
        main_inputs = {"inputs_embeds": mm, "input_ids": ids_ov}
        main_inputs.update(self._lora_ov)
        emb_buf = self._run(
            sh.s_main, sh.b_main, main_inputs, ["embeddings"], out_bufs=[emb_buf])[0]
        # copy out of the reused buffer so callers may retain results across encode() calls
        return emb_buf.numpy().copy()


# ══════════════════════════════════════════════════════════════════════════════
# Sample inputs (self-contained, mirror the demo/standalone)
# ══════════════════════════════════════════════════════════════════════════════
def make_sample_images():
    from PIL import Image
    h, w = INPUT_IMAGE_SIZE                            # raw input shape [H, W]; the graph resizes to the target
    size = (w, h)                                      # PIL Image.new takes (width, height)
    imgs = {
        "[img] red square": Image.new("RGB", size, (220, 30, 30)),
        "[img] blue square": Image.new("RGB", size, (30, 60, 220)),
    }
    board = np.zeros((h, w, 3), dtype=np.uint8)
    tile = 64
    for i in range(0, h, tile):
        for j in range(0, w, tile):
            if (i // tile + j // tile) % 2 == 0:
                board[i:i + tile, j:j + tile] = 255
    imgs["[img] checkerboard"] = Image.fromarray(board, "RGB")
    return imgs


def make_sample_audios():
    t = np.linspace(0.0, AUDIO_SECONDS, int(AUDIO_SR * AUDIO_SECONDS), endpoint=False)
    rng = np.random.default_rng(0)
    return {
        "[aud] 440 Hz tone": 0.5 * np.sin(2 * np.pi * 440 * t),
        "[aud] 880 Hz tone": 0.5 * np.sin(2 * np.pi * 880 * t),
        "[aud] white noise": 0.2 * rng.standard_normal(t.shape),
    }


def build_audio_ids(n_tokens: int) -> np.ndarray:
    """[audio_start, <audio> * n_tokens, audio_end] — n_tokens = number of pooled audio frames."""
    return np.array([[Cfg.audio_start_token_id, *([Cfg.audio_token_id] * n_tokens),
                      Cfg.audio_end_token_id]], dtype=np.int32)


# ══════════════════════════════════════════════════════════════════════════════
# Cross-modal demo
# ══════════════════════════════════════════════════════════════════════════════
def print_similarity_matrix(labels, matrix):
    width = max(len(l) for l in labels)
    print(" " * (width + 2) + " ".join(f"{i:>6d}" for i in range(len(labels))))
    for i, label in enumerate(labels):
        row = " ".join(f"{matrix[i, j]:6.2f}" for j in range(len(labels)))
        print(f"{label:<{width}}  {row}")


def run_demo(task, ort, tokenizer, text_prefix):
    print(f"\n{'=' * 72}\n  ORT cross-modal demo (task={task})\n{'=' * 72}")
    texts = ["Mars is often called the Red Planet because of its reddish surface.",
             "A solid bright red colored square image.",
             "A pure sine wave musical tone at constant pitch."]
    labels, vectors = [], []
    # accumulated ort.encode() wall time per modality (excludes tokenize / preprocess)
    encode_time = 0.0

    def collect(label, vec):
        labels.append(label)
        vectors.append(vec)
        print(f"  {label:<30s} dim={vec.shape[-1]:<4d} ||v||={np.linalg.norm(vec):.4f}")

    print("\nText embeddings:")
    enc = tokenizer([f"{text_prefix}{t}" for t in texts], padding=True, truncation=True,
                    max_length=MAX_SEQ_LEN, return_tensors="np")
    t0 = time.time()
    tvecs = ort.encode(enc["input_ids"])
    text_elapsed = time.time() - t0
    encode_time += text_elapsed
    for t, v in zip(texts, tvecs):
        snip = (t[:34] + "\u2026") if len(t) > 35 else t
        collect(f"[txt] {snip}", v)
    print(f"  Time Cost: {text_elapsed:.3f} Seconds ({len(texts)} texts)")

    # fixed vision grid -> constant image-token count
    n_img_tokens = IMAGE_HEIGHT_FACTOR * IMAGE_WIDTH_FACTOR
    print("\nImage embeddings:")
    image_elapsed = 0.0
    for label, img in make_sample_images().items():
        enc = tokenizer(f"{text_prefix}{IMAGE_PLACEHOLDER * n_img_tokens}", return_tensors="np")
        vin = prepare_vision_inputs(img)                       # raw [1,3,H,W]; resize/patchify in-graph
        t0 = time.time()
        v = ort.encode(enc["input_ids"], vision_inputs=vin)
        image_elapsed += time.time() - t0
        collect(label, v[0])
    encode_time += image_elapsed
    print(f"  Time Cost: {image_elapsed:.3f} Seconds")

    print("\nAudio embeddings:")
    audio_elapsed = 0.0
    for label, wave in make_sample_audios().items():
        ain, n_aud_tokens = prepare_audio_inputs(wave)         # raw waveform; STFT/mel + chunking in-graph
        ids = build_audio_ids(n_aud_tokens)                    # pooled-frame count == audio tokens
        t0 = time.time()
        v = ort.encode(ids, audio_inputs=ain)
        audio_elapsed += time.time() - t0
        collect(label, v[0])
    encode_time += audio_elapsed
    print(f"  Time Cost: {audio_elapsed:.3f} Seconds")

    emb = np.stack(vectors)
    sim = emb @ emb.T
    print("\nCross-modal cosine-similarity matrix:\n")
    print_similarity_matrix(labels, sim)
    print("\nColumn index = row order above.")
    print(f"\nTotal Encode Time Cost: {encode_time:.3f} Seconds ({len(labels)} embeddings)")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    import contextlib
    import io
    from transformers import AutoTokenizer
    from transformers import logging as hf_logging
    hf_logging.set_verbosity_error()

    # always export/test every task module listed in config.json
    tasks = _CONFIG["task_names"]

    if DO_EXPORT:
        # shared task-agnostic graphs, then the per-task provider(s): FUSED -> one all-task Omni_LoRA;
        # SPLIT -> one Omni_LoRA per task.
        export_shared_graphs()
        if FUSE_LORA_INTO_EMBED:
            export_fused_provider(tasks)
        else:
            # load the full model state dict ONCE and reuse it across every task provider
            sd_full = {k: v.float() for k, v in load_file(str(MODEL_DIR / "model.safetensors")).items()}
            for task in tasks:
                export_task_provider(task, sd=sd_full)
        # bake the runtime constants into the tiny combine graphs so the ONNX inference reads them from
        # metadata (get_modelmeta().custom_metadata_map) instead of config.json / tokenizer.json.
        meta = runtime_metadata_dict(tasks)
        for g in ("Omni_Combine_Image.onnx", "Omni_Combine_Audio.onnx"):
            bake_metadata_props(shared_dir() / g, meta)
        print(f"[meta] baked runtime metadata into combine graphs: {sorted(meta)}")

    if not RUN_DEMO:
        print("\nExport complete (no run requested).")
        return

    # only the tokenizer stays on CPU (text -> input_ids); image/audio front-ends are inlined.
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)

    # the 5 shared sessions are loaded ONCE and reused by every task (only the cached tensors differ)
    shared = OmniShared(tasks)
    for task in tasks:
        text_prefix = "Query: " if task == "retrieval" else "Document: "
        print(f"\n{'#' * 72}\n# Task: {task}\n{'#' * 72}")
        ort = OmniORT(task, shared)
        if RUN_DEMO:
            run_demo(task, ort, tokenizer, text_prefix)


if __name__ == "__main__":
    main()
