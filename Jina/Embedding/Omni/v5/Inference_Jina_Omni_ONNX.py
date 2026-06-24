"""
Inference_Jina_Omni_ONNX.py — ONNX Runtime inference for jina-embeddings-v5-omni-nano
=====================================================================================
Pure inference over the ONNX graphs exported by Export_Jina_Omni.py: text, image and
audio embedded into ONE shared embedding space. No torch / safetensors / config.json — only
onnxruntime + numpy (+ a CPU tokenizer for text and PIL for the image demo). Every model and
preprocessing constant is read from the ONNX custom metadata baked at export time.

Pipeline (per encode(), single-modality):
    TEXT  -> Omni_Embed -> Omni_Main
    IMAGE -> Omni_Embed + Omni_Vision -> Omni_Combine_Image -> Omni_Main
    AUDIO -> Omni_Audio -> Omni_Combine_Audio -> Omni_Main

Plugin-LoRA: the per-task tensors (text LoRA, vision merger, audio projector, task tokens) come from
the Omni_LoRA provider graph — FUSED mode runs one shared all-task provider selected by task_index,
SPLIT mode loads one provider per task. The provider runs ONCE per task at startup; its 17 OrtValues
are cached and bound on every encode(). All sessions chain through IOBinding / OrtValue (only the final
embedding is materialized).

The image/audio front-ends (resize + normalize + patchify; Whisper log-mel via STFT + chunking) are
inlined in the ONNX graphs, so runtime preprocessing only lays out raw RGB [1,3,H,W] and zero-pads the
waveform (+ feature_len). FUSED vs SPLIT is auto-detected from the shared provider graph; the ORT
settings MUST match the export values.

PARITY-CRITICAL (do not change): the audio/merger sublayers use EXACT (erf) GELU, so ORT
optimization.enable_gelu_approximation MUST stay OFF (it rewrites exact GELU to tanh, ~3e-3 drift).
"""

from __future__ import annotations

import os
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import onnxruntime

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG  (runtime environment only -- every model + preprocessing constant is read from the ONNX
#          metadata baked at export time, so no config.json is needed: just the tokenizer + ORT)
# ══════════════════════════════════════════════════════════════════════════════
MODEL_DIR        = Path("/home/DakeQQ/Downloads/jina-embeddings-v5-omni-nano")   # tokenizer files only
OUTPUT_ROOT      = Path(__file__).resolve().parent / "Jina_Omni_Optimized"       # exported ONNX root

# FUSED (one shared all-task Omni_LoRA selected by task_index) vs SPLIT (one provider per task) is
# auto-detected from whether the shared provider graph exists -- no flag to keep in sync with export.
FUSE_LORA_INTO_EMBED = (OUTPUT_ROOT / "Omni_LoRA.onnx").exists()

AUDIO_SECONDS    = 5.0               # demo-only: length (s) of the generated sample tones

ORT_LOG          = False
ORT_FP16         = False             # FP16 ORT session opts (qdq accuracy level + fp16 cast optimizers)
ORT_Accelerate_Providers = []        # e.g. ['CUDAExecutionProvider'] or ['DmlExecutionProvider']; empty = CPU only
MAX_THREADS      = 0                 # 0 = let ORT decide
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


# ══════════════════════════════════════════════════════════════════════════════
# RUNTIME CONFIG  (read ONCE from the ONNX custom metadata; baked at export time by
#                  Export_Jina_Omni.py -- the exporter is the single source of truth)
# ══════════════════════════════════════════════════════════════════════════════
class RuntimeConfig:
    """Model + preprocessing constants the inference needs, read from an ONNX graph's custom metadata
    (get_modelmeta().custom_metadata_map). Replaces every config.json / tokenizer.json lookup."""

    def __init__(self, meta: dict):
        if not meta or "embed_dim" not in meta:
            raise RuntimeError(
                "Exported ONNX graphs carry no runtime metadata. Re-run Export_Jina_Omni.py "
                "(it bakes the runtime constants into the combine graphs).")
        self.embed_dim = int(meta["embed_dim"])                      # final embedding dim (Omni_Main out)
        self.max_seq_len = int(meta["max_seq_len"])                  # text truncation length
        self.audio_sr = int(meta["audio_sr"])                        # Whisper sample rate
        self.audio_n_fft = int(meta["audio_n_fft"])                  # Whisper n_fft
        self.audio_hop = int(meta["audio_hop"])                      # Whisper hop_length
        self.audio_max_seconds = float(meta["audio_max_seconds"])    # truncation cap (chunk_length)
        self.audio_n_window = int(meta["audio_n_window"])            # half-chunk frames (chunk = 2*n_window)
        self.audio_start_token_id = int(meta["audio_start_token_id"])
        self.audio_token_id = int(meta["audio_token_id"])
        self.audio_end_token_id = int(meta["audio_end_token_id"])
        self.image_height_factor = int(meta["image_height_factor"])  # target_h = factor*patch*merge
        self.image_width_factor = int(meta["image_width_factor"])    # target_w = factor*patch*merge
        self.input_image_size = [int(meta["input_image_height"]), int(meta["input_image_width"])]
        self.image_placeholder = meta["image_placeholder"]           # <image> token string
        self.task_names = [t for t in meta["task_names"].split(",") if t]

    @property
    def n_image_tokens(self) -> int:
        """Fixed merged-token count for one image (= factor_h * factor_w)."""
        return self.image_height_factor * self.image_width_factor

    @classmethod
    def from_session(cls, session) -> "RuntimeConfig":
        return cls(session.get_modelmeta().custom_metadata_map)


def task_dir(task: str) -> Path:
    return OUTPUT_ROOT / task


# ══════════════════════════════════════════════════════════════════════════════
# Per-task runtime-input names (the Omni_LoRA provider emits exactly these 17 tensors, in this order).
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
# Provider output order: 8 LoRA + 6 merger + 2 projector + 1 task-token table = 17 tensors.
LORA_PROVIDER_OUTPUT_NAMES = (
    LORA_INPUT_NAMES + MERGER_INPUT_NAMES + PROJECTOR_INPUT_NAMES + [TASK_TOKEN_INPUT_NAME]
)


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


def prepare_audio_inputs(waveform, cfg):
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
    max_samples = int(cfg.audio_max_seconds * cfg.audio_sr)
    R = min(wav.shape[0], max_samples)                          # Whisper truncates to chunk_length

    # zero-pad to a multiple of hop with an n_fft zero tail so the last real frame's STFT window sees
    # zeros (matches Whisper's zero-pad).
    L_pad = ((R + cfg.audio_n_fft + cfg.audio_hop - 1) // cfg.audio_hop) * cfg.audio_hop
    wav_pad = np.zeros(L_pad, dtype=np.int16)
    wav_pad[:R] = wav[:R]

    # Whisper valid-frame count (verified n_valid = R // hop); the graph slices its mel to this.
    feature_len = R // cfg.audio_hop

    # audio token count from feature_len: per-chunk after-conv2 sum, then stride-2 pool.
    n_window2 = cfg.audio_n_window * 2                          # 200 mel frames per chunk
    num_chunks = max(1, (feature_len + n_window2 - 1) // n_window2)
    raw_chunk_lens = np.clip(feature_len - np.arange(num_chunks) * n_window2, 0, n_window2)
    encoded_len = int(((raw_chunk_lens - 1) // 2 + 1).sum())
    num_tokens = encoded_len // 2
    return {
        "waveform": np.ascontiguousarray(wav_pad.reshape(1, 1, L_pad), dtype=np.int16),
        "feature_len": np.array([feature_len], dtype=np.int64),
    }, num_tokens


# ══════════════════════════════════════════════════════════════════════════════
# ORT SESSIONS & IOBINDING RUNTIME
# ══════════════════════════════════════════════════════════════════════════════
def shared_dir() -> Path:
    return OUTPUT_ROOT


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
    """Holds the 6 task-agnostic ONNX sessions + IOBindings, loaded ONCE and reused by every task's
    OmniORT. FUSED mode also owns the shared all-task Omni_LoRA provider (selected by task_index);
    SPLIT mode loads a per-task provider in each OmniORT."""

    def __init__(self):
        d = shared_dir()
        so = make_session_opts()
        providers = ORT_PROVIDERS
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
        # every model + preprocessing constant (and the task list) is read from the ONNX metadata baked
        # into the combine graphs at export time -- no config.json. Read it once here and reuse.
        self.cfg = RuntimeConfig.from_session(self.s_combine_audio)
        self.tasks = self.cfg.task_names
        self.task_index_map = {t: i for i, t in enumerate(self.tasks)}


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
        emb_buf = self._cached_out("embeddings", (b, self.shared.cfg.embed_dim), np.float32)
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
def make_sample_images(cfg):
    from PIL import Image
    h, w = cfg.input_image_size                        # raw input shape [H, W]; the graph resizes to the target
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


def make_sample_audios(cfg):
    t = np.linspace(0.0, AUDIO_SECONDS, int(cfg.audio_sr * AUDIO_SECONDS), endpoint=False)
    rng = np.random.default_rng(0)
    return {
        "[aud] 440 Hz tone": 0.5 * np.sin(2 * np.pi * 440 * t),
        "[aud] 880 Hz tone": 0.5 * np.sin(2 * np.pi * 880 * t),
        "[aud] white noise": 0.2 * rng.standard_normal(t.shape),
    }


def build_audio_ids(n_tokens: int, cfg) -> np.ndarray:
    """[audio_start, <audio> * n_tokens, audio_end] — n_tokens = number of pooled audio frames."""
    return np.array([[cfg.audio_start_token_id, *([cfg.audio_token_id] * n_tokens),
                      cfg.audio_end_token_id]], dtype=np.int32)


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
    cfg = ort.shared.cfg
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
                    max_length=cfg.max_seq_len, return_tensors="np")
    t0 = time.time()
    tvecs = ort.encode(enc["input_ids"])
    text_elapsed = time.time() - t0
    encode_time += text_elapsed
    for t, v in zip(texts, tvecs):
        snip = (t[:34] + "\u2026") if len(t) > 35 else t
        collect(f"[txt] {snip}", v)
    print(f"  Time Cost: {text_elapsed:.3f} Seconds ({len(texts)} texts)")

    # fixed vision grid -> constant image-token count
    n_img_tokens = cfg.n_image_tokens
    print("\nImage embeddings:")
    image_elapsed = 0.0
    for label, img in make_sample_images(cfg).items():
        enc = tokenizer(f"{text_prefix}{cfg.image_placeholder * n_img_tokens}", return_tensors="np")
        vin = prepare_vision_inputs(img)                       # raw [1,3,H,W]; resize/patchify in-graph
        t0 = time.time()
        v = ort.encode(enc["input_ids"], vision_inputs=vin)
        image_elapsed += time.time() - t0
        collect(label, v[0])
    encode_time += image_elapsed
    print(f"  Time Cost: {image_elapsed:.3f} Seconds")

    print("\nAudio embeddings:")
    audio_elapsed = 0.0
    for label, wave in make_sample_audios(cfg).items():
        ain, n_aud_tokens = prepare_audio_inputs(wave, cfg)    # raw waveform; STFT/mel + chunking in-graph
        ids = build_audio_ids(n_aud_tokens, cfg)               # pooled-frame count == audio tokens
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

    # the exported graphs must already exist (run Export_Jina_Omni.py first)
    missing = [n for n in ("Omni_Embed.onnx", "Omni_Vision.onnx", "Omni_Audio.onnx",
                           "Omni_Combine_Image.onnx", "Omni_Combine_Audio.onnx", "Omni_Main.onnx")
               if not (shared_dir() / n).exists()]
    if FUSE_LORA_INTO_EMBED and not (shared_dir() / "Omni_LoRA.onnx").exists():
        missing.append("Omni_LoRA.onnx")
    if missing:
        raise FileNotFoundError(
            f"Missing exported ONNX graphs in {shared_dir()}: {missing}. "
            f"Run Export_Jina_Omni.py first.")

    # only the tokenizer stays on CPU (text -> input_ids); image/audio front-ends are inlined.
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)

    # the shared sessions are loaded ONCE; the task list and every model constant come from the ONNX
    # metadata (read inside OmniShared) -- no config.json.
    shared = OmniShared()
    if not FUSE_LORA_INTO_EMBED:
        # SPLIT mode: each task needs its own provider graph (checked now the task list is known).
        split_missing = [t for t in shared.tasks if not (task_dir(t) / "Omni_LoRA.onnx").exists()]
        if split_missing:
            raise FileNotFoundError(
                f"Missing per-task Omni_LoRA.onnx for tasks {split_missing}. "
                f"Run Export_Jina_Omni.py first.")
    for task in shared.tasks:
        text_prefix = "Query: " if task == "retrieval" else "Document: "
        print(f"\n{'#' * 72}\n# Task: {task}\n{'#' * 72}")
        ort = OmniORT(task, shared)
        run_demo(task, ort, tokenizer, text_prefix)


if __name__ == "__main__":
    main()
