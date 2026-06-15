"""
Jina Embeddings v5 — Shared Backbone with LoRA (Fused or Split mode)
=====================================================================
Architecture (FUSE_LORA_INTO_EMBED = True):
    1 fused Embed+LoRA ONNX  — token embedding + all-task LoRA (int32 index selects task)
    1 shared Main ONNX       — backbone with LoRA as *inputs*

Architecture (FUSE_LORA_INTO_EMBED = False):
    1 Embed ONNX             — token embedding only
    N task LoRA ONNX files   — one per task, each outputs the 8 LoRA tensors
    1 shared Main ONNX       — backbone with LoRA as *inputs*

LoRA tensors can optionally be quantized (Q4, Q8, F16, F32) to reduce
ONNX model size. Quantization settings mirror the KV cache quantization
style from the Qwen export scripts.

Embed_LoRA ONNX outputs (fused mode):
    hidden_states [batch, seq_len, hidden_size]
    + 8 stacked LoRA tensors across num_layers layers

Split mode: Embed ONNX outputs hidden_states; task LoRA ONNX outputs 8 LoRA tensors.
"""

import gc
import json
import os
import time

import numpy as np
import onnxruntime
import torch
import torch.nn.functional as F
from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer
from transformers.models.qwen3 import Qwen3Model


MODEL_PATH              = r"/home/DakeQQ/Downloads/jina-embeddings-v5-text-small"                    # Path to the pretrained Jina v5 model directory
ONNX_OUTPUT_DIR         = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Jina_ONNX")      # Output directory for exported ONNX files
os.makedirs(ONNX_OUTPUT_DIR, exist_ok=True)
onnx_model_EmbedLoRA    = os.path.join(ONNX_OUTPUT_DIR, 'Embedding_Embed_LoRA.onnx')
onnx_model_Embed        = os.path.join(ONNX_OUTPUT_DIR, 'Embedding_Embed.onnx')
onnx_model_Main         = os.path.join(ONNX_OUTPUT_DIR, 'Embedding_Main.onnx')


# Export config
DO_EXPORT               = True              # Whether to export the ONNX models
PREVENT_F16_OVERFLOW    = False             # Apply overflow scaling in RMSNorm for FP16 stability
MAX_SEQ_LEN             = 8192              # Maximum token sequence length (clamped to model's max_position_embeddings)
OPSET                   = 18                # ONNX opset version for export

# LoRA architecture config
FUSE_LORA_INTO_EMBED    = True              # True: 1 fused Embed+LoRA ONNX (all tasks); False: 1 Embed + N separate task LoRA ONNXs

# Quantization config — separate dtype for embed vs LoRA, shared algorithm params
# Fused mode:  EMBED_QUANT_DTYPE → embed_tokens.weight;  LORA_QUANT_DTYPE → LoRA tensors (both inside one ONNX)
# Split mode:  EMBED_QUANT_DTYPE → Embed ONNX;           LORA_QUANT_DTYPE → per-task LoRA ONNXs
EMBED_QUANT_DTYPE       = "F16"             # "ROTARY_Q4" | "ROTARY_Q4_CUDA" | "Q8" | "Q8_CUDA" | "ROTARY_Q8" | "ROTARY_Q8_CUDA" | "F16" | "F32"
LORA_QUANT_DTYPE        = "F16"             # "ROTARY_Q4" | "ROTARY_Q4_CUDA" | "Q8" | "Q8_CUDA" | "ROTARY_Q8" | "ROTARY_Q8_CUDA" | "F16" | "F32"

# Shared quantization algorithm parameters (apply to both EMBED and LORA when quantized)
LORA_QUANT_GROUP_SIZE   = 16                # Group size for Q4 and Q8 (when USE_HADAMARD or USE_SHUFFLE enabled) per-group quantization. Smaller = more accurate. Must divide last_dim evenly.
LORA_USE_HADAMARD       = True              # True = More Accuracy. Apply enhanced randomized Walsh-Hadamard mixing within each group before quantization. Works for Q4 and Q8 modes.
LORA_HADAMARD_SEED      = 9527              # Seed for the deterministic Rademacher sign pattern used by the enhanced Hadamard transform.
LORA_USE_CLIP           = True              # Clip outliers to mean ± LORA_CLIP_SIGMA*std before quantization. Works for Q4 and Q8 modes.
LORA_CLIP_SIGMA         = 3.0               # Clip threshold in standard deviations. Lower = more aggressive clipping. 2.5-3.5 recommended.
LORA_USE_SHUFFLE        = True              # True = More Accuracy. Interleave channels across groups so that high-variance channels are evenly distributed. Works for Q4 and Q8 modes.
LORA_USE_SYM            = False             # True = Less storage. True: symmetric quantization (no bias, absmax-based); False: asymmetric (min-max with bias). Works for Q4 and Q8 modes.
LORA_USE_FLOAT16_SCALE  = True              # Whether to use float16 for scale and bias in all quantized modes (Q4, Q8, and ROTARY variants).

# Runtime config
ORT_LOG                 = False             # Enable verbose ONNX Runtime logging
ORT_FP16                = False             # Use FP16 optimizations in ONNX Runtime session
ORT_Accelerate_Providers = []               # E.g. ['CUDAExecutionProvider'] or ['DmlExecutionProvider']; empty = CPU only
MAX_THREADS             = 0                 # ORT inter/intra-op thread count; 0 = let ORT decide
DEVICE_ID               = 0                 # GPU device index for CUDA/DML providers

# Fused LoRA adapter task_index mapping (int32 input):
#   0 = classification
#   1 = clustering
#   2 = retrieval
#   3 = text-matching


# Demo inputs
TEST_QUERIES = [
    "What are the effects of climate change on oceans?",
    "How does machine learning work?",
]

TEST_DOCUMENTS = [
    "Climate change has led to rising sea levels, ocean acidification, and disruption of marine ecosystems worldwide.",
    "Machine learning is a subset of AI where algorithms learn patterns from data to make predictions without explicit programming.",
    "The French Revolution began in 1789 and fundamentally transformed French society and politics.",
    "Photosynthesis converts sunlight into chemical energy in plants using chlorophyll.",
]


# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS & EXPECTED CONFIG
# ══════════════════════════════════════════════════════════════════════════════

MODEL_CONFIG_KEYS = (
    "vocab_size",
    "hidden_size",
    "intermediate_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "rms_norm_eps",
    "rope_theta",
    "max_position_embeddings",
)
ADAPTER_CONFIG_KEYS = ("r", "lora_alpha")

LoRA_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)
LoRA_TARGET_GROUPS = {
    "q_proj": "self_attn",
    "k_proj": "self_attn",
    "v_proj": "self_attn",
    "o_proj": "self_attn",
    "gate_proj": "mlp",
    "up_proj": "mlp",
    "down_proj": "mlp",
}
LoRA_NORM_SOURCES = {
    "q_proj": "attn_input",
    "k_proj": "attn_input",
    "v_proj": "attn_input",
    "o_proj": None,
    "gate_proj": "mlp_input",
    "up_proj": "mlp_input",
    "down_proj": None,
}

# Names of the 8 stacked LoRA input tensors
LORA_INPUT_NAMES = [
    "lora_qkv_a",
    "lora_qkv_b",
    "lora_o_a",
    "lora_o_b",
    "lora_gate_up_a",
    "lora_gate_up_b",
    "lora_down_a",
    "lora_down_b",
]

# Supported LoRA quantization dtypes
SUPPORTED_LORA_QUANT_DTYPES = (
    "ROTARY_Q4", "ROTARY_Q4_CUDA", "Q8", "Q8_CUDA",
    "ROTARY_Q8", "ROTARY_Q8_CUDA", "F16", "F32"
)


def get_lora_onnx_path(task_name):
    """Return ONNX file path for a split-mode task LoRA model."""
    return os.path.join(ONNX_OUTPUT_DIR, f'Embedding_LoRA_{task_name}.onnx')


# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def load_json(path):
    with open(path, 'r', encoding='utf-8') as file:
        return json.load(file)


def resolve_max_seq_len(model_config):
    if MAX_SEQ_LEN > model_config.max_position_embeddings:
        print(
            f"\n[Warning] MAX_SEQ_LEN ({MAX_SEQ_LEN}) exceeds config.max_position_embeddings "
            f"({model_config.max_position_embeddings}); clamping to the model limit."
        )
    return min(MAX_SEQ_LEN, model_config.max_position_embeddings)


def resolve_model_config_value(model_config, key):
    if hasattr(model_config, key):
        return True, getattr(model_config, key)
    config_dict = model_config.to_dict()
    if key in config_dict:
        return True, config_dict[key]
    if key == 'rope_theta':
        rope_parameters = getattr(model_config, 'rope_parameters', None) or config_dict.get('rope_parameters')
        if isinstance(rope_parameters, dict) and 'rope_theta' in rope_parameters:
            return True, rope_parameters['rope_theta']
    return False, None


def load_expected_model_config(model_path):
    """Read model architecture values from config.json at the given path."""
    config_data = load_json(os.path.join(model_path, "config.json"))
    result = {}
    for key in MODEL_CONFIG_KEYS:
        if key in config_data:
            result[key] = config_data[key]
        elif key == "rope_theta":
            rope_params = config_data.get("rope_parameters")
            if isinstance(rope_params, dict) and "rope_theta" in rope_params:
                result[key] = rope_params["rope_theta"]
    return result


def load_expected_adapter_config(model_path, task_names):
    """Read adapter rank/alpha from the first available adapter_config.json."""
    for task in task_names:
        cfg_path = os.path.join(model_path, "adapters", task, "adapter_config.json")
        if os.path.isfile(cfg_path):
            cfg = load_json(cfg_path)
            return {k: cfg[k] for k in ADAPTER_CONFIG_KEYS if k in cfg}
    return {}


def validate_model_config(model_config, expected_model_config):
    for key, expected in expected_model_config.items():
        found, actual = resolve_model_config_value(model_config, key)
        if not found:
            print(f"\n[Warning] Could not resolve config.{key}; skipping validation for this field.")
            continue
        if actual != expected:
            print(f"\n[Warning] config.{key} = {actual} differs from config.json value {expected}.")


def validate_adapter_config(adapter_config, expected_adapter_config):
    for key, expected in expected_adapter_config.items():
        actual = adapter_config.get(key)
        if actual is None:
            print(f"\n[Warning] adapter_config missing key {key!r}.")
        elif actual != expected:
            print(f"\n[Warning] adapter_config[{key!r}] = {actual} differs from reference value {expected}.")


def collect_layer_LoRA_scales(qwen_model):
    norm_factor = qwen_model.config.hidden_size ** 0.5
    layer_scales = []
    for layer in qwen_model.layers:
        layer_scales.append({
            'attn_input': (layer.input_layernorm.weight.float().unsqueeze(0) * norm_factor).contiguous(),
            'mlp_input': (layer.post_attention_layernorm.weight.float().unsqueeze(0) * norm_factor).contiguous(),
        })
    return layer_scales


def build_task_LoRA_stacked(task_name, model_config, layer_scales, expected_adapter_config):
    """
    Build the 8 stacked LoRA tensors for a task.
    Returns dict: {tensor_name: np.ndarray[num_layers, ...]}.
    """
    print(f"\nLoading LoRA tensors for task '{task_name}' ...")
    adapter_dir = os.path.join(MODEL_PATH, 'adapters', task_name)
    adapter_config = load_json(os.path.join(adapter_dir, 'adapter_config.json'))
    adapter_model_path = os.path.join(adapter_dir, 'adapter_model.safetensors')
    scaling = adapter_config['lora_alpha'] / adapter_config['r']
    validate_adapter_config(adapter_config, expected_adapter_config)

    # Collect per-layer tensors
    all_qkv_a, all_qkv_b = [], []
    all_o_a, all_o_b = [], []
    all_gate_up_a, all_gate_up_b = [], []
    all_down_a, all_down_b = [], []

    with safe_open(adapter_model_path, framework='pt', device='cpu') as handle:
        key_set = set(handle.keys())
        for layer_idx in range(model_config.num_hidden_layers):
            scale_map = layer_scales[layer_idx]
            layer_lora = {}
            for module_name in LoRA_TARGET_MODULES:
                module_group = LoRA_TARGET_GROUPS[module_name]
                lora_key_prefix = f'base_model.model.layers.{layer_idx}.{module_group}.{module_name}'
                lora_a_key = f'{lora_key_prefix}.lora_A.weight'
                lora_b_key = f'{lora_key_prefix}.lora_B.weight'
                if lora_a_key not in key_set or lora_b_key not in key_set:
                    raise KeyError(f'Missing LoRA tensors for task {task_name!r}: {lora_a_key}, {lora_b_key}')

                lora_a = handle.get_tensor(lora_a_key).float()
                lora_b = handle.get_tensor(lora_b_key).float() * scaling
                norm_source = LoRA_NORM_SOURCES[module_name]
                if norm_source is not None:
                    lora_a = lora_a * scale_map[norm_source]

                layer_lora[module_name] = (lora_a, lora_b)

            # Fuse q/k/v LoRA
            qkv_a = torch.cat([layer_lora['q_proj'][0], layer_lora['k_proj'][0], layer_lora['v_proj'][0]], dim=0)
            qkv_b = torch.block_diag(layer_lora['q_proj'][1], layer_lora['k_proj'][1], layer_lora['v_proj'][1])

            all_qkv_a.append(qkv_a)
            all_qkv_b.append(qkv_b)
            all_o_a.append(layer_lora['o_proj'][0])
            all_o_b.append(layer_lora['o_proj'][1])

            # Fuse gate/up LoRA
            gate_up_a = torch.cat([layer_lora['gate_proj'][0], layer_lora['up_proj'][0]], dim=0)
            gate_up_b = torch.block_diag(layer_lora['gate_proj'][1], layer_lora['up_proj'][1])
            all_gate_up_a.append(gate_up_a)
            all_gate_up_b.append(gate_up_b)

            all_down_a.append(layer_lora['down_proj'][0])
            all_down_b.append(layer_lora['down_proj'][1])

    # Stack into [num_layers, ...] tensors
    stacked = {
        'lora_qkv_a': torch.stack(all_qkv_a).detach().numpy(),
        'lora_qkv_b': torch.stack(all_qkv_b).detach().numpy(),
        'lora_o_a': torch.stack(all_o_a).detach().numpy(),
        'lora_o_b': torch.stack(all_o_b).detach().numpy(),
        'lora_gate_up_a': torch.stack(all_gate_up_a).detach().numpy(),
        'lora_gate_up_b': torch.stack(all_gate_up_b).detach().numpy(),
        'lora_down_a': torch.stack(all_down_a).detach().numpy(),
        'lora_down_b': torch.stack(all_down_b).detach().numpy(),
    }

    total_bytes = sum(arr.nbytes for arr in stacked.values())
    print(f"  LoRA adapter size for '{task_name}': {total_bytes / 1024 / 1024:.1f} MB")
    for name, arr in stacked.items():
        print(f"    {name}: {arr.shape}")

    return stacked


def prepare_input_texts(texts, task_name, prompt_name, sentence_transformer_config):
    prompts = sentence_transformer_config.get('prompts', {})
    default_prompt_name = sentence_transformer_config.get('default_prompt_name', 'document')
    if task_name == 'retrieval':
        prompt_name = default_prompt_name if prompt_name is None else prompt_name
        if prompt_name not in prompts:
            raise ValueError(f'Unknown retrieval prompt_name: {prompt_name}')
        prefix = prompts[prompt_name]
        return [f'{prefix}{text}' for text in texts]
    if prompt_name is None:
        return list(texts)
    if prompt_name not in prompts:
        raise ValueError(f'Unknown prompt_name: {prompt_name}')
    prefix = prompts[prompt_name]
    return [f'{prefix}{text}' for text in texts]


def tokenize_texts(tokenizer, texts, task_name, prompt_name, sentence_transformer_config, max_length):
    prepared_texts = prepare_input_texts(texts, task_name, prompt_name, sentence_transformer_config)
    batch = tokenizer(
        prepared_texts,
        return_tensors='np',
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    return prepared_texts, batch['input_ids'].astype(np.int32)


# ══════════════════════════════════════════════════════════════════════════════
# LORA QUANTIZER (mirrors KVQuantizer from Qwen_Export.py)
# ══════════════════════════════════════════════════════════════════════════════

def _next_power_of_two(n):
    value = 1
    while value < n:
        value *= 2
    return value


def normalize_lora_quant_settings(last_dim):
    """Validate and normalize LoRA quant settings once a representative last_dim is known."""
    global LORA_QUANT_GROUP_SIZE

    if LORA_QUANT_DTYPE not in SUPPORTED_LORA_QUANT_DTYPES:
        raise ValueError(f"Unsupported LORA_QUANT_DTYPE: {LORA_QUANT_DTYPE}")

    quantized = {"Q8", "Q8_CUDA", "ROTARY_Q8", "ROTARY_Q8_CUDA", "ROTARY_Q4", "ROTARY_Q4_CUDA"}
    rotary = {"ROTARY_Q4", "ROTARY_Q4_CUDA", "ROTARY_Q8", "ROTARY_Q8_CUDA"}
    notes = []

    if LORA_QUANT_DTYPE in rotary and last_dim % 2 != 0:
        raise ValueError(f"{LORA_QUANT_DTYPE} requires an even last_dim, got {last_dim}.")
    if LORA_QUANT_DTYPE in {"Q8_CUDA", "ROTARY_Q8_CUDA"} and last_dim % 4 != 0:
        raise ValueError(f"{LORA_QUANT_DTYPE} requires last_dim divisible by 4, got {last_dim}.")
    if LORA_QUANT_DTYPE == "ROTARY_Q4_CUDA" and last_dim % 8 != 0:
        raise ValueError(f"{LORA_QUANT_DTYPE} requires last_dim divisible by 8, got {last_dim}.")

    if LORA_QUANT_DTYPE in quantized:
        if LORA_QUANT_GROUP_SIZE <= 0:
            raise ValueError(f"LORA_QUANT_GROUP_SIZE must be positive, got {LORA_QUANT_GROUP_SIZE}.")
        if LORA_QUANT_GROUP_SIZE > last_dim:
            notes.append(f"[Warning] LORA_QUANT_GROUP_SIZE ({LORA_QUANT_GROUP_SIZE}) > last_dim ({last_dim}); clamping.")
            LORA_QUANT_GROUP_SIZE = last_dim
        elif LORA_QUANT_GROUP_SIZE < last_dim and last_dim % LORA_QUANT_GROUP_SIZE != 0:
            original = LORA_QUANT_GROUP_SIZE
            LORA_QUANT_GROUP_SIZE = max(g for g in range(1, LORA_QUANT_GROUP_SIZE + 1) if last_dim % g == 0)
            notes.append(f"[Warning] LORA_QUANT_GROUP_SIZE ({original}) doesn't divide last_dim ({last_dim}); using {LORA_QUANT_GROUP_SIZE}.")
    elif any((LORA_USE_HADAMARD, LORA_USE_CLIP, LORA_USE_SHUFFLE, LORA_USE_SYM, LORA_USE_FLOAT16_SCALE)):
        notes.append("[Info] Quant-only flags are ignored when LORA_QUANT_DTYPE is F16 or F32.")

    return notes


class LoRAQuantizer:
    """Unified LoRA weight quantizer mirroring KVQuantizer from Qwen_Export.py.

    Supports all KV cache quantization modes applied to LoRA weight tensors:
    - ROTARY_Q4 / ROTARY_Q4_CUDA: rotary + 4-bit quantization
    - Q8 / Q8_CUDA: 8-bit quantization
    - ROTARY_Q8 / ROTARY_Q8_CUDA: rotary + 8-bit quantization

    Precision-enhancement techniques:
    1. Rotary transform (ROTARY_* modes): orthogonal pairwise rotation spreading outlier energy
    2. Enhanced Hadamard transform (LORA_USE_HADAMARD): Walsh-Hadamard within groups
    3. Channel shuffle (LORA_USE_SHUFFLE): interleave channels across groups
    4. Residual bias correction (asymmetric modes): reduces systematic dequant drift
    5. Sigma clipping (LORA_USE_CLIP): clip outliers before quantization
    """

    def __init__(self, last_dim, group_size, is_q4=False, is_rotary=False, is_cuda=False,
                 use_sym=False, use_hadamard=False, use_clip=False, clip_sigma=3.0, use_shuffle=False):
        self.last_dim = last_dim
        self.last_dim_half = last_dim // 2
        self.is_q4 = is_q4
        self.is_rotary = is_rotary
        self.is_cuda = is_cuda
        self.use_sym = use_sym
        self.use_hadamard = use_hadamard
        self.use_clip = use_clip
        self.clip_sigma = clip_sigma
        self.use_shuffle = use_shuffle
        self.use_residual_bias_correction = not use_sym

        # Quantization range
        if use_sym:
            self.SIGNED_QMIN = -8 if is_q4 else -128
            self.SIGNED_QMAX = 7 if is_q4 else 127
            self.QMAX = float(self.SIGNED_QMAX)
        else:
            self.SIGNED_QMIN = None
            self.SIGNED_QMAX = None
            self.QMAX = 15.0 if is_q4 else 255.0
        self.inv_qmax = 1.0 / self.QMAX

        # Group parameters
        self.is_grouped = is_q4 or ((use_hadamard or use_shuffle) and group_size < last_dim)
        if not self.is_grouped and not is_q4:
            self.use_hadamard = False
            self.use_shuffle = False
        self.group_size = group_size if self.is_grouped else last_dim
        self.num_groups = last_dim // self.group_size if self.is_grouped else 1

        # Rotary transform buffers
        if is_rotary:
            sqrt2 = 2.0 ** 0.5
            inv_sqrt2 = 1.0 / sqrt2
            self.rot_cos = inv_sqrt2
            fwd_sin = torch.cat([torch.full((last_dim // 2,), -inv_sqrt2), torch.full((last_dim // 2,), inv_sqrt2)])
            self.rot_sin = fwd_sin

        # Hadamard transform buffers
        if self.use_hadamard:
            self.hadamard_size = _next_power_of_two(self.group_size)
            self.hadamard_pad = self.hadamard_size - self.group_size
            self.hadamard_inv_sqrt = self.hadamard_size ** -0.5
            sign_gen = torch.Generator()
            sign_gen.manual_seed(LORA_HADAMARD_SEED)
            hadamard_sign = torch.randint(0, 2, (self.group_size,), generator=sign_gen, dtype=torch.int64)
            self.hadamard_sign = hadamard_sign.float().mul_(2.0).sub_(1.0)
            self._hadamard_levels = []
            w = self.hadamard_size
            while w > 1:
                h = w // 2
                self._hadamard_levels.append((w, h))
                w = h

        # Shuffle permutation buffers
        if self.use_shuffle:
            perm = torch.arange(last_dim).view(self.num_groups, self.group_size).T.contiguous().view(-1)
            inv_perm = torch.empty_like(perm)
            inv_perm[perm] = torch.arange(last_dim)
            self.shuffle_idx = perm
            self.unshuffle_idx = inv_perm

    # ── Hadamard transform ───────────────────────────────────────────────
    def _apply_hadamard(self, x, inverse=False):
        if not self.use_hadamard:
            return x
        if not inverse:
            x = x * self.hadamard_sign
        if self.hadamard_pad:
            x = F.pad(x, (0, self.hadamard_pad))
        for width, half in self._hadamard_levels:
            x = x.view(*x.shape[:-1], -1, width)
            even, odd = torch.split(x, [half, half], dim=-1)
            x = torch.cat([even + odd, even - odd], dim=-1)
            x = x.view(*x.shape[:-2], -1)
        x = x * self.hadamard_inv_sqrt
        if self.hadamard_pad:
            x = x[..., :self.group_size]
        if inverse:
            x = x * self.hadamard_sign
        return x

    # ── Rotary transform ─────────────────────────────────────────────────
    def _rotate_forward(self, x):
        """Apply π/4 rotation to spread outlier energy across dimension pairs."""
        x1 = x[..., :self.last_dim_half]
        x2 = x[..., self.last_dim_half:]
        rot_x = torch.cat([
            x1 * self.rot_cos + x2 * (-1.0 / (2.0 ** 0.5)),
            x1 * (1.0 / (2.0 ** 0.5)) + x2 * self.rot_cos
        ], dim=-1)
        return rot_x

    def _rotate_inverse(self, x):
        """Inverse π/4 rotation."""
        x1 = x[..., :self.last_dim_half]
        x2 = x[..., self.last_dim_half:]
        rot_x = torch.cat([
            x1 * self.rot_cos + x2 * (1.0 / (2.0 ** 0.5)),
            x1 * (-1.0 / (2.0 ** 0.5)) + x2 * self.rot_cos
        ], dim=-1)
        return rot_x

    # ── Clipping ─────────────────────────────────────────────────────────
    def _clip_to_sigma(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = (x - mean).square().mean(dim=-1, keepdim=True)
        std = var.sqrt()
        bound = self.clip_sigma * std
        return x.clamp(mean - bound, mean + bound)

    # ── Quantization ─────────────────────────────────────────────────────
    def quantize(self, tensor):
        """Quantize a float32 tensor along the last dimension.

        Returns dict with keys: 'data', 'scale', optionally 'bias'.
        """
        x = tensor.float()

        # 1. Rotary transform
        if self.is_rotary:
            x = self._rotate_forward(x)

        # 2. Shuffle
        if self.use_shuffle:
            x = x.index_select(-1, self.shuffle_idx)

        # 3. Hadamard within groups
        if self.use_hadamard:
            orig_shape = x.shape
            x = x.view(*orig_shape[:-1], self.num_groups, self.group_size)
            x = self._apply_hadamard(x)
            x = x.view(*orig_shape)

        # 4. Group reshape for quantization
        orig_shape = x.shape
        x_grouped = x.view(*orig_shape[:-1], self.num_groups, self.group_size)

        # 5. Clip
        if self.use_clip:
            x_grouped = self._clip_to_sigma(x_grouped)

        # 6. Quantize
        if self.use_sym:
            absmax = x_grouped.abs().amax(dim=-1, keepdim=True)
            scale = absmax / self.QMAX
            scale = scale.clamp(min=1e-10)
            x_quant = torch.round(x_grouped / scale).clamp(self.SIGNED_QMIN, self.SIGNED_QMAX).to(torch.int32)
            if self.is_q4:
                x_stored = torch.remainder(x_quant, 16).to(torch.uint8)
            elif self.is_cuda:
                x_stored = torch.remainder(x_quant, 256).to(torch.uint8)
            else:
                x_stored = x_quant.to(torch.int8)
            scale = scale.squeeze(-1)
            if LORA_USE_FLOAT16_SCALE:
                scale = scale.half()
            result = {'data': x_stored, 'scale': scale}
        else:
            block_min, block_max = torch.aminmax(x_grouped, dim=-1, keepdim=True)
            scale = (block_max - block_min) / self.QMAX
            scale = scale.clamp(min=1e-10)
            x_normalized = (x_grouped - block_min) / scale
            x_packed = torch.round(x_normalized)
            # Residual bias correction
            if self.use_residual_bias_correction:
                block_residual = x_grouped - (x_packed * scale + block_min)
                block_min = block_min + block_residual.mean(dim=-1, keepdim=True)
            if self.is_q4:
                x_stored = x_packed.to(torch.uint8)
            elif self.is_cuda:
                x_stored = x_packed.to(torch.uint8)
            else:
                x_stored = x_packed.to(torch.uint8)
            scale = scale.squeeze(-1)
            block_min = block_min.squeeze(-1)
            if LORA_USE_FLOAT16_SCALE:
                scale = scale.half()
                block_min = block_min.half()
            result = {'data': x_stored, 'scale': scale, 'bias': block_min}

        # 7. Q4 packing (2 nibbles -> 1 byte)
        if self.is_q4:
            data = result['data']
            data = data.view(*orig_shape[:-1], self.num_groups, self.group_size // 2, 2)
            low, high = data.unbind(-1)
            result['data'] = (low + high * 16).to(torch.uint8)

        # 8. CUDA packing (4 uint8 -> 1 int32)
        if self.is_cuda:
            data = result['data']
            packed_dim = self.group_size // 4 if not self.is_q4 else (self.group_size // 2) // 4
            data = data.view(*orig_shape[:-1], self.num_groups, packed_dim, 4)
            x0, x1, x2, x3 = data.unbind(-1)
            x0, x1, x2, x3 = x0.to(torch.int32), x1.to(torch.int32), x2.to(torch.int32), x3.to(torch.int32)
            result['data'] = (x0 + x1 * 256 + x2 * 65536 + (x3 - 128) * 16777216).to(torch.int32)

        return result

    # ── Dequantization ───────────────────────────────────────────────────
    def dequantize(self, quant_result):
        """Dequantize back to float32 tensor with original shape."""
        data = quant_result['data']
        scale = quant_result['scale']
        bias = quant_result.get('bias', None)

        if LORA_USE_FLOAT16_SCALE:
            scale = scale.float()
            if bias is not None:
                bias = bias.float()

        # 1. CUDA unpack (int32 -> 4 uint8)
        if self.is_cuda:
            x_i32 = data
            r3 = x_i32 % 16777216
            x3 = (x_i32 - r3) // 16777216 + 128
            x2 = r3 // 65536
            r2 = r3 % 65536
            x1 = r2 // 256
            x0 = r2 % 256
            data = torch.stack([x0, x1, x2, x3], dim=-1)
            if self.is_q4:
                packed_dim = (self.group_size // 2) // 4
                data = data.reshape(*data.shape[:-2], self.num_groups, self.group_size // 2)
            else:
                data = data.reshape(*data.shape[:-2], self.num_groups, self.group_size)

        # 2. Q4 unpack (1 byte -> 2 nibbles)
        if self.is_q4 and not self.is_cuda:
            # data shape: [..., num_groups, group_size//2]
            low = data % 16
            high = data // 16
            data = torch.stack([low, high], dim=-1).reshape(*data.shape[:-1], self.group_size)

        if self.is_q4 and self.is_cuda:
            low = data % 16
            high = data // 16
            data = torch.stack([low, high], dim=-1).reshape(*data.shape[:-1], self.group_size)

        # 3. Dequantize values
        if self.use_sym:
            if self.is_q4:
                # Decode signed Q4: stored as uint8 mod 16, recover [-8, 7]
                x_float = (torch.remainder(data.to(torch.int16) + 8, 16) - 8).float()
            elif self.is_cuda:
                # Decode signed Q8 from uint8: stored mod 256, recover [-128, 127]
                x_float = (torch.remainder(data.to(torch.int16) + 128, 256) - 128).float()
            else:
                # int8 directly
                x_float = data.float()
            x = x_float * scale.unsqueeze(-1)
        else:
            x_float = data.float()
            x = x_float * scale.unsqueeze(-1) + bias.unsqueeze(-1)

        # 4. Reshape back from groups
        orig_last_dim = self.num_groups * self.group_size
        x = x.reshape(*x.shape[:-2], orig_last_dim)

        # 5. Inverse Hadamard
        if self.use_hadamard:
            x = x.view(*x.shape[:-1], self.num_groups, self.group_size)
            x = self._apply_hadamard(x, inverse=True)
            x = x.view(*x.shape[:-2], orig_last_dim)

        # 6. Inverse shuffle
        if self.use_shuffle:
            x = x.index_select(-1, self.unshuffle_idx)

        # 7. Inverse rotary
        if self.is_rotary:
            x = self._rotate_inverse(x)

        return x


def quantize_lora_tensors(task_lora_dict):
    """Quantize all 8 LoRA tensor stacks according to LORA_QUANT_DTYPE settings.

    Uses the full KV cache quantization algorithm from Qwen_Export.py:
    rotary, hadamard, shuffle, clip, residual bias correction, Q4/Q8/CUDA packing.

    Args:
        task_lora_dict: dict of {name: np.ndarray} with 8 stacked LoRA tensors

    Returns:
        If F32/F16: same dict (possibly cast to float16)
        If quantized: dict with keys name + '_data', name + '_scale', [name + '_bias']
    """
    if LORA_QUANT_DTYPE not in SUPPORTED_LORA_QUANT_DTYPES:
        raise ValueError(f"Unsupported LORA_QUANT_DTYPE: {LORA_QUANT_DTYPE}")

    if LORA_QUANT_DTYPE == "F32":
        return task_lora_dict

    if LORA_QUANT_DTYPE == "F16":
        return {k: v.astype(np.float16) for k, v in task_lora_dict.items()}

    is_q4 = LORA_QUANT_DTYPE in ("ROTARY_Q4", "ROTARY_Q4_CUDA")
    is_rotary = LORA_QUANT_DTYPE in ("ROTARY_Q4", "ROTARY_Q4_CUDA", "ROTARY_Q8", "ROTARY_Q8_CUDA")
    is_cuda = LORA_QUANT_DTYPE in ("Q8_CUDA", "ROTARY_Q8_CUDA", "ROTARY_Q4_CUDA")

    result = {}
    for name, arr in task_lora_dict.items():
        tensor = torch.from_numpy(arr).float()
        last_dim = tensor.shape[-1]

        quantizer = LoRAQuantizer(
            last_dim=last_dim,
            group_size=LORA_QUANT_GROUP_SIZE,
            is_q4=is_q4,
            is_rotary=is_rotary,
            is_cuda=is_cuda,
            use_sym=LORA_USE_SYM,
            use_hadamard=LORA_USE_HADAMARD,
            use_clip=LORA_USE_CLIP,
            clip_sigma=LORA_CLIP_SIGMA,
            use_shuffle=LORA_USE_SHUFFLE,
        )

        quant = quantizer.quantize(tensor)
        result[name + '_data'] = quant['data'].numpy()
        result[name + '_scale'] = quant['scale'].numpy()
        if 'bias' in quant:
            result[name + '_bias'] = quant['bias'].numpy()

    total_bytes = sum(v.nbytes for v in result.values())
    print(f"  Quantized LoRA ({LORA_QUANT_DTYPE}, group={LORA_QUANT_GROUP_SIZE}, sym={LORA_USE_SYM}, "
          f"hadamard={LORA_USE_HADAMARD}, shuffle={LORA_USE_SHUFFLE}): {total_bytes / 1024 / 1024:.2f} MB")
    return result


def dequantize_lora_numpy(quant_dict):
    """Dequantize a quantized LoRA dict back to float32 numpy arrays.

    Args:
        quant_dict: dict from quantize_lora_tensors (with _data/_scale/_bias keys)

    Returns:
        dict of {name: np.ndarray float32} with original 8 LoRA tensor names
    """
    if LORA_QUANT_DTYPE in ("F32", "F16"):
        if LORA_QUANT_DTYPE == "F16":
            return {k: v.astype(np.float32) for k, v in quant_dict.items()}
        return quant_dict

    is_q4 = LORA_QUANT_DTYPE in ("ROTARY_Q4", "ROTARY_Q4_CUDA")
    is_rotary = LORA_QUANT_DTYPE in ("ROTARY_Q4", "ROTARY_Q4_CUDA", "ROTARY_Q8", "ROTARY_Q8_CUDA")
    is_cuda = LORA_QUANT_DTYPE in ("Q8_CUDA", "ROTARY_Q8_CUDA", "ROTARY_Q4_CUDA")

    result = {}
    for name in LORA_INPUT_NAMES:
        data = torch.from_numpy(quant_dict[name + '_data'])
        scale = torch.from_numpy(quant_dict[name + '_scale'])
        bias_arr = quant_dict.get(name + '_bias', None)
        bias = torch.from_numpy(bias_arr) if bias_arr is not None else None

        # Infer last_dim from scale shape and group structure
        # scale shape: [..., num_groups] or [..., num_groups] for grouped
        num_groups = scale.shape[-1]
        last_dim = num_groups * LORA_QUANT_GROUP_SIZE

        quantizer = LoRAQuantizer(
            last_dim=last_dim,
            group_size=LORA_QUANT_GROUP_SIZE,
            is_q4=is_q4,
            is_rotary=is_rotary,
            is_cuda=is_cuda,
            use_sym=LORA_USE_SYM,
            use_hadamard=LORA_USE_HADAMARD,
            use_clip=LORA_USE_CLIP,
            clip_sigma=LORA_CLIP_SIGMA,
            use_shuffle=LORA_USE_SHUFFLE,
        )

        quant_result = {'data': data, 'scale': scale}
        if bias is not None:
            quant_result['bias'] = bias

        dequant = quantizer.dequantize(quant_result)
        result[name] = dequant.numpy()

    return result


# ══════════════════════════════════════════════════════════════════════════════
# MODEL CLASSES
# ══════════════════════════════════════════════════════════════════════════════

def _parse_quant_dtype(dtype_str):
    """Return (is_quantized, is_f16, is_q4, is_rotary, is_cuda) from a quant dtype string."""
    quantized_set = ("Q8", "Q8_CUDA", "ROTARY_Q8", "ROTARY_Q8_CUDA", "ROTARY_Q4", "ROTARY_Q4_CUDA")
    is_quantized = dtype_str in quantized_set
    is_f16 = dtype_str == "F16"
    is_q4 = dtype_str in ("ROTARY_Q4", "ROTARY_Q4_CUDA")
    is_rotary = dtype_str in ("ROTARY_Q4", "ROTARY_Q4_CUDA", "ROTARY_Q8", "ROTARY_Q8_CUDA")
    is_cuda = dtype_str in ("Q8_CUDA", "ROTARY_Q8_CUDA", "ROTARY_Q4_CUDA")
    return is_quantized, is_f16, is_q4, is_rotary, is_cuda


def _make_quantizer(dtype_str, last_dim):
    """Create a LoRAQuantizer configured for the given dtype and last_dim."""
    _, _, is_q4, is_rotary, is_cuda = _parse_quant_dtype(dtype_str)
    return LoRAQuantizer(
        last_dim=last_dim, group_size=LORA_QUANT_GROUP_SIZE,
        is_q4=is_q4, is_rotary=is_rotary, is_cuda=is_cuda,
        use_sym=LORA_USE_SYM, use_hadamard=LORA_USE_HADAMARD,
        use_clip=LORA_USE_CLIP, clip_sigma=LORA_CLIP_SIGMA, use_shuffle=LORA_USE_SHUFFLE,
    )


class JINA_EMBED_LORA_FUSED(torch.nn.Module):
    """
    Combined token embedding + fused LoRA adapter in a single ONNX-exportable module.

    Embedding quantization controlled by EMBED_QUANT_DTYPE.
    LoRA quantization controlled by LORA_QUANT_DTYPE.
    Each can be independently set to F32, F16, or any quantized mode.

    Inputs:
        input_ids:  [batch, seq_len] int32 token IDs
        task_index: [1] int32 scalar — index of the target task

    Outputs:
        hidden_states: [batch, seq_len, hidden_size] float32 token embeddings
        + 8 LoRA tensors for the selected task, each [num_layers, ...]
    """

    def __init__(self, qwen_model, all_task_lora_list):
        super().__init__()
        self.embed_is_quantized, self.embed_is_f16, _, _, _ = _parse_quant_dtype(EMBED_QUANT_DTYPE)
        self.lora_is_quantized, self.lora_is_f16, _, _, _ = _parse_quant_dtype(LORA_QUANT_DTYPE)

        # ── Embedding weights ────────────────────────────────────────────
        embed_weight = qwen_model.embed_tokens.weight.float()
        if self.embed_is_quantized:
            embed_last_dim = embed_weight.shape[-1]
            self._embed_quantizer = _make_quantizer(EMBED_QUANT_DTYPE, embed_last_dim)
            quant = self._embed_quantizer.quantize(embed_weight)
            self.register_buffer('embed_data', quant['data'].contiguous())
            self.register_buffer('embed_scale', quant['scale'].contiguous())
            if 'bias' in quant:
                self.register_buffer('embed_bias', quant['bias'].contiguous())
        elif self.embed_is_f16:
            self.register_buffer('embed_weight', embed_weight.half().contiguous())
        else:
            self.embed_tokens = qwen_model.embed_tokens.float()

        # ── LoRA weights ─────────────────────────────────────────────────
        if self.lora_is_quantized:
            self._lora_quantizers = {}
            for name in LORA_INPUT_NAMES:
                fused = torch.from_numpy(
                    np.stack([task_lora[name] for task_lora in all_task_lora_list], axis=0)
                ).float()
                last_dim = fused.shape[-1]
                q = _make_quantizer(LORA_QUANT_DTYPE, last_dim)
                self._lora_quantizers[name] = q
                quant = q.quantize(fused)
                self.register_buffer(name + '_data', quant['data'].contiguous())
                self.register_buffer(name + '_scale', quant['scale'].contiguous())
                if 'bias' in quant:
                    self.register_buffer(name + '_bias', quant['bias'].contiguous())
        elif self.lora_is_f16:
            for name in LORA_INPUT_NAMES:
                fused = torch.from_numpy(
                    np.stack([task_lora[name] for task_lora in all_task_lora_list], axis=0)
                ).half().contiguous()
                self.register_buffer(name, fused)
        else:
            for name in LORA_INPUT_NAMES:
                fused = torch.from_numpy(
                    np.stack([task_lora[name] for task_lora in all_task_lora_list], axis=0)
                ).float().contiguous()
                self.register_buffer(name, fused)

    def _dequant_embed(self, indices):
        """Gather quantized embedding rows for needed tokens, then dequantize."""
        quant_result = {'data': self.embed_data[indices], 'scale': self.embed_scale[indices]}
        if hasattr(self, 'embed_bias'):
            quant_result['bias'] = self.embed_bias[indices]
        return self._embed_quantizer.dequantize(quant_result)

    def _dequant_lora(self, name, task_index):
        """Gather quantized LoRA data for the needed task, then dequantize."""
        quantizer = self._lora_quantizers[name]
        quant_result = {
            'data': getattr(self, name + '_data')[task_index],
            'scale': getattr(self, name + '_scale')[task_index],
        }
        if not LORA_USE_SYM:
            bias = getattr(self, name + '_bias', None)
            if bias is not None:
                quant_result['bias'] = bias[task_index]
        return quantizer.dequantize(quant_result).squeeze(0)

    def forward(self, input_ids, task_index):
        # ── Embedding lookup ─────────────────────────────────────────────
        if self.embed_is_quantized:
            hidden_states = self._dequant_embed(input_ids)
        elif self.embed_is_f16:
            hidden_states = F.embedding(input_ids, self.embed_weight.float())
        else:
            hidden_states = self.embed_tokens(input_ids)

        # ── LoRA tensor selection ────────────────────────────────────────
        if self.lora_is_quantized:
            lora_outputs = tuple(
                self._dequant_lora(name, task_index)
                for name in LORA_INPUT_NAMES
            )
        elif self.lora_is_f16:
            lora_outputs = tuple(
                getattr(self, name)[task_index].squeeze(0).float()
                for name in LORA_INPUT_NAMES
            )
        else:
            lora_outputs = tuple(
                getattr(self, name)[task_index].squeeze(0)
                for name in LORA_INPUT_NAMES
            )

        return (hidden_states,) + lora_outputs


class JINA_EMBED(torch.nn.Module):
    """Token embedding only (split mode). Outputs hidden_states in float32.

    Quantization controlled by EMBED_QUANT_DTYPE independently from LORA_QUANT_DTYPE.
    """

    def __init__(self, qwen_model):
        super().__init__()
        self.is_quantized, self.is_f16, _, _, _ = _parse_quant_dtype(EMBED_QUANT_DTYPE)

        embed_weight = qwen_model.embed_tokens.weight.float()
        if self.is_quantized:
            embed_last_dim = embed_weight.shape[-1]
            self._embed_quantizer = _make_quantizer(EMBED_QUANT_DTYPE, embed_last_dim)
            quant = self._embed_quantizer.quantize(embed_weight)
            self.register_buffer('embed_data', quant['data'].contiguous())
            self.register_buffer('embed_scale', quant['scale'].contiguous())
            if 'bias' in quant:
                self.register_buffer('embed_bias', quant['bias'].contiguous())
        elif self.is_f16:
            self.register_buffer('embed_weight', embed_weight.half().contiguous())
        else:
            self.embed_tokens = qwen_model.embed_tokens.float()

    def forward(self, input_ids):
        if self.is_quantized:
            quant_result = {'data': self.embed_data[input_ids], 'scale': self.embed_scale[input_ids]}
            if hasattr(self, 'embed_bias'):
                quant_result['bias'] = self.embed_bias[input_ids]
            return self._embed_quantizer.dequantize(quant_result)
        elif self.is_f16:
            return F.embedding(input_ids, self.embed_weight.float())
        else:
            return self.embed_tokens(input_ids)


class JINA_LORA_TASK(torch.nn.Module):
    """Single-task LoRA weight provider (split mode).

    Stores the 8 stacked LoRA tensors for one task as constant buffers.
    No inputs required — outputs are fixed for the task.

    If LORA_QUANT_DTYPE is quantized (Q8/Q8_CUDA/ROTARY_Q8/ROTARY_Q8_CUDA/ROTARY_Q4/ROTARY_Q4_CUDA),
    stores quantized data + scale (+ bias) and dequantizes on forward using the full
    KV-style algorithm (rotary, hadamard, shuffle). If F16/F32, stores directly.
    """

    def __init__(self, task_lora_dict):
        super().__init__()
        self.is_quantized = LORA_QUANT_DTYPE in (
            "Q8", "Q8_CUDA", "ROTARY_Q8", "ROTARY_Q8_CUDA", "ROTARY_Q4", "ROTARY_Q4_CUDA"
        )

        if self.is_quantized:
            quant = quantize_lora_tensors(task_lora_dict)
            for key, arr in quant.items():
                self.register_buffer(key, torch.from_numpy(arr).contiguous())
            # Pre-create quantizers (avoid torch.Generator in forward during ONNX trace)
            self._quantizers = {}
            for name in LORA_INPUT_NAMES:
                last_dim = task_lora_dict[name].shape[-1]
                self._quantizers[name] = _make_quantizer(LORA_QUANT_DTYPE, last_dim)
        elif LORA_QUANT_DTYPE == "F16":
            for name in LORA_INPUT_NAMES:
                self.register_buffer(name, torch.from_numpy(task_lora_dict[name]).half().contiguous())
        else:
            for name in LORA_INPUT_NAMES:
                self.register_buffer(name, torch.from_numpy(task_lora_dict[name]).float().contiguous())

    def forward(self):
        if self.is_quantized:
            outputs = []
            for name in LORA_INPUT_NAMES:
                data = getattr(self, name + '_data')
                scale = getattr(self, name + '_scale')
                bias = getattr(self, name + '_bias', None) if not LORA_USE_SYM else None
                quant_result = {'data': data, 'scale': scale}
                if bias is not None:
                    quant_result['bias'] = bias
                dequant = self._quantizers[name].dequantize(quant_result)
                outputs.append(dequant)
            return tuple(outputs)
        else:
            return tuple(getattr(self, name).float() for name in LORA_INPUT_NAMES)


class JINA_MAIN(torch.nn.Module):
    """
    Shared transformer backbone: rotary embeddings + attention layers + pooling.

    LoRA tensors are passed as INPUTS (not baked-in constants), allowing a single
    exported ONNX to serve all tasks by swapping the LoRA input tensors.
    """

    def __init__(self, qwen_model, num_heads, num_key_value_heads, head_dim, num_layers, hidden_size, max_seq_len):
        super().__init__()
        self.model = qwen_model

        # ── Attention geometry ───────────────────────────────────────────
        self.head_dim = head_dim
        self.head_dim_half = head_dim // 2
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_key_value_groups = num_heads // num_key_value_heads
        self.qk_heads = num_heads + num_key_value_heads
        self.total_qkv_heads = self.qk_heads + num_key_value_heads
        self.qkv_split_sizes = [self.qk_heads, num_key_value_heads]
        self.qk_split_sizes = [num_heads, num_key_value_heads]
        self.num_layers = num_layers

        # ── Pre-computed attention mask (upper triangle -> -128) ───────────
        self.register_buffer('attention_mask_template', (1 - torch.tril(torch.ones(1, 1, 1, max_seq_len, max_seq_len, dtype=torch.int8))) * -128, persistent=False)

        # ── Pre-computed rotary tables ───────────────────────────────────
        position_ids = torch.arange(max_seq_len, dtype=torch.float32).unsqueeze(-1)
        inv_freq = qwen_model.rotary_emb.inv_freq.float()
        idx_theta = (position_ids * inv_freq).unsqueeze(1).unsqueeze(1).unsqueeze(0)
        cos = torch.cos(idx_theta)
        sin = torch.sin(idx_theta)
        self.register_buffer('cos_rotary_pos_emb', torch.cat([cos, cos], dim=-1).half(), persistent=False)
        self.register_buffer('sin_rotary_pos_emb', torch.cat([-sin, sin], dim=-1).half(), persistent=False)

        # ── RMSNorm constants ────────────────────────────────────────────
        self.overflow_scale = torch.tensor([0.01], dtype=torch.float32)
        hidden_rms_norm = self.model.layers[0].input_layernorm
        qk_rms_norm = self.model.layers[0].self_attn.q_norm
        hidden_rms_norm_eps = float(getattr(hidden_rms_norm, 'variance_epsilon', getattr(hidden_rms_norm, 'eps', 1e-6)))
        qk_rms_norm_eps = float(getattr(qk_rms_norm, 'variance_epsilon', getattr(qk_rms_norm, 'eps', hidden_rms_norm_eps)))
        hidden_rms_norm_eps = hidden_size * hidden_rms_norm_eps
        qk_rms_norm_eps = self.head_dim * qk_rms_norm_eps
        if PREVENT_F16_OVERFLOW:
            hidden_rms_norm_eps *= self.overflow_scale.square()
            qk_rms_norm_eps *= self.overflow_scale.square()
        self.register_buffer('hidden_rms_norm_eps', torch.tensor([hidden_rms_norm_eps], dtype=torch.float32))
        self.register_buffer('qk_rms_norm_eps', torch.tensor([qk_rms_norm_eps], dtype=torch.float32))
        self.register_buffer('final_norm_weight', self.model.norm.weight.view(1, 1, -1) * (hidden_size ** 0.5))

        # ── Fuse base weights ────────────────────────────────────────────
        self._fuse_weights(hidden_size)
        self.o_proj_in_features = self.model.layers[0].self_attn.o_proj.in_features
        self.mlp_split = [self.model.layers[0].mlp.down_proj.in_features] * 2

    # ══════════════════════════════════════════════════════════════════════
    # Weight Fusion (runs once at init)
    # ══════════════════════════════════════════════════════════════════════
    def _fuse_weights(self, hidden_size):
        scale_factor = self.head_dim ** -0.25
        norm_factor = hidden_size ** 0.5
        norm_factor_qk = self.head_dim ** 0.5

        with torch.no_grad():
            for layer in self.model.layers:
                self._fuse_qkv_projection(layer, scale_factor, norm_factor, norm_factor_qk)
                self._fuse_gate_up_projection(layer, norm_factor)
            del self.model.norm

    def _fuse_qkv_projection(self, layer, scale_factor, norm_factor, norm_factor_qk):
        attn = layer.self_attn
        q_proj = attn.q_proj
        k_proj = attn.k_proj
        v_proj = attn.v_proj

        in_features = int(q_proj.in_features)
        out_features = int(q_proj.out_features + k_proj.out_features + v_proj.out_features)
        has_bias = any(proj.bias is not None for proj in (q_proj, k_proj, v_proj))

        qkv = torch.nn.Linear(in_features, out_features, bias=has_bias)
        qkv.weight.copy_(torch.cat([q_proj.weight, k_proj.weight, v_proj.weight], dim=0))

        if has_bias:
            def _get_bias(proj):
                if proj.bias is None:
                    return torch.zeros(proj.out_features, dtype=qkv.weight.dtype)
                return proj.bias
            qkv.bias.copy_(torch.cat([_get_bias(q_proj), _get_bias(k_proj), _get_bias(v_proj)], dim=0))

        combined_scale = scale_factor * norm_factor_qk
        attn.q_norm.weight.mul_(combined_scale)
        attn.k_norm.weight.mul_(combined_scale)
        q_norm_repeated = attn.q_norm.weight.repeat(self.num_heads)
        k_norm_repeated = attn.k_norm.weight.repeat(self.num_key_value_heads)
        attn.qk_norm_weight = torch.nn.Parameter(
            torch.cat([q_norm_repeated, k_norm_repeated], dim=0).view(1, 1, 1, -1, self.head_dim),
            requires_grad=False,
        )

        input_norm_weight = layer.input_layernorm.weight.unsqueeze(0) * norm_factor
        qkv.weight.mul_(input_norm_weight)

        attn.qkv = qkv
        del attn.q_proj, attn.k_proj, attn.v_proj
        del attn.q_norm, attn.k_norm
        del layer.input_layernorm

    def _fuse_gate_up_projection(self, layer, norm_factor):
        post_norm_weight = layer.post_attention_layernorm.weight.unsqueeze(0) * norm_factor
        gate = layer.mlp.gate_proj
        up = layer.mlp.up_proj

        gate_up = torch.nn.Linear(gate.in_features, gate.out_features + up.out_features, bias=False)
        gate_up.weight.copy_(torch.cat([
            gate.weight * post_norm_weight,
            up.weight * post_norm_weight,
        ], dim=0))

        layer.mlp.gate_up_proj = gate_up
        del layer.mlp.gate_proj, layer.mlp.up_proj, layer.post_attention_layernorm

    # ══════════════════════════════════════════════════════════════════════
    # Utility Methods
    # ══════════════════════════════════════════════════════════════════════
    def _rms_norm(self, x, eps):
        if PREVENT_F16_OVERFLOW:
            x = x * self.overflow_scale
        return x * torch.rsqrt(x.square().sum(-1, keepdim=True) + eps)

    def _rotate_half(self, x, batch_size):
        x = x.view(batch_size, -1, 1, self.qk_heads, 2, self.head_dim_half)
        x = x.flip(-2)
        return x.view(batch_size, -1, 1, self.qk_heads, self.head_dim)

    @staticmethod
    def _apply_LoRA(x, lora_a, lora_b):
        return F.linear(F.linear(x, lora_a), lora_b)

    def forward(
        self,
        hidden_states,
        lora_qkv_a,
        lora_qkv_b,
        lora_o_a,
        lora_o_b,
        lora_gate_up_a,
        lora_gate_up_b,
        lora_down_a,
        lora_down_b,
    ):
        batch_size = hidden_states.shape[0]
        seq_len = hidden_states.shape[1]

        # ── Rotary embeddings ────────────────────────────────────────────
        rotary_pos_emb_cos = self.cos_rotary_pos_emb[:, :seq_len].float()
        rotary_pos_emb_sin = self.sin_rotary_pos_emb[:, :seq_len].float()

        # ── Causal attention bias (pre-computed int8, slice + cast) ────────
        attention_bias = self.attention_mask_template[..., :seq_len, :seq_len].to(torch.float32)

        # ── Transformer layers ───────────────────────────────────────────
        for layer_idx, layer in enumerate(self.model.layers):
            residual = hidden_states
            hidden_states = self._rms_norm(hidden_states, self.hidden_rms_norm_eps)

            # Index into stacked LoRA tensors for this layer
            qkv_a = lora_qkv_a[layer_idx]
            qkv_b = lora_qkv_b[layer_idx]
            o_a = lora_o_a[layer_idx]
            o_b = lora_o_b[layer_idx]
            gate_up_a = lora_gate_up_a[layer_idx]
            gate_up_b = lora_gate_up_b[layer_idx]
            down_a = lora_down_a[layer_idx]
            down_b = lora_down_b[layer_idx]

            qkv = layer.self_attn.qkv(hidden_states) + self._apply_LoRA(hidden_states, qkv_a, qkv_b)
            qkv = qkv.reshape(batch_size, -1, 1, self.total_qkv_heads, self.head_dim)
            qk, v = torch.split(qkv, self.qkv_split_sizes, dim=-2)

            qk = self._rms_norm(qk, self.qk_rms_norm_eps) * layer.self_attn.qk_norm_weight
            qk_rot = qk * rotary_pos_emb_cos + self._rotate_half(qk, batch_size) * rotary_pos_emb_sin

            q, k = torch.split(qk_rot, self.qk_split_sizes, dim=-2)
            q = q.reshape(batch_size, -1, self.num_key_value_heads, self.num_key_value_groups, self.head_dim)
            q = q.permute(0, 2, 3, 1, 4)

            k = k.permute(0, 3, 2, 4, 1)
            v = v.transpose(1, 3)

            attn = torch.matmul(q, k) + attention_bias
            attn = torch.softmax(attn, dim=-1)
            attn = torch.matmul(attn, v)

            attn = attn.permute(0, 3, 1, 2, 4).reshape(batch_size, -1, self.o_proj_in_features)
            hidden_states = residual + layer.self_attn.o_proj(attn) + self._apply_LoRA(attn, o_a, o_b)

            residual = hidden_states
            hidden_states = self._rms_norm(hidden_states, self.hidden_rms_norm_eps)
            gate_up_delta = self._apply_LoRA(hidden_states, gate_up_a, gate_up_b)
            gate_up = layer.mlp.gate_up_proj(hidden_states) + gate_up_delta
            gate, up = torch.split(gate_up, self.mlp_split, dim=-1)
            mlp_hidden = layer.mlp.act_fn(gate) * up
            hidden_states = residual + layer.mlp.down_proj(mlp_hidden) + self._apply_LoRA(mlp_hidden, down_a, down_b)

        # ── Final RMSNorm ────────────────────────────────────────────────
        last_hidden_state = self._rms_norm(hidden_states, self.hidden_rms_norm_eps) * self.final_norm_weight

        # ── Last-token pooling + L2 normalize ────────────────────────────
        pooled = last_hidden_state[:, -1, :]
        embeddings = F.normalize(pooled, p=2, dim=-1)

        return embeddings, last_hidden_state


# ══════════════════════════════════════════════════════════════════════════════
# EXPORT
# ══════════════════════════════════════════════════════════════════════════════

if DO_EXPORT:
    print('Export start ...')
    with torch.inference_mode():

        # ══════════════════════════════════════════════════════════════════
        # Load Model & Extract Config
        # ══════════════════════════════════════════════════════════════════
        model_config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
        task_names = list(model_config.task_names)
        max_seq_len = resolve_max_seq_len(model_config)

        num_layers          = model_config.num_hidden_layers
        num_heads           = model_config.num_attention_heads
        num_key_value_heads = model_config.num_key_value_heads
        head_dim            = model_config.head_dim
        hidden_size         = model_config.hidden_size

        expected_model_config = load_expected_model_config(MODEL_PATH)
        expected_adapter_config = load_expected_adapter_config(MODEL_PATH, task_names)

        print(f'\nModel config (from {MODEL_PATH}):')
        for k, v in expected_model_config.items():
            print(f'  {k}: {v}')
        print(f'Adapter config: {expected_adapter_config}')
        print(f'\nFUSE_LORA_INTO_EMBED: {FUSE_LORA_INTO_EMBED}')
        print(f'EMBED_QUANT_DTYPE: {EMBED_QUANT_DTYPE}')
        print(f'LORA_QUANT_DTYPE: {LORA_QUANT_DTYPE}')

        if FUSE_LORA_INTO_EMBED:
            print(f'\nArchitecture: 1 fused Embed+LoRA + 1 shared Main ONNX (int32 task_index)')
        else:
            print(f'\nArchitecture: 1 Embed + {len(task_names)} task LoRA + 1 shared Main ONNX (split mode)')
        print(f'Tasks: {task_names}')
        print(f'Task index map: {dict(enumerate(task_names))}')
        print(f'Max sequence length: {max_seq_len}')

        # ══════════════════════════════════════════════════════════════════
        # Load Base Model & Build LoRA Tensors
        # ══════════════════════════════════════════════════════════════════
        print('\nLoading shared base Qwen3Model backbone ...')
        qwen_model = Qwen3Model.from_pretrained(
            MODEL_PATH,
            config=model_config,
            dtype=torch.float32,
            low_cpu_mem_usage=True,
        ).eval()

        validate_model_config(model_config, expected_model_config)
        layer_scales = collect_layer_LoRA_scales(qwen_model)

        task_lora = {}
        for task_name in task_names:
            task_lora[task_name] = build_task_LoRA_stacked(task_name, model_config, layer_scales, expected_adapter_config)

        all_task_lora_list = [task_lora[name] for name in task_names]

        # Validate quant settings against representative dims
        representative_last_dim = list(task_lora[task_names[0]].values())[0].shape[-1]
        for note in normalize_lora_quant_settings(representative_last_dim):
            print(f"\n{note}")
        if EMBED_QUANT_DTYPE not in SUPPORTED_LORA_QUANT_DTYPES:
            raise ValueError(f"Unsupported EMBED_QUANT_DTYPE: {EMBED_QUANT_DTYPE}")

        # ══════════════════════════════════════════════════════════════════
        # Build Modules & Export
        # ══════════════════════════════════════════════════════════════════
        dummy_batch_size = 2
        dummy_seq_len = min(16, max_seq_len)

        if FUSE_LORA_INTO_EMBED:
            # ── Fused Mode: Embed + all-task LoRA in one ONNX ────────────
            if EMBED_QUANT_DTYPE not in ("F32", "F16") or LORA_QUANT_DTYPE not in ("F32", "F16"):
                print(f'\nApplying quantization (fused mode):')
                print(f'  Embed: {EMBED_QUANT_DTYPE}  |  LoRA: {LORA_QUANT_DTYPE}')
                print(f'  Weights stored quantized in ONNX; dequantized in forward pass.')

            model_EmbedLoRA = JINA_EMBED_LORA_FUSED(qwen_model, all_task_lora_list).eval()

            model_Main = JINA_MAIN(
                qwen_model,
                num_heads=num_heads,
                num_key_value_heads=num_key_value_heads,
                head_dim=head_dim,
                num_layers=num_layers,
                hidden_size=hidden_size,
                max_seq_len=max_seq_len,
            ).eval()

            gc.collect()

            # ── Export: Fused Embed + LoRA ────────────────────────────────
            print('\nExport start [fused Embed + LoRA] ...')
            input_ids = torch.ones((dummy_batch_size, dummy_seq_len), dtype=torch.int32)
            task_index = torch.tensor([0], dtype=torch.int32)

            output_names_EmbedLoRA = ['hidden_states'] + LORA_INPUT_NAMES

            torch.onnx.export(
                model_EmbedLoRA,
                (input_ids, task_index),
                onnx_model_EmbedLoRA,
                input_names=['input_ids', 'task_index'],
                output_names=output_names_EmbedLoRA,
                dynamic_axes={
                    'input_ids': {0: 'batch', 1: 'seq_len'},
                    'hidden_states': {0: 'batch', 1: 'seq_len'},
                },
                opset_version=OPSET,
                dynamo=False,
            )
            del input_ids, task_index
            gc.collect()
            print('Export done [fused Embed + LoRA]!')

            # ── Export: Shared Main ───────────────────────────────────────
            print('\nExport start [shared Main with LoRA inputs] ...')
            hidden_states = torch.ones((dummy_batch_size, dummy_seq_len, hidden_size), dtype=torch.float32)

            first_task_lora = task_lora[task_names[0]]
            lora_tensors = tuple(
                torch.from_numpy(first_task_lora[name]).float()
                for name in LORA_INPUT_NAMES
            )

            all_inputs = (hidden_states,) + lora_tensors
            input_names_Main = ['hidden_states'] + LORA_INPUT_NAMES
            output_names_Main = ['embeddings', 'last_hidden_state']

            dynamic_axes = {
                'hidden_states': {0: 'batch', 1: 'seq_len'},
                'embeddings': {0: 'batch'},
                'last_hidden_state': {0: 'batch', 1: 'seq_len'},
            }

            torch.onnx.export(
                model_Main,
                all_inputs,
                onnx_model_Main,
                input_names=input_names_Main,
                output_names=output_names_Main,
                dynamic_axes=dynamic_axes,
                opset_version=OPSET,
                dynamo=False,
            )

            del model_Main, model_EmbedLoRA, hidden_states, lora_tensors, all_inputs
            gc.collect()
            print('Export done [shared Main]!')

            # ── Size Summary ─────────────────────────────────────────────
            embed_lora_size = os.path.getsize(onnx_model_EmbedLoRA)
            main_size = os.path.getsize(onnx_model_Main)
            total = embed_lora_size + main_size
            print(f'\n  {"=" * 56}')
            print(f'  SIZE SUMMARY (Fused Mode, EMBED={EMBED_QUANT_DTYPE}, LORA={LORA_QUANT_DTYPE})')
            print(f'  {"=" * 56}')
            print(f'  Embed+LoRA:     {embed_lora_size / 1024 / 1024:>8.1f} MB  ({len(task_names)} tasks)')
            print(f'  Main:           {main_size / 1024 / 1024:>8.1f} MB')
            print(f'  {"-" * 44}')
            print(f'  TOTAL:          {total / 1024 / 1024:>8.1f} MB')
            print()

        else:
            # ── Split Mode: Separate Embed + per-task LoRA ONNXs ─────────
            model_Embed = JINA_EMBED(qwen_model).eval()

            model_Main = JINA_MAIN(
                qwen_model,
                num_heads=num_heads,
                num_key_value_heads=num_key_value_heads,
                head_dim=head_dim,
                num_layers=num_layers,
                hidden_size=hidden_size,
                max_seq_len=max_seq_len,
            ).eval()

            gc.collect()

            # ── Export: Embed (token embedding only) ─────────────────────
            print('\nExport start [Embed only] ...')
            input_ids = torch.ones((dummy_batch_size, dummy_seq_len), dtype=torch.int32)

            torch.onnx.export(
                model_Embed,
                (input_ids,),
                onnx_model_Embed,
                input_names=['input_ids'],
                output_names=['hidden_states'],
                dynamic_axes={
                    'input_ids': {0: 'batch', 1: 'seq_len'},
                    'hidden_states': {0: 'batch', 1: 'seq_len'},
                },
                opset_version=OPSET,
                dynamo=False,
            )
            del input_ids
            gc.collect()
            print('Export done [Embed only]!')

            # ── Export: Per-task LoRA models ──────────────────────────────
            for task_name in task_names:
                print(f'\nExport start [LoRA task: {task_name}] ...')
                lora_model = JINA_LORA_TASK(task_lora[task_name]).eval()
                lora_onnx_path = get_lora_onnx_path(task_name)

                torch.onnx.export(
                    lora_model,
                    (),
                    lora_onnx_path,
                    input_names=[],
                    output_names=LORA_INPUT_NAMES,
                    dynamic_axes={},
                    opset_version=OPSET,
                    dynamo=False,
                )
                del lora_model
                gc.collect()
                print(f'Export done [LoRA task: {task_name}]!')

            # ── Export: Shared Main ───────────────────────────────────────
            print('\nExport start [shared Main with LoRA inputs] ...')
            hidden_states = torch.ones((dummy_batch_size, dummy_seq_len, hidden_size), dtype=torch.float32)

            first_task_lora = task_lora[task_names[0]]
            lora_tensors = tuple(
                torch.from_numpy(first_task_lora[name]).float()
                for name in LORA_INPUT_NAMES
            )

            all_inputs = (hidden_states,) + lora_tensors
            input_names_Main = ['hidden_states'] + LORA_INPUT_NAMES
            output_names_Main = ['embeddings', 'last_hidden_state']

            dynamic_axes = {
                'hidden_states': {0: 'batch', 1: 'seq_len'},
                'embeddings': {0: 'batch'},
                'last_hidden_state': {0: 'batch', 1: 'seq_len'},
            }

            torch.onnx.export(
                model_Main,
                all_inputs,
                onnx_model_Main,
                input_names=input_names_Main,
                output_names=output_names_Main,
                dynamic_axes=dynamic_axes,
                opset_version=OPSET,
                dynamo=False,
            )

            del model_Main, model_Embed, hidden_states, lora_tensors, all_inputs
            gc.collect()
            print('Export done [shared Main]!')

            # ── Size Summary ─────────────────────────────────────────────
            embed_size = os.path.getsize(onnx_model_Embed)
            main_size = os.path.getsize(onnx_model_Main)
            total = embed_size + main_size
            print(f'\n  {"=" * 56}')
            print(f'  SIZE SUMMARY (Split Mode, EMBED={EMBED_QUANT_DTYPE}, LORA={LORA_QUANT_DTYPE})')
            print(f'  {"=" * 56}')
            print(f'  Embed:          {embed_size / 1024 / 1024:>8.1f} MB')
            for task_name in task_names:
                lora_path = get_lora_onnx_path(task_name)
                lora_size = os.path.getsize(lora_path)
                total += lora_size
                print(f'  LoRA [{task_name:16s}]: {lora_size / 1024 / 1024:>8.1f} MB')
            print(f'  Main:           {main_size / 1024 / 1024:>8.1f} MB')
            print(f'  {"-" * 44}')
            print(f'  TOTAL:          {total / 1024 / 1024:>8.1f} MB')
            print()

    print(
        '\nExport done!\n\n'
        'Start running the Jina Embedding by ONNXRuntime.\n'
        'Now loading . . . it could cost minutes.'
    )
else:
    model_config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
    task_names = list(model_config.task_names)
    max_seq_len = resolve_max_seq_len(model_config)
    hidden_size = model_config.hidden_size


# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS (ORT)
# ══════════════════════════════════════════════════════════════════════════════

def bind_ort_out_buf(binding, names, values):
    for name, val in zip(names, values):
        binding.bind_ortvalue_output(name, val)


def create_ort_with_numpy(array, device, device_id):
    return onnxruntime.OrtValue.ortvalue_from_numpy(np.ascontiguousarray(array), device, device_id)


def create_ort_with_shape(shape, dtype, device, device_id):
    return onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros(shape, dtype=dtype), device, device_id)


def create_session(model_path, _session_opts, _providers, _provider_options, _disabled_optimizers):
    return onnxruntime.InferenceSession(
        model_path,
        sess_options=_session_opts,
        providers=_providers,
        provider_options=_provider_options,
        disabled_optimizers=_disabled_optimizers,
    )


def get_in_names(session):
    return [x.name for x in session.get_inputs()]


def get_out_names(session):
    return [x.name for x in session.get_outputs()]


def run(session, binding):
    session.run_with_iobinding(binding, run_options=run_options)


# ══════════════════════════════════════════════════════════════════════════════
# ORT SESSION & RUNTIME OPTIONS
# ══════════════════════════════════════════════════════════════════════════════

session_opts = onnxruntime.SessionOptions()
run_options  = onnxruntime.RunOptions()

for opt in (session_opts, run_options):
    opt.log_severity_level  = 0 if ORT_LOG else 4
    opt.log_verbosity_level = 4

session_opts.inter_op_num_threads     = MAX_THREADS
session_opts.intra_op_num_threads     = MAX_THREADS
session_opts.execution_mode           = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
session_opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL

_session_configs = {
    'session.set_denormal_as_zero':                  '1',
    'session.intra_op.allow_spinning':               '1',
    'session.inter_op.allow_spinning':               '1',
    'session.enable_quant_qdq_cleanup':              '1',
    'session.qdq_matmulnbits_accuracy_level':        '2' if ORT_FP16 else '4',
    'session.use_device_allocator_for_initializers': '1',
    'session.graph_optimizations_loop_level':        '2',
    'optimization.enable_gelu_approximation':        '1',
    'optimization.minimal_build_optimizations':      '',
    'optimization.enable_cast_chain_elimination':    '1',
    'optimization.disable_specified_optimizers':
        'CastFloat16Transformer;FuseFp16InitializerToFp32NodeTransformer' if ORT_FP16 else ''
}
for k, v in _session_configs.items():
    session_opts.add_session_config_entry(k, v)

run_options.add_run_config_entry('disable_synchronize_execution_providers', '0')

disabled_optimizers = ['CastFloat16Transformer', 'FuseFp16InitializerToFp32NodeTransformer'] if ORT_FP16 else None


# ══════════════════════════════════════════════════════════════════════════════
# EXECUTION PROVIDER CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

if 'CUDAExecutionProvider' in ORT_Accelerate_Providers:
    provider_options = [{'device_id': DEVICE_ID}]
    device_type = 'cuda'
elif 'DmlExecutionProvider' in ORT_Accelerate_Providers:
    provider_options = [{'device_id': DEVICE_ID}]
    device_type = 'dml'
else:
    provider_options = None
    device_type = 'cpu'

packed_settings = {
    '_session_opts':        session_opts,
    '_providers':           ORT_Accelerate_Providers if ORT_Accelerate_Providers else ['CPUExecutionProvider'],
    '_provider_options':    provider_options,
    '_disabled_optimizers': disabled_optimizers,
}

# Separate session options for EmbedLoRA / LoRA models to prevent constant-folding
# memory explosion. The dequantization ops (rotary, hadamard, shuffle, Q4 unpack) have
# all-constant inputs, so ORT tries to evaluate them at session creation time,
# materializing dozens of full-size intermediate float32 tensors simultaneously (60GB+).
# NOTE: ORT_ENABLE_BASIC *still* includes constant folding. We must use ORT_DISABLE_ALL
# and explicitly list ConstantFolding in disabled_optimizers to fully prevent it.
session_opts_lora = onnxruntime.SessionOptions()
session_opts_lora.log_severity_level        = 0 if ORT_LOG else 4
session_opts_lora.log_verbosity_level       = 4
session_opts_lora.inter_op_num_threads      = MAX_THREADS
session_opts_lora.intra_op_num_threads      = MAX_THREADS
session_opts_lora.execution_mode            = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
session_opts_lora.graph_optimization_level  = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
for k, v in _session_configs.items():
    session_opts_lora.add_session_config_entry(k, v)

# Explicitly disable ConstantFolding (belt-and-suspenders with ORT_DISABLE_ALL)
disabled_optimizers_lora = ['ConstantFolding', 'ConstantSharing']
if disabled_optimizers:
    disabled_optimizers_lora += disabled_optimizers

packed_settings_lora = {
    '_session_opts':        session_opts_lora,
    '_providers':           ORT_Accelerate_Providers if ORT_Accelerate_Providers else ['CPUExecutionProvider'],
    '_provider_options':    provider_options,
    '_disabled_optimizers': disabled_optimizers_lora,
}


# ══════════════════════════════════════════════════════════════════════════════
# LOAD ONNX SESSIONS
# ══════════════════════════════════════════════════════════════════════════════

if FUSE_LORA_INTO_EMBED:
    # --- Fused Mode: Embed + LoRA ---
    # Use packed_settings_lora to avoid constant folding the dequant ops
    ort_session_EmbedLoRA = create_session(onnx_model_EmbedLoRA, **packed_settings_lora)
    in_name_EmbedLoRA     = get_in_names(ort_session_EmbedLoRA)
    out_name_EmbedLoRA    = get_out_names(ort_session_EmbedLoRA)
    binding_EmbedLoRA     = ort_session_EmbedLoRA.io_binding()

    print(f"\nUsable Providers: {ort_session_EmbedLoRA.get_providers()}")
else:
    # --- Split Mode: Embed + per-task LoRA ---
    # Embed also contains dequant ops for quantized embedding weights
    ort_session_Embed = create_session(onnx_model_Embed, **packed_settings_lora)
    in_name_Embed     = get_in_names(ort_session_Embed)
    out_name_Embed    = get_out_names(ort_session_Embed)
    binding_Embed     = ort_session_Embed.io_binding()

    # Load all task LoRA sessions, pre-run once, and cache outputs.
    # LoRA models have no dynamic inputs (constant-output), so we run them
    # once at startup and reuse the cached OrtValues for every encode() call.
    cached_lora_outputs = {}
    for task_name in task_names:
        lora_path = get_lora_onnx_path(task_name)
        session = create_session(lora_path, **packed_settings_lora)
        binding = session.io_binding()
        out_names = get_out_names(session)
        for name in out_names:
            binding.bind_output(name, device_type, DEVICE_ID)
        session.run_with_iobinding(binding, run_options=run_options)
        cached_lora_outputs[task_name] = binding.get_outputs()
        del session, binding

    print(f"\nUsable Providers: {ort_session_Embed.get_providers()}")

# --- Main ---
ort_session_Main = create_session(onnx_model_Main, **packed_settings)
binding_Main     = ort_session_Main.io_binding()
in_name_Main     = get_in_names(ort_session_Main)
out_name_Main    = get_out_names(ort_session_Main)

print(f"Main model inputs: {in_name_Main}")


# ══════════════════════════════════════════════════════════════════════════════
# MODEL METADATA & TASK CONFIG
# ══════════════════════════════════════════════════════════════════════════════

task_index_map = {task: idx for idx, task in enumerate(task_names)}

# OrtValue shape cache for reusable output buffers
_shape_cache = {}


def get_cached_buffer(name, shape, dtype):
    cache_key = (name, tuple(int(dim) for dim in shape), np.dtype(dtype).str)
    if cache_key not in _shape_cache:
        _shape_cache[cache_key] = create_ort_with_shape(shape, dtype, device_type, DEVICE_ID)
    return _shape_cache[cache_key]


# ══════════════════════════════════════════════════════════════════════════════
# TOKENIZER & SENTENCE TRANSFORMER CONFIG
# ══════════════════════════════════════════════════════════════════════════════

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
sentence_transformer_config = load_json(os.path.join(MODEL_PATH, 'config_sentence_transformers.json'))


# ══════════════════════════════════════════════════════════════════════════════
# ENCODE FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def encode(input_ids, task_name, return_last_hidden_state=False):
    """Encode input_ids using Embed(+LoRA) -> Main pipeline.

    Supports both fused and split LoRA modes transparently.

    Args:
        input_ids: np.ndarray [batch, seq_len] int32
        task_name: str — one of the task names in task_index_map
        return_last_hidden_state: if True, also return the full last hidden state

    Returns:
        embeddings: np.ndarray [batch, hidden_size]
        (optional) last_hidden_state: np.ndarray [batch, seq_len, hidden_size]
    """
    batch_size, seq_len = input_ids.shape
    task_idx = task_index_map[task_name]

    # ── Output buffers ───────────────────────────────────────────────────
    embeddings_buf = get_cached_buffer('embeddings', (batch_size, hidden_size), np.float32)
    last_hidden_state_buf = get_cached_buffer('last_hidden_state', (batch_size, seq_len, hidden_size), np.float32)

    if FUSE_LORA_INTO_EMBED:
        # ── Fused Mode: single EmbedLoRA session ─────────────────────────
        input_ids_ort = create_ort_with_numpy(input_ids.astype(np.int32), device_type, DEVICE_ID)
        task_idx_ort = create_ort_with_numpy(np.array([task_idx], dtype=np.int32), device_type, DEVICE_ID)

        binding_EmbedLoRA.bind_ortvalue_input(in_name_EmbedLoRA[0], input_ids_ort)
        binding_EmbedLoRA.bind_ortvalue_input(in_name_EmbedLoRA[1], task_idx_ort)

        for name in out_name_EmbedLoRA:
            binding_EmbedLoRA.bind_output(name, device_type, DEVICE_ID)

        run(ort_session_EmbedLoRA, binding_EmbedLoRA)
        embed_lora_outputs = binding_EmbedLoRA.get_outputs()

        # ── Main ─────────────────────────────────────────────────────────
        binding_Main.bind_ortvalue_input(in_name_Main[0], embed_lora_outputs[0])
        for i in range(len(LORA_INPUT_NAMES)):
            binding_Main.bind_ortvalue_input(in_name_Main[1 + i], embed_lora_outputs[1 + i])

    else:
        # ── Split Mode: Embed + cached LoRA outputs ──────────────────────
        input_ids_ort = create_ort_with_numpy(input_ids.astype(np.int32), device_type, DEVICE_ID)

        # Step 1: Embed
        binding_Embed.bind_ortvalue_input(in_name_Embed[0], input_ids_ort)
        for name in out_name_Embed:
            binding_Embed.bind_output(name, device_type, DEVICE_ID)
        run(ort_session_Embed, binding_Embed)
        hidden_states_ort = binding_Embed.get_outputs()[0]

        # Step 2: Use pre-cached LoRA outputs (no re-run needed)
        lora_outputs = cached_lora_outputs[task_name]

        # ── Main ─────────────────────────────────────────────────────────
        binding_Main.bind_ortvalue_input(in_name_Main[0], hidden_states_ort)
        for i in range(len(LORA_INPUT_NAMES)):
            binding_Main.bind_ortvalue_input(in_name_Main[1 + i], lora_outputs[i])

    bind_ort_out_buf(binding_Main, out_name_Main, [embeddings_buf, last_hidden_state_buf])
    run(ort_session_Main, binding_Main)

    embeddings = embeddings_buf.numpy()

    if return_last_hidden_state:
        return embeddings, last_hidden_state_buf.numpy()

    return embeddings



# ══════════════════════════════════════════════════════════════════════════════
# INFERENCE DEMO
# ══════════════════════════════════════════════════════════════════════════════

def format_similarity_matrix(similarity_matrix, row_labels, column_labels):
    header = f"{'':30s}" + ''.join(f" {label:>10s}" for label in column_labels)
    lines = [header, '  ' + '-' * (len(header) - 2)]
    for label, row in zip(row_labels, similarity_matrix):
        short_label = label[:30]
        row_values = ''.join(f' {value:10.4f}' for value in row)
        lines.append(f'{short_label:30s}{row_values}')
    return '\n'.join(lines)


for task_name in task_names:
    print(f"\n{'=' * 70}")
    print(f"  Task: {task_name}")
    print(f"{'=' * 70}")
    print(f'\n  Running inference for task: {task_name}\n')

    query_texts, query_input_ids = tokenize_texts(
        tokenizer, TEST_QUERIES, task_name,
        'query' if task_name == 'retrieval' else None,
        sentence_transformer_config, max_seq_len,
    )
    document_texts, document_input_ids = tokenize_texts(
        tokenizer, TEST_DOCUMENTS, task_name,
        'document' if task_name == 'retrieval' else None,
        sentence_transformer_config, max_seq_len,
    )

    start_time = time.time()
    query_embeddings = encode(query_input_ids, task_name)
    query_elapsed = time.time() - start_time

    start_time = time.time()
    document_embeddings = encode(document_input_ids, task_name)
    document_elapsed = time.time() - start_time

    print(f'  Query embeddings shape: {query_embeddings.shape}')
    print(f'  Document embeddings shape: {document_embeddings.shape}')

    similarity_matrix = query_embeddings @ document_embeddings.T
    row_labels = [text.replace('Query: ', '')[:30] for text in query_texts]
    column_labels = [f'Doc{i}' for i in range(len(document_texts))]

    print('\n  Similarity Matrix:')
    print('  ' + format_similarity_matrix(similarity_matrix, row_labels, column_labels).replace('\n', '\n  '))

    print()
    for idx, query_text in enumerate(query_texts):
        best_idx = int(np.argmax(similarity_matrix[idx]))
        best_score = float(similarity_matrix[idx, best_idx])
        print(f'  Query: "{query_text}"')
        print(f'    -> Best (score={best_score:.4f}): "{document_texts[best_idx]}"')

    print(f'\n  Timing: queries {query_elapsed:.3f}s, documents {document_elapsed:.3f}s')
