"""
Jina Embeddings v5 — Shared Backbone with Fused Embed+LoRA (2 ONNX files)
===========================================================================
Architecture:
    1 fused Embed+LoRA ONNX  — token embedding + all-task LoRA (int32 index selects task)
    1 shared Main ONNX       — backbone with LoRA as *inputs*

At runtime: load Embed_LoRA + Main once. Pass input_ids + int32 task_index
to Embed_LoRA to get hidden_states and the target task's LoRA tensors,
then feed them into Main.

The fused Embed_LoRA module concatenates all per-task LoRA weights along a
new task dimension (dim=0) and uses Gather (index_select) to extract the
target task. Combined with token embedding in a single ONNX graph.

Embed_LoRA ONNX inputs:
    input_ids     [batch, seq_len] int32
    task_index    [1] int32

Embed_LoRA ONNX outputs:
    hidden_states [batch, seq_len, hidden_size]
    + 8 stacked LoRA tensors across 28 layers:
    lora_qkv_a     [num_layers, 96, hidden_size]
    lora_qkv_b     [num_layers, qkv_out, 96]
    lora_o_a       [num_layers, rank, o_in]
    lora_o_b       [num_layers, hidden_size, rank]
    lora_gate_up_a [num_layers, 2*rank, hidden_size]
    lora_gate_up_b [num_layers, 2*intermediate_size, 2*rank]
    lora_down_a    [num_layers, rank, intermediate_size]
    lora_down_b    [num_layers, hidden_size, rank]
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
onnx_model_Main         = os.path.join(ONNX_OUTPUT_DIR, 'Embedding_Main.onnx')


# Export config
DO_EXPORT               = True              # Whether to export the ONNX models
PREVENT_F16_OVERFLOW    = False             # Apply overflow scaling in RMSNorm for FP16 stability
MAX_SEQ_LEN             = 8192              # Maximum token sequence length (clamped to model's max_position_embeddings)
OPSET                   = 18                # ONNX opset version for export

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
# MODEL CLASSES
# ══════════════════════════════════════════════════════════════════════════════

class JINA_EMBED_LORA_FUSED(torch.nn.Module):
    """
    Combined token embedding + fused LoRA adapter in a single ONNX-exportable module.

    Token embedding is applied in float32. All task LoRA weights are concatenated
    along a new task dimension (dim=0). An int32 task_index input selects the
    target task's weights via gather.

    Inputs:
        input_ids:  [batch, seq_len] int32 token IDs
        task_index: [1] int32 scalar — index of the target task

    Outputs:
        hidden_states: [batch, seq_len, hidden_size] float32 token embeddings
        + 8 LoRA tensors for the selected task, each [num_layers, ...]
    """

    def __init__(self, qwen_model, all_task_lora_list):
        super().__init__()
        self.embed_tokens = qwen_model.embed_tokens.float()
        for name in LORA_INPUT_NAMES:
            fused = torch.from_numpy(
                np.stack([task_lora[name] for task_lora in all_task_lora_list], axis=0)
            ).float().contiguous()
            self.register_buffer(name, fused)

    def forward(self, input_ids, task_index):
        hidden_states = self.embed_tokens(input_ids)
        return (
            hidden_states,
            self.lora_qkv_a[task_index].squeeze(0),
            self.lora_qkv_b[task_index].squeeze(0),
            self.lora_o_a[task_index].squeeze(0),
            self.lora_o_b[task_index].squeeze(0),
            self.lora_gate_up_a[task_index].squeeze(0),
            self.lora_gate_up_b[task_index].squeeze(0),
            self.lora_down_a[task_index].squeeze(0),
            self.lora_down_b[task_index].squeeze(0)
        )


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

        # ── Pre-computed attention mask (upper triangle → -128) ───────────
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

        print(f'\nArchitecture: 1 fused Embed+LoRA + 1 shared Main ONNX (int32 task_index)')
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

        # ══════════════════════════════════════════════════════════════════
        # Build Modules
        # ══════════════════════════════════════════════════════════════════
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

        # ══════════════════════════════════════════════════════════════════
        # Export: Fused Embed + LoRA
        # ══════════════════════════════════════════════════════════════════
        print('\nExport start [fused Embed + LoRA] ...')
        dummy_batch_size = 2
        dummy_seq_len = min(16, max_seq_len)
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

        # ══════════════════════════════════════════════════════════════════
        # Export: Shared Main (with LoRA inputs)
        # ══════════════════════════════════════════════════════════════════
        print('\nExport start [shared Main with LoRA inputs] ...')
        hidden_states = torch.ones((dummy_batch_size, dummy_seq_len, hidden_size), dtype=torch.float32)

        # Use first task's LoRA as the dummy input shapes for export
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

        # ══════════════════════════════════════════════════════════════════
        # Size Summary
        # ══════════════════════════════════════════════════════════════════
        embed_lora_size = os.path.getsize(onnx_model_EmbedLoRA)
        main_size = os.path.getsize(onnx_model_Main)
        total = embed_lora_size + main_size
        print(f'\n  ════════════════════════════════════════════')
        print(f'  SIZE SUMMARY')
        print(f'  ════════════════════════════════════════════')
        print(f'  Embed+LoRA:     {embed_lora_size / 1024 / 1024:>8.1f} MB  ({len(task_names)} tasks)')
        print(f'  Main:           {main_size / 1024 / 1024:>8.1f} MB')
        print(f'  ────────────────────────────────────────────')
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


# ══════════════════════════════════════════════════════════════════════════════
# LOAD ONNX SESSIONS
# ══════════════════════════════════════════════════════════════════════════════

# --- Embed + LoRA ---
ort_session_EmbedLoRA = create_session(onnx_model_EmbedLoRA, **packed_settings)
in_name_EmbedLoRA     = get_in_names(ort_session_EmbedLoRA)
out_name_EmbedLoRA    = get_out_names(ort_session_EmbedLoRA)

# --- Main ---
ort_session_Main = create_session(onnx_model_Main, **packed_settings)
binding_Main     = ort_session_Main.io_binding()
in_name_Main     = get_in_names(ort_session_Main)
out_name_Main    = get_out_names(ort_session_Main)

# --- IOBinding for EmbedLoRA (avoid numpy round-trip) ---
binding_EmbedLoRA = ort_session_EmbedLoRA.io_binding()

print(f"\nUsable Providers: {ort_session_EmbedLoRA.get_providers()}")
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
    """Encode input_ids using Embed+LoRA -> Main pipeline.

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

    # ── Step 1: Embed + LoRA (IOBinding, outputs stay as OrtValues) ──────
    input_ids_ort = create_ort_with_numpy(input_ids.astype(np.int32), device_type, DEVICE_ID)
    task_idx_ort = create_ort_with_numpy(np.array([task_idx], dtype=np.int32), device_type, DEVICE_ID)

    binding_EmbedLoRA.bind_ortvalue_input(in_name_EmbedLoRA[0], input_ids_ort)
    binding_EmbedLoRA.bind_ortvalue_input(in_name_EmbedLoRA[1], task_idx_ort)

    for name in out_name_EmbedLoRA:
        binding_EmbedLoRA.bind_output(name, device_type, DEVICE_ID)

    run(ort_session_EmbedLoRA, binding_EmbedLoRA)
    embed_lora_outputs = binding_EmbedLoRA.get_outputs()

    # ── Step 2: Main (pass OrtValues directly, no numpy round-trip) ──────
    binding_Main.bind_ortvalue_input(in_name_Main[0], embed_lora_outputs[0])

    for i in range(len(LORA_INPUT_NAMES)):
        binding_Main.bind_ortvalue_input(in_name_Main[1 + i], embed_lora_outputs[1 + i])

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
