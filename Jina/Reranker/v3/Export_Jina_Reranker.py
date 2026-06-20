import gc
import os
import time

import numpy as np
import onnx
import onnxruntime
import torch
from onnxruntime.capi import _pybind_state as C
from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer
from transformers.models.qwen3 import Qwen3Model


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

MODEL_DIR                  = r'/home/DakeQQ/Downloads/jina-reranker-v3'
ONNX_OUTPUT_DIR            = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Reranker_ONNX")
os.makedirs(ONNX_OUTPUT_DIR, exist_ok=True)

onnx_model_Embed           = os.path.join(ONNX_OUTPUT_DIR, 'Reranker_Embed.onnx')
onnx_model_Main            = os.path.join(ONNX_OUTPUT_DIR, 'Reranker_Main.onnx')
onnx_model_Concat          = os.path.join(ONNX_OUTPUT_DIR, 'Reranker_Concat.onnx')
onnx_model_Score           = os.path.join(ONNX_OUTPUT_DIR, 'Reranker_Score.onnx')

# Export config
DO_EXPORT                  = True
PREVENT_F16_OVERFLOW       = False          # Apply overflow scaling in RMSNorm for FP16 stability
MAX_SEQ_LEN                = 8192           # Maximum token sequence length (clamped to config.max_position_embeddings)
OPSET                      = 18             # ONNX opset version for export

# Special token IDs (from Jina Reranker v3)
DOC_EMBED_TOKEN_ID         = 151670         # <|embed_token|>
QUERY_EMBED_TOKEN_ID       = 151671         # <|rerank_token|>

# ── Inference runtime config (used by the post-export verification demo) ──────
ORT_LOG                    = False
ORT_FP16                   = False          # FP16 ORT optimizations (needs ARM64-v8.2a+ for CPU)
ORT_Accelerate_Providers   = []             # ['CUDAExecutionProvider', 'DmlExecutionProvider', 'OpenVINOExecutionProvider']
MAX_THREADS                = 0              # 0 = auto
DEVICE_ID                  = 0

MAX_QUERY_LENGTH           = 1024           # Max tokens per query (for truncation in block mode)
MAX_DOC_LENGTH             = 4096           # Max tokens per document (for truncation in block mode)
BLOCK_SIZE                 = 128            # Max documents per block in block-wise mode

SPECIAL_TOKENS = {
    "query_embed_token": "<|rerank_token|>",
    "doc_embed_token":   "<|embed_token|>",
}


# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def resolve_max_seq_len(model_config, max_seq_len):
    """Clamp max_seq_len to the model's max_position_embeddings."""
    if max_seq_len > model_config.max_position_embeddings:
        print(
            f"\n[Warning] MAX_SEQ_LEN ({max_seq_len}) exceeds config.max_position_embeddings "
            f"({model_config.max_position_embeddings}); clamping to the model limit."
        )
    return min(max_seq_len, model_config.max_position_embeddings)


def replace_gelu_with_tanh_approximation(module):
    """Recursively replace exact GELU with tanh-approximated GELU for ONNX friendliness."""
    for name, child in module.named_children():
        if isinstance(child, torch.nn.GELU):
            setattr(module, name, torch.nn.GELU(approximate='tanh'))
            print(f"  Replaced GELU at: {name}")
        else:
            replace_gelu_with_tanh_approximation(child)


def load_projector_weights(safetensors_path):
    """Load projector Linear weights from the safetensors file.

    Returns:
        proj_0_weight: [hidden_size // 2, hidden_size]  float32
        proj_2_weight: [proj_dim, hidden_size // 2]     float32
    """
    with safe_open(safetensors_path, framework="pt", device="cpu") as f:
        proj_0_weight = f.get_tensor("projector.0.weight").float()
        proj_2_weight = f.get_tensor("projector.2.weight").float()
    return proj_0_weight, proj_2_weight


def embed_size_metadata(onnx_path, metadata):
    """Bake integer model geometry into an ONNX model's custom metadata_props.

    Applied only to the small Score graph (never the multi-GB backbone).
    Existing keys are overwritten so re-exports stay idempotent.
    """
    model = onnx.load(onnx_path)
    existing = {prop.key: idx for idx, prop in enumerate(model.metadata_props)}
    for key, value in metadata.items():
        if key in existing:
            model.metadata_props[existing[key]].value = str(value)
        else:
            entry = model.metadata_props.add()
            entry.key   = str(key)
            entry.value = str(value)
    onnx.save(model, onnx_path)


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 1: RERANKER_EMBED
# ══════════════════════════════════════════════════════════════════════════════

class RERANKER_EMBED(torch.nn.Module):
    """Token embedding layer in float32.

    Input:  input_ids      [B, S]       int32
    Output: hidden_states  [B, S, D]    float32
    """

    def __init__(self, qwen_model):
        super().__init__()
        self.embed_tokens = qwen_model.embed_tokens.float()

    def forward(self, input_ids):
        return self.embed_tokens(input_ids)


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 2: RERANKER_MAIN  (fused backbone + folded post-process)
# ══════════════════════════════════════════════════════════════════════════════

class RERANKER_MAIN(torch.nn.Module):
    """Fused transformer backbone + folded post-process head.

    Single-pass full-sequence forward (no KV cache). RoPE tables and the causal
    mask are pre-computed as buffers. The graph runs end-to-end with no host
    round-trip:

        1. transformer backbone       (28 layers; fused QKV, gate-up, etc.)
        2. special-token extraction   (one combined mask + slice)
        3. final RMSNorm              (gathered rows only; weight folded into linear1)
        4. projector                  (Linear -> ReLU -> Linear, then sliced doc/query)

    Cosine scoring is deferred to RERANKER_SCORE so one scoring graph serves
    both the single- and multi-block cases; Main emits only the projections.

    Inputs:
        hidden_states  [1, S, D]      float32   (token embeddings)
        input_ids      [1, S]         int32
    Outputs:
        query_proj   [1, proj_dim]    float32
        doc_proj     [N, proj_dim]    float32
    """

    def __init__(self, qwen_model, num_heads, num_key_value_heads, head_dim,
                 num_layers, hidden_size, max_seq_len,
                 proj_0_weight, proj_2_weight, doc_token_id, query_token_id):
        super().__init__()
        self.model = qwen_model

        # ── Attention geometry ───────────────────────────────────────────
        self.head_dim             = head_dim
        self.head_dim_half        = head_dim // 2
        self.num_heads            = num_heads
        self.num_key_value_heads  = num_key_value_heads
        self.num_key_value_groups = num_heads // num_key_value_heads
        self.qk_heads             = num_heads + num_key_value_heads
        self.total_qkv_heads      = self.qk_heads + num_key_value_heads
        self.qkv_split_sizes      = [self.qk_heads, num_key_value_heads]
        self.qk_split_sizes       = [num_heads, num_key_value_heads]
        self.num_layers           = num_layers
        self.hidden_size          = hidden_size

        # ── Pre-computed causal mask (upper triangle → -128, int8) ───────
        self.register_buffer(
            'attention_mask',
            (1 - torch.tril(torch.ones(1, 1, 1, max_seq_len, max_seq_len, dtype=torch.int8))) * -128,
            persistent=False,
        )

        # ── Pre-computed rotary tables (half precision, cast at runtime) ─
        position_ids = torch.arange(max_seq_len, dtype=torch.float32).unsqueeze(-1)
        inv_freq     = qwen_model.rotary_emb.inv_freq.float()
        idx_theta    = (position_ids * inv_freq).unsqueeze(1).unsqueeze(1).unsqueeze(0)
        cos_table    = torch.cos(idx_theta)
        sin_table    = torch.sin(idx_theta)
        self.register_buffer(
            'cos_rotary_pos_emb',
            torch.cat([cos_table, cos_table], dim=-1).half(),
            persistent=False,
        )
        self.register_buffer(
            'sin_rotary_pos_emb',
            torch.cat([-sin_table, sin_table], dim=-1).half(),
            persistent=False,
        )

        # ── RMSNorm constants (sum-based, pre-scaled epsilon) ────────────
        self.overflow_scale = torch.tensor([0.01], dtype=torch.float32)
        hidden_rms_norm     = self.model.layers[0].input_layernorm
        qk_rms_norm         = self.model.layers[0].self_attn.q_norm
        hidden_rms_norm_eps = float(getattr(hidden_rms_norm, 'variance_epsilon', getattr(hidden_rms_norm, 'eps', 1e-6)))
        qk_rms_norm_eps     = float(getattr(qk_rms_norm, 'variance_epsilon', getattr(qk_rms_norm, 'eps', hidden_rms_norm_eps)))
        hidden_rms_norm_eps = hidden_size * hidden_rms_norm_eps
        qk_rms_norm_eps     = head_dim * qk_rms_norm_eps
        if PREVENT_F16_OVERFLOW:
            hidden_rms_norm_eps *= self.overflow_scale.square()
            qk_rms_norm_eps     *= self.overflow_scale.square()
        self.register_buffer('hidden_rms_norm_eps', torch.tensor([hidden_rms_norm_eps], dtype=torch.float32))
        self.register_buffer('qk_rms_norm_eps', torch.tensor([qk_rms_norm_eps], dtype=torch.float32))

        # ── Final norm weight (NOT absorbed into lm_head) ────────────────
        self.register_buffer(
            'final_norm_weight',
            self.model.norm.weight.view(1, 1, -1) * (hidden_size ** 0.5),
        )

        # ── Fuse & reshape weights for efficient inference ───────────────
        replace_gelu_with_tanh_approximation(self.model)
        self._fuse_weights(hidden_size)
        self.o_proj_in_features = self.model.layers[0].self_attn.o_proj.in_features
        self.mlp_split          = [self.model.layers[0].mlp.down_proj.in_features] * 2

        # ── Folded post-process head (extraction + projector + scoring) ──
        self._build_postprocess(proj_0_weight, proj_2_weight, doc_token_id, query_token_id)

    # ══════════════════════════════════════════════════════════════════════
    # Post-Process Head Construction (runs once at init)
    # ══════════════════════════════════════════════════════════════════════
    def _build_postprocess(self, proj_0_weight, proj_2_weight,
                           doc_token_id, query_token_id):
        """Build the folded post-process head (extraction + projector).

        The final RMSNorm's per-feature weight is a diagonal scale before the
        linear1 matmul, so it is pre-multiplied into linear1.weight. The
        row-wise RMS scaling reuses _rms_norm with hidden_rms_norm_eps (the
        final-norm eps is identical), so no separate norm op is needed.
        """
        # ── Fold the deferred final-norm per-feature weight into linear1 ──
        fold_norm_weight = self.final_norm_weight.detach().reshape(1, -1).float()  # [1, D]
        self.linear1 = torch.nn.Linear(proj_0_weight.shape[1], proj_0_weight.shape[0], bias=False)
        self.linear1.weight.copy_(proj_0_weight * fold_norm_weight)
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(proj_2_weight.shape[1], proj_2_weight.shape[0], bias=False)
        self.linear2.weight.copy_(proj_2_weight)
        self.doc_token_id   = int(doc_token_id)
        self.query_token_id = int(query_token_id)

    # ══════════════════════════════════════════════════════════════════════
    # Weight Fusion (runs once at init)
    # ══════════════════════════════════════════════════════════════════════
    def _fuse_weights(self, hidden_size):
        """Merge projections and absorb layer norms into weight matrices."""
        scale_factor   = self.head_dim ** -0.25
        norm_factor    = hidden_size ** 0.5
        norm_factor_qk = self.head_dim ** 0.5

        with torch.no_grad():
            for layer in self.model.layers:
                self._fuse_qkv_projection(layer, scale_factor, norm_factor, norm_factor_qk)
                self._fuse_gate_up_projection(layer, norm_factor)
            del self.model.norm

    def _fuse_qkv_projection(self, layer, scale_factor, norm_factor, norm_factor_qk):
        """Fuse Q, K, V projections and absorb input LayerNorm + QK norms."""
        attn   = layer.self_attn
        q_proj = attn.q_proj
        k_proj = attn.k_proj
        v_proj = attn.v_proj

        # ── Create merged QKV linear ─────────────────────────────────
        in_features  = int(q_proj.in_features)
        out_features = int(q_proj.out_features + k_proj.out_features + v_proj.out_features)
        has_bias     = any(p.bias is not None for p in (q_proj, k_proj, v_proj))

        qkv = torch.nn.Linear(in_features, out_features, bias=has_bias)
        qkv.weight.copy_(torch.cat([q_proj.weight, k_proj.weight, v_proj.weight], dim=0))

        if has_bias:
            def _get_bias(proj):
                return proj.bias if proj.bias is not None else torch.zeros(
                    proj.out_features, dtype=qkv.weight.dtype)
            qkv.bias.copy_(torch.cat([
                _get_bias(q_proj), _get_bias(k_proj), _get_bias(v_proj),
            ], dim=0))

        # ── Fuse QK norms (absorb attention scale factors) ───────────
        combined_scale = scale_factor * norm_factor_qk
        attn.q_norm.weight.mul_(combined_scale)
        attn.k_norm.weight.mul_(combined_scale)
        q_norm_repeated = attn.q_norm.weight.repeat(self.num_heads)
        k_norm_repeated = attn.k_norm.weight.repeat(self.num_key_value_heads)
        attn.qk_norm_weight = torch.nn.Parameter(
            torch.cat([q_norm_repeated, k_norm_repeated], dim=0).view(1, 1, 1, -1, self.head_dim),
            requires_grad=False,
        )

        # ── Absorb input LayerNorm into QKV weights ─────────────────
        input_norm_weight = layer.input_layernorm.weight.unsqueeze(0) * norm_factor
        qkv.weight.mul_(input_norm_weight)

        attn.qkv = qkv
        del attn.q_proj, attn.k_proj, attn.v_proj
        del attn.q_norm, attn.k_norm
        del layer.input_layernorm

    def _fuse_gate_up_projection(self, layer, norm_factor):
        """Fuse gate and up projections, absorbing post-attention LayerNorm."""
        post_norm_weight = layer.post_attention_layernorm.weight.unsqueeze(0) * norm_factor
        gate = layer.mlp.gate_proj
        up   = layer.mlp.up_proj

        gate_up = torch.nn.Linear(gate.in_features, gate.out_features + up.out_features, bias=False)
        gate_up.weight.copy_(torch.cat([
            gate.weight * post_norm_weight,
            up.weight   * post_norm_weight,
        ], dim=0))

        layer.mlp.gate_up_proj = gate_up
        del layer.mlp.gate_proj, layer.mlp.up_proj, layer.post_attention_layernorm

    # ══════════════════════════════════════════════════════════════════════
    # Utility Methods
    # ══════════════════════════════════════════════════════════════════════
    def _rms_norm(self, x, eps):
        """Sum-based RMSNorm (with optional overflow scaling)."""
        if PREVENT_F16_OVERFLOW:
            x = x * self.overflow_scale
        return x * torch.rsqrt(x.square().sum(-1, keepdim=True) + eps)

    def _rotate_half(self, x, batch_size):
        """Flip-based rotate_half: view + flip(-2) instead of split + negate + concat."""
        x = x.view(batch_size, -1, 1, self.qk_heads, 2, self.head_dim_half)
        x = x.flip(-2)
        return x.view(batch_size, -1, 1, self.qk_heads, self.head_dim)

    def _project(self, x):
        """Projector head: Linear -> ReLU -> Linear (the deferred final-norm
        per-feature weight is folded into linear1)."""
        return self.linear2(self.relu(self.linear1(x)))

    # ══════════════════════════════════════════════════════════════════════
    # Forward Pass
    # ══════════════════════════════════════════════════════════════════════
    def forward(self, hidden_states, input_ids):
        batch_size = hidden_states.shape[0]
        seq_len    = hidden_states.shape[1]

        # ── Slice pre-computed RoPE tables to seq_len ────────────────────
        rotary_pos_emb_cos = self.cos_rotary_pos_emb[:, :seq_len].float()
        rotary_pos_emb_sin = self.sin_rotary_pos_emb[:, :seq_len].float()

        # ── Slice pre-computed causal mask ────────────────────────────────
        attention_mask = self.attention_mask[..., :seq_len, :seq_len].to(torch.float32)

        # ── Transformer layers ───────────────────────────────────────────
        for layer in self.model.layers:

            # ── Self-Attention ───────────────────────────────────────
            residual      = hidden_states
            hidden_states = self._rms_norm(hidden_states, self.hidden_rms_norm_eps)

            # Fused QKV projection & reshape
            qkv   = layer.self_attn.qkv(hidden_states)
            qkv   = qkv.reshape(batch_size, -1, 1, self.total_qkv_heads, self.head_dim)
            qk, v = torch.split(qkv, self.qkv_split_sizes, dim=-2)

            # QK normalization + rotary embedding
            qk     = self._rms_norm(qk, self.qk_rms_norm_eps) * layer.self_attn.qk_norm_weight
            qk_rot = qk * rotary_pos_emb_cos + self._rotate_half(qk, batch_size) * rotary_pos_emb_sin

            # Split Q, K — reshape Q for GQA (no repeat_kv allocation)
            q, k = torch.split(qk_rot, self.qk_split_sizes, dim=-2)
            q    = q.reshape(batch_size, -1, self.num_key_value_heads, self.num_key_value_groups, self.head_dim)
            q    = q.permute(0, 2, 3, 1, 4)               # (B, KVH, G, S, D)

            k = k.permute(0, 3, 2, 4, 1)                  # (B, KVH, 1, D, S)
            v = v.transpose(1, 3)                          # (B, KVH, 1, S, D)

            # Scaled dot-product attention with causal mask
            attn = torch.matmul(q, k) + attention_mask
            attn = torch.softmax(attn, dim=-1)
            attn = torch.matmul(attn, v)

            # Output projection & residual
            attn = attn.permute(0, 3, 1, 2, 4).reshape(batch_size, -1, self.o_proj_in_features)
            hidden_states = residual + layer.self_attn.o_proj(attn)

            # ── Feed-Forward Network ─────────────────────────────────
            residual      = hidden_states
            hidden_states = self._rms_norm(hidden_states, self.hidden_rms_norm_eps)

            gate_up       = layer.mlp.gate_up_proj(hidden_states)
            gate, up      = torch.split(gate_up, self.mlp_split, dim=-1)
            hidden_states = residual + layer.mlp.down_proj(layer.mlp.act_fn(gate) * up)

        # ── Fold batch=1 to [S, D]; the final RMSNorm is deferred to the
        #    gathered special-token rows below (not all S positions). ───────
        hidden_states = hidden_states.reshape(-1, self.hidden_size)

        # ══════════════════════════════════════════════════════════════════
        # Folded Post-Process  (special-token extraction + deferred final
        #     RMSNorm + projector) — no host round-trip
        # ══════════════════════════════════════════════════════════════════
        flat_ids = input_ids.reshape(-1)                           # [S]

        # ── Combined special-token extraction: doc & query are the only ids
        #    >= doc_token_id; NonZero returns positions ascending and the query
        #    token trails every doc, so the vector splits by slicing. ───────
        special_pos = torch.where(flat_ids >= self.doc_token_id)[0]  # [N+1]

        # ── Gather rows FIRST, then the deferred final RMSNorm on just those
        #    N+1 rows. RMSNorm is per-row, so gather-then-norm == norm-then-
        #    gather but skips the S-(N+1) unused rows. ─────────────────────
        special_hidden = hidden_states.index_select(0, special_pos)  # [N+1, D]
        special_hidden = self._rms_norm(special_hidden, self.hidden_rms_norm_eps)

        # ── Project all N+1 rows in one matmul (row-wise projector), then
        #    slice doc/query. ───────────────────────────────────────────────
        special_proj = self._project(special_hidden)       # [N+1, proj_dim]
        doc_proj     = special_proj[:-1]                   # [N, proj_dim]
        query_proj   = special_proj[[-1]]                   # [1, proj_dim]

        # ── Scoring is deferred to RERANKER_SCORE; Main emits only projections.
        return query_proj, doc_proj


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 3: RERANKER_CONCAT  (pairwise projection concatenation)
# ══════════════════════════════════════════════════════════════════════════════

class RERANKER_CONCAT(torch.nn.Module):
    """Pairwise concatenation of two projection tensors along axis 0.

    ONNX graphs take a fixed input count, so the variable-length concat over
    per-block projections is folded across the block list (accumulator + next
    block) at inference time.

    Inputs:
        embed_0  [R0, proj_dim]             float32
        embed_1  [R1, proj_dim]             float32
    Output:
        concat_embed  [R0 + R1, proj_dim]   float32
    """

    def forward(self, embed_0, embed_1):
        concat_embed = torch.cat([embed_0, embed_1], dim=0)
        return concat_embed,


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 4: RERANKER_SCORE  (multi-block query combination + cosine scoring)
# ══════════════════════════════════════════════════════════════════════════════

class RERANKER_SCORE(torch.nn.Module):
    """Query combination + cosine scoring + block-weight computation.

    The single scoring graph for the whole pipeline. It both consumes a
    per-block weight for every query row (weighted combination done in-graph)
    and produces the scored block's weight (max normalised score).

    Usage:
      - single / per-block scoring → block_weights = [1.0]; the lone query row
        passes through unscaled (plain cosine).
      - multi-block final scoring → per-block query rows with their weights;
        weighted and summed in-graph (cosine is scale-invariant, so the
        weighted SUM equals the official weighted MEAN), then scored.

    Inputs:
        query_projs    [num_blocks, proj_dim]   float32
        block_weights  [num_blocks]             float32
        doc_projs      [total_docs, proj_dim]   float32
    Outputs:
        scores         [total_docs]             float32
        block_weight   [1]                      float32   (max normalised score)
    """

    def __init__(self, eps=1e-10):
        super().__init__()
        self.eps = float(eps)

    def forward(self, query_projs, block_weights, doc_projs):
        # ── Weight each query row, then sum → combined query. The weighted SUM
        #    equals the official weighted MEAN (cosine is scale-invariant). ──
        combined_query = (query_projs * block_weights.unsqueeze(-1)).sum(dim=0, keepdim=True)   # [1, proj_dim]

        # ── Cosine similarity vs all documents. The two norm sqrts fuse into
        #    one: sqrt(a)·sqrt(b) == sqrt(a·b) for sums of squares. ───────────
        dot    = torch.matmul(combined_query, doc_projs.transpose(0, 1)).reshape(-1)  # [total_docs]
        q_sq   = (combined_query * combined_query).sum(dim=-1)   # [1]
        d_sq   = (doc_projs * doc_projs).sum(dim=-1)             # [total_docs]
        denom  = torch.sqrt(q_sq * d_sq) + self.eps              # [total_docs]
        scores = dot / denom                                     # [total_docs]

        # ── Block weight = max normalised score ────────────────────────
        block_weight = ((1.0 + scores) * 0.5).amax(dim=-1, keepdim=True)  # [1]
        return scores, block_weight


# ══════════════════════════════════════════════════════════════════════════════
# EXPORT
# ══════════════════════════════════════════════════════════════════════════════

if DO_EXPORT:
    print('Export start ...')
    with torch.inference_mode():

        # ══════════════════════════════════════════════════════════════════
        # Load Model Config
        # ══════════════════════════════════════════════════════════════════
        print('\nLoading model config ...')
        model_config = AutoConfig.from_pretrained(MODEL_DIR, trust_remote_code=True)

        num_layers   = model_config.num_hidden_layers
        num_heads    = model_config.num_attention_heads
        num_kv_heads = model_config.num_key_value_heads
        head_dim     = model_config.head_dim
        hidden_size  = model_config.hidden_size
        proj_dim     = hidden_size // 2     # 512 for jina-reranker-v3
        max_seq_len  = resolve_max_seq_len(model_config, MAX_SEQ_LEN)

        print(f'  num_hidden_layers:      {num_layers}')
        print(f'  num_attention_heads:    {num_heads}')
        print(f'  num_key_value_heads:    {num_kv_heads}')
        print(f'  head_dim:               {head_dim}')
        print(f'  hidden_size:            {hidden_size}')
        print(f'  max_seq_len (export):   {max_seq_len}')
        print(f'  PREVENT_F16_OVERFLOW:   {PREVENT_F16_OVERFLOW}')

        # ══════════════════════════════════════════════════════════════════
        # Load Backbone Model
        # ══════════════════════════════════════════════════════════════════
        print('\nLoading Qwen3 backbone ...')
        qwen_model = Qwen3Model.from_pretrained(
            MODEL_DIR,
            config=model_config,
            dtype=torch.float32,
            low_cpu_mem_usage=True,
        ).eval()
        print('  Backbone loaded.')

        # ══════════════════════════════════════════════════════════════════
        # Load Projector Weights
        # ══════════════════════════════════════════════════════════════════
        safetensors_path = os.path.join(MODEL_DIR, 'model.safetensors')
        print(f'\nLoading projector weights from: {safetensors_path}')
        proj_0_weight, proj_2_weight = load_projector_weights(safetensors_path)
        print(f'  projector.0.weight: {list(proj_0_weight.shape)}')   # [512, 1024]
        print(f'  projector.2.weight: {list(proj_2_weight.shape)}')   # [512, 512]

        # ══════════════════════════════════════════════════════════════════
        # Build Exportable Modules
        # ══════════════════════════════════════════════════════════════════
        print('\nBuilding RERANKER_EMBED ...')
        model_Embed = RERANKER_EMBED(qwen_model).eval()

        print('Building RERANKER_MAIN (fusing weights + folded post-process) ...')
        model_Main = RERANKER_MAIN(
            qwen_model,
            num_heads=num_heads,
            num_key_value_heads=num_kv_heads,
            head_dim=head_dim,
            num_layers=num_layers,
            hidden_size=hidden_size,
            max_seq_len=max_seq_len,
            proj_0_weight=proj_0_weight,
            proj_2_weight=proj_2_weight,
            doc_token_id=DOC_EMBED_TOKEN_ID,
            query_token_id=QUERY_EMBED_TOKEN_ID,
        ).eval()
        del proj_0_weight, proj_2_weight
        gc.collect()

        # ══════════════════════════════════════════════════════════════════
        # Export: Reranker_Embed.onnx
        # Shape:  [1, S] int32 → [1, S, 1024] float32
        # ══════════════════════════════════════════════════════════════════
        print('\nExporting Reranker_Embed.onnx ...')
        dummy_seq_len = min(16, max_seq_len)
        input_ids = torch.ones((1, dummy_seq_len), dtype=torch.int32)

        torch.onnx.export(
            model_Embed,
            (input_ids,),
            onnx_model_Embed,
            input_names=['input_ids'],
            output_names=['hidden_states'],
            dynamic_axes={
                'input_ids':     {0: 'batch', 1: 'seq_len'},
                'hidden_states': {0: 'batch', 1: 'seq_len'},
            },
            opset_version=OPSET,
            dynamo=False,
        )
        del input_ids
        print(f'  Saved: {onnx_model_Embed}')

        # ══════════════════════════════════════════════════════════════════
        # Export: Reranker_Main.onnx  (backbone + folded post-process)
        # Shape:  [1, S, 1024] f32 + [1, S] int32
        #         → query_proj [1, 512] f32, doc_proj [N, 512] f32
        #   (cosine scoring is deferred to Reranker_Score.onnx)
        # ══════════════════════════════════════════════════════════════════
        print('\nExporting Reranker_Main.onnx (merged backbone + post-process) ...')
        hidden_states  = torch.ones((1, dummy_seq_len, hidden_size), dtype=torch.float32)
        # Dummy input_ids must contain ≥1 query token and ≥1 doc token so the
        # in-graph extraction (NonZero/Gather) traces with non-empty gathers.
        input_ids_main = torch.ones((1, dummy_seq_len), dtype=torch.int32)
        input_ids_main[0, 3]  = DOC_EMBED_TOKEN_ID
        input_ids_main[0, 7]  = DOC_EMBED_TOKEN_ID
        input_ids_main[0, 11] = DOC_EMBED_TOKEN_ID
        input_ids_main[0, -1] = QUERY_EMBED_TOKEN_ID

        torch.onnx.export(
            model_Main,
            (hidden_states, input_ids_main),
            onnx_model_Main,
            input_names=['hidden_states', 'input_ids'],
            output_names=['query_proj', 'doc_proj'],
            dynamic_axes={
                'hidden_states': {0: 'batch', 1: 'seq_len'},
                'input_ids':     {0: 'batch', 1: 'seq_len'},
                'doc_proj':      {0: 'num_docs'},
            },
            opset_version=OPSET,
            dynamo=False,
        )
        del hidden_states, input_ids_main
        del model_Main, model_Embed, qwen_model
        gc.collect()
        print(f'  Saved: {onnx_model_Main}')

        # ══════════════════════════════════════════════════════════════════
        # Export: Reranker_Concat.onnx  (pairwise projection concatenation)
        # Shape:  [R0, 512] f32 + [R1, 512] f32 → [R0 + R1, 512] f32
        #   Folded across the block list at inference time to replace the
        #   host-side np.concatenate over per-block projections.
        # ══════════════════════════════════════════════════════════════════
        print('\nExporting Reranker_Concat.onnx ...')
        model_Concat   = RERANKER_CONCAT().eval()
        concat_embed_0 = torch.randn((3, proj_dim), dtype=torch.float32)
        concat_embed_1 = torch.randn((2, proj_dim), dtype=torch.float32)

        torch.onnx.export(
            model_Concat,
            (concat_embed_0, concat_embed_1),
            onnx_model_Concat,
            input_names=['embed_0', 'embed_1'],
            output_names=['concat_embed'],
            dynamic_axes={
                'embed_0':      {0: 'rows_0'},
                'embed_1':      {0: 'rows_1'},
                'concat_embed': {0: 'rows_total'},
            },
            opset_version=OPSET,
            dynamo=False,
        )
        del concat_embed_0, concat_embed_1, model_Concat
        print(f'  Saved: {onnx_model_Concat}')

        # ══════════════════════════════════════════════════════════════════
        # Export: Reranker_Score.onnx  (block combination + cosine scoring)
        # Shape:  query_projs    [num_blocks, 512] f32,
        #         block_weights  [num_blocks]      f32,
        #         doc_projs      [total_docs, 512] f32
        #         → scores       [total_docs]      f32,
        #           block_weight  [1]              f32
        # ══════════════════════════════════════════════════════════════════
        print('\nExporting Reranker_Score.onnx ...')
        model_Score         = RERANKER_SCORE().eval()
        dummy_query_projs   = torch.randn((2, proj_dim), dtype=torch.float32)
        dummy_block_weights = torch.ones((2,), dtype=torch.float32)
        dummy_doc_projs     = torch.randn((5, proj_dim), dtype=torch.float32)

        torch.onnx.export(
            model_Score,
            (dummy_query_projs, dummy_block_weights, dummy_doc_projs),
            onnx_model_Score,
            input_names=['query_projs', 'block_weights', 'doc_projs'],
            output_names=['scores', 'block_weight'],
            dynamic_axes={
                'query_projs':   {0: 'num_blocks'},
                'block_weights': {0: 'num_blocks'},
                'doc_projs':     {0: 'total_docs'},
                'scores':        {0: 'total_docs'},
            },
            opset_version=OPSET,
            dynamo=False,
        )
        del dummy_query_projs, dummy_block_weights, dummy_doc_projs, model_Score
        gc.collect()
        print(f'  Saved: {onnx_model_Score}')

        # ══════════════════════════════════════════════════════════════════
        # Bake model geometry into Reranker_Score.onnx metadata
        # ══════════════════════════════════════════════════════════════════
        #   Inference reads hidden_size / max_seq_len / proj_dim from the Score
        #   graph metadata (the multi-GB backbone is never touched).
        embed_size_metadata(onnx_model_Score, {
            'hidden_size': hidden_size,
            'max_seq_len': max_seq_len,
            'proj_dim':    proj_dim,
        })
        print(f'  Embedded size metadata into: {onnx_model_Score}')

        # ══════════════════════════════════════════════════════════════════
        # Size Summary
        # ══════════════════════════════════════════════════════════════════
        embed_size     = os.path.getsize(onnx_model_Embed)
        main_size      = os.path.getsize(onnx_model_Main)
        concat_size    = os.path.getsize(onnx_model_Concat)
        score_size     = os.path.getsize(onnx_model_Score)
        total_size     = embed_size + main_size + concat_size + score_size

        print(f'\n  {"=" * 50}')
        print(f'  SIZE SUMMARY')
        print(f'  {"=" * 50}')
        print(f'  Reranker_Embed.onnx:     {embed_size / 1024 / 1024:>8.1f} MB')
        print(f'  Reranker_Main.onnx:      {main_size / 1024 / 1024:>8.1f} MB')
        print(f'  Reranker_Concat.onnx:    {concat_size / 1024 / 1024:>8.1f} MB')
        print(f'  Reranker_Score.onnx:     {score_size / 1024 / 1024:>8.1f} MB')
        print(f'  {"-" * 42}')
        print(f'  TOTAL:                   {total_size / 1024 / 1024:>8.1f} MB')
        print(f'  {"=" * 50}')
        print(f'  (Projector folded into Reranker_Main.onnx — no separate file)')

    print('\nExport done!')


# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════  ONNX RUNTIME INFERENCE  ══════════════════════════
# ══════════════════════════════════════════════════════════════════════════════
#   End-to-end verification demo for the freshly-exported models. Reuses the
#   export's geometry globals (hidden_size / max_seq_len / proj_dim) directly.
#
#   Pipeline:
#       tokenize → embed → [ backbone → extract → projector ] → score → rank
#                          └──────── Main graph ───────┘   (scoring)


# ══════════════════════════════════════════════════════════════════════════════
# PROMPT FORMATTING (mirrors official modeling.py)
# ══════════════════════════════════════════════════════════════════════════════

def sanitize_input(text, special_tokens):
    """Strip special marker tokens from raw user text to prevent injection."""
    for token in special_tokens.values():
        text = text.replace(token, "")
    return text


def format_docs_prompts_func(query, docs, instruction=None,
                             special_tokens=None, no_thinking=True):
    """Build the chat-template prompt for Jina Reranker v3 (mirrors modeling.py).

    Args:
        query:          search query string
        docs:           list of document strings
        instruction:    optional ranking instruction
        special_tokens: dict mapping token roles to token strings
        no_thinking:    if True, append <think>\\n\\n</think>\\n\\n
    """
    if special_tokens is None:
        special_tokens = SPECIAL_TOKENS

    query = sanitize_input(query, special_tokens)
    docs  = [sanitize_input(doc, special_tokens) for doc in docs]

    prefix = (
        "<|im_start|>system\n"
        "You are a search relevance expert who can determine a ranking of the "
        "passages based on how relevant they are to the query. "
        "If the query is a question, how relevant a passage is depends on how "
        "well it answers the question. If not, try to analyze the intent of "
        "the query and assess how well each passage satisfies the intent. "
        "If an instruction is provided, you should follow the instruction "
        "when determining the ranking."
        "<|im_end|>\n<|im_start|>user\n"
    )
    suffix = "<|im_end|>\n<|im_start|>assistant\n"
    if no_thinking:
        suffix += "<think>\n\n</think>\n\n"

    doc_emb_token   = special_tokens["doc_embed_token"]
    query_emb_token = special_tokens["query_embed_token"]

    prompt = (
        f"I will provide you with {len(docs)} passages, each indicated by a "
        f"numerical identifier. Rank the passages based on their relevance to "
        f"query: {query}\n"
    )

    if instruction:
        prompt += f'<instruct>\n{instruction}\n</instruct>\n'

    doc_parts = [
        f'<passage id="{i}">\n{doc}{doc_emb_token}\n</passage>'
        for i, doc in enumerate(docs)
    ]
    prompt += "\n".join(doc_parts) + "\n"
    prompt += f"<query>\n{query}{query_emb_token}\n</query>"

    return prefix + prompt + suffix


# ══════════════════════════════════════════════════════════════════════════════
# ORT HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def bind_ort_out_buf(binding, names, values):
    """Bind OrtValue outputs by name."""
    for name, val in zip(names, values):
        binding.bind_ortvalue_output(name, val)


def create_ort_with_shape(shape, dtype, device, device_id):
    """Create a zero-filled OrtValue with the given shape."""
    return onnxruntime.OrtValue.ortvalue_from_numpy(
        np.zeros(shape, dtype=dtype), device, device_id)


def create_ort_with_numpy(array, device, device_id):
    """Create an OrtValue from a numpy array (ensures contiguous layout)."""
    return onnxruntime.OrtValue.ortvalue_from_numpy(
        np.ascontiguousarray(array), device, device_id)


def create_session(model_path, _session_opts, _providers, _provider_options,
                   _disabled_optimizers):
    """Create an ORT InferenceSession with standard options."""
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


# Map ORT NodeArg.type strings to numpy dtypes so every IO-binding buffer
# matches the model's exported precision (fp32 or fp16), not a hard-coded type.
_ONNX_TO_NUMPY_DTYPE = {
    'tensor(float)':   np.float32,
    'tensor(float16)': np.float16,
    'tensor(double)':  np.float64,
    'tensor(int64)':   np.int64,
    'tensor(int32)':   np.int32,
    'tensor(int16)':   np.int16,
    'tensor(int8)':    np.int8,
    'tensor(uint8)':   np.uint8,
    'tensor(bool)':    np.bool_,
}


def onnx_dtype_to_numpy(onnx_type):
    """Map an ORT NodeArg type string (e.g. 'tensor(float16)') to a numpy dtype."""
    try:
        return _ONNX_TO_NUMPY_DTYPE[onnx_type]
    except KeyError:
        raise ValueError(f"Unsupported ONNX tensor type: {onnx_type!r}")


def get_in_dtypes(session):
    """Numpy dtypes of a session's inputs, parallel to get_in_names()."""
    return [onnx_dtype_to_numpy(x.type) for x in session.get_inputs()]


def get_out_dtypes(session):
    """Numpy dtypes of a session's outputs, parallel to get_out_names()."""
    return [onnx_dtype_to_numpy(x.type) for x in session.get_outputs()]


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
        'CastFloat16Transformer;FuseFp16InitializerToFp32NodeTransformer' if ORT_FP16 else '',
}
for k, v in _session_configs.items():
    session_opts.add_session_config_entry(k, v)

run_options.add_run_config_entry('disable_synchronize_execution_providers', '0')

disabled_optimizers = (
    ['CastFloat16Transformer', 'FuseFp16InitializerToFp32NodeTransformer']
    if ORT_FP16 else None
)


# ══════════════════════════════════════════════════════════════════════════════
# EXECUTION PROVIDER CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

if "OpenVINOExecutionProvider" in ORT_Accelerate_Providers:
    provider_options = [{
        'device_type':              'CPU',
        'precision':                'ACCURACY',
        'num_of_threads':           MAX_THREADS if MAX_THREADS != 0 else 8,
        'num_streams':              1,
        'enable_opencl_throttling': False,
        'enable_qdq_optimizer':     False,
        'disable_dynamic_shapes':   False,
    }]
    device_type      = 'cpu'
    _ort_device_type = C.OrtDevice.cpu()

elif "CUDAExecutionProvider" in ORT_Accelerate_Providers:
    provider_options = [{
        'device_id':                          DEVICE_ID,
        'gpu_mem_limit':                      24 * (1024 ** 3),
        'arena_extend_strategy':              'kNextPowerOfTwo',
        'cudnn_conv_algo_search':             'EXHAUSTIVE',
        'sdpa_kernel':                        '2',
        'use_tf32':                           '1',
        'fuse_conv_bias':                     '0',
        'cudnn_conv_use_max_workspace':       '1',
        'cudnn_conv1d_pad_to_nc1d':           '0',
        'tunable_op_enable':                  '0',
        'tunable_op_tuning_enable':           '0',
        'tunable_op_max_tuning_duration_ms':  10,
        'do_copy_in_default_stream':          '0',
        'enable_cuda_graph':                  '0',
        'prefer_nhwc':                        '0',
        'enable_skip_layer_norm_strict_mode': '0',
        'use_ep_level_unified_stream':        '0',
    }]
    device_type      = 'cuda'
    _ort_device_type = C.OrtDevice.cuda()

elif "DmlExecutionProvider" in ORT_Accelerate_Providers:
    provider_options = [{
        'device_id':                  DEVICE_ID,
        'performance_preference':     'high_performance',
        'device_filter':              'gpu',
        'disable_metacommands':       'false',
        'enable_graph_capture':       'false',
        'enable_graph_serialization': 'false',
    }]
    device_type      = 'dml'
    _ort_device_type = C.OrtDevice.dml()

else:
    provider_options = None
    device_type      = 'cpu'
    _ort_device_type = C.OrtDevice.cpu()

packed_settings = {
    '_session_opts':        session_opts,
    '_providers':           (ORT_Accelerate_Providers
                             if ORT_Accelerate_Providers
                             else ['CPUExecutionProvider']),
    '_provider_options':    provider_options,
    '_disabled_optimizers': disabled_optimizers,
}

_ort_device_type = C.OrtDevice(
    _ort_device_type, C.OrtDevice.default_memory(), DEVICE_ID)


# ══════════════════════════════════════════════════════════════════════════════
# LOAD ONNX SESSIONS
# ══════════════════════════════════════════════════════════════════════════════

print('\nLoading ONNX sessions ...')

ort_session_Embed     = create_session(onnx_model_Embed,     **packed_settings)
ort_session_Main      = create_session(onnx_model_Main,      **packed_settings)
ort_session_Score     = create_session(onnx_model_Score,     **packed_settings)

binding_Embed         = ort_session_Embed.io_binding()
binding_Main          = ort_session_Main.io_binding()
binding_Score         = ort_session_Score.io_binding()

in_name_Embed         = get_in_names(ort_session_Embed)
out_name_Embed        = get_out_names(ort_session_Embed)
in_name_Main          = get_in_names(ort_session_Main)
out_name_Main         = get_out_names(ort_session_Main)
in_name_Score         = get_in_names(ort_session_Score)
out_name_Score        = get_out_names(ort_session_Score)
# Dtypes read from each model's I/O signature so every buffer adopts the
# model's own precision (fp32 or fp16), not a hard-coded float32.
in_dtype_Embed        = get_in_dtypes(ort_session_Embed)
out_dtype_Embed       = get_out_dtypes(ort_session_Embed)
in_dtype_Main         = get_in_dtypes(ort_session_Main)
out_dtype_Main        = get_out_dtypes(ort_session_Main)
in_dtype_Score        = get_in_dtypes(ort_session_Score)
out_dtype_Score       = get_out_dtypes(ort_session_Score)

# ══════════════════════════════════════════════════════════════════════════════
# MODEL METADATA
# ══════════════════════════════════════════════════════════════════════════════
#   The demo reuses the export's geometry globals (hidden_size, max_seq_len,
#   proj_dim) directly — no metadata / config round-trip needed.


# ══════════════════════════════════════════════════════════════════════════════
# SHAPE-CACHED ORTVALUE BUFFERS
# ══════════════════════════════════════════════════════════════════════════════

_shape_cache = {}


def get_cached_buffer(name, shape, dtype):
    """Return a reusable OrtValue output buffer, creating one if shape is new."""
    cache_key = (name, tuple(int(d) for d in shape), np.dtype(dtype).str)
    if cache_key not in _shape_cache:
        _shape_cache[cache_key] = create_ort_with_shape(
            shape, dtype, device_type, DEVICE_ID)
    return _shape_cache[cache_key]

# Cached unit block-weight ([1.0]) for single / per-block scoring: a lone query
# row weighted by 1.0 passes through unscaled (plain cosine). Uploaded once.
ones1_ort = create_ort_with_numpy(
    np.ones((1,), dtype=in_dtype_Score[1]), device_type, DEVICE_ID)

# ══════════════════════════════════════════════════════════════════════════════
# TOKENIZER
# ══════════════════════════════════════════════════════════════════════════════

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
tokenizer.padding_side = 'left'
if tokenizer.pad_token is None:
    tokenizer.pad_token    = tokenizer.unk_token
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)


# ══════════════════════════════════════════════════════════════════════════════
# SINGLE-BATCH PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def _embed_and_main(input_ids_np):
    """Embed → Main, keeping the projections resident on the device.

    The Embed output buffer is bound directly as the Main input, and the Main
    output OrtValues are returned as-is for direct binding into Reranker_Score
    (no numpy round-trip).

    Returns:
        query_proj_buf: OrtValue [1, proj_dim]        (on device)
        doc_proj_buf:   OrtValue [num_docs, proj_dim] (on device)
        num_docs:       int
    """
    batch_size, seq_len = input_ids_np.shape
    num_docs = int(np.count_nonzero(input_ids_np == DOC_EMBED_TOKEN_ID))

    # One input_ids OrtValue, shared by both graphs (Main needs it for the
    # in-graph special-token extraction).
    input_ids_ort = create_ort_with_numpy(input_ids_np, device_type, DEVICE_ID)

    # ── Embed: input_ids → hidden_states (kept on device, reused buffer) ──
    hidden_buf = get_cached_buffer(
        'hidden_states', (batch_size, seq_len, hidden_size), out_dtype_Embed[0])

    binding_Embed.bind_ortvalue_input(in_name_Embed[0], input_ids_ort)
    binding_Embed.bind_ortvalue_output(out_name_Embed[0], hidden_buf)
    run(ort_session_Embed, binding_Embed)

    # ── Main: backbone + in-graph extraction + projector ───────────────
    #   hidden_buf is bound directly as Main's input (zero-copy device
    #   handoff); input_ids_ort is reused for the in-graph token extraction.
    query_proj_buf = get_cached_buffer('query_proj', (1, proj_dim),        out_dtype_Main[0])
    doc_proj_buf   = get_cached_buffer('doc_proj',   (num_docs, proj_dim), out_dtype_Main[1])

    binding_Main.bind_ortvalue_input(in_name_Main[0], hidden_buf)       # hidden_states
    binding_Main.bind_ortvalue_input(in_name_Main[1], input_ids_ort)    # input_ids
    bind_ort_out_buf(binding_Main, out_name_Main,
                     [query_proj_buf, doc_proj_buf])
    run(ort_session_Main, binding_Main)

    return query_proj_buf, doc_proj_buf, num_docs


def _score_on_device(query_projs_ort, block_weights_ort, doc_projs_ort, total_docs):
    """Reranker_Score with all inputs already resident on the device.

    Inputs are bound directly (no numpy round-trip); both outputs go into
    shape-cached, reused device buffers.

    Returns:
        score_buf:        OrtValue [total_docs] (shape-cached, reused)
        block_weight_buf: OrtValue [1]          (shape-cached, reused)
    """
    score_buf        = get_cached_buffer('scores',       (total_docs,), out_dtype_Score[0])
    block_weight_buf = get_cached_buffer('block_weight', (1,),          out_dtype_Score[1])

    binding_Score.bind_ortvalue_input(in_name_Score[0], query_projs_ort)    # query_projs
    binding_Score.bind_ortvalue_input(in_name_Score[1], block_weights_ort)  # block_weights
    binding_Score.bind_ortvalue_input(in_name_Score[2], doc_projs_ort)      # doc_projs
    bind_ort_out_buf(binding_Score, out_name_Score, [score_buf, block_weight_buf])
    run(ort_session_Score, binding_Score)

    return score_buf, block_weight_buf


# ══════════════════════════════════════════════════════════════════════════════
# BLOCK-WISE RERANKING (mirrors official Jina rerank for oversized inputs)
# ══════════════════════════════════════════════════════════════════════════════

def _truncate_texts(query, documents):
    """Truncate query and documents to fit within token limits.

    Returns:
        query:        (possibly truncated) query string
        docs:         list of (possibly truncated) document strings
        doc_lengths:  list of token counts per document
        query_length: token count for query
    """
    docs        = []
    doc_lengths = []
    for doc in documents:
        doc_tokens = tokenizer(doc, truncation=True, max_length=MAX_DOC_LENGTH)
        if len(doc_tokens['input_ids']) >= MAX_DOC_LENGTH:
            doc = tokenizer.decode(doc_tokens['input_ids'])
        doc_lengths.append(len(doc_tokens['input_ids']))
        docs.append(doc)

    query_tokens = tokenizer(query, truncation=True, max_length=MAX_QUERY_LENGTH)
    if len(query_tokens['input_ids']) >= MAX_QUERY_LENGTH:
        query = tokenizer.decode(query_tokens['input_ids'])
    query_length = len(query_tokens['input_ids'])

    return query, docs, doc_lengths, query_length


# Lazily-loaded Concat session for the multi-block path; the common
# single-block case never loads it. Reranker_Score is loaded eagerly above.
_concat_state = {}


def _load_concat_session():
    """Create (once) and return the Concat session, io-binding + I/O names."""
    if not _concat_state:
        sess = create_session(onnx_model_Concat, **packed_settings)
        _concat_state['session']   = sess
        _concat_state['binding']   = sess.io_binding()
        _concat_state['in_names']  = get_in_names(sess)
        _concat_state['out_names'] = get_out_names(sess)
    return (_concat_state['session'],  _concat_state['binding'],
            _concat_state['in_names'], _concat_state['out_names'])


def _concat_fold(session, binding, in_names, out_names, projs):
    """Fold a list of [*, proj_dim] arrays into one via the pairwise Concat graph.

    Folds across the list (accumulator + next block, axis 0). A single-element
    list is returned as is. Multi-block cold path only.

    Returns:
        concat: numpy array [sum(rows_i), proj_dim]
    """
    acc = np.ascontiguousarray(projs[0])
    proj_dim_ = acc.shape[1]
    concat_out_dtype = get_out_dtypes(session)[0]
    for nxt in projs[1:]:
        nxt = np.ascontiguousarray(nxt)
        out_rows = acc.shape[0] + nxt.shape[0]

        acc_ort    = create_ort_with_numpy(acc, device_type, DEVICE_ID)
        nxt_ort    = create_ort_with_numpy(nxt, device_type, DEVICE_ID)
        concat_buf = get_cached_buffer(
            'concat_embed', (out_rows, proj_dim_), concat_out_dtype)

        binding.bind_ortvalue_input(in_names[0], acc_ort)
        binding.bind_ortvalue_input(in_names[1], nxt_ort)
        binding.bind_ortvalue_output(out_names[0], concat_buf)
        run(session, binding)

        acc = concat_buf.numpy().copy()
    return acc


def _tokenize_block(query, block_docs, instruction):
    """Format + tokenize one block's prompt → int32 input_ids [1, S]."""
    prompt = format_docs_prompts_func(
        query, block_docs, instruction=instruction,
        special_tokens=SPECIAL_TOKENS, no_thinking=True,
    )
    tokens = tokenizer(
        prompt, return_tensors='np', padding=False)['input_ids'].astype(in_dtype_Embed[0])

    if tokens.shape[1] > max_seq_len:
        raise ValueError(
            f"Prompt length ({tokens.shape[1]}) exceeds MAX_SEQ_LEN "
            f"({max_seq_len}). Reduce the number of documents per block "
            f"or increase MAX_SEQ_LEN in the export script and re-export."
        )
    return tokens


def _pack_blocks(docs_trunc, doc_lengths, query_length):
    """Pack documents into blocks that each fit within max_seq_len.

    Mirrors the official Jina capacity heuristic (the query is counted twice,
    as it appears both in the prompt body and the trailing <query> tag).

    Returns:
        blocks: list of blocks, each a list of document strings
    """
    # length_capacity accounts for the query appearing twice in the prompt
    length_capacity  = max_seq_len - 2 * query_length
    current_capacity = length_capacity

    blocks     = []
    block_docs = []
    for doc_len, doc in zip(doc_lengths, docs_trunc):
        block_docs.append(doc)
        current_capacity -= doc_len

        if len(block_docs) >= BLOCK_SIZE or current_capacity <= MAX_DOC_LENGTH:
            blocks.append(block_docs)
            block_docs       = []
            current_capacity = length_capacity

    if block_docs:
        blocks.append(block_docs)
    return blocks


def _score_single_block(query, block_docs, instruction):
    """Hot path: one block scored end-to-end on the device.

    Embed → Main → Score run back-to-back (zero-copy between graphs); only the
    final scores are copied back. One query row ⇒ plain cosine, so the block
    scores are the final scores.
    """
    tokens = _tokenize_block(query, block_docs, instruction)
    query_proj_buf, doc_proj_buf, num_docs = _embed_and_main(tokens)
    score_buf, _ = _score_on_device(
        query_proj_buf, ones1_ort, doc_proj_buf, num_docs)
    return score_buf.numpy().copy()


def _score_multi_block(query, blocks, instruction):
    """Cold path: documents span several blocks (mirrors official Jina).

    Each block is scored on the device so Reranker_Score emits its weight
    in-graph; the per-block projections are copied back, folded by
    Reranker_Concat, then the same Score graph combines the block-weighted
    query rows and rescores cosine over every document.
    """
    sess_concat, bind_concat, concat_in, concat_out = _load_concat_session()

    all_query_projs = []
    all_doc_projs   = []
    block_weights   = []
    for block_docs in blocks:
        tokens = _tokenize_block(query, block_docs, instruction)
        query_proj_buf, doc_proj_buf, num_docs = _embed_and_main(tokens)
        _, block_weight_buf = _score_on_device(
            query_proj_buf, ones1_ort, doc_proj_buf, num_docs)
        # Copy out of the shape-cached buffers before the next block reuses them.
        all_query_projs.append(query_proj_buf.numpy().copy())   # [1, proj_dim]
        all_doc_projs.append(doc_proj_buf.numpy().copy())       # [num_docs, proj_dim]
        block_weights.append(float(block_weight_buf.numpy().reshape(-1)[0]))

    # ── Assemble the per-block projections with Reranker_Concat ──────────
    doc_projs   = _concat_fold(
        sess_concat, bind_concat, concat_in, concat_out, all_doc_projs)
    query_projs = _concat_fold(
        sess_concat, bind_concat, concat_in, concat_out, all_query_projs)

    # ── Final scoring: block weights applied in-graph, then cosine ───────
    block_weights_ort = create_ort_with_numpy(
        np.asarray(block_weights, dtype=in_dtype_Score[1]), device_type, DEVICE_ID)
    query_projs_ort   = create_ort_with_numpy(query_projs, device_type, DEVICE_ID)
    doc_projs_ort     = create_ort_with_numpy(doc_projs,   device_type, DEVICE_ID)

    score_buf, _ = _score_on_device(
        query_projs_ort, block_weights_ort, doc_projs_ort, doc_projs.shape[0])
    return score_buf.numpy().copy()


def rerank(query, documents, instruction=None, top_n=None):
    """Rerank documents by relevance to query.

    Single-batch when all docs fit in MAX_SEQ_LEN, else block-wise (mirrors
    official Jina rerank):
        1. Truncate query and docs individually
        2. Pack docs into blocks that fit max_seq_len
        3. Score each block (Score emits its weight)
        4. Weight block query embeddings by max normalized block score
        5. Sum the weighted query embeddings across blocks (in-graph)
        6. Recompute final cosine scores over all document embeddings

    Returns a list of {index, document, relevance_score} dicts, sorted
    descending by relevance.
    """
    query_trunc, docs_trunc, doc_lengths, query_length = _truncate_texts(
        query, documents)

    blocks = _pack_blocks(docs_trunc, doc_lengths, query_length)

    if len(blocks) == 1:
        # Single block (common case): lone query row ⇒ plain cosine, so the
        # block scores are the final scores.
        final_scores = _score_single_block(query_trunc, blocks[0], instruction)
    else:
        final_scores = _score_multi_block(query_trunc, blocks, instruction)

    # Sort descending
    order = np.argsort(final_scores)[::-1]

    if top_n is None:
        top_n = len(documents)
    else:
        top_n = min(top_n, len(documents))

    results = [
        {
            'index':           int(order[i]),
            'document':        documents[int(order[i])],
            'relevance_score': float(final_scores[int(order[i])]),
        }
        for i in range(top_n)
    ]

    return results


# ══════════════════════════════════════════════════════════════════════════════
# DEMO INFERENCE
# ══════════════════════════════════════════════════════════════════════════════

query = "Organic skincare products for sensitive skin"
documents = [
    "Organic skincare for sensitive skin with aloe vera and chamomile.",
    "New makeup products for a natural look, including foundations and lip colors.",
    "Bio-degradable cleaning products for eco-friendly households.",
    "Recycled paper products for office and home use.",
    "Luxury bath and body: high-end soaps, lotions, and fragrances.",
]

print(f'\n  Query: {query}')

start_time = time.time()
results = rerank(query, documents)
inference_time = time.time() - start_time

# ── Display ranking results ──────────────────────────────────────────────
print(f'\n{"─" * 60}')
print(f'  Ranking Results')
print(f'{"─" * 60}')
for rank, r in enumerate(results, 1):
    print(f'  #{rank}  [doc {r["index"]}]  score={r["relevance_score"]:+.6f}')
    print(f'        {r["document"][:70]}')
print(f'{"─" * 60}')
print(f'  Time Cost: {inference_time:.3f} seconds')
print(f'{"─" * 60}')
