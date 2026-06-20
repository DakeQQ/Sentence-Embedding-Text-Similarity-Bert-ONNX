import os
import time

import numpy as np
import onnxruntime
from onnxruntime.capi import _pybind_state as C
from transformers import AutoTokenizer


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

MODEL_DIR                  = r'/home/DakeQQ/Downloads/jina-reranker-v3'
ONNX_DIR                   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Reranker_Optimized")

onnx_model_Embed           = os.path.join(ONNX_DIR, 'Reranker_Embed.onnx')
onnx_model_Main            = os.path.join(ONNX_DIR, 'Reranker_Main.onnx')
onnx_model_Concat          = os.path.join(ONNX_DIR, 'Reranker_Concat.onnx')
onnx_model_Score           = os.path.join(ONNX_DIR, 'Reranker_Score.onnx')

# Special token IDs
DOC_EMBED_TOKEN_ID         = 151670         # <|embed_token|>
QUERY_EMBED_TOKEN_ID       = 151671         # <|rerank_token|>

# Runtime config
ORT_LOG                    = False
ORT_FP16                   = False          # FP16 ORT optimizations (needs ARM64-v8.2a+ for CPU)
ORT_Accelerate_Providers   = []             # ['CUDAExecutionProvider', 'DmlExecutionProvider', 'OpenVINOExecutionProvider']
MAX_THREADS                = 0              # 0 = auto
DEVICE_ID                  = 0

# Reranker config
MAX_QUERY_LENGTH           = 1024           # Max tokens per query (for truncation in block mode)
MAX_DOC_LENGTH             = 4096           # Max tokens per document (for truncation in block mode)
BLOCK_SIZE                 = 128            # Max documents per block in block-wise mode

SPECIAL_TOKENS = {
    "query_embed_token": "<|rerank_token|>",
    "doc_embed_token":   "<|embed_token|>",
}


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
    """Build the chat-template prompt for Jina Reranker v3 (mirrors official modeling.py)."""
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


# Map ORT NodeArg type strings onto numpy dtypes so every IO-binding buffer
# matches the models' exported precision (fp32 / fp16) — the same script serves
# either export.
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

print('Loading ONNX sessions ...')

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
# Dtypes read from each model's I/O signature so every buffer below adopts the
# models' own precision (fp32 / fp16).
in_dtype_Embed        = get_in_dtypes(ort_session_Embed)
out_dtype_Embed       = get_out_dtypes(ort_session_Embed)
in_dtype_Main         = get_in_dtypes(ort_session_Main)
out_dtype_Main        = get_out_dtypes(ort_session_Main)
in_dtype_Score        = get_in_dtypes(ort_session_Score)
out_dtype_Score       = get_out_dtypes(ort_session_Score)

# ══════════════════════════════════════════════════════════════════════════════
# MODEL METADATA  (read from the Reranker_Score ONNX custom metadata)
# ══════════════════════════════════════════════════════════════════════════════
#   Config is baked into Reranker_Score.onnx at export time, so the runtime
#   reads it from the ONNX metadata (no AutoConfig / config.json). max_seq_len
#   is already clamped to min(8192, max_position_embeddings).

score_metadata = ort_session_Score.get_modelmeta().custom_metadata_map

hidden_size    = int(score_metadata['hidden_size'])
max_seq_len    = int(score_metadata['max_seq_len'])
proj_dim       = int(score_metadata['proj_dim'])     # 512 for jina-reranker-v3


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

# Cached unit block-weight [1.0] for single-block / per-block scoring: a lone
# query row weighted by 1.0 passes through unscaled ⇒ plain cosine. Uploaded
# once and reused.
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

    The Embed output is bound directly as Main's input and Main's outputs are
    returned as-is so they can be bound straight into Reranker_Score — no numpy
    round-trip. Special-token extraction + projection happen inside Main;
    scoring is deferred to Reranker_Score.

    Returns (query_proj_buf [1, proj_dim], doc_proj_buf [num_docs, proj_dim] —
    both on device, num_docs).
    """
    batch_size, seq_len = input_ids_np.shape
    num_docs = int(np.count_nonzero(input_ids_np == DOC_EMBED_TOKEN_ID))

    # input_ids shared by both graphs (Main needs it for token extraction).
    input_ids_ort = create_ort_with_numpy(input_ids_np, device_type, DEVICE_ID)

    # Embed: input_ids → hidden_states (on device, reused buffer).
    hidden_buf = get_cached_buffer(
        'hidden_states', (batch_size, seq_len, hidden_size), out_dtype_Embed[0])

    binding_Embed.bind_ortvalue_input(in_name_Embed[0], input_ids_ort)
    binding_Embed.bind_ortvalue_output(out_name_Embed[0], hidden_buf)
    run(ort_session_Embed, binding_Embed)

    # Main: backbone + in-graph extraction + projector. hidden_buf is bound
    # directly as Main's input (zero-copy); input_ids_ort is reused.
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

    On the single-block hot path the query / doc projections are the very
    OrtValues Main just wrote (no numpy round-trip). Score weights the query
    rows by block_weights, sums them into the combined query, cosine-scores the
    docs, AND emits the block weight (max normalised score) — all in-graph.

    Returns (score_buf [total_docs], block_weight_buf [1]), both shape-cached
    device buffers (the CPU EP honours pre-sized output buffers).
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
    """Truncate query and documents individually to their token limits.

    Returns (query, docs, doc_lengths, query_length).
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


# Concat session, lazily loaded for the multi-block path only (folds the
# per-block projections); the common single-block case never loads it.
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
    """Concatenate a list of [*, proj_dim] arrays via the pairwise Concat graph.

    ONNX graphs have a fixed input count, so the variable-length concat is done
    by folding the pairwise graph across the list (accumulator + next, axis 0).
    Returns the [sum(rows), proj_dim] array; multi-block cold path only.
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

    Mirrors the official Jina capacity heuristic: the query is counted twice —
    it appears in both the prompt body and the trailing <query> tag.
    """
    # query counted twice (see docstring)
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
    """Hot path: a single block scored end-to-end on the device.

    One query row ⇒ the in-graph weighted sum is the query itself ⇒ plain
    cosine, so the block scores are the final scores.
    """
    tokens = _tokenize_block(query, block_docs, instruction)
    query_proj_buf, doc_proj_buf, num_docs = _embed_and_main(tokens)
    score_buf, _ = _score_on_device(
        query_proj_buf, ones1_ort, doc_proj_buf, num_docs)
    return score_buf.numpy().copy()


def _score_multi_block(query, blocks, instruction):
    """Cold path: documents span several blocks (mirrors official Jina).

    Each block is scored on device so Score emits its block weight; Concat folds
    the per-block projections, then the same Score graph combines the
    block-weighted query rows and rescores cosine over every document.
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

    # Assemble the per-block projections with Concat.
    doc_projs   = _concat_fold(
        sess_concat, bind_concat, concat_in, concat_out, all_doc_projs)
    query_projs = _concat_fold(
        sess_concat, bind_concat, concat_in, concat_out, all_query_projs)

    # Final scoring: Score scales each block's query row by its weight, sums
    # them into the combined query, and cosine-scores all docs. Cosine is
    # scale-invariant, so this weighted SUM matches the official weighted MEAN.
    block_weights_ort = create_ort_with_numpy(
        np.asarray(block_weights, dtype=in_dtype_Score[1]), device_type, DEVICE_ID)
    query_projs_ort   = create_ort_with_numpy(query_projs, device_type, DEVICE_ID)
    doc_projs_ort     = create_ort_with_numpy(doc_projs,   device_type, DEVICE_ID)

    score_buf, _ = _score_on_device(
        query_projs_ort, block_weights_ort, doc_projs_ort, doc_projs.shape[0])
    return score_buf.numpy().copy()


def rerank(query, documents, instruction=None, top_n=None):
    """Rerank documents by relevance to query.

    Single-block when all docs fit in max_seq_len, else block-wise (mirrors
    official Jina): truncate → pack into blocks → score each block (Score emits
    its weight) → weight and sum the block query embeddings in-graph →
    recompute cosine over all document embeddings. Returns dicts sorted by
    descending relevance (top_n if given).
    """
    query_trunc, docs_trunc, doc_lengths, query_length = _truncate_texts(
        query, documents)

    blocks = _pack_blocks(docs_trunc, doc_lengths, query_length)

    if len(blocks) == 1:
        # Single block (common case): the block scores are the final scores.
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

# Display ranking results.
print(f'\n{"─" * 60}')
print(f'  Ranking Results')
print(f'{"─" * 60}')
for rank, r in enumerate(results, 1):
    print(f'  #{rank}  [doc {r["index"]}]  score={r["relevance_score"]:+.6f}')
    print(f'        {r["document"][:70]}')
print(f'{"─" * 60}')
print(f'  Time Cost: {inference_time:.3f} seconds')
print(f'{"─" * 60}')
