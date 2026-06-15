"""
Jina Embeddings v5 — ONNX Inference Script
============================================
Pure inference using the exported Embed+LoRA and Main ONNX models.

Architecture at runtime:
    1. Embed+LoRA ONNX: token embedding + task-selected LoRA tensors (via int32 task_index)
    2. Main ONNX: shared backbone that accepts hidden_states + LoRA tensors as inputs

Task index mapping (int32 input to Embed+LoRA):
    0 = classification
    1 = clustering
    2 = retrieval
    3 = text-matching
"""

import json
import os
import time

import numpy as np
import onnxruntime
from transformers import AutoConfig, AutoTokenizer


# ══════════════════════════════════════════════════════════════════════════════
# PATHS
# ══════════════════════════════════════════════════════════════════════════════

MODEL_PATH              = r"/home/DakeQQ/Downloads/jina-embeddings-v5-text-small"                       # Path to the pretrained Jina v5 model directory
ONNX_OUTPUT_DIR         = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Jina_Optimized")    # Directory containing exported ONNX files
onnx_model_EmbedLoRA    = os.path.join(ONNX_OUTPUT_DIR, 'Embedding_Embed_LoRA.onnx')
onnx_model_Main         = os.path.join(ONNX_OUTPUT_DIR, 'Embedding_Main.onnx')


# ══════════════════════════════════════════════════════════════════════════════
# RUNTIME CONFIG
# ══════════════════════════════════════════════════════════════════════════════

MAX_SEQ_LEN             = 8192              # Maximum token sequence length. Keep the same value as exported model.
ORT_LOG                 = False             # Enable verbose ONNX Runtime logging
ORT_FP16                = False             # Use FP16 optimizations in ONNX Runtime session
ORT_Accelerate_Providers = []               # E.g. ['CUDAExecutionProvider'] or ['DmlExecutionProvider']; empty = CPU only
MAX_THREADS             = 0                 # ORT inter/intra-op thread count; 0 = let ORT decide
DEVICE_ID               = 0                 # GPU device index for CUDA/DML providers


# ══════════════════════════════════════════════════════════════════════════════
# DEMO INPUTS
# ══════════════════════════════════════════════════════════════════════════════

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
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

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
# ORT HELPER FUNCTIONS
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

# Separate session options for EmbedLoRA models to prevent constant-folding
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
# LOAD MODEL CONFIG
# ══════════════════════════════════════════════════════════════════════════════

model_config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
task_names = list(model_config.task_names)
max_seq_len = resolve_max_seq_len(model_config)
hidden_size = model_config.hidden_size


# ══════════════════════════════════════════════════════════════════════════════
# LOAD ONNX SESSIONS
# ══════════════════════════════════════════════════════════════════════════════

print('Loading ONNX models ... this may take a moment.')

# --- Embed + LoRA ---
# Use packed_settings_lora to avoid constant folding the dequant ops
ort_session_EmbedLoRA = create_session(onnx_model_EmbedLoRA, **packed_settings_lora)
in_name_EmbedLoRA     = get_in_names(ort_session_EmbedLoRA)
out_name_EmbedLoRA    = get_out_names(ort_session_EmbedLoRA)

# --- Main ---
ort_session_Main = create_session(onnx_model_Main, **packed_settings)
binding_Main     = ort_session_Main.io_binding()
in_name_Main     = get_in_names(ort_session_Main)
out_name_Main    = get_out_names(ort_session_Main)

# --- IOBinding for EmbedLoRA ---
binding_EmbedLoRA = ort_session_EmbedLoRA.io_binding()

print(f"\nUsable Providers: {ort_session_EmbedLoRA.get_providers()}")
print(f"Main model inputs: {in_name_Main}")


# ══════════════════════════════════════════════════════════════════════════════
# TASK CONFIG & BUFFER CACHE
# ══════════════════════════════════════════════════════════════════════════════

task_index_map = {task: idx for idx, task in enumerate(task_names)}

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
