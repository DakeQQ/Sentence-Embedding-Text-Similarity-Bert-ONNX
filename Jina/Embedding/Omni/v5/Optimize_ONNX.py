import os
import gc
import glob
import json
import site
import numpy as np
import onnx
import onnx.version_converter
from onnx import helper, numpy_helper
from pathlib import Path
from onnxslim import slim
from onnxruntime.transformers.optimizer import optimize_model
from onnxruntime.quantization import (
    matmul_nbits_quantizer,  # onnxruntime >= 1.22.0
    quant_utils,
    QuantType,
    quantize_dynamic,
)


# ==============================================================================
# Path Settings
# ==============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# The omni export writes its 7 shared graphs into Jina_ONNX_Omni/shared; optimize them
# into a parallel Jina_ONNX_Omni_Optimized/shared (point the inference OUTPUT_ROOT there).
original_folder_path = os.path.join(SCRIPT_DIR, 'Jina_Omni_ONNX')
quanted_folder_path = os.path.join(SCRIPT_DIR, 'Jina_Omni_Optimized')
os.makedirs(quanted_folder_path, exist_ok=True)

# Model directory (config.json) — used only for the per-tower attention-fusion hints below.
MODEL_DIR = '/home/DakeQQ/Downloads/jina-embeddings-v5-omni-nano'


# ==============================================================================
# General Settings
# ==============================================================================
use_openvino     = False                 # Set True for OpenVINO optimization.
SAVE_TWO_PARTS   = False                 # If True, save the model into 2 parts.
upgrade_opset    = 0                     # Optional opset upgrade. Set 0 to disable.

# The omni towers are pre-fused/folded custom graphs (GQA-broadcast attention, flip-RoPE,
# affine-free norms) that the Python BERT fusion passes don't match and can crash on
# (onnxruntime simplified-layernorm bug). Run the ORT C++ graph optimizer only.
ONLY_ONNXRUNTIME = True


# ==============================================================================
# Model-Dtype Mapping
# ==============================================================================
# 'q8_dynamic' = INT8 dynamic per-channel quantization (Q4.py vision recipe). int4 noticeably
#                hurts the vision tower, so it gets int8 + the TransposeOptimizer disabled.
# The LoRA provider + combine routers now flow through the same optimize/quant pipeline as every
# other graph (no more verbatim copy). They default to float32 (optimize-only) so the LoRA
# provider's fp32 constant outputs and the combine routers' fp32 routing still match the consuming
# graphs' fp32 inputs; their runtime metadata_props are re-attached after processing (see
# _carry_over_metadata_props). Override per graph freely, but note fp16/int* on the LoRA/combine
# graphs changes their output dtype and needs matching dtypes at the consuming graph inputs.
MODEL_DTYPE = {
    "Omni_Embed":         "int4",        # token-embedding table        [int4, int8, float32, float16]
    "Omni_Vision":        "q8_dynamic",  # vision tower  (Qwen3VL)      INT8 dynamic per-channel (Q4.py)
    "Omni_Audio":         "int4",        # audio tower   (Qwen2.5-Omni) [int4, int8, float32, float16]
    "Omni_Main":          "int4",        # text backbone (EuroBERT)     [int4, int8, float32, float16]
    "Omni_LoRA":          "int4",        # LoRA/merger/projector provider (constant outputs)  [float32, float16, int4, int8]
    "Omni_Combine_Image": "float32",     # tiny routing graph + runtime metadata              [float32, float16, int4, int8]
    "Omni_Combine_Audio": "float32",     # tiny routing graph + runtime metadata              [float32, float16, int4, int8]
}


# ==============================================================================
# Manual Quantization Settings
# ==============================================================================
quant_int4       = False                 # Quant to int4 (not used by auto settings).
quant_int8       = False                 # Global default, overridden per model.
quant_float16    = False                 # Global default, overridden per model.
keep_io_dtype    = False                 # Must be True for mixed-precision.
fp16_op_block_list = [
    'DynamicQuantizeLinear',
    'DequantizeLinear',
    'DynamicQuantizeMatMul',
    'Range',
    'MatMulIntegerToFloat',
]


# ==============================================================================
# Int4 matmul_nbits_quantizer Settings
# ==============================================================================
algorithm        = "k_quant"             # ["DEFAULT", "RTN", "HQQ", "k_quant"]
bits             = 4                     # [4, 8]; 8 is not recommended.
block_size       = 16                    # [16, 32, 64, 128, 256]; smaller => more accuracy.
accuracy_level   = 4                     # 0:default, 1:fp32, 2:fp16, 3:bf16, 4:int8
quant_symmetric  = False                 # False may yield more accuracy.
nodes_to_exclude = None                  # Example: ["/layers.0/mlp/down_proj/MatMul"]


# ==============================================================================
# Architecture dims (per-tower attention-fusion hints read from the model config, so the
# script adapts to any model size instead of hardcoding nano/small literals)
# ==============================================================================
def _load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


_CONFIG = _load_json(os.path.join(MODEL_DIR, 'config.json'))
_VCFG = _CONFIG['vision_config']
_TCFG = _CONFIG['text_config']
_ACFG = _CONFIG['audio_config']

# model_name -> (num_attention_heads, hidden_size) handed to optimize_model. Every other
# graph (embed lookup / LoRA provider / combine routers) has no attention -> (0, 0).
_MODEL_DIMS = {
    "Omni_Vision": (_VCFG["num_heads"],               _VCFG["hidden_size"]),
    "Omni_Audio":  (_ACFG["encoder_attention_heads"], _ACFG["d_model"]),
    "Omni_Main":   (_TCFG["num_attention_heads"],     _TCFG["hidden_size"]),
}


# ==============================================================================
# Helper Functions
# ==============================================================================
def _is_transformer_block(name):
    return name in ("Omni_Vision", "Omni_Audio", "Omni_Main")


def _is_embed_block(name):
    return name == "Omni_Embed" or name == "Omni_LoRA"


def _is_vision_block(name):
    return name == "Omni_Vision"


def _opt_level(name):
    return 1 if use_openvino else 2


def _num_heads(name):
    return _MODEL_DIMS.get(name, (0, 0))[0]


def _hidden_size(name):
    return _MODEL_DIMS.get(name, (0, 0))[1]


def _carry_over_metadata_props(src_onnx, dst_model):
    """Re-attach the source graph's metadata_props to the processed model.

    optimize_model/slim drop metadata_props; the combine routers carry the runtime metadata the
    inference reads back, so copy any source metadata onto the optimized model before saving.
    No-op for graphs without metadata (the towers / embed / LoRA provider carry none).
    """
    src = onnx.load(src_onnx, load_external_data=False)
    if not src.metadata_props:
        return
    existing = {p.key for p in dst_model.metadata_props}
    for prop in src.metadata_props:
        if prop.key in existing:
            continue
        entry = dst_model.metadata_props.add()
        entry.key = prop.key
        entry.value = prop.value


def _convert_gemm_to_matmul(model):
    """Rewrite constant-weight Gemm (nn.Linear exported in 2D) as MatMul (+ Add) so its weight
    becomes a MatMul B initializer the MatMulNBits quantizer can quantize. The vision tower's
    linears export as 2D Gemm (MatMulNBits ignores Gemm), whereas Main/Audio already use MatMul.
    Dynamic-weight Gemm/MatMul (e.g. the per-task merger/projector input weights) is left as-is.
    """
    graph = model.graph
    inits = {i.name: i for i in graph.initializer}
    new_nodes = []
    converted = 0
    for node in graph.node:
        if node.op_type != "Gemm" or len(node.input) < 2 or node.input[1] not in inits:
            new_nodes.append(node)
            continue
        attrs = {a.name: a for a in node.attribute}
        alpha = attrs["alpha"].f if "alpha" in attrs else 1.0
        beta = attrs["beta"].f if "beta" in attrs else 1.0
        trans_a = attrs["transA"].i if "transA" in attrs else 0
        trans_b = attrs["transB"].i if "transB" in attrs else 0
        if trans_a:  # nn.Linear never transposes the dynamic activation; bail out safely
            new_nodes.append(node)
            continue
        weight = numpy_helper.to_array(inits[node.input[1]])
        if trans_b:
            weight = weight.T
        if alpha != 1.0:
            weight = weight * np.asarray(alpha, dtype=weight.dtype)
        w_name = node.input[1] + "_to_matmul"
        graph.initializer.append(numpy_helper.from_array(np.ascontiguousarray(weight), w_name))
        has_bias = len(node.input) >= 3 and node.input[2]
        mm_out = node.output[0] if not has_bias else (node.name or w_name) + "_mm"
        new_nodes.append(helper.make_node(
            "MatMul", [node.input[0], w_name], [mm_out], name=(node.name or w_name) + "_MatMul"))
        if has_bias:
            bias = node.input[2]
            if beta != 1.0 and bias in inits:
                bias_val = numpy_helper.to_array(inits[bias])
                bias = node.input[2] + "_beta"
                graph.initializer.append(numpy_helper.from_array(
                    np.ascontiguousarray(bias_val * np.asarray(beta, dtype=bias_val.dtype)), bias))
            new_nodes.append(helper.make_node(
                "Add", [mm_out, bias], [node.output[0]], name=(node.name or w_name) + "_Add"))
        converted += 1
    if converted:
        del graph.node[:]
        graph.node.extend(new_nodes)
        # Drop initializers orphaned by the rewrite (the original transposed Gemm weights).
        used = {inp for n in graph.node for inp in n.input}
        used.update(o.name for o in graph.output)
        keep = [i for i in graph.initializer if i.name in used]
        if len(keep) != len(graph.initializer):
            del graph.initializer[:]
            graph.initializer.extend(keep)
    print(f"  Converted {converted} constant-weight Gemm -> MatMul for quantization.")
    return model


def _patch_optimizer_file(disable_transpose):
    """Toggle disabling ORT's TransposeOptimizer in onnxruntime's optimizer.py (mirrors Q4.py).

    The vision tower's layout transposes get reordered by the TransposeOptimizer in a way that
    degrades vision accuracy/performance, so it is disabled while the vision graph is optimized
    and restored immediately afterwards. optimize_model exposes no disabled_optimizers argument,
    hence the in-place source toggle.
    """
    optimizer_path = os.path.join(site.getsitepackages()[-1], "onnxruntime/transformers/optimizer.py")
    if disable_transpose:
        old = "disabled_optimizers=disabled_optimizers"
        new = 'disabled_optimizers=["TransposeOptimizer", "TransposeOptimizer_CPUExecutionProvider"]'
    else:
        old = 'disabled_optimizers=["TransposeOptimizer", "TransposeOptimizer_CPUExecutionProvider"]'
        new = "disabled_optimizers=disabled_optimizers"
    try:
        with open(optimizer_path, "r", encoding="utf-8") as f:
            content = f.read()
        if old in content:
            with open(optimizer_path, "w", encoding="utf-8") as f:
                f.write(content.replace(old, new))
            print(f"  {'Disabled' if disable_transpose else 'Restored'} TransposeOptimizer ({optimizer_path})")
    except OSError as e:
        print(f"  Warning: could not patch optimizer.py ({e}); continuing without it.")


def process_vision_quantization(src_path, dst_path):
    """INT8 dynamic per-channel quantization for the vision tower (Q4.py recipe). int8 preserves
    more vision accuracy than int4. quantize_dynamic only quantizes MatMul (IntegerOps registry has
    no Gemm), so the vision tower's constant Gemm linears are first rewritten to MatMul.
    """
    model = quant_utils.load_model_with_shape_infer(Path(src_path))
    _convert_gemm_to_matmul(model)
    print("Applying dynamic INT8 quantization for the vision model...")
    quantize_dynamic(
        model_input=model,
        model_output=dst_path,
        per_channel=True,
        reduce_range=False,
        weight_type=QuantType.QUInt8,
        extra_options={
            "ActivationSymmetric": False,
            "WeightSymmetric": False,
            "EnableSubgraph": True,
            "ForceQuantizeNoInputCheck": False,
            "MatMulConstBOnly": True,
        },
        nodes_to_exclude=None,
        use_external_data_format=SAVE_TWO_PARTS,
    )
    del model
    gc.collect()


# ==============================================================================
# Core Processing Function
# ==============================================================================
def process_single_model(
    model_path,
    quanted_model_path,
    model_name,
    algorithm,
    bits,
    block_size,
    quant_int4_flag,
    quant_int8_flag,
    quant_float16_flag,
    quant_q8_vision_flag,
    keep_io_flag,
    op_block_list,
):
    """Process a single ONNX file: quantize / optimize / slim."""
    be_optimized = False

    # ------------------------------------------------------------------
    # Branch 0: Vision INT8 dynamic per-channel quantization (Q4.py recipe)
    # ------------------------------------------------------------------
    if quant_q8_vision_flag:
        process_vision_quantization(model_path, quanted_model_path)

    # ------------------------------------------------------------------
    # Branch 1: Integer quantization (int4 / int8)
    # ------------------------------------------------------------------
    elif (quant_int4_flag or quant_int8_flag) and (_is_embed_block(model_name) or _is_transformer_block(model_name)):
        if _is_embed_block(model_name):
            op_types = ["Gather"]
            quant_axes = [1]
            algo = "DEFAULT"
            blk_size = 16
            bit = 4
        else:
            op_types = ["MatMul"]
            quant_axes = [0]
            if quant_int8_flag:
                algo = "DEFAULT"
                bit = 8
            else:
                algo = algorithm
                bit = bits
            blk_size = block_size




        model = quant_utils.load_model_with_shape_infer(Path(model_path))
        # Vision's tower linears export as 2D Gemm (MatMulNBits only sees MatMul); rewrite the
        # constant-weight Gemm to MatMul so they get int4-quantized like Main/Audio.
        if _is_transformer_block(model_name):
            _convert_gemm_to_matmul(model)
        axes_tuple = tuple((op_types[i], quant_axes[i]) for i in range(len(op_types)))

        if algo == "RTN":
            quant_config = matmul_nbits_quantizer.RTNWeightOnlyQuantConfig(
                quant_format=quant_utils.QuantFormat.QOperator,
                op_types_to_quantize=tuple(op_types),
            )
        elif algo == "HQQ":
            quant_config = matmul_nbits_quantizer.HQQWeightOnlyQuantConfig(
                bits=bit,
                block_size=blk_size,
                axis=quant_axes[0],
                quant_format=quant_utils.QuantFormat.QOperator,
                op_types_to_quantize=tuple(op_types),
                quant_axes=axes_tuple,
            )
        elif algo == "k_quant":
            quant_config = matmul_nbits_quantizer.KQuantWeightOnlyQuantConfig(
                quant_format=quant_utils.QuantFormat.QOperator,
                op_types_to_quantize=tuple(op_types),
            )
        else:
            quant_config = matmul_nbits_quantizer.DefaultWeightOnlyQuantConfig(
                block_size=blk_size,
                is_symmetric=quant_symmetric,
                accuracy_level=accuracy_level,
                quant_format=quant_utils.QuantFormat.QOperator,
                op_types_to_quantize=tuple(op_types),
                quant_axes=axes_tuple,
            )

        quant_config.bits = bit
        quant = matmul_nbits_quantizer.MatMulNBitsQuantizer(
            model,
            block_size=blk_size,
            is_symmetric=quant_symmetric,
            accuracy_level=accuracy_level,
            quant_format=quant_utils.QuantFormat.QOperator,
            op_types_to_quantize=tuple(op_types),
            quant_axes=axes_tuple,
            algo_config=quant_config,
            nodes_to_exclude=nodes_to_exclude,
        )
        quant.process()
        quant.model.save_model_to_file(quanted_model_path, True)

    # ------------------------------------------------------------------
    # Branch 2: Float16 conversion
    # ------------------------------------------------------------------
    elif quant_float16_flag:
        print("Optimizing model before Float16 conversion...")
        be_optimized = True
        model = optimize_model(
            model_path,
            use_gpu=False,
            opt_level=_opt_level(model_name),
            num_heads=_num_heads(model_name),
            hidden_size=_hidden_size(model_name),
            verbose=False,
            model_type='bert',
            only_onnxruntime=ONLY_ONNXRUNTIME,
        )
        print("Converting model to Float16...")
        model.convert_float_to_float16(
            keep_io_types=keep_io_flag,
            force_fp16_initializers=True,
            use_symbolic_shape_infer=True,
            max_finite_val=32767.0,
            op_block_list=op_block_list,
        )
        model.save_model_to_file(quanted_model_path, use_external_data_format=SAVE_TWO_PARTS)

    # ------------------------------------------------------------------
    # Branch 3: Float32 (optimize only, no quantization)
    # ------------------------------------------------------------------
    else:
        print("Target dtype is float32: optimizing without quantization...")
        be_optimized = True
        model = optimize_model(
            model_path,
            use_gpu=False,
            opt_level=_opt_level(model_name),
            num_heads=_num_heads(model_name),
            hidden_size=_hidden_size(model_name),
            verbose=False,
            model_type='bert',
            only_onnxruntime=ONLY_ONNXRUNTIME,
        )
        model.save_model_to_file(quanted_model_path, use_external_data_format=SAVE_TWO_PARTS)

    # ------------------------------------------------------------------
    # Post-quantization optimization pass
    # ------------------------------------------------------------------
    if not be_optimized:
        print("Running additional ONNX Runtime optimization on quantized model...")
        # Disable ORT's TransposeOptimizer for the vision tower: it reorders the tower's layout
        # transposes in a way that degrades vision accuracy/performance (mirrors Q4.py).
        disable_transpose = _is_vision_block(model_name)
        if disable_transpose:
            _patch_optimizer_file(disable_transpose=True)
        try:
            model = optimize_model(
                quanted_model_path,
                use_gpu=False,
                opt_level=_opt_level(model_name),
                num_heads=_num_heads(model_name),
                hidden_size=_hidden_size(model_name),
                verbose=False,
                model_type='bert',
                only_onnxruntime=ONLY_ONNXRUNTIME,
            )
            model.save_model_to_file(quanted_model_path, use_external_data_format=SAVE_TWO_PARTS)
        finally:
            if disable_transpose:
                _patch_optimizer_file(disable_transpose=False)

    # ------------------------------------------------------------------
    # Slim pass
    # ------------------------------------------------------------------
    slim(
        model=quanted_model_path,
        output_model=quanted_model_path,
        no_shape_infer=False,
        skip_fusion_patterns=False,
        no_constant_folding=False,
        save_as_external_data=SAVE_TWO_PARTS,
        verbose=False,
    )

    # ------------------------------------------------------------------
    # Final load/save: optional opset upgrade + metadata carry-over
    # ------------------------------------------------------------------
    # optimize_model/slim drop metadata_props, so re-attach the source graph's metadata (the
    # combine routers carry the runtime keys the inference reads back) before the final save.
    m = onnx.load(quanted_model_path)
    if upgrade_opset > 0:
        print(f"Upgrading Opset to {upgrade_opset}...")
        try:
            m = onnx.version_converter.convert_version(m, upgrade_opset)
        except Exception as e:
            print(f"Could not upgrade opset due to an error: {e}. Saving model with original opset.")
    _carry_over_metadata_props(model_path, m)
    onnx.save(m, quanted_model_path, save_as_external_data=SAVE_TWO_PARTS)
    del m
    gc.collect()


# ==============================================================================
# Main Processing Loop
# ==============================================================================

for model_name, target_dtype in MODEL_DTYPE.items():
    print(f"\n--- Processing model: {model_name} ---")

    model_path = os.path.join(original_folder_path, f"{model_name}.onnx")
    quanted_model_path = os.path.join(quanted_folder_path, f"{model_name}.onnx")

    if not os.path.exists(model_path):
        print(f"Warning: Model file not found at {model_path}. Skipping.")
        continue

    # Reset per-iteration quantization flags
    quant_int4 = (target_dtype == "int4")
    quant_int8 = (target_dtype == "int8")
    quant_float16 = (target_dtype == "float16")
    quant_q8_vision = (target_dtype == "q8_dynamic")

    print(f"Selected target dtype for {model_name}: {target_dtype}")
    print(f"quant_int8={quant_int8}, quant_float16={quant_float16}, "
          f"quant_q8_vision={quant_q8_vision}, keep_io_dtype={keep_io_dtype}")

    process_single_model(
        model_path, quanted_model_path, model_name, algorithm,
        bits, block_size, quant_int4, quant_int8,
        quant_float16, quant_q8_vision, keep_io_dtype, fp16_op_block_list,
    )


# ==============================================================================
# Cleanup
# ==============================================================================
print("Cleaning up temporary *.onnx.data files...")
pattern = os.path.join(quanted_folder_path, '*.onnx.data')
files_to_delete = glob.glob(pattern)
for file_path in files_to_delete:
    try:
        os.remove(file_path)
        print(f"Deleted {file_path}")
    except Exception as e:
        print(f"Error deleting {file_path}: {e}")

print("--- All models processed successfully! ---")
