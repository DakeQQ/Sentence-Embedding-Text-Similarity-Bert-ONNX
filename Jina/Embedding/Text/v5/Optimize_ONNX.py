import os
import gc
import glob
import onnx
import onnx.version_converter
from pathlib import Path
from onnxslim import slim
from onnxruntime.transformers.optimizer import optimize_model
from onnxruntime.quantization import (
    matmul_nbits_quantizer,  # onnxruntime >= 1.22.0
    quant_utils
)


# ==============================================================================
# Path Settings
# ==============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
original_folder_path = os.path.join(SCRIPT_DIR, 'Jina_ONNX')
quanted_folder_path = os.path.join(SCRIPT_DIR, 'Jina_Optimized')
os.makedirs(quanted_folder_path, exist_ok=True)


# ==============================================================================
# General Settings
# ==============================================================================
use_openvino     = False                 # Set True for OpenVINO optimization.
SAVE_TWO_PARTS   = False                 # If True, save the model into 2 parts.
upgrade_opset    = 0                     # Optional opset upgrade. Set 0 to disable.


# ==============================================================================
# Model-Dtype Mapping
# ==============================================================================
MODEL_DTYPE = {
    "Embedding_Embed_LoRA": "int4",      # [int4, float32, float16]
    "Embedding_Main":       "int8",      # [int8, float32, float16]
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
block_size       = 32                    # [16, 32, 64, 128, 256]; smaller => more accuracy.
accuracy_level   = 4                     # 0:default, 1:fp32, 2:fp16, 3:bf16, 4:int8
quant_symmetric  = False                 # False may yield more accuracy.
nodes_to_exclude = None                  # Example: ["/layers.0/mlp/down_proj/MatMul"]


# ==============================================================================
# Helper Functions
# ==============================================================================
def _is_transformer_block(name):
    return name in ("LLM_Main", "LLM_Vision", "Embedding_Main")


def _is_embed_block(name):
    return name in ("LLM_Embed", "Embedding_Embed_LoRA")


def _opt_level(name):
    return 1 if use_openvino else 2


def _num_heads(name):
    if "small" in name:
        return 16
    elif "nano" in name:
        return 12
    return 0


def _hidden_size(name):
    if "small" in name:
        return 1024
    elif "nano" in name:
        return 768
    return 0


# ==============================================================================
# Core Processing Function
# ==============================================================================
def process_single_model(
    model_path,
    quanted_model_path,
    model_name,
    bits,
    block_size,
    quant_int4_flag,
    quant_int8_flag,
    quant_float16_flag,
    keep_io_flag,
    op_block_list,
):
    """Process a single ONNX file: quantize / optimize / slim."""
    be_optimized = False

    # ------------------------------------------------------------------
    # Branch 1: Integer quantization (int4 / int8)
    # ------------------------------------------------------------------
    if (quant_int4_flag or quant_int8_flag) and (_is_embed_block(model_name) or _is_transformer_block(model_name)):
        if _is_embed_block(model_name):
            op_types = ["Gather"]
            quant_axes = [1]
            algo = "DEFAULT"
        else:
            op_types = ["MatMul"]
            quant_axes = [0]
            algo = algorithm_copy

        model = quant_utils.load_model_with_shape_infer(Path(model_path))
        axes_tuple = tuple((op_types[i], quant_axes[i]) for i in range(len(op_types)))

        if algo == "RTN":
            quant_config = matmul_nbits_quantizer.RTNWeightOnlyQuantConfig(
                quant_format=quant_utils.QuantFormat.QOperator,
                op_types_to_quantize=tuple(op_types),
            )
        elif algo == "HQQ":
            quant_config = matmul_nbits_quantizer.HQQWeightOnlyQuantConfig(
                bits=bits,
                block_size=block_size,
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
                block_size=block_size,
                is_symmetric=quant_symmetric,
                accuracy_level=accuracy_level,
                quant_format=quant_utils.QuantFormat.QOperator,
                op_types_to_quantize=tuple(op_types),
                quant_axes=axes_tuple,
            )

        quant_config.bits = bits
        quant = matmul_nbits_quantizer.MatMulNBitsQuantizer(
            model,
            block_size=block_size,
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
            only_onnxruntime=use_openvino,
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
            only_onnxruntime=use_openvino,
        )
        model.save_model_to_file(quanted_model_path, use_external_data_format=SAVE_TWO_PARTS)

    # ------------------------------------------------------------------
    # Post-quantization optimization pass
    # ------------------------------------------------------------------
    if not be_optimized:
        print("Running additional ONNX Runtime optimization on quantized model...")
        model = optimize_model(
            quanted_model_path,
            use_gpu=False,
            opt_level=_opt_level(model_name),
            num_heads=_num_heads(model_name),
            hidden_size=_hidden_size(model_name),
            verbose=False,
            model_type='bert',
            only_onnxruntime=use_openvino,
        )
        model.save_model_to_file(quanted_model_path, use_external_data_format=SAVE_TWO_PARTS)

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
    # Optional opset upgrade
    # ------------------------------------------------------------------
    if upgrade_opset > 0:
        print(f"Upgrading Opset to {upgrade_opset}...")
        try:
            m = onnx.load(quanted_model_path)
            converted_model = onnx.version_converter.convert_version(m, upgrade_opset)
            onnx.save(converted_model, quanted_model_path, save_as_external_data=SAVE_TWO_PARTS)
            del m, converted_model
            gc.collect()
        except Exception as e:
            print(f"Could not upgrade opset due to an error: {e}. Saving model with original opset.")
            m = onnx.load(quanted_model_path)
            onnx.save(m, quanted_model_path, save_as_external_data=SAVE_TWO_PARTS)
            del m
            gc.collect()
    else:
        m = onnx.load(quanted_model_path)
        onnx.save(m, quanted_model_path, save_as_external_data=SAVE_TWO_PARTS)
        del m
        gc.collect()


# ==============================================================================
# Main Processing Loop
# ==============================================================================
algorithm_copy = algorithm

for model_name, target_dtype in MODEL_DTYPE.items():
    print(f"\n--- Processing model: {model_name} ---")

    # Reset per-iteration quantization flags
    quant_int4 = (target_dtype == "int4")
    quant_int8 = (target_dtype == "int8")
    quant_float16 = (target_dtype == "float16")

    print(f"Selected target dtype for {model_name}: {target_dtype}")
    print(f"quant_int8={quant_int8}, quant_float16={quant_float16}, keep_io_dtype={keep_io_dtype}")

    model_path = os.path.join(original_folder_path, f"{model_name}.onnx")
    quanted_model_path = os.path.join(quanted_folder_path, f"{model_name}.onnx")

    if not os.path.exists(model_path):
        print(f"Warning: Model file not found at {model_path}. Skipping.")
        continue

    process_single_model(
        model_path, quanted_model_path, model_name,
        bits, block_size, quant_int4, quant_int8,
        quant_float16, keep_io_dtype, fp16_op_block_list,
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
