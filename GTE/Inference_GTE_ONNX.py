import re
import time
import numpy as np
import onnxruntime


onnx_model_A = r"/home/DakeQQ/Downloads/GTE_Optimized/Model_GTE.onnx"                       # The exported onnx model save path.
vocab_path = '/home/DakeQQ/Downloads/nlp_gte_sentence-embedding_chinese-small/vocab.txt'    # Set the path where the GTE model vocab.txt stored.
sentence_1 = "吃完海鲜可以喝牛奶吗?"                                                            # The sentence for similarity test.
sentence_2 = "不可以，早晨喝牛奶不科学"                                                          # The sentence for similarity test.

ORT_Accelerate_Providers = ['CPUExecutionProvider']       # If you have accelerate devices for : ['CUDAExecutionProvider', 'TensorrtExecutionProvider', 'CoreMLExecutionProvider', 'DmlExecutionProvider', 'OpenVINOExecutionProvider', 'ROCMExecutionProvider', 'MIGraphXExecutionProvider', 'AzureExecutionProvider']
                                                          # else keep empty.
MAX_THREADS = 8                                           # Max CPU parallel threads.
DEVICE_ID = 0                                             # The GPU id, default to 0.
TOKEN_UNKNOWN = 100                                       # The model parameter, do not edit it.
TOKEN_BEGIN = 101                                         # The model parameter, do not edit it.
TOKEN_END = 102                                           # The model parameter, do not edit it.


if "OpenVINOExecutionProvider" in ORT_Accelerate_Providers:
    provider_options = [
        {
            'device_type': 'CPU',                         # [CPU, NPU, GPU, GPU.0, GPU.1]]
            'precision': 'ACCURACY',                      # [FP32, FP16, ACCURACY]
            'num_of_threads': MAX_THREADS,
            'num_streams': 1,
            'enable_opencl_throttling': True,
            'enable_qdq_optimizer': False                 # Enable it carefully
        }
    ]
elif "CUDAExecutionProvider" in ORT_Accelerate_Providers:
    provider_options = [
        {
            'device_id': DEVICE_ID,
            'gpu_mem_limit': 8 * 1024 * 1024 * 1024,      # 8 GB
            'arena_extend_strategy': 'kNextPowerOfTwo',
            'cudnn_conv_algo_search': 'EXHAUSTIVE',
            'cudnn_conv_use_max_workspace': '1',
            'do_copy_in_default_stream': '1',
            'cudnn_conv1d_pad_to_nc1d': '1',
            'enable_cuda_graph': '0',                     # Set to '0' to avoid potential errors when enabled.
            'use_tf32': '0'
        }
    ]
else:
    # Please config by yourself for others providers.
    provider_options = None


# Read the model vocab.
with open(vocab_path, 'r', encoding='utf-8') as file:
    vocab = file.readlines()
vocab = np.array([line.strip() for line in vocab], dtype=np.str_)


# For GTE Model
def tokenizer(input_string, max_input_words, is_dynamic):
    input_ids = np.zeros(max_input_words, dtype=np.int32)
    input_string = re.findall(r'[\u4e00-\u9fa5]|[a-zA-Z]+', input_string.lower())
    input_ids[0] = TOKEN_BEGIN
    full = max_input_words - 1
    ids_len = 1
    for i in input_string:
        indices = np.where(vocab == i)[0]
        if len(indices) > 0:
            input_ids[ids_len] = indices[0]
            ids_len += 1
            if ids_len == full:
                break
        else:
            for j in list(i):
                indices = np.where(vocab == j)[0]
                if len(indices) > 0:
                    input_ids[ids_len] = indices[0]
                else:
                    input_ids[ids_len] = TOKEN_UNKNOWN
                ids_len += 1
                if ids_len == full:
                    break
    input_ids[ids_len] = TOKEN_END
    if is_dynamic:
        input_ids = input_ids[:ids_len + 1]
    return input_ids.reshape(1, -1)


print("\nRun GTE model by ONNXRuntime.")
session_opts = onnxruntime.SessionOptions()
session_opts.log_severity_level = 3         # error level, it an adjustable value.
session_opts.inter_op_num_threads = 0       # Run different nodes with num_threads. Set 0 for auto.
session_opts.intra_op_num_threads = 0       # Under the node, execute the operators with num_threads. Set 0 for auto.
session_opts.enable_cpu_mem_arena = True    # True for execute speed; False for less memory usage.
session_opts.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
session_opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
session_opts.add_session_config_entry("session.intra_op.allow_spinning", "1")
session_opts.add_session_config_entry("session.inter_op.allow_spinning", "1")
session_opts.add_session_config_entry("session.set_denormal_as_zero", "1")

ort_session_A = onnxruntime.InferenceSession(onnx_model_A, sess_options=session_opts, providers=ORT_Accelerate_Providers, provider_options=provider_options)
print(f"\nUsable Providers: {ort_session_A.get_providers()}")
shape_value_in = ort_session_A._inputs_meta[0].shape[-1]
in_name_A0 = ort_session_A.get_inputs()[0].name
out_name_A0 = ort_session_A.get_outputs()[0].name
if isinstance(shape_value_in, str):
    max_input_words = 512                   # Default value, you can adjust it.
    is_dynamic = True
else:
    max_input_words = shape_value_in
    is_dynamic = False

# Run the cosine similarity
start_time = time.time()

input_ids = tokenizer(sentence_1, max_input_words, is_dynamic)
output_1 = ort_session_A.run([out_name_A0], {in_name_A0: input_ids})[0]

input_ids = tokenizer(sentence_2, max_input_words, is_dynamic)
output_2 = ort_session_A.run([out_name_A0], {in_name_A0: input_ids})[0]

cos_similarity = np.dot(output_1, output_2) / np.sqrt(np.dot(output_1, output_1) * np.dot(output_2, output_2))
print(f"\nThe Cosine Similarity between: \n\n1.'{sentence_1}' \n2.'{sentence_2}' \n\nScore = {cos_similarity:.3f}\n\nTime Cost: {time.time() - start_time:.3f} seconds")


