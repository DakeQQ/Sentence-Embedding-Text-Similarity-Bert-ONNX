import re
import time
import torch
import numpy as np
import onnxruntime
from transformers import AutoModel

model_path = r"/home/DakeQQ/Downloads/nlp_gte_sentence-embedding_chinese-large"    # Path to the entire downloaded GTE model project.
onnx_model_A = r"/home/DakeQQ/Downloads/GTE_ONNX/Model_GTE.onnx"                   # The exported onnx model save path.
vocab_path = f'{model_path}/vocab.txt'                                             # Set the path where the GTE model vocab.txt stored.
sentence_1 = "吃完海鲜可以喝牛奶吗?"                                                   # The sentence for similarity test.
sentence_2 = "不可以，早晨喝牛奶不科学"                                                 # The sentence for similarity test.

DYNAMIC_AXES = False          # Whether both are set to True or False, they must still be less than MAX_INPUT_WORDS.
MAX_INPUT_WORDS = 1024        # The maximum input words for the input sentence.
TOKEN_UNKNOWN = 100           # The model parameter, do not edit it.
TOKEN_BEGIN = 101             # The model parameter, do not edit it.
TOKEN_END = 102               # The model parameter, do not edit it.


# Read the model vocab.
with open(vocab_path, 'r', encoding='utf-8') as file:
    vocab = file.readlines()
vocab = np.array([line.strip() for line in vocab], dtype=np.str_)


class BERT(torch.nn.Module):
    def __init__(self, bert_model, max_seq_len):
        super(BERT, self).__init__()
        self.bert_model = bert_model

        num_head = bert_model.encoder.layer._modules["0"].attention.self.num_attention_heads
        head_dim = bert_model.encoder.layer._modules["0"].attention.self.attention_head_size
        hidden_size = bert_model.encoder.layer._modules["0"].attention.self.all_head_size
        scale_factor = float(head_dim ** -0.25)

        for layer in self.bert_model.encoder.layer:
            layer.attention.self.query.weight.data *= scale_factor
            layer.attention.self.query.bias.data *= scale_factor
            layer.attention.self.key.weight.data *= scale_factor
            layer.attention.self.key.bias.data *= scale_factor

            layer.attention.self.query.weight.data = layer.attention.self.query.weight.data.view(num_head, head_dim, hidden_size).transpose(1, 2).contiguous()
            layer.attention.self.key.weight.data = layer.attention.self.key.weight.data.view(num_head, head_dim, hidden_size).transpose(1, 2).contiguous()
            layer.attention.self.value.weight.data = layer.attention.self.value.weight.data.view(num_head, head_dim, hidden_size).transpose(1, 2).contiguous()
            layer.attention.self.query.bias.data = layer.attention.self.query.bias.data.view(num_head, 1, head_dim).contiguous()
            layer.attention.self.key.bias.data = layer.attention.self.key.bias.data.view(num_head, 1, head_dim).contiguous()
            layer.attention.self.value.bias.data = layer.attention.self.value.bias.data.view(num_head, 1, head_dim).contiguous()
            layer.attention.output.dense.weight.data = layer.attention.output.dense.weight.data.view(hidden_size, num_head, head_dim).permute(1, 2, 0).contiguous()
            layer.attention.output.dense.bias.data = layer.attention.output.dense.bias.data.view(1, 1, -1).contiguous()

        self.bert_model.embeddings.token_type_embeddings.weight.data = self.bert_model.embeddings.token_type_embeddings.weight.data[[0], :max_seq_len].unsqueeze(-1)
        self.bert_model.embeddings.position_embeddings.weight.data = self.bert_model.embeddings.position_embeddings.weight.data[:max_seq_len, :].unsqueeze(0)

    def forward(self, input_ids: torch.IntTensor, ids_len: torch.IntTensor):
        if not DYNAMIC_AXES:
            input_ids = input_ids[:, :ids_len]
        hidden_states = self.bert_model.embeddings.LayerNorm(self.bert_model.embeddings.word_embeddings(input_ids) + self.bert_model.embeddings.token_type_embeddings.weight.data[:, :ids_len] + self.bert_model.embeddings.position_embeddings.weight.data[:, :ids_len])
        for layer in self.bert_model.encoder.layer:
            q = torch.matmul(hidden_states, layer.attention.self.query.weight) + layer.attention.self.query.bias
            k = (torch.matmul(hidden_states, layer.attention.self.key.weight) + layer.attention.self.key.bias).transpose(1, 2).contiguous()
            v = torch.matmul(hidden_states, layer.attention.self.value.weight) + layer.attention.self.value.bias
            attn_out = torch.matmul(torch.nn.functional.softmax(torch.matmul(q, k), dim=-1), v)
            attn_out = torch.matmul(attn_out, layer.attention.output.dense.weight).sum(dim=0, keepdim=True) + layer.attention.output.dense.bias
            hidden_states += attn_out
            hidden_states = layer.attention.output.LayerNorm(hidden_states)
            hidden_states = layer.output.LayerNorm(layer.output.dense(layer.intermediate.intermediate_act_fn(layer.intermediate.dense(hidden_states))) + hidden_states)
        return hidden_states[0, 0]


print("\nExport Start...")
with torch.inference_mode():
    model = AutoModel.from_pretrained(model_path, torch_dtype=torch.float32).eval()
    input_ids = torch.zeros((1, MAX_INPUT_WORDS), dtype=torch.int32)
    ids_len = torch.ones(1, dtype=torch.int64)
    model = BERT(model, MAX_INPUT_WORDS)
    torch.onnx.export(model,
                      (input_ids, ids_len),
                      onnx_model_A,
                      input_names=['text_ids', 'ids_len'],
                      output_names=['encoder_output'],
                      dynamic_axes={
                          'text_ids': {1: 'ids_len'}
                      } if DYNAMIC_AXES else None,
                      do_constant_folding=True,
                      opset_version=17)
del model
del input_ids
print("\nExport Done!")


# For GTE Model
def tokenizer(input_string, max_input_words, is_dynamic):
    input_ids = np.zeros((1, max_input_words), dtype=np.int32)
    input_string = re.findall(r'[\u4e00-\u9fa5]|[a-zA-Z]+|[^\w\s]', input_string.lower())
    input_ids[0] = TOKEN_BEGIN
    full = max_input_words - 1
    ids_len = 1
    for i in input_string:
        indices = np.where(vocab == i)[0]
        if len(indices) > 0:
            input_ids[:, ids_len] = indices[0]
            ids_len += 1
            if ids_len == full:
                break
        else:
            for j in list(i):
                indices = np.where(vocab == j)[0]
                if len(indices) > 0:
                    input_ids[:, ids_len] = indices[0]
                else:
                    input_ids[:, ids_len] = TOKEN_UNKNOWN
                ids_len += 1
                if ids_len == full:
                    break
    input_ids[:, ids_len] = TOKEN_END
    ids_len += 1
    if is_dynamic:
        input_ids = input_ids[:, :ids_len]
    return input_ids, np.array([ids_len], dtype=np.int64)


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

ort_session_A = onnxruntime.InferenceSession(onnx_model_A, sess_options=session_opts, providers=['CPUExecutionProvider'], provider_options=None)
shape_value_in = ort_session_A._inputs_meta[0].shape[-1]
in_name_A0 = ort_session_A.get_inputs()[0].name
in_name_A1 = ort_session_A.get_inputs()[1].name
out_name_A0 = ort_session_A.get_outputs()[0].name
if isinstance(shape_value_in, str):
    max_input_words = 1024                  # Default value, you can adjust it.
    is_dynamic = True
else:
    max_input_words = shape_value_in
    is_dynamic = False

# Run the cosine similarity
start_time = time.time()

input_ids, ids_len = tokenizer(sentence_1, max_input_words, is_dynamic)
output_1 = ort_session_A.run([out_name_A0], {in_name_A0: input_ids, in_name_A1: ids_len})[0]

input_ids, ids_len = tokenizer(sentence_2, max_input_words, is_dynamic)
output_2 = ort_session_A.run([out_name_A0], {in_name_A0: input_ids, in_name_A1: ids_len})[0]

cos_similarity = np.dot(output_1, output_2) / np.sqrt(np.dot(output_1, output_1) * np.dot(output_2, output_2))
print(f"\nThe Cosine Similarity between: \n\n1.'{sentence_1}' \n2.'{sentence_2}' \n\nScore = {cos_similarity:.3f}\n\nTime Cost: {time.time() - start_time:.3f} Seconds")
