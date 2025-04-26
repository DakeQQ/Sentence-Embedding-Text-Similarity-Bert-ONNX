# Sentence-Embedding-Text-Similarity-Bert-ONNX
 - Utilizes ONNX Runtime to get the sentence embedding vectors and similarity.
 - 利用 ONNX Runtime 获取句子嵌入向量和相似度。

## Supported Models:
- GTE Series: Tiny, Small, Base, Large, [Chinese](https://modelscope.cn/models/iic/nlp_gte_sentence-embedding_chinese-large), [English](https://modelscope.cn/models/iic/nlp_gte_sentence-embedding_english-large), [Multi-lingual](https://modelscope.cn/models/iic/gte_sentence-embedding_multilingual-base).


## 性能 Performance  
| OS           | Device       | Backend           | Model        | Time Cost in Seconds|
|:------------:|:------------:|:-----------------:|:------------:|:------------------------------------------------:|
| Ubuntu-24.04 | Laptop      | CPU <br> i7-1165G7 | GTE-Large-Chinese <br> f32 | 0.11                               |
| Ubuntu-24.04 | Laptop      | CPU <br> i7-1165G7 | GTE-Large-Chinese <br> q8f32 | 0.035                            |



## 支持的模型：
- GTE 系列: Tiny, Small, Base, Large, [中文](https://modelscope.cn/models/iic/nlp_gte_sentence-embedding_chinese-large), [英文](https://modelscope.cn/models/iic/nlp_gte_sentence-embedding_english-large), [多语言](https://modelscope.cn/models/iic/gte_sentence-embedding_multilingual-base).
