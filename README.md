# Neural Network Architectures for Sequence Tasks

This repository contains implementations of various neural network architectures used for sequence-to-sequence tasks, such as machine translation. The models are currently trained on an **English to Hindi** translation task, but the codebase is flexible and can be adapted to any other language pair. **Inference code** is also included to demonstrate the translation capabilities. The table below provides a comparison of these models, highlighting their purposes, advantages, disadvantages, and the original papers they were introduced in.

## Model Comparison

| Model | Purpose | Advantages | Disadvantages | Paper | Image |
|:---|:---|:---|:---|:---|:---|
| **RNN** | Sequential data processing | Handles variable length input. | Vanishing gradient problem; difficulty capturing long-term dependencies. | [Rumelhart et al., 1986](https://www.nature.com/articles/323533a0) | <img src="Architectures/1_RNN/rnn.png" width="300"> |
| **LSTM** | Address vanishing gradient in RNNs | Captures long-term dependencies; handles vanishing gradient problem better than RNNs. | Complex structure; more parameters to train; slower than RNNs. | [Hochreiter & Schmidhuber, 1997](https://www.bioinf.jku.at/publications/older/2604.pdf) | <img src="Architectures/2_LSTM/lstm.png" width="300"> |
| **GRU** | Simpler LSTM alternative | Faster training than LSTM; fewer parameters; comparable performance to LSTM. | Can be less powerful than LSTM in some very deep or complex contexts. | [Cho et al., 2014](https://arxiv.org/abs/1406.1078) | <img src="Architectures/3_GRU/gru.png" width="300"> |
| **Bahdanau Attention** | Dynamic focusing on source context | Improves performance on long sentences by focusing on relevant parts of input. | Computationally expensive; alignment computed at every step. | [Bahdanau et al., 2014](https://arxiv.org/abs/1409.0473) | <img src="Architectures/4_Bahdanau_Attention/bahdanau%20attention].png" width="300"> |
| **Luong Attention** | Refined attention mechanisms | Offers global and local attention mechanisms; different scoring functions. | Complexity varies depending on the specific alignment function used. | [Luong et al., 2015](https://arxiv.org/abs/1508.04025) | <img src="Architectures/5_Luong_attention/Luoung%20attention.png" width="300"> |
| **Transformer** | Parallelized sequence processing | Highly parallelizable; captures global dependencies effectively; state-of-the-art results. | Quadratic memory complexity with respect to sequence length; requires large amounts of data. | [Vaswani et al., 2017](https://arxiv.org/abs/1706.03762) | <img src="Architectures/Transformers/transformer_architecture.png" width="300"> |

## Directory Structure
- `Architectures/`: Contains specific model implementations and diagrams.
- `data/`: Data storage.
- `model/`, `models/`, `runs/`: Checkpoints and training logs.
---
## MAP of Models Used
<img src="Architectures/map.png">