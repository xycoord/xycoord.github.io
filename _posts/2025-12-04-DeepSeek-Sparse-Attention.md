---
layout: post
title: DeepSeek Sparse Attention
description: This post explains the DeepSeek Sparse Attention mechanism introduced in DeepSeek V3.2 to reduce attention cost in long contexts.
---

The vanilla attention algorithm[^AIAYN] scales quadratically in context length due to the pairwise Key/Query dot products. This makes long contexts—a requirement for complex reasoning and long agentic tasks—computationally expensive. However, each token typically attends to only a small fraction of prior tokens. Computing accurate attention scores for these relevant positions requires high numerical precision (BF16) and separate computations per head. But identifying which positions are irrelevant can use a much cheaper approximation: lower precision (FP8) and a single shared score across all heads.
DeepSeek's new sparse attention mechanism, called DeepSeek Sparse Attention (DSA), introduces a lightweight *Lightning Indexer* that efficiently identifies the most relevant tokens for each query. The main MLA attention then computes precise attention scores over only this selected subset, dramatically reducing computational cost whilst maintaining model quality.

<!--more-->

In my explanation I lean heavily on the DeepSeek V3.2 Paper[^DS32Paper], and include extra insights from the code[^DS32Code]. The dimensions I quote are for DeepSeek V3.2 671B.

# The Lightning Indexer
The Lightning Indexer approximates attention scores with a Multi-Query Attention-like architecture: multiple query heads ($H^I=64$) attend to a single shared key, with both projected to low-dimensional (128-dim) FP8 representations.
For each new token, let $\mathbf x_t$ be the normalised residual input to the attention block:
1. Project to get $\mathbf q^I_t$, $\mathbf k^I_t$ and $\mathbf w^I_t$

	$$
	\begin{align*}
	\mathbf q^I_t = &\text{RMSNorm}(\mathbf x_t^\top \mathbf W^\text{LoRA}) \mathbf W^I_Q 
	\\
	\mathbf k^I_t = &\text{LayerNorm}(\mathbf x_t^\top \mathbf W^I_K)
	\\
	\mathbf w^I_t = &\frac{1}{\sqrt{H^I d^I}}\cdot \mathbf x_t^\top \mathbf W^I_W
	\end{align*}
	$$

	The LayerNorm on keys reduces dynamic range, preparing them for efficient FP8 quantisation. For queries, the indexer reuses the already-RMSNorm'd LoRA projection from MLA (rather than creating its own compression path), which provides both the same quantisation benefits and parameter efficiency from sharing the compression.
2. Apply RoPE to $\mathbf q^I_t$ and $\mathbf k^I_t$

	They apply RoPE to only the first 64 of the 128 dimensions. The remaining dimensions are position-independent, allowing them to capture pure content matches without the distance decay typically seen in attention patterns.

3. Quantise $\mathbf q^I_t$ and $\mathbf k^I_t$ to FP8

	This process splits each vector into an FP8 vector and a scaling factor which can efficiently be reapplied post-dot-product. While this quantisation limits the dynamic range between elements of a single vector, it preserves the dynamic range across vectors. In order to minimise dynamic range clipping within a vector, they apply a Hadamard transform to the vectors before quantisation which produces a vector with lower dynamic range that behaves identically in a dot product. In the implementation, DeepSeek fold the scale factor for $\mathbf q^I_t$ into the head weights. 

4. Store $\mathbf k^I_t$ in a cache with the keys for all prior tokens 

	 This is much like a typical KV cache. They have separate caches for the FP8 vector and the scaling factors.

5. Compute index scores $I_{t,s}$ for each prior token, indexed with $s$:

	$$I_{t,s}= \sum_{j=1}^{H^I}w^I_{t,j}\cdot\text{ReLU}(\mathbf q^I_{t,j}\cdot\mathbf k^I_s)$$

	In the implementation, the quantisation scale factors (q-scale and k-scale) are moved outside the ReLU and summation respectively, which is valid since they're always positive.

6. Take Top-k (k=2048) indexes for use in MLA

Even though the indexer is also quadratic in context length, it's lightning fast compared to running the full MLA:
- Half the number of heads (64 vs 128)
- Shares keys between heads like MQA
- Dot products done in fp8 (vs bf16)
- 2/3 the head dimension (128 vs 192)
- Smaller Cache per token leads to faster cache reads from memory 

Once context exceeds 2048 tokens, MLA processes only the selected top-2048, making attention cost per new token constant regardless of total context length.

# Integrating into MLA
In Multi-head Latent Attention (MLA), the keys are compressed into a single latent vector before being stored in the KV cache. Therefore, fetching one latent from memory effectively retrieves the necessary key information for *all* attention heads simultaneously. The DSA indexer leverages this by selecting a single set of token indices shared across all heads. This maximizes hardware efficiency because every fetched latent block is utilized by every head, eliminating wasted memory bandwidth from loading unused tokens[^NSA]. Crucially, the query projection can be fused with the latent decompression: computing $(\mathbf q_t \mathbf W_{kv}^b) \cdot \mathbf c_t$ instead of $\mathbf q_t \cdot (\mathbf W_{kv}^b \mathbf c_t)$, where $\mathbf c_t$ is the compressed latent and $\mathbf W_{kv}^b$ is the decompression matrix. This means the system never materialises the full uncompressed keys, keeping everything memory-efficient.

# Training
A key feature of DSA is that it doesn't significantly change which tokens are attended to, because it only ignores tokens with negligible attention scores. This means the indexer can be added to a pretrained model without restarting pretraining from scratch. They use two training phases.

## Dense Warm-up
Beginning with frozen DeepSeek V3.1-Terminus parameters, they train only the indexer to align with the model's attention distribution. The target attention scores have already been softmaxed to create a distribution. Therefore, they sum attention scores across all heads and L1-normalise across the sequence to combine these into target distributions $p_t$. They apply softmax to $I_t$ to convert it to a distribution like $p_t$, then minimise the KL divergence:

$$\mathcal L^I = \sum_t \mathbb D_{\text{KL}}(p_{t,:}\Vert\text{Softmax}(I_{t,:}))$$

## Sparse Training
They then unfreeze the full model and train it with standard language modelling loss whilst the indexer continues learning separately. 

The indexer is detached from the main model's computational graph and trained only on its KL alignment loss. However, alignment now occurs only over the top-k selected tokens $\mathcal S_t = \{s\mid I_{t,s}\in\text{Top-K}(I_{t,:})\}$:

$$\mathcal{L}^I = \sum_t \mathbb{D}_{\text{KL}}(p_{t,\mathcal{S}_t}\Vert\text{Softmax}(I_{t,\mathcal{S}_t}))$$

This allows the main model to adapt to sparse attention patterns whilst the indexer learns to identify the most relevant tokens.

# Takeaways
My explanation of DSA covers many small implementation details. The key insights are these:
1. Models spend significant effort computing attention scores for tokens with minimal influence on the attention output
2. A mechanism to predict low attention scores allows the model to ignore the values of these tokens and operate much more efficiently
3. Sharing index selection across MLA heads leads to more efficient cache recall.

I wonder how further insights into attention patterns may allow yet more aggressive culling of keys and values for extra-long contexts.

## Speculation: Prospective Heavy-Hitter Identification
It's a well studied phenomenon that models store summary information in the residual streams of specific tokens (such as sentence summaries in full-stops)[^Sinks]. Such tokens are far more frequently attended to than their neighbours (Heavy-hitters).  Extensive research on KV cache eviction (H2O[^H2O], Scissorhands[^Scissorhands], SAGE-KV[^SAGEKV]) exploit this with a *retrospective* approach: observing which tokens consistently receive attention and evicting the rest. This works well for recent context but may be confused by tangents in a conversation—if something hasn't been queried recently, it gets evicted regardless of its original importance. I wonder whether a *prospective* approach could work better: train a lightweight predictor that identifies tokens containing "interesting" information at encoding time, using their key representations to predict whether they'll be useful to *any* future query. This would allow the model to remember  important events that happened hundreds of thousands of tokens ago—for instance, "I researched semiconductor regulations" or "I identified a security vulnerability"—and maintain their values in the cache whilst evicting the vast unimportant details from the distant past.

This trades exact token recall for a more human-like memory system: the heavy-hitter values maintain high-level summaries and awareness of what happened, whilst tool-based search retrieves specific details when needed. For million-token agentic contexts, maintaining the gist of conversation history seems more valuable than perfect token-level recall, especially when the full text remains accessible through retrieval tools.

# References


[^AIAYN]: Vaswani, et al. (2017). Attention is all you need [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)

[^DS32Paper]: DeepSeek-AI (2025). DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models [arXiv:2512.02556](https://arxiv.org/abs/2512.02556)

[^DS32Code]: DeepSeek-AI (2025). DeepSeek V3.2 671B Code [Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp/tree/main/inference)

[^MLA]: DeepSeek-AI (2024). DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model [arXiv:2405.04434](https://arxiv.org/abs/2405.04434)

[^NSA]: Yuan, et al. (2025). Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention [arXiv:2502.11089](https://arxiv.org/abs/2502.11089)

[^Sinks]: Zhang, et al. (2025). Attention Sinks: A 'Catch, Tag, Release' Mechanism for Embeddings [arXiv:2502.00919](https://arxiv.org/abs/2502.00919)

[^H2O]: Zhang, et al. (2023). H_2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models [arXiv:2306.14048](https://arxiv.org/abs/2306.14048)

[^Scissorhands]: Liu, et al. (2023). Scissorhands: Exploiting the Persistence of Importance Hypothesis for LLM KV Cache Compression at Test Time [arXiv:2305.17118](https://arxiv.org/abs/2305.17118)

[^SAGEKV]: Wang, et al. (2025). LLMs Know What to Drop: Self-Attention Guided KV Cache Eviction for Efficient Long-Context Inference [arXiv:2503.08879](https://arxiv.org/abs/2503.08879)