---
layout: post
title: Direct Logit Attribution
description: "This post explains Direct Logit Attribution and Logit Lens: key tools in the initial mechanistic investigation of transformer behaviour."
---
Transformer language models generate text by producing a probability distribution over the next token. This distribution is derived from a vector of *logits* (one value per token in the vocabulary) where higher values indicate tokens the model considers more likely. But how does the model arrive at these logits? Which internal components are responsible for promoting or suppressing particular tokens?

Direct Logit Attribution (DLA) is an interpretability method which allows us to answer these questions by decomposing the output logits into a sum of contributions from each component of a transformer (attention layers and MLPs). For a given prompt, we can determine exactly how much each component *directly* contributes to the logit of any token -- revealing, for instance, that a specific MLP directly promotes "Paris" as the next token after "The Eiffel Tower is in the city of".

<!--more-->

Following the level of abstraction in [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html), we can define the transformer architecture as:
```
tokens = encode(prompt)
residual_stream = 0
residual_stream += embed(tokens)
for layer in layers:
	residual_stream += layer.attention(residual_stream)
	residual_stream += layer.mlp(residual_stream)
ln_embedding = layer_norm_final(residual_stream)
logits = unembed(ln_embedding)
```

The key observation is that the model only ever manipulates the residual stream by adding to it. Thus the final state of the residual stream is the sum of the outputs of each component (MLPs and Attention). To get the logits, the residual stream is then passed through a LayerNorm and then the unembedding (a linear map). 

LayerNorm is not quite linear, but as we'll see in the next section, for a given forward pass it can be treated as one. This allows us to decompose the logits into a sum of per-component contributions exactly.

# LayerNorm
Before unembedding, LayerNorm is applied to the residual stream $\mathbf{x}$ independently at each token position:

$$\text{LayerNorm}(\mathbf x, \boldsymbol\gamma, \boldsymbol\beta) = \boldsymbol\gamma \odot \left(\frac{\mathbf x - \mu(\mathbf x)}{\sigma(\mathbf x)}\right) + \boldsymbol\beta$$

where $\boldsymbol\gamma$ and $\boldsymbol\beta$ are learnt parameters. 

The first thing to note is that $\mu(\mathbf x)$ may be linearly decomposed. Let $\mathbf x = \sum_i \mathbf c_i$ where $c_i$ is the output of layer $i$ including the embedding. The mean is taken over the embedding dimension, thus:

$$
\begin{align*}
\mu(\mathbf x) =& \mu\left(\sum_i \mathbf c_i\right)\\
=& \frac{1}{d}\sum_{j=1}^d \sum_i \mathbf c_{i,j}\\
=& \sum_i \frac{1}{d}\sum_{j=1}^d \mathbf c_{i,j}\\
=& \sum_i \mu(\mathbf c_i)
\end{align*}
$$

Hence, 

$$\text{LayerNorm}(\mathbf x, \boldsymbol\gamma, \boldsymbol\beta) = \boldsymbol\gamma \odot \left(\frac{\sum_i (\mathbf c_i - \mu(\mathbf c_i))}{\sigma(\mathbf x)}\right) + \boldsymbol\beta$$

The $\sigma$ term makes the transformation non-linear in $\mathbf{x}$ (and hence $\mathbf c_i$). However, for decomposing a specific forward pass, we fix $\sigma^\ast = \sigma(\mathbf x)$ to the value actually computed. Division by this fixed scalar is linear, so the decomposition is exact. 

$$= \boldsymbol\gamma \odot \left(\frac{\sum_i (\mathbf c_i - \mu(\mathbf c_i))}{\sigma^\ast}\right) + \boldsymbol\beta$$

*Note: The ARENA tutorials describe this fixing as an "approximation". This framing only makes sense if interpreting DLA causally -- i.e., predicting what would happen if a component were ablated. But such causal interpretation is problematic for deeper reasons: ablating a component would also affect all downstream computation, not just $\sigma$. DLA is best understood as decomposing the logits into **direct** contributions, not as causal analysis.*

Since, the element wise multiplication by $\boldsymbol\gamma$ is also linear, we can decompose the first part by pulling the sum outside:

$$= \boldsymbol\beta + \sum_i \boldsymbol\gamma \odot \left(\frac{\mathbf c_i - \mu(\mathbf c_i)}{\sigma^\ast}\right)$$

The $\boldsymbol\beta$ contribution is constant and quite separate from the rest of the LayerNorm. It encodes unigram token statistics -- how common is this token?

# RMSNorm
Many modern transformer architectures such as Llama and Qwen replace LayerNorm with RMSNorm defined:

$$\text{RMSNorm}(\mathbf x, \boldsymbol\gamma) = \boldsymbol\gamma \odot \frac{\mathbf x}{\text{RMS}(\mathbf x)}$$

where $\boldsymbol\gamma$ is a learnt parameter and

$$\text{RMS}(\mathbf x) = \sqrt{\epsilon+\frac{1}{d}\sum_{i=1}^d \mathbf x_i^2}$$

The RMS term is non-linear for the same reason as $\sigma$: it involves a square root of a sum of squares. As before, we fix $\text{RMS}^* = \text{RMS}(\mathbf{x})$ from the forward pass, making the transformation linear and the decomposition exact. 

Thus the decomposition is:

$$\text{RMSNorm}(\mathbf x, \boldsymbol\gamma) = \sum_i\boldsymbol\gamma \odot \frac{\mathbf c_i}{\text{RMS}^\ast}$$

**Important note:** When using DLA, check the type of norm the architecture of your model uses with `print(model)`. There are often subtle implementation details in custom versions of LayerNorm and RMSNorm which are important to replicate for maximal accuracy.

From now on I'll assume a LayerNorm architecture.

# DLA Algorithm
1. **Forward pass:** Compute component outputs including the residual stream state $\mathbf x$ before the final LayerNorm
2. **Record fixed stats**: $\sigma^*=\sigma(\mathbf x)$
3. **LayerNorm:** Centre and apply the linear transform to each component contribution using $\sigma^*$ and $\boldsymbol\gamma$
4. **$\beta$ contribution**: append $\beta$ as a separate contribution
5. **Unembed:** Apply the unembedding matrix to obtain logit contributions

# Code Sample (using [NNSight](https://nnsight.net/) and LayerNorm)

In addition to the code extracts provided here, complete implemenations can be found in [this notebook](https://colab.research.google.com/drive/1WBfSycdPDkNnbtQXeIpiqPwhEHGEBL1v?usp=sharing).

```python
prompt = "The Eiffel Tower is in the city of"

layers = model.transformer.h
eps = model.transformer.ln_f.eps
gamma = model.transformer.ln_f.weight.data
beta = model.transformer.ln_f.bias.data
unembedding = model.lm_head.weight.data

residual_contributions = []

# 1. Forward pass
# Save only the residual stream for the last token with [:, -1, :]
with model.trace(prompt) as tracer:
	# Record Embeddings
	token_embed = model.transformer.wte.output[:, -1, :]
	pos_embed = model.transformer.wpe.output[:, -1, :]
	residual_contributions.append((token_embed + pos_embed).save())
	
	# Record Attention and MLP contributions
	for layer in layers:
		residual_contributions.append(layer.attn.output[0][:, -1, :].save())
		residual_contributions.append(layer.mlp.output[:, -1, :].save())
	
  	# 2. Record fixed stats
	final_residual = layers[-1].output[0][:, -1, :]
	var = final_residual.var(dim=-1, unbiased=False, keepdim=True)
	sigma = torch.sqrt(var + eps).save()

batch_size, d_model = residual_contributions[0].shape

# 3. Apply LayerNorm with fixed sigma
def layer_norm_f(contrib):
	centered_contrib = contrib - contrib.mean(dim=-1, keepdim=True)
	return gamma * (centered_contrib / sigma)

ln_f_contributions = list(map(layer_norm_f, residual_contributions))

# 4. Beta contribution
beta_contribution = beta.unsqueeze(0).expand(batch_size, d_model)
ln_f_contributions.append(beta_contribution)

# 5. Unembed
def unembed(contrib):
    return contrib @ unembedding.T

logit_contributions = list(map(unembed, ln_f_contributions))
```

Since DLA gives us an exact decomposition of the logits we can check the sum against the final logits.

```python
summed_logits = sum(logit_contributions)

with model.trace(prompt):
    actual_logits = model.lm_head.output[:, -1, :].save()

assert torch.allclose(summed_logits, actual_logits, atol=1e-3), \
    "Mismatch! The sum of DLA components does not equal the model's final logits."

print("Success: The decomposition matches the model's output!")
```

So far, we've looked at the case where the components we're interested in are individual layers. This is called **layer attribution**. However, we can decompose the model into larger or smaller components for different granularities of attribution.

# Logit Lens
If we want to know what the model 'believes' at an intermediate layer, we can read the residual stream after layer $l$ and transform it to logits. This approach is called **Logit Lens**. The [original post](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens) explores a broader range of metrics and visualisations than covered here, and is well worth reading for further approaches.

A key gotcha is that Logit Lens implementations typically don't fix the LayerNorm standard deviation but compute it fresh for the intermediate residuals. This approach answers the question: what would the logits be if the later layers made no contributions to the residual stream? This contrasts with the DLA aim of decomposing the logits exactly into contributions from each component (i.e. the contributions should sum to the logits).

Further, since the aim of Logit Lens is to make a statement about the model's 'belief', it's common to apply the softmax to the logits to give a probability distribution. This is not the case with DLA since the non-linearity of softmax breaks the decomposition.

```python
prompt = "The Eiffel Tower is in the city of"

token_embed = model.transformer.wte
position_embed = model.transformer.wpe
layers = model.transformer.h
ln_f = model.transformer.ln_f
unembed = model.lm_head	

logit_trajectory = []
prob_trajectory = []

with model.trace(prompt) as tracer:
	# Save only the residual stream for the last token with [:, -1, :]
	embedding = token_embed.output[:, -1, :] + position_embed.output[:, -1, :]

	# Apply LayerNorm (computing σ fresh from this intermediate state) 
	# and Unembedding
	embed_logits = unembed(ln_f(embedding)).save()
	# Apply Softmax to get the probability distribution
	embed_probs = embed_logits.softmax(dim=-1).save()
	logit_trajectory.append(embed_logits)
	prob_trajectory.append(embed_probs)

	for layer in layers:
		residual_stream = layer.output[0][:, -1, :]
		intermediate_logits = unembed(ln_f(residual_stream)).save()
		intermediate_probs = intermediate_logits.softmax(dim=-1).save()
		logit_trajectory.append(intermediate_logits)
		prob_trajectory.append(intermediate_probs)
```

# Attention Heads
The output of an attention layer is just the sum of the outputs of its heads. Hence we can equivalently add the output of each head to the residual stream independently, where each output is a residual contribution. Thus we can use DLA to find the logit contributions for each head -- called **head attribution**. You might do this for a particular attention layer after identifying it has a large contribution.

# Logit Difference
We're often interested not so much in the raw logits but in the relationship between two logits: typically a correct and an incorrect token. Where the incorrect token is chosen to control for the other tasks the model is doing. The metric for this is the logit difference: `correct_logit - incorrect_logit`. This is linear so we can also look at the contributions of each component to the difference: `correct_logit_contribution - incorrect_logit_contribution`.

Applying the unembedding matrix for each component can be an expensive matrix multiplication due to large vocab sizes. It's also largely pointless since we only want the contributions to two of the tokens. Efficiently, we can multiply only by the vector entries for those tokens. 

$$(\mathbf x_\text{ln}^\top W_U)_\text{correct} = \mathbf x_\text{ln}^\top \mathbf u_\text{correct}$$

Even more efficiently, we can precompute the difference between these vectors to create a logit-difference projection: projecting directly from the post-LayerNorm stream to the logit-difference scalar.

$$
\begin{align*}
\text{logit diff}=&(\mathbf x_\text{ln}^\top W_U)_\text{correct} - (\mathbf x_\text{ln}^\top W_U)_\text{incorrect}
\\=& \mathbf x_\text{ln}^\top \mathbf u_\text{correct} - \mathbf x_\text{ln}^\top \mathbf u_\text{incorrect}
\\=& \mathbf x_\text{ln}^\top (\mathbf u_\text{correct} -  \mathbf u_\text{incorrect})
\\=& \mathbf x_\text{ln}^\top \mathbf u_\text{diff}
\end{align*}
$$

```python
prompt = "The Eiffel Tower is in the city of"

layers = model.transformer.h
unembedding = model.lm_head.weight.data
gamma = model.transformer.ln_f.weight.data
beta = model.transformer.ln_f.bias.data

correct_direction = unembedding[correct_token_id]
incorrect_direction = unembedding[incorrect_token_id]
diff_direction = correct_direction - incorrect_direction

residual_contributions = []

# 1. Forward pass
# Save only the residual stream for the last token with [:, -1, :]
with model.trace(prompt) as tracer:
	# Record Embeddings
	token_embed = model.transformer.wte.output[:, -1, :]
	pos_embed = model.transformer.wpe.output[:, -1, :]
	residual_contributions.append((token_embed + pos_embed).save())
	
	# Record Attention and MLP contributions
	for layer in layers:
		residual_contributions.append(layer.attn.output[0][:, -1, :].save())
		residual_contributions.append(layer.mlp.output[:, -1, :].save())
	
	# 2. Record std
	final_residual = layers[-1].output[0][:, -1, :]
	var = final_residual.var(dim=-1, unbiased=False, keepdim=True)
	sigma = torch.sqrt(var + eps).save()

batch_size, d_model = residual_contributions[0].shape

# 3. Apply LayerNorm with fixed sigma
def layer_norm_f(contrib):
	centered_contrib = contrib - contrib.mean(dim=-1, keepdim=True)
	return gamma * (centered_contrib / sigma)

ln_f_contributions = list(map(layer_norm_f, residual_contributions))

# 4. Beta contribution
beta_contribution = beta.unsqueeze(0).expand(batch_size, d_model)
ln_f_contributions.append(beta_contribution)

# 5. Unembed
def unembed(contrib):
    return (contrib @ diff_direction.T).item()

logit_diffs = list(map(unembed, ln_f_contributions))
```

# Plotting and Interpretation
To illustrate these techniques, let's examine the logit contributions for a concrete example. Below is a plot of the contributions of each layer to the logit difference. The model is GPT2 (LayerNorm) and the prompt is: "The Eiffel Tower is in the city of".

<div class="wide-plot">
{% include plots/logit_contributions_gpt2.html %}
</div>

The first thing to note is the large negative effect of $\boldsymbol\beta$. This is due to the fact that " London" is a more common token than " Paris" ($\boldsymbol\beta$ encode unigram statistics).

We can also observe the three large (+~0.3) contributions from MLPs 9, 10 and 11---late in the network. Notably, there are no significant *direct* contributions to the logits from attention layers. This tells us that these MLPs are crucial is processing any contextual information gathered through attention in order to make meaningful contributions to the logits.

Interestingly, the contribution is roughly equally split across these three consecutive layers. Since none of them individually would overcome the $\boldsymbol\beta$ contribution, this suggests distributed computation rather than redundancy -- each layer appears necessary for the final prediction.

However, DLA alone cannot tell us *what* these MLPs are computing or *how* they're using the contextual information. To draw more concrete conclusions about the mechanisms, we would need further analysis with tools such as activation patching or neuron interpretation. To draw more concrete conclusions, we would need further analysis with tools such as activation patching. 

## Qwen Plot

Below is the plot for the same experiment done with Qwen2.5-0.5B, a slightly larger model which uses RMSNorm.

<div class="wide-plot">
{% include plots/logit_contributions_qwen.html %}
</div>

The results here are far more concentrated, with most contribution coming from just 4 components in the final layers. Since this model has no $\boldsymbol\beta$ offset, the unigram statistics must be encoded within the model's weights. The large negative contributions from the final two MLPs (~-2.7 and ~-3.1) suggest these layers may encode unigram statistics, though confirming this would require further investigation.

In contrast to GPT2, we see a large positive contribution directly from an attention layer (~6.3 from L21.attn). This is the single largest contribution in the entire decomposition. This tells us that information favouring " Paris" over " London" is being moved from earlier token positions and written directly to the residual stream by this attention layer - this attention output is the dominant signal determining the final prediction. It would be interesting to run DLA for the heads of this layer for a more precise analysis.

The contribution from the following MLP is significant (~3.9) although its exact role is unclear.

# Conclusion

Direct Logit Attribution provides a powerful lens for understanding transformer models by decomposing their final predictions into quantifiable contributions. It offers exact decomposition, flexibility across different granularities, and requires only a single forward pass.

However, DLA has important limitations. It answers "what contributes" and "how much", but not "how" or "why". Specifically:

- **Single token focus:** DLA examines contributions at a specific token position but cannot reveal how information from other tokens in the sequence is being processed or moved
- **Only final outputs:** By examining contributions to the final residual stream, DLA misses the intermediate computations and information flow between components
- **No mechanism insight:** DLA cannot tell us what computation a component is performing or what features it's responding to
- **Interpretation requires care:** What DLA reveals varies case-by-case depending on whether contributions are concentrated or distributed. Each analysis requires reasoning from first principles to form and narrow down hypotheses without drawing overly strong conclusions

For these reasons, DLA is best viewed as a starting point for mechanistic interpretability -- a way to efficiently identify which components warrant deeper investigation through complementary techniques such as activation patching and analysis of attention patterns in specific heads.

*Thanks to Elias Sandmann for their review and suggestions, especially in the Logit Lens section.*