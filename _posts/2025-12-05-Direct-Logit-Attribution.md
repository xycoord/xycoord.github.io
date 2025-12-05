---
layout: post
title: Direct Logit Attribution
description: "This post explains Direct Logit Attribution and Logit Lens: key tools in the initial mechanistic investigation of transformer behaviour."
---

Direct Logit Attribution (DLA) is an interpretability method which allows us to understand the contributions of different components of the model to the output logits for a given prompt.

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

The key observation is that the model only ever manipulates the residual stream by adding to it. Thus the final state of the residual stream is the sum of the outputs of each component (MLPs and Attention). To get the logits, the residual stream is then passed through a LayerNorm (a *roughly* linear map) and then the unembedding (a linear map). In the next section, we'll look at decomposing and approximating LayerNorm into a linear map on the residual stream and a further logit contribution.
Since these transformations composed form a linear map, it is legitimate to apply them component wise, and then compute the sum afterwards.

This gives an approximately equivalent formulation of the architecture:
```
tokens = encode(prompt)
residual_contributions = []
embed(tokens) : residual_contributions
for layer in layers:
	layer.attention(sum(residual_contributions)) : residual_contributions
	layer.mlp(sum(residual_contributions)) : residual_contributions
ln_contributions = map(layer_norm_final, residual_contributions)
logit_contributions = map(unembed, ln_contributions)
logits = sum(logit_contributions)
```
*Note: `x : list` is notational short-hand for appending to a list.*

This is not an efficient formulation, but it is a useful one.

In this formulation, `logit_contributions`, is a list of tensors with the same dimensions as `logits` where each tensor corresponds to a component (attention or mlp) of the model. It describes the effect that component has on the logits exactly.

## LayerNorm
Before unembedding, LayerNorm is applied to the residual stream $\mathbf{x}$ independently at each token position:

$$\text{LayerNorm}(\mathbf x, \boldsymbol\gamma, \boldsymbol\beta) = \boldsymbol\gamma \odot \left(\frac{\mathbf x - \mu(\mathbf x)}{\sigma(\mathbf x)}\right) + \boldsymbol\beta$$

where $\boldsymbol\gamma$ and $\boldsymbol\beta$ are learnt parameters. You could think of this as normalising each token's residual vector (centring and scaling to unit variance), then applying an elementwise affine transformation. However, following the framework we've set out, it's more meaningful to rearrange this formula into three parts:

$$=\underbrace{\boldsymbol\gamma \odot \frac{\mathbf x}{\sigma(\mathbf x)}}_{\text{"linear" transform}} + \underbrace{\boldsymbol\gamma \odot \frac{-\mu(\mathbf x)}{\sigma(\mathbf x)}}_{\text{mean contribution}}+\boldsymbol\beta$$

The second and third terms are additive contributions to the logits, similar to those from MLPs and attention layers. The first term is a *roughly* linear transformation on the residual stream.

The $\sigma$ term makes this first part non-linear in $\mathbf{x}$. However, in reality, there are normally just a few directions utilised by any particular component. As such $\sigma(\mathbf x)$ is only weakly dependent on that component since it is taken across *all* directions. Consequently, we can approximate component dependent $\sigma(\mathbf x)$ with a fixed $\sigma^\ast$ taken from the actual forward pass. Using fixed $\sigma^\ast$ makes the normalisation of component outputs a linear operation. And composing with the learnt elementwise scaling $\boldsymbol\gamma$ gives us a linear approximation to LayerNorm that we can apply to component outputs independently.

For the mean contribution, there is no approximation involved. By using the actual $\mu^\ast$ and $\sigma^\ast$ from the forward pass, we calculate this term exactly as it appears in the LayerNorm decomposition formula defined above. Since this term undergoes the same $\boldsymbol\gamma/\sigma^\ast$ scaling as the other contributions, we can simply append $-\mu^\ast$ to the list of residual contributions before applying the scaling.

The $\boldsymbol\beta$ contribution is constant and quite separate from the rest of the LayerNorm. It encodes unigram token statistics—how common is this token?

## RMSNorm
Many modern transformer architectures such as Llama and Qwen replace LayerNorm with RMSNorm defined:

$$\text{RMSNorm}(\mathbf x, \boldsymbol\gamma) = \boldsymbol\gamma \odot \frac{\mathbf x}{\text{RMS}(\mathbf x)}$$

where $\boldsymbol\gamma$ is a learnt parameter and

$$\text{RMS}(\mathbf x) = \sqrt{\epsilon+\frac{1}{n}\sum_{i=1}^n \mathbf x_i^2}$$

Unlike LayerNorm, there are no offset factors/new contributions - only scaling. This makes the DLA implementation much simpler. The RMS term behaves like the $\sigma$ term in that it is only weakly dependent on each output of the prior components. Thus, we approximate it by fixing $\text{RMS}^\ast=\text{RMS}(\mathbf x)$ of the final state of the residual stream. The $\boldsymbol\gamma$ term behaves much the same, as a scaling constant.

**Important note:** When using DLA, check the type of norm the architecture of your model uses with `print(model)`. There are often subtle implementation details in custom versions of LayerNorm and RMSNorm which are important to replicate for maximal accuracy.

From now on I'll assume a LayerNorm architecture.

## DLA Algorithm
1. **Forward pass:** Compute component outputs including the residual stream state $\mathbf x$ before the final LayerNorm
2. **Compute fixed stats**: $\mu^\ast=\mu(\mathbf x)$, $\sigma^\ast=\sigma(\mathbf x)$
3. **Mean contribution**: append $-\mu^\ast$​
4. **Linearised LayerNorm:** Apply the linear transform to each component output using $\sigma^\ast$
5. **$\boldsymbol\beta$ contribution**: append $\boldsymbol\beta$
6. **Unembed:** Apply the unembedding matrix to obtain logit contributions

## Code Sample (using [NNSight](https://nnsight.net/) and LayerNorm)

In addition to the code extracts provided here, complete implemenations can be found in [this notebook](https://colab.research.google.com/drive/1kGT6XpgEb8ceMyNaUjo--hneCuUdKz1m?usp=sharing).

```python
prompt = "The Eiffel Tower is in the city of"

layers = model.transformer.h
gamma = model.transformer.ln_f.weight.data
beta = model.transformer.ln_f.bias.data
unembedding = model.lm_head.weight.data

residual_contributions = []

# 1. Forward pass
with model.trace(prompt) as tracer:
	# Save only the residual stream for the last token with [:, -1, :]
	token_embed = model.transformer.wte.output[:, -1, :]
	pos_embed = model.transformer.wpe.output[:, -1, :]
	residual_contributions.append((token_embed + pos_embed).save())
	for layer in layers:
		residual_contributions.append(layer.attn.output[0][:, -1, :].save())
		residual_contributions.append(layer.mlp.output[:, -1, :].save())
	final_residual = layers[-1].output[0][:, -1, :].save()

# 2. Compute fixed stats (and get learnt params)
mean = final_residual.mean(dim=-1, keepdim=True)
std = torch.sqrt(final_residual.var(dim=-1, unbiased=False, keepdim=True) + 1e-5)

batch_size, d_model = final_residual.shape 

# 3. Mean contribution
mean_contribution = - mean.expand(batch_size, d_model)
residual_contributions.append(mean_contribution)

# 4. Apply Linearised LayerNorm
def linearised_ln_f(contrib):
    return gamma * (contrib / std)

ln_f_contributions = list(map(linearised_ln_f, residual_contributions))

# 5. Beta contribution
beta_contribution = beta.unsqueeze(0).expand(batch_size, d_model)
ln_f_contributions.append(beta_contribution)

# 6. Unembed
def unembed(contrib):
    return contrib @ unembedding.T

logit_contributions = list(map(unembed, ln_f_contributions))
```

So far, we've looked at the case where the components we're interested in are individual layers. This is called **layer attribution**. However, we can decompose the model into larger or smaller components for different granularities of attribution.

## Logit Lens
If we want to know the contribution of layers $0\dots l$, we can read the residual stream after layer $l$ and transforming that, effectively performing the sum before the output transformations. We can interpret this as: what does the model believe the next token should be at layer $l$ ? This approach is called Logit Lens ([nostalgebraist, 2020](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens)).

A key gotcha is that Logit Lens implementations typically don't fix the LayerNorm stats but compute them fresh for the intermediate residuals. This approach answers the question: what would the logits be if the later layers made no contributions to the residual stream? This contrasts the DLA aim of decomposing the logits exactly into contributions from each component (i.e. the contributions should sum to the logits).

```python
prompt = "The Eiffel Tower is in the city of"

token_embed = model.transformer.wte
position_embed = model.transformer.wpe
layers = model.transformer.h
ln_f = model.transformer.ln_f
unembed = model.lm_head	

logit_trajectory = []

with model.trace(prompt) as tracer:
	# Save only the residual stream for the last token with [:, -1, :]
	embedding = token_embed.output[:, -1, :] + position_embed.output[:, -1, :]

	# Apply fresh LayerNorm and Unembedding
	embed_logits = unembed(ln_f(embedding)).save()
	logit_trajectory.append(embed_logits)

	for layer in layers:
		residual_stream = layer.output[0][:, -1, :]
		intermediate_logits = unembed(ln_f(residual_stream)).save()
		logit_trajectory.append(intermediate_logits)
```
## Attention Heads
The output of an attention layer is just the sum of the outputs of its heads. Hence we can equivalently add the output of each head to the residual stream independently, where each output is a residual contribution. Thus we can use DLA to find the logit contributions for each head—called **head attribution**.

## Logit Difference
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

correct_token_id = model.tokenizer.encode(" Paris")
incorrect_token_id = model.tokenizer.encode(" London")
correct_direction = unembedding[correct_token_id]
incorrect_direction = unembedding[incorrect_token_id]
diff_direction = correct_direction - incorrect_direction

residual_contributions = []

# 1. Forward pass
with model.trace(prompt) as tracer:
	# Save only the residual stream for the last token with [:, -1, :]
	token_embed = model.transformer.wte.output[:, -1, :]
	pos_embed = model.transformer.wpe.output[:, -1, :]
	residual_contributions.append((token_embed + pos_embed).save())
	for layer in layers:
		residual_contributions.append(layer.attn.output[0][:, -1, :].save())
		residual_contributions.append(layer.mlp.output[:, -1, :].save())
	final_residual = layers[-1].output[0][:, -1, :].save()

# 2. Compute fixed stats (and get learnt params)
mean = final_residual.mean(dim=-1, keepdim=True)
std = torch.sqrt(final_residual.var(dim=-1, unbiased=False, keepdim=True) + 1e-5)

batch_size, d_model = final_residual.shape 

# 3. Mean contribution
mean_contribution = - mean.expand(batch_size, d_model)
residual_contributions.append(mean_contribution)

# 4. Apply Linearised LayerNorm
def linearised_ln_f(contrib):
    return gamma * (contrib / std)

ln_f_contributions = list(map(linearised_ln_f, residual_contributions))

# 5. Beta contribution
beta_contribution = beta.unsqueeze(0).expand(batch_size, d_model)
ln_f_contributions.append(beta_contribution)

# 6. Unembed
def unembed(contrib):
    return (contrib @ diff_direction.T)

logit_diffs = list(map(unembed, ln_f_contributions))
```

# Plotting and Interpretation
To illustrate these techniques, let's examine the logit contributions for a concrete example. Below is a plot of the contributions of each layer to the logit difference. The model is GPT2 (LayerNorm) and the prompt is: "The Eiffel Tower is in the city of".

<div class="wide-plot">
{% include plots/logit_contributions_gpt2.html %}
</div>

The first thing to note is the large negative effect of $\boldsymbol\beta$. This is due to the fact that " London" is a more common token than " Paris" ($\boldsymbol\beta$ encode unigram statistics).

We can also observe the three large (+~0.3) contributions from MLPs 9, 10 and 11---late in the network. Notably, there are no significant *direct* contributions to the logits from attention layers. This tells us that these MLPs are crucial is processing any contextual information gathered through attention in order to make meaningful contributions to the logits.

Interestingly, the contribution is roughly equally split across these three consecutive layers. Since none of them individually would overcome the $\boldsymbol\beta$ contribution, this suggests distributed computation rather than redundancy—each layer appears necessary for the final prediction.

However, DLA alone cannot tell us *what* these MLPs are computing or *how* they're using the contextual information. To draw more concrete conclusions about the mechanisms, we would need further analysis with tools such as activation patching or neuron interpretation.To draw more concrete conclusions, we would need further analysis with tools such as activation patching. 

## Qwen Plot

Below is the plot for the same experiment done with Qwen2.5-0.5B, a slightly larger model which uses RMSNorm.

<div class="wide-plot">
{% include plots/logit_contributions_qwen.html %}
</div>

The results here are far more concentrated, with most contribution coming from just 4 components in the final layers. Since this model has no $\boldsymbol\beta$ offset, the unigram statistics must be encoded within the model's weights. The large negative contributions from the final two MLPs (~-2.7 and ~-3.1) suggest these layers may encode unigram statistics, though confirming this would require further investigation.

In contrast to GPT-2, we see a large positive contribution directly from an attention layer (~6.3 from L21.attn). This is the single largest contribution in the entire decomposition. This tells us that information favouring " Paris" over " London" is being moved from earlier token positions and written directly to the residual stream by this attention layer - this attention output is the dominant signal determining the final prediction. It would be interesting to run DLA for the heads of this layer for a more precise analysis.

The contribution from the following MLP is significant (~3.9) although its exact role is unclear.

# Conclusion

Direct Logit Attribution provides a powerful lens for understanding transformer models by decomposing their final predictions into quantifiable contributions. It offers exact decomposition (modulo the LayerNorm approximation), flexibility across different granularities, and requires only a single forward pass.

However, DLA has important limitations. It answers "what contributes" and "how much", but not "how" or "why". Specifically:

- **Single token focus:** DLA examines contributions at a specific token position but cannot reveal how information from other tokens in the sequence is being processed or moved
- **Only final outputs:** By examining contributions to the final residual stream, DLA misses the intermediate computations and information flow between components
- **No mechanism insight:** DLA cannot tell us what computation a component is performing or what features it's responding to
- **Interpretation requires care:** What DLA reveals varies case-by-case depending on whether contributions are concentrated or distributed. Each analysis requires reasoning from first principles to form and narrow down hypotheses without drawing overly strong conclusions

For these reasons, DLA is best viewed as a starting point for mechanistic interpretability—a way to efficiently identify which components warrant deeper investigation through complementary techniques such as activation patching and analysis of attention patterns in specific heads.