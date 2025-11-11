---
layout: post
title: "From Hours to Seconds: Optimising BPE Tokeniser Training" 
---

Training a tokeniser on even modest datasets can be surprisingly slow. When I implemented the standard chunked BPE algorithm (following [Andrej Karpathy's tutorial](https://www.youtube.com/watch?v=zduSFxRajkE)) and ran it on a 100MB dataset, the estimated training time was several hours. For context, I'm building a toy language model from scratch. The tokeniser is just a small part of the puzzle, and here I was, watching my CPU churn through what seemed like it should be a straightforward algorithm. 

This article details the six optimisations to the vanilla BPE algorithm I implemented - how and why they work as well as their performance gains. When combined they bring that training time down to just 13.3s! The full implementations are available on [GitHub](https://github.com/xycoord/Language-Modelling/tree/main/src/lm_tokenizers) so you can follow along with the optimisations as we go.

<!--more-->

For readers unfamiliar with the BPE algorithm, let’s begin there.

# Introduction to BPE

The Byte Pair Encoding (BPE) tokeniser algorithm is one of the most popular algorithms for tokenisation. 

The aim of tokenisation is to compress the sequence length of a piece of text by expressing collections of characters as single tokens (integers). For LLMs, this compression can greatly improve efficiency since the core attention mechanism is O(sequence length^2). However, poor tokenisation can: unnecessarily increase the size of the embedding table (vocab size * embedding dim); over-compress common phrases such that it becomes difficult for the model to learn more atomic semantic information.

This leads us to the specification:

> For a given distribution of text and a given vocab size, maximally compress the sequence length while keeping tokens as linguistically meaningful units.
> 

Byte Pair Encoding (BPE) is one of the most popular algorithms for training a tokeniser to meet this specification. We'll return to the constraint on "linguistically meaningful units" in the section on chunking.

# Vanilla BPE

We begin by encoding the text as UTF-8 bytes. This gives us a minimal 256 token vocabulary which can express any unicode character. Compressing the length of this sequence is done by merging pairs of tokens to form a new token following a set of learnt rules. Each rule is of the form:

$$
(a,b)\leftrightarrow c
\quad\text{where} \quad 
a,b<c
$$

The constraint $a,b<c$ means that new tokens always have higher IDs than existing tokens. This provides an ordering of the rules such that $c$ may be involved in future merges allowing significant compression. Each merge decreases the sequence length by one.

To detokenise (aka decode tokens), the rules are applied in reverse. Each token in the vocabulary corresponds to a sequence of bytes. Therefore, decoders are often simply a dictionary from tokens to byte sequences or strings.

Vanilla BPE tries to learn a set of merge rules to satisfy the specification:

> For a given distribution of text and a given vocab size, maximally compress the sequence length.
> 

The algorithm is:

> 1. Find the most common adjacent pair of tokens in the sequence
> 2. Mint a new token and record the rule
> 3. Apply the new rule to the token sequence
> 4. Repeat until we have the desired vocab_size

In practice, step 1 involves two distinct operations that will become important for optimisation:

> 1. a. Count the occurrences of each unique adjacent pair in the sequence
> 1. b. Find the maximum count

In python we can implement this as:

```python
Token = int
TokenPair = tuple[Token, Token]
merges: dict[TokenPair, Token] = {}
vocab: dict[Token, bytes] = {token: bytes([token]) for token in range(self.vocab_size)}

def train(self, token_seq: list[Token], target_vocab_size: int):
				next_token = 256
        while next_token < target_vocab_size:

            pair_counts = count_pairs(token_seq)
            
            if not pair_counts:
                break # no more pairs to merge, we're done

            most_common_pair = max(pair_counts, key=pair_counts.get)
            
            new_token = next_token
            merges[most_common_pair] = new_token
            vocab[new_token] = (vocab[most_common_pair[0]] + 
										            vocab[most_common_pair[1]])

            token_seq = merge_pair(token_seq, most_common_pair, new_token)

            next_token += 1
```

```python
def count_pairs(token_seq: list[Token]) -> dict[TokenPair, int]:
    counts = {}
    for pair in zip(token_seq, token_seq[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts
    
def merge_pair(token_seq: list[Token], pair: TokenPair, new_token: Token) -> list[Token]:
    new_token_seq = []
    i = 0
    while i < len(token_seq):
        if token_seq[i] == pair[0] and i < len(token_seq) - 1 and token_seq[i+1] == pair[1]:
            new_token_seq.append(new_token)
            i += 2
        else:
            new_token_seq.append(token_seq[i])
            i += 1
    return new_token_seq
```

# Chunking

While vanilla BPE effectively compresses sequences, it can create problematic tokens when given a large vocabulary size. If phrases like "on the other hand" or "at the end of the day" appear frequently in training data, they may become single tokens. This violates our requirement of "keeping tokens as linguistically meaningful units", causing two problems for the downstream language model:

1. The model must learn that these phrase tokens decompose into constituent parts - a difficult task that relies on seeing variations or typos in training data
2. If the model fails to learn this relationship, it will likely be confused when presented with parts, variations, or typos of these phrases. For example, for the context “at the end of the” it may not learn that “day” is a likely next token.

Chunking solves this by breaking training text into segments (typically words) and preventing merges across segment boundaries. This ensures tokens remain linguistically meaningful units rather than arbitrary frequent sequences.

The choice of how to chunk text (defined by a regex) can be crucial to the performance of the language model using the tokeniser. For instance, how should you chunk numbers? one option is to chunk entire numbers like words. This would mean that common numbers such as 1024 would likely get their own token while neighbouring 1023 might not  (perhaps tokenising as [10,23] or [102,3]). This can make it difficult for models to perform arithmetic since the algorithm becomes complex with many edge cases. Alternatively, it has become common to chunk on digits, such that each digit is tokenised independently. This has been shown to improve model performance on arithmetic.

Designing effective chunking patterns is an art form in itself. Techniques have evolved as practitioners discovered limitations with particular tokenisation schemes and extend support to languages with quite different structures from English. While essential to downstream model performance, these details are beyond our scope - we'll focus on the algorithmic optimisations that apply regardless of chunking strategy. [https://arxiv.org/abs/2402.14903](https://arxiv.org/abs/2402.14903)

---

The rest of this article describes six optimisations to the chunked BPE algorithm. Each version of the algorithm is equivalent in that they produce identical outputs to the original - just 2000x faster*. Most are algorithmic improvements - reducing the big O of the loop - and all can be implemented in Python - they don't require a low-level language. The optimisations follow a common pattern: identifying where redundant work is being performed and finding ways to cache, track, or avoid these repeated computations.

# Benchmarks

In order to keep track of the real world performance improvements of each optimisation I use a variety of benchmarks. The chunked implementation described above is O(dataset size * new tokens) and different optimisations help reduce either factor. Therefore, I use two datasets of different sizes and a range of vocab sizes for benchmarking.

1. A corpus of books from Project Gutenberg with total size 114MB.
2. The Hugging Face Wikipedia dataset (English) with total size 18.29GB.

*To show how performance improves with each optimisation, I'll use a consistent benchmark of the Gutenberg dataset with vocab size 1000 (774 new tokens). Keep in mind this isn't fully representative since most optimisations reduce the Big-O and thus shine more with larger datasets and vocab sizes.

| Dataset | Vocab Size | Preprocessing Time | BPE Training Time |
| --- | --- | --- | --- |
| Gutenberg | 1000 | 34.1s | 30338s (8.42hrs) |

This baseline time was extrapolated assuming linear increase with vocab size. I didn’t fancy waiting 8 hours for the result.

Let’s now look at each optimisation in turn.

# Chunk Deduplication

The frequent repetition of words in language means chunks frequently repeat in the training text. The chunks “the” and “a” may appear almost as many times as you have sentences! Executing `count_pairs` and `merge_pairs` on all chunks therefore requires considerable repeated work. It is more efficient to first deduplicate the chunks such that each unique chunk is processed only once. To ensure that the most common pair remains accurate, we must keep track of the number of copies of each chunk in the original text, and multiply accordingly in `count_pairs`.

Vanilla BPE processes every token in the dataset once per iteration, making it O(vocab_size × total_tokens). After chunk deduplication, we have O(vocab_size × unique_chunks × avg_chunk_length), and crucially `unique_chunks` grows **sub-linearly** with dataset size due to [Heaps' law](https://en.wikipedia.org/wiki/Heaps%27_law) in natural language. This means all subsequent training operations become sub-linear in dataset size.

When implementing this we use a type alias `WeightedChunk = tuple[list[Token], int]` , and perform deduplication with the python class `Counter` .

```python
def preprocess_train(self, text: str) -> list[WeightedChunk]:
    text_chunks = re.findall(self.split_regex, text)
    chunk_counts = Counter(text_chunks)
    chunks = [(list(chunk_text.encode("utf-8")), num_copies) for chunk_text, num_copies in chunk_counts.items()]
    return chunks
```

```python
def count_pairs(token_seq: list[Token], num_copies: int = 1, 
                counts: Optional[dict[TokenPair, int]] = None
                ) -> dict[TokenPair, int]:
    counts = {} if counts is None else counts
    for pair in zip(token_seq, token_seq[1:]):
        counts[pair] = counts.get(pair, 0) + num_copies
    return counts

def count_pairs_chunked(chunks: list[WeightedChunk]) -> dict[TokenPair, int]:
    counts = {}
    for token_chunk, num_copies in chunks:
        count_pairs(token_chunk, num_copies, counts)
    return counts
```

We can exploit the same property of language when encoding new text (converting text to tokens using our learned rules). By caching the encodings of chunks in a dictionary as we process them, we avoid re-tokenising the same words repeatedly. In my code, the cache starts fresh for each call of `encode`, however, I expect that many production tokenisers will begin with a precomputed cache of the top chunks encountered in training.

| Dataset | Vocab Size | Preprocessing Time | BPE Training Time |
| --- | --- | --- | --- |
| Gutenberg | 1000 | 15.9s | 796s (13.3mins) |

The preprocessing time improvement (from 34.1s to 15.9s) occurs because deduplication happens before expensive byte encoding and list conversion operations.

---

Profiling the new training loop reveals that the majority of work is spent roughly equally between functions `count_pairs_chunked` and `merge_pairs` .  These are both O(chunks), performing a full pass over the chunks on each iteration. The next two optimisations work together to remove this full pass, instead focusing only on the changes made. On each iteration, most chunks don't contain the most common pair (especially later in training), therefore O(changes) << O(chunks), offering significant speedup potential.

# Incremental Counting

On each iteration, the only counts which change are those of pairs involved in the merges. Therefore, instead of re-counting the pairs from scratch each time, we can instead track the changes to the counts: if a pair is added to a chunk we increase its count by `copies` and if it’s removed we decrease its count by `copies`. We track these changes in a dictionary `deltas` : pair → delta.

During each merge there are up to 5 pairs to consider: 3 that disappear and 2 that appear (fewer at boundaries). Suppose the token sequence is:

$\dots, a,b,c,d,\dots$

and the merge is: $(b,c)\rarr e$,

$(a,b),(b,c),(c,d)$  decrease by `copies`

$\dots,a,e,d,\dots$

$(a,e), (e,d)$ increase by `copies`

Using this logic we can write a new version of `merge_pairs` :

```python
def merge_pair_track_deltas(token_seq: list[Token], pair: TokenPair, new_token: Token, copies: int) -> tuple[list[Token], dict[TokenPair, int]]:
    new_token_seq = []
    pair_deltas = defaultdict(int)
    if len(token_seq)< 2:
        return token_seq, pair_deltas

    i = 0
    while i + 1 < len(token_seq):
        if token_seq[i] == pair[0] and token_seq[i+1] == pair[1]:
            pair_deltas[(token_seq[i], token_seq[i+1])] -= copies        # (b, c)
            if i > 0:
                pair_deltas[(new_token_seq[-1], token_seq[i])] -= copies # (a, b)
                pair_deltas[(new_token_seq[-1], new_token)] += copies    # (a, e)
            if i + 2 < len(token_seq):
                pair_deltas[(token_seq[i+1], token_seq[i+2])] -= copies  # (c, d)
                pair_deltas[(new_token, token_seq[i+2])] += copies       # (e, d)

            new_token_seq.append(new_token)
            i += 2
        else:
            new_token_seq.append(token_seq[i])
            i += 1

    # If the last pair is not merged, its second token still needs to be added    
    if i < len(token_seq):
        new_token_seq.append(token_seq[i])

    return new_token_seq, pair_deltas

def apply_deltas(target: dict[TokenPair, int], deltas: dict[TokenPair, int]) -> None:
    for pair, delta in deltas.items():
        target[pair] += delta
```

This optimisation removes the `count_pairs` function call from the training loop, since the work is now done in `merge_pairs_track_deltas`  and `apply_deltas`. `count_pairs` is still used for initialising the `pair_counts`.

# Track Pair Locations

Merge pairs has to do a full search of all chunks each iteration to find the instances of the pair it’s merging. We can remove this search by keeping track of the chunks in which each pair exists. `pair_chunk_locations` is a dict from pairs to the set of the indices of the chunks containing them
(e.g., `{(97, 97): {0, 5, 12, ...}}`).

This also only changes when we perform a merge so we also track it iteratively like `pair_counts`. For each chunk, we can use the deltas from the merge of that chunk to help maintain `pair_chunk_locations` . If the delta for a pair is positive, that pair must contain the new token so it didn’t exist previously in the chunk and we add the chunk index to that pair’s set. If it's negative, the pair may not have been removed entirely from the chunk (multiple occurrences of the same pair), so we need to double-check before removing the chunk index from its set.

```python
    def _update_pair_chunk_locations(self, chunk_index: int, chunk: list[Token], deltas: dict[TokenPair, int]) -> None:
        for pair, delta in deltas.items():
            if delta > 0:
                self._pair_chunk_locations[pair].add(chunk_index)
            elif delta < 0:
                is_pair_still_in_chunk = any(p == pair for p in zip(chunk, chunk[1:]))
                if not is_pair_still_in_chunk:
                    self._pair_chunk_locations[pair].discard(chunk_index)
                    if not self._pair_chunk_locations[pair]:
                        del self._pair_chunk_locations[pair]
```

## Bringing It Together

Incremental counting alone can only halve the runtime since it eliminates one of two equally expensive O(unique_chunks) operations. However, `merge_pairs` still searches all chunks, so the Big-O remains unchanged. Together with pair location tracking, both operations become O(chunks_containing_pair), where chunks_containing_pair << unique_chunks, especially as training progresses and pairs become more specialised ([Zipf’s law](https://en.wikipedia.org/wiki/Zipf%27s_law)). This is where the fundamental algorithmic improvement occurs.

Since this optimisation requires the hairy maintenance of a new data structure, I elected to implement a worker class `IncrementalBPEWorker` to enforce the invariant relationship between `chunks` and `_pair_chunk_locations` . This invariant is:

- For every pair `p` and chunk index `i`, `p` exists in `chunks[i]`, iff `i` is in `_pair_chunk_locations[p]`
- No empty sets exist in `_pair_chunk_locations.values()`
- Only chunks containing the target pair are modified during merge operations

The class performs a full index of the pairs, populating `_pair_chunk_locations` on initialisation and has the primary method `merge_pair_incremental` which brings together these two optimisations, keeping the main training loop clean.

```python
def merge_pair_incremental(self, pair_to_merge: TokenPair, new_token: Token) -> dict[TokenPair, int]:
        total_pair_deltas = defaultdict(int)

        if pair_to_merge not in self._pair_chunk_locations:
            return total_pair_deltas

        # list() to avoid modifying the iterator while iterating
        for chunk_index in list(self._pair_chunk_locations[pair_to_merge]):
            chunk, copies = self.chunks[chunk_index]

            chunk, deltas = merge_pair_track_deltas(chunk, pair_to_merge, new_token, copies)

						self.chunks[chunk_index] = (chunk, copies)
            apply_deltas(total_pair_deltas, deltas)
            self._update_pair_chunk_locations(chunk_index, chunk, deltas)

        return total_pair_deltas
```

# Reduce Churn

My original implementation of `merge_pairs_track_deltas` was a pure function which built a new list of tokens on each call. The old lists then became garbage causing expensive calls to the garbage collector. Replacing this with an imperative implementation that edits the lists in place greatly reduced the churn. I was sad to give up the elegant functional code but the performance gain was worth it.

I discovered this inefficiency by profiling the training loop and noticing that a trivial line was taking a disproportionately long time - a tell-tale sign the garbage collector was running. I confirmed this by explicitly calling the garbage collector which shifted the blame.

```python
def merge_pair_track_deltas_in_place(token_seq: list[Token], pair: TokenPair, new_token: Token, copies: int) -> dict[TokenPair, int]:
    pair_deltas = defaultdict(int)
    
    # We use two pointers: a 'read' pointer to scan the original sequence,
    # and a 'write' pointer to build the new sequence in the same list.
    write_idx = 0
    read_idx = 0

    while read_idx < len(token_seq):
        # Check for a pair match at the current read position
        is_match = (
            read_idx + 1 < len(token_seq) and
            token_seq[read_idx] == pair[0] and
            token_seq[read_idx+1] == pair[1]
        )
        
        if is_match:
            # A pair was found. Calculate deltas BEFORE overwriting the data.
            pair_deltas[(token_seq[read_idx], token_seq[read_idx+1])] -= copies
            if write_idx > 0:
                pair_deltas[(token_seq[write_idx-1], token_seq[read_idx])] -= copies
                pair_deltas[(token_seq[write_idx-1], new_token)] += copies
            if read_idx + 2 < len(token_seq):
                pair_deltas[(token_seq[read_idx+1], token_seq[read_idx+2])] -= copies
                pair_deltas[(new_token, token_seq[read_idx+2])] += copies

            # Perform the in-place write
            token_seq[write_idx] = new_token
            write_idx += 1
            read_idx += 2  # Skip the two tokens we just merged
        else:
            # No match, just copy the token from read to write position.
            if read_idx != write_idx:
                token_seq[write_idx] = token_seq[read_idx]
            write_idx += 1
            read_idx += 1
            
    # Truncate the list to remove the old, leftover data at the end.
    if write_idx < len(token_seq):
        del token_seq[write_idx:]
    
    return pair_deltas
```

| Dataset | Vocab Size | Preprocessing Time | BPE Training Time |
| --- | --- | --- | --- |
| Gutenberg | 1000 | 15.9s | 14.1s |
| Gutenberg | 10000 | 15.9s | 128.3s |
| Gutenberg | 50000 | 15.9s | 1442s |

# Fast Max

Now that `merge_pairs` is heavily optimised, profiling revealed that the line:

```python
most_common_pair = max(pair_counts, key=pair_counts.get)
```

is now the bottleneck. It's an O(unique_pairs) operation and in most cases O(unique_pairs) > O(chunks_containing_pair). However, we can reduce this by maintaining a second data structure. In addition to `pair_counts` which is a dict (key: pair, value: count), we introduce an equivalent sorted dict `counts_to_pairs` (key: count, value: set of pairs) using `SortedDict` from `sortedcontainers`. Finding the most common pair is now O(1) because the dict is sorted! Maintaining `counts_to_pairs` requires O(log k) work per delta update, where k is the number of distinct count values. Since deltas are sparse, typically O(chunks_containing_pair), the amortised cost per iteration is much smaller than O(unique_pairs).

Since the two data structures are tracking the same thing and require complex maintenance, I abstracted them into a class `PairCountsTracker` with methods `__init__` , `apply_deltas` and `get_most_common_pair` . It ensures invariants:

- For every pair `p`: `p` is in `pair_counts` with count `c` iff `p` is in `counts_to_pairs[c]`
- All values in `pair_counts` are > 0
- No empty sets exist in `counts_to_pairs.values()`

We still need to maintain `pair_counts` since it allows us to quickly find which count a given pair has (and thus to which set in `counts_to_pairs` it belongs).

```python
def apply_deltas(self, deltas: dict[TokenPair, int]) -> None:
        for pair, delta in deltas.items():
            if delta == 0:
                continue
            
            old_count = self.pair_counts.get(pair, 0)
            new_count = old_count + delta

            if new_count < 0:
                raise ValueError(f"New count is negative: {new_count}, violating invariant")

            if old_count > 0: # pair existed previously
                old_set = self.counts_to_pairs[old_count]
                old_set.remove(pair)
                # clean up
                if not old_set: # remove empty set (essential to remove false max counts)
                    del self.counts_to_pairs[old_count]
                if new_count <= 0: # pair no longer exists
                    del self.pair_counts[pair]
            
            if new_count > 0: # pair exists now
                self.pair_counts[pair] = new_count
                if new_count not in self.counts_to_pairs:
                    self.counts_to_pairs[new_count] = set()
                self.counts_to_pairs[new_count].add(pair)
                
```

| Dataset | Vocab Size | Preprocessing Time | BPE Training Time |
| --- | --- | --- | --- |
| Gutenberg | 1000 | 15.9s | 13.3s |
| Gutenberg | 10000 | 15.9s | 16.8s |
| Gutenberg | 50000 | 15.9s | 19.2s |
| Gutenberg | 100000 | 15.9s | 24.0s |
| Wikipedia | 1000 | 638s | 395s |

## Algorithmic Transformation

The benchmark results reveal a fundamental shift in how the algorithm works. Notice how training time barely increases with vocab_size (from 13.3s at 1000 tokens to just 24.0s at 100,000 tokens) - this dramatic improvement stems from moving computational work from repeated per-iteration operations to one-time initialisation:

1. **Initial pair counting**: Populating `pair_counts` - O(unique_chunks × avg_chunk_length)
2. **Pair location tracking**: Building `_pair_chunk_locations` index - O(unique_chunks × avg_chunk_length)
3. **Fast max**: Initialising `counts_to_pairs` sorted structure - O(unique_pairs + k log k) where k is the number of distinct count values. This is very small since k ≪ unique_pairs ≤ 256².

This transforms the algorithm from repeating expensive operations vocab_size times to performing small incremental updates on pre-built data structures.

# Parallelisation

Most of the work in the training loop now happens in the `merge_pairs_incremental` call which processes each chunk independently. This suggests that the algorithm would be well suited to parallelisation. However, there are a couple of challenges to take into account.

First, moving data between processes is expensive in Python because it must be pickled and unpickled. Therefore, we want to minimise moving chunks around. I elected to setup n persistent workers on separate processes which each hold a split of the chunks. These are just instances of the `IncrementalBPEWorker` class from earlier. For each merge, the merge instruction (pair, new_token) is sent to each worker and each returns only deltas. Notably, the chunks must be transferred to their processes once at the start of training which is relatively expensive.

Second, as training progresses, the number of merges per new token decreases significantly ([Zipf’s Law](https://en.wikipedia.org/wiki/Zipf%27s_law)). For a smaller dataset (~100MB), the cost of parallelisation quickly outweighs the time per `merge_pairs` . This means the parallelised version is actually slower. To overcome this, I implemented an adaptive parallel version which switches to sequential merging part way through. It times each merge and uses a windowed average to choose when to switch. Unfortunately, performance is very sensitive to the threshold which requires tuning to the specific hardware.

A major issue with this approach is that it requires transfer of the chunks back to the main process and a re-index of `_pair_chunk_locations` during the transition back to serial processing. These are expensive operations which now need to be performed twice. They are O(chunks) meaning that as the dataset grows - improving the benefit of parallelisation - they become more expensive limiting the benefit of parallelisation. Furthermore, the later the switch from parallel to serial, the cheaper it is, because more merges have occurred and `chunks` is smaller. This makes the optimal switch threshold sensitive also to the size of the dataset, making it impossible to perfectly tune without running full scale training. 

Despite these limitations, the adaptive approach still provides benefits for larger datasets. I observed a 4x speedup over fully sequential with the Wikipedia dataset and a vocab size of 1000 tokens.

| Dataset | Vocab Size | Preprocessing Time | BPE Training Time |
| --- | --- | --- | --- |
| Wikipedia | 1000 | 638s | 91.7s |
| Wikipedia | 10000 | 638s | 289s |
| Wikipedia | 50000 | 638s | 426s |

# The New Bottleneck - Regex

Looking at our latest results, we see that preprocessing time (638s) now significantly outweighs training time (426s). Profiling this revealed that the chunking with regex dominates this stage. 

There are no obvious ways I could find to speed this up. While fast linear-time regex engines like Rust's regex crate and Go's regex exist, they don't support the complex features (e.g. negative lookaheads) required by widely-used tokeniser patterns like in GPT4 and Llama 3. These use pattern features only supported by engines like Python's regex or Rust's fancy-regex (used in tiktoken), which sacrifice linear-time guarantees for feature completeness. Moreover, the preprocessing is at best linear in dataset size, meaning it will dominate more as the size of the dataset is increased. 

However, the preprocessing is far more amenable to parallelisation than the main training phase, so long as the training text is already split into documents which can be preprocessed independently. My experiments ran on only 8 threads and preprocessing could easily make use of far more.

# Conclusion

The results speak for themselves. We've achieved truly dramatic performance improvements through relatively simple algorithmic optimisations. Taking BPE training from over 8 hours down to seconds represents a 2000x speedup that makes tokeniser training practically instant for most use cases. Even the large Wikipedia dataset (18GB) completes in under 18 minutes total with Fast Max optimisation, or about 12 minutes with parallelisation. For the vast majority of practitioners training custom tokenisers, these times are more than sufficient - you could retrain your tokeniser dozens of times while experimenting with different split patterns.

However, our success has revealed where the true computational bottleneck lies - in the regex preprocessing. This stage remains linear in dataset size while our optimised training is now sub-linear thanks to chunk deduplication. This means regex will inevitably dominate for large enough datasets, and our optimisations have made training so efficient that this appears even on moderate-sized datasets.

This insight changes the rewrite strategy. Any effort to port to a lower-level language such as Rust should begin with benchmarking regex performance using `fancy-regex` to determine if the speedup justifies the effort. Even then, it might only be worth rewriting the preprocessing step since the training loop is already highly optimised.

For most applications, the FastMaxBPETokeniser is probably more than fast enough. And let's be real - tokeniser training is likely a tiny cost compared to actually training the language model.