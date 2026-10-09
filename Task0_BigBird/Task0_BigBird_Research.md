# Task 0: Research About BigBird Model

## 1. Introduction

BigBird is a Transformer architecture designed to process long sequences efficiently.
It was introduced to overcome one of the main limitations of standard Transformer
models such as BERT: the quadratic computational and memory cost of full
self-attention.

In standard self-attention, every token attends to every other token in the sequence.
For a sequence of length n, this produces O(n²) attention complexity.

BigBird replaces full attention with a sparse attention mechanism, reducing this
dependency from quadratic to approximately linear with respect to sequence length.

---

## 2. Motivation

Models such as BERT work well for many NLP tasks, but full self-attention becomes
expensive when the input document is very long.

This creates difficulties when processing:

- Research papers
- Legal documents
- Long reports
- Books
- Medical records
- Long question-answering contexts

BigBird was designed to preserve the advantages of Transformers while making much
longer sequences practical.

---

## 3. BigBird Architecture

BigBird uses a block-sparse attention mechanism instead of allowing every token to
attend to every other token.

Its sparse attention pattern combines three main types of attention:

### 3.1 Local / Sliding-Window Attention

Each token attends to nearby tokens.

This captures local context because neighboring words are often strongly related.

### 3.2 Global Attention

A small number of global tokens can attend to the entire sequence and can be
attended to by other tokens.

These tokens help information travel between distant parts of a long document.

### 3.3 Random Attention

Tokens also attend to a number of randomly selected tokens.

Random connections allow information to move between distant parts of the
sequence without requiring full attention.

Together, local, global, and random attention provide both local and long-range
information while keeping the attention matrix sparse.

---

## 4. Block-Sparse Attention

Instead of computing all token-to-token relationships, BigBird organizes attention
into sparse blocks.

Only selected blocks of the attention matrix are computed.

This significantly reduces memory and computational requirements and enables the
model to process substantially longer sequences than traditional full-attention
Transformers.

---

## 5. Computational Complexity

### BERT

BERT uses full self-attention.

Complexity:

O(n²)

As the sequence length increases, memory and computational requirements grow
quadratically.

### Longformer

Longformer combines local sliding-window attention with task-specific global
attention.

Its attention complexity scales approximately linearly with sequence length when
the attention window is fixed.

### BigBird

BigBird combines:

- Local/sliding-window attention
- Global attention
- Random attention

With a fixed sparse attention configuration, its attention cost scales approximately
linearly with sequence length rather than quadratically.

Therefore, both BigBird and Longformer are better suited than standard BERT for
long documents.

---

## 6. BigBird vs BERT vs Longformer

| Feature | BERT | Longformer | BigBird |
|---|---|---|---|
| Attention | Full | Sparse | Sparse |
| Local Attention | Full connectivity | Sliding window | Sliding/local |
| Global Attention | Not sparse/global-specific | Yes | Yes |
| Random Attention | No | No | Yes |
| Complexity | O(n²) | Approximately linear | Approximately linear |
| Long Documents | Limited | Suitable | Suitable |

The distinctive feature of BigBird is the combination of local, global, and random
attention patterns.

---

## 7. Applications

BigBird is particularly useful for tasks involving long input sequences, including:

- Long-document classification
- Question answering
- Document summarization
- Scientific and academic documents
- Legal document analysis
- Medical and clinical documents
- Genomics and biological sequences

---

## 8. Advantages

1. Handles substantially longer sequences than standard full-attention Transformers.
2. Reduces the quadratic attention dependency to linear scaling under a fixed sparse pattern.
3. Requires less attention memory for long sequences.
4. Maintains both local and long-range information through multiple sparse attention patterns.
5. Can improve performance on NLP tasks requiring long context.
6. Retains important theoretical properties of full-attention Transformers.

---

## 9. Limitations

1. Sparse attention is more complicated than standard full attention.
2. The attention pattern and implementation are more complex.
3. For short sequences, using BigBird may provide little advantage over simpler Transformer models.
4. Sparse attention does not compute every possible token-to-token relationship directly.
5. Model effectiveness can depend on the task and attention configuration.

---

## 10. Performance Characteristics

The original BigBird research showed that its sparse attention mechanism reduces
the quadratic dependency of full attention to linear scaling.

The model can process sequences significantly longer than those practical with
standard BERT-like full attention using comparable hardware.

This makes BigBird particularly valuable when important information is distributed
across large documents.

---

## Conclusion

BigBird addresses one of the major limitations of traditional Transformers:
quadratic self-attention complexity.

Instead of computing attention between every pair of tokens, BigBird combines
local/sliding-window, global, and random attention in a block-sparse pattern.

This reduces attention cost while retaining access to both nearby and distant
information, making BigBird a strong Transformer architecture for long-document
NLP applications.

## References

- Zaheer et al. (2020), "Big Bird: Transformers for Longer Sequences."
- Beltagy, Peters, and Cohan (2020), "Longformer: The Long-Document Transformer."