# Cellula NLP Internship — Week 3

## Alaa Boghdady

This repository contains the Week 3 tasks for the Cellula Technologies NLP Internship.

---

## Task 0 — BigBird Transformer Research

A research report covering:

- BigBird architecture and motivation
- Limitations of standard quadratic self-attention
- Sparse attention mechanisms
- Local, global, and random attention
- Block-sparse attention patterns
- Computational complexity
- Comparison with BERT and Longformer
- Long-document applications
- Advantages and limitations

Research file:

`Task0_BigBird/Task0_BigBird_Research.md`

---

## Task 1 — Personal RAG Knowledge Base

A Retrieval-Augmented Generation prototype built using professional information from my CV.

### Pipeline

PDF CV
→ Document Loading
→ Text Chunking
→ Sentence Embeddings
→ FAISS Vector Database
→ Semantic Retrieval
→ Grounded Professional Context
→ Source Identification

### Technologies

- Python
- LangChain
- Hugging Face Sentence Transformers
- all-MiniLM-L6-v2
- FAISS
- PyPDF

### How It Works

1. The CV is loaded from PDF.
2. The document is divided into overlapping text chunks.
3. Each chunk is converted into a semantic embedding.
4. The embeddings are stored in a FAISS vector database.
5. The user enters a question about the professional profile.
6. FAISS performs semantic similarity search.
7. The most relevant professional context is returned.
8. The original document and page are displayed as the source.

### Hallucination Reduction

The final retrieval layer returns information directly grounded in the source document rather than generating unsupported information.

This improves transparency and allows the user to identify the source of the retrieved information.

### Current Limitation

A local FLAN-T5-small generation layer was experimentally tested. However, because the lightweight local model produced inconsistent answer quality, the final prototype prioritizes grounded semantic retrieval and source transparency.

A stronger instruction-following LLM can be integrated as the generation layer in future development.

### Run

Install dependencies:

```bash
pip install -r Task1_RAG/requirements.txt
