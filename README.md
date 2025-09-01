# JHQ: Johnson-Lindenstrauss Enhanced Hierarchical Quantization

Implementation for **VLDB 2026** submission: *JHQ: Johnson-Lindenstrauss Enhanced Hierarchical Quantization for High-Dimensional Approximate Nearest Neighbor Search*

## Quick Start

```bash
git clone --recurse-submodules https://github.com/jiabhan/JHQ.git
```

**Note**: We use [FAISS](https://github.com/facebookresearch/faiss) as a third-party library for baseline comparisons. The `--recurse-submodules` flag will automatically download FAISS and other dependencies.

- Third party library: `/external`
- Code implementation: `/jhqlib`
- Example test script: `/examples`

To set the dataset and index folder, in root folder's `CMakeList.txt`, modify the follow lines:
```
# Modify this to your desired dataset dir
set(DATA_DIR /datasets/)
# Modify this to your desired index dir
set(INDEX_DIR /index/)
```

## Datasets

### Test Datasets (100K vectors)
- [OpenAI3-3072-100K](https://huggingface.co/datasets/Qdrant/dbpedia-entities-openai3-text-embedding-3-large-3072-100K)
- [OpenAI3-1024-100K](https://huggingface.co/datasets/Qdrant/dbpedia-entities-openai3-text-embedding-3-large-1024-100K)

### Full Evaluation Datasets
- [OpenAI3-1536](https://huggingface.co/datasets/Qdrant/dbpedia-entities-openai3-text-embedding-3-large-1536-1M)
- [OpenAI3-3072](https://huggingface.co/datasets/Qdrant/dbpedia-entities-openai3-text-embedding-3-large-3072-1M)
- [Arxiv-Abstracts-768](https://huggingface.co/datasets/macrocosm/arxiv_abstracts)
- [Vogue-768](https://huggingface.co/datasets/tonyassi/vogue933k-embeddings)
- [BGE-M3-1024](https://huggingface.co/datasets/Upstash/wikipedia-2024-06-bge-m3)
- [Stella-TREC24-1024](https://huggingface.co/datasets/ielabgroup/stella_trec24_biogen_embedding)
