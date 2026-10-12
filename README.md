# Morphing Vectors

**Recency-adaptive quantization tiers for approximate nearest-neighbor (ANN) indexes.**

Morphing Vectors is a from-scratch ANN index that stores each vector at a quantization precision that depends on how recently it was inserted or queried. Hot vectors stay at high precision. As vectors age, they are lazily migrated to coarser product-quantization tiers. The search graph's topology never changes; only the precision of the payload attached to each node does.

> Companion code for the paper *"Morphing Vectors: Recency-Adaptive Quantization Tiers for Approximate Nearest-Neighbor Indexes"* (Shehan Liyanaarachchi).
> Extends the Morphing Pages encoding-transition idea from columnar storage to vector indexing, and shares an architectural thread with the QUANT-DB hybrid SQL/semantic query engine.

---

## Motivation

Standard ANN indexes (HNSW, IVF, PQ-based systems) apply one fixed precision to every stored vector, regardless of how often or how recently it is queried. In recency-skewed workloads such as financial time-series embeddings, logs, and sensor streams, this wastes storage on old data that is rarely touched.

Morphing Vectors treats **per-vector precision as a time-varying property** driven by a recency signal:

- New vectors are inserted at the finest tier.
- Vectors that go untouched past an age boundary migrate to coarser tiers.
- Retrieval "touches" a vector and keeps it hot.
- Migration is O(1) per vector (tiers use independently trained quantizers, so no residual chaining) and never rebuilds the graph.

## Key Ideas

| Idea | Summary |
|---|---|
| **Recency tiers** | Hot / Warm / Cold tiers, each with its own independently trained product quantizer. |
| **Fixed topology** | The graph is built once. Migration swaps only a node's payload in place. |
| **Lazy, amortized migration** | A priority heap schedules due migrations, processed in bounded batches per operation. |
| **Tier-aware routing** | Routing nodes (layer ≥ 1) are pinned at the finest tier and exempt from age-based coarsening. |

## Everything Is Built From Scratch

No FAISS, hnswlib, or scikit-learn is used inside the system under test. All arithmetic is plain NumPy so the tier-migration engine can hook directly into every component (for example, in-place precision swaps that a library's opaque index would not expose).

| Component | File | What's custom |
|---|---|---|
| Product quantizer | `core/quantizer.py` | k-means++ seeding, Lloyd's algorithm, mini-batch variant for large N, encode/decode, error tracking |
| Hierarchical graph | `core/graph_index.py` | Multi-layer HNSW-style graph: random layer assignment, greedy descent, bounded beam search, closest-M neighbor selection |
| Recency tier manager | `core/tiers.py` | Recency tracking, age-boundary assignment, heap-scheduled lazy migration, hub pinning |
| Storage layer | `core/storage.py` | Per-vector encoding metadata and targeted re-encoding |
| Top-level system | `core/index.py` | `MorphingVectorIndex` composing quantizer, graph, and tier manager |
| Baseline | `core/baseline.py` | `FlatIndex`: same components at a single fixed tier, for controlled comparison |
| Theory | `core/theory.py` | Proposition statements and numeric validators |

> **Scope note.** The novel contribution is the recency-tiered precision mechanism and its cost properties. The quantizer and graph are engineering choices that make the mechanism attributable and mutable in place. They are **not** claimed as algorithmic improvements over PQ or HNSW. A production deployment would reasonably sit on top of FAISS or hnswlib.

---

## How It Works

```
Insert(v, x):   encode x at Tier 0  ->  add node to graph  ->  register in tier manager  ->  run due migrations
Query(q, k):    graph search  ->  touch returned vectors (reset recency)  ->  run due migrations  ->  return top-k
```

Default tier configuration (d = 32):

| Tier | Subvectors (M) | Bits (b) | Bytes/vector | Age boundary |
|---|---|---|---|---|
| Hot (0) | 16 | 8 | 16 | 150 ticks |
| Warm (1) | 8 | 5 | 5 | 500 ticks |
| Cold (2) | 2 | 4 | 1 | ∞ |

Graph parameters: M = 16, ef_construction = 100, query ef = 150.

---

## Results

Evaluated on a **temporally correlated synthetic workload** (slowly drifting center plus per-vector noise, so insertion order correlates with spatial position). The primary external baseline is FAISS `IndexHNSWFlat`, run on identical data and ground truth.

### Recall@10

| System | N = 2,000 | N = 20,000 |
|---|---|---|
| FAISS IndexHNSWFlat (external reference) | 1.00 | 1.00 |
| FlatFine (from-scratch, no tiering) | 0.92 | 0.44 |
| **Morphing Vectors** (from-scratch, tiered) | 0.80 | 0.08 |
| FlatCoarse (from-scratch, no tiering) | 0.07 | 0.01 |

At N = 20,000, tiered recall on **recent-age queries** is **0.42**, close to FlatFine's recent-age performance. That is the comparison the tiering mechanism should be judged on.

### Storage

| System | N = 2,000 | N = 20,000 |
|---|---|---|
| **Morphing Vectors** | 8.6 KB | 41.6 KB |
| FlatFine (always finest) | 32.0 KB | 320.0 KB |
| FlatCoarse (always coarsest) | 2.0 KB | 20.0 KB |

That is roughly **3.7× to 7.7× savings vs. FlatFine**, including the cost of pinning hub nodes at full precision.

### Migration cost

Steady-state per-insert time grew from 3.3 ms (N = 2,000) to 4.0 ms (N = 20,000): a 10× increase in N produced about 1.2× growth in per-insert cost, consistent with the amortized-cost claim (Proposition 2).

### Reading the numbers honestly

The results separate two distinct gaps:

- **Gap A (FAISS vs. FlatFine):** graph implementation quality. Our closest-M neighbor selection lacks HNSW's diversity-aware heuristic. Raising M from 16 to 32 and ef from 150 to 300 recovered only 0.44 → 0.48 at roughly 2× build cost, which points to neighbor-selection quality rather than search budget.
- **Gap B (FlatFine vs. Morphing Vectors):** the tiering mechanism's own recall cost, paid for the storage savings above. This is the gap the contribution should be judged against.

---

## Design Pitfalls Found (Generalizable Findings)

These are documented in the paper as part of the contribution.

1. **Routing-node staleness under tiering.** This occurred twice. First, the early entry points of the original flat graph were the oldest nodes and the first to be coarsened. Later, the hierarchical graph's upper-layer routing nodes were coarsened by the same age-based policy as leaf vectors, which initially dropped tiered recall to 0.015 at N = 20,000. *Fix:* pin every node in layer ≥ 1 at the finest tier. *Principle:* any tiered index needs tier-aware routing infrastructure, since distortion in routing nodes corrupts queries globally rather than locally.
2. **Synthetic benchmark validity.** An i.i.d. Gaussian dataset has no correlation between insertion time and position, which makes recency tiering look like it fails when the premise is simply untestable. Temporally correlated data is a *requirement* for evaluating this class of system.

---

## Theory

| Claim | Status | Location |
|---|---|---|
| **Lemma 1 (Monotonic error):** coarser tiers have non-decreasing reconstruction error | Proved (cell-merging argument) | `core/theory.py::validate_monotonic_error` |
| **Proposition 1 (Recall ordering):** quantization-induced recall-miss probability is non-decreasing in tier coarseness | One-sided case only (exact query, quantized database); quantization term only, excluding graph-search error | `core/theory.py` |
| **Proposition 2 (Amortized cost):** per-operation migration cost is O(B log N), independent of how many vectors are eligible | Proof sketch plus empirical scaling check | `eval/scaling_test.py` |

## Quick Start

```python
import numpy as np
from core.index import MorphingVectorIndex

d = 32
index = MorphingVectorIndex(dim=d)

# Insert vectors in time order (newest at the finest tier)
for i, x in enumerate(np.random.randn(2000, d).astype("float32")):
    index.insert(i, x)

# Query: results are "touched," which keeps them hot
q = np.random.randn(d).astype("float32")
results = index.query(q, k=10, ef=150)
print(results)
```

> The snippet shows intended usage. Check the constructor and method signatures in `core/index.py` and adjust if they differ.

## Reproducing the Results

```bash
# Main recall / storage benchmark (FlatFine, FlatCoarse, Morphing Vectors, optional FAISS)
python3 -m eval.benchmark

# Amortized migration-cost scaling check (Proposition 2)
python3 -m eval.scaling_test

# Correctness tests
python3 -m pytest tests/
```

## Repository Layout

```
morphing_vectors/
├── core/
│   ├── quantizer.py     # ProductQuantizer, TieredQuantizer (from-scratch PQ + mini-batch k-means)
│   ├── graph_index.py   # Hierarchical navigable small-world graph
│   ├── tiers.py         # RecencyTierManager (the novel contribution)
│   ├── storage.py       # MorphingStorage (per-vector encoding state)
│   ├── index.py         # MorphingVectorIndex (top-level system)
│   ├── baseline.py      # FlatIndex (same-components fixed-tier baseline)
│   └── theory.py        # Proposition statements + numeric validators
├── eval/
│   ├── benchmark.py     # recall/storage benchmark + temporally correlated workload generator
│   └── scaling_test.py  # amortized migration cost scaling check
├── tests/
    └── test_basic.py    # correctness tests

```

---

## Limitations

- **Synthetic data only.** SIFT1M, GIST1M, and Deep1M were not available in the development environment. Recall numbers are not yet representative of real-world datasets.
- **Neighbor selection.** The closest-M heuristic must be upgraded to HNSW's full diversity-aware selection. This is the dominant remaining cause of the gap to FAISS.
- **Proposition 1 is partial.** It covers only the one-sided case and only the quantization-error term. Graph-search error, empirically the larger term at this scale, is not yet folded into a combined bound.
- **Scale.** Current evaluation is at N = 2,000 and 20,000. Production-scale claims require N = 100K to 1M.
- **Language.** Python/NumPy is sufficient to validate the tiering mechanism, not to make absolute-latency claims.

## Roadmap

- [ ] Implement HNSW's diversity-aware neighbor selection (top priority)
- [ ] Evaluate on SIFT1M / GIST1M / Deep1M
- [ ] Re-run all tables at N = 100K to 1M
- [ ] Combined recall bound covering quantization and graph-search error
- [ ] Two-sided (quantized query) extension of Proposition 1
- [ ] Real financial time-series embeddings from the QUANT-DB pipeline
- [ ] Pruned-assignment k-means (SuperKMeans-style) to speed up training
- [ ] Rust/C++ reimplementation to separate interpreter overhead from algorithmic gaps

## Related Work

Builds on product quantization (Jégou et al., 2011) and HNSW (Malkov & Yashunin, 2018). Related adaptive-indexing work includes Limousine, LeaFi, and DARTH. PDX (physical layout at fixed precision) and SuperKMeans (faster clustering) are orthogonal and potentially composable. See the paper for the full discussion.



## Author

**Shehan Liyanaarachchi**, Independent Researcher

## License

Add your license here (e.g., MIT or Apache-2.0) and include a `LICENSE` file in the repo root.
