# Reflection — Lab 19

**Tên:** Nguyễn Tuấn Anh - 2A202601669  
**Cohort:** K3  
**Path đã chạy:** lite

---

## Reflection (≤ 200 words)

> Trên golden set 50 queries, mode nào thắng ở loại query nào (`exact` /
> `paraphrase` / `mixed`), và tại sao? Khi nào bạn **không** dùng hybrid
> (i.e. khi nào pure BM25 hoặc pure vector là lựa chọn đúng)?

On the 50-query golden set, each search mode exhibits distinct strengths across query categories:

- **Exact Queries (Keyword / BM25 wins):** BM25 excels on precise technical terms, identifiers, and rare keywords (e.g., specific CLI commands or error codes) where exact lexical matching provides sharp TF-IDF discrimination without dense embedding dilution.
- **Paraphrase Queries (Semantic Vector wins):** Dense embeddings dominate when queries use synonyms, descriptive phrasing, or Vietnamese conceptual expressions without literal keyword overlap, mapping semantic intent into shared vector space.
- **Mixed Queries (Hybrid RRF wins):** Reciprocal Rank Fusion (RRF) delivers the highest overall Precision@10 by fusing sparse lexical precision with dense semantic recall, eliminating blind spots of either standalone ranker.

**When NOT to use Hybrid Search:**
1. **Strict Ultra-Low Latency / Resource Constraints:** In high-throughput systems (P99 < 5ms) or edge environments where embedding inference and dual-index ranking add unacceptable latency and compute cost.
2. **Deterministic Identifier Lookups:** Pure keyword search or primary key lookup is superior for structured entity lookups (e.g., user IDs, SKU codes, exact hashes) where semantic fuzziness risks returning irrelevant false positives.

---

## Điều ngạc nhiên nhất khi làm lab này

Sự kết hợp giữa RRF (Reciprocal Rank Fusion) và Point-in-Time joins của Feast giúp giải quyết trọn vẹn cả bài toán grounding kiến thức ngữ nghĩa lẫn bài toán phục vụ real-time profile với độ trễ P99 < 10ms.

---

## Bonus challenge

- [x] Đã làm bonus (xem `bonus/`)
- [ ] Pair work với: _<tên đồng đội nếu có>_
