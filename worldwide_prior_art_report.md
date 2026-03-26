# Worldwide Prior-Art Review

Invention under review: deterministic gather of logically managed key-value objects into execution-ready tiles or descriptors for autoregressive neural network inference, supported by residency-first hierarchical management.

Review date: 2026-03-26

This report is a preliminary technical prior-art review intended to support claim drafting. It is not a legal invalidity opinion or freedom-to-operate opinion.

## Bottom-Line Assessment

The closest worldwide prior art is not generic caching. The nearest art is the emerging body of LLM-serving systems and filings that already disclose one or more of:

- blockized or paged KV storage,
- logical-to-physical indirection for KV cache,
- prefix or prompt reuse,
- offloading or multi-tier KV placement,
- distributed KV reuse,
- compression or quantization of KV state, and
- asynchronous loading or prefetching of KV state.

The strongest novelty position remains the same as previously identified:

- deterministic gather as the central execution-preparation mechanism,
- using logically managed KV objects rather than mere page tables or flat per-sequence tensors,
- with representation-aware source binding across multiple tiers,
- with epoch-safe validation, rebind, or publication of source descriptors,
- and with output forms that include descriptor programs, DMA chains, stream descriptors, and hardware front-end command structures in addition to materialized tiles.

If the independent claims stop at blockized KV storage, logical mapping, prefix sharing, or multi-tier placement, the novelty risk is high. If the independent claims require deterministic gather with source validation and execution-ready generation from shared, tiered, and format-diverse logical objects, the position is materially stronger.

## Search Scope

The review considered worldwide patent and non-patent literature categories, including:

- published papers and preprints,
- open-source or official technical documentation,
- public technical reports,
- patent publications from multiple jurisdictions accessible through global search services.

The principal search themes were:

- paged KV cache and block table attention,
- dynamic KV memory management,
- prompt or prefix attention-state reuse,
- multi-tier KV offload and prefetch,
- distributed KV reuse or migration,
- KV compression and quantization,
- inference-time gathering or descriptor-based execution preparation.

## Closest Non-Patent Literature

### 1. PagedAttention / vLLM

Reference:
- Woosuk Kwon et al., “Efficient Memory Management for Large Language Model Serving with PagedAttention,” arXiv:2309.06180. [arXiv](https://arxiv.org/abs/2309.06180)
- vLLM documentation, “Paged Attention.” [vLLM docs](https://docs.vllm.ai/en/stable/design/paged_attention/)

Why it matters:
- This is the primary prior-art risk for any claims directed to non-contiguous KV storage, blockized KV objects, logical indirection, and sharing across requests.
- vLLM publicly teaches paged KV caches and indirection-based access to KV blocks.

What it appears to disclose:
- partitioning KV cache into blocks,
- virtual-memory-like indirection,
- efficient reuse of KV blocks,
- support for complex sampling or request sharing.

What it does not clearly disclose, based on the current review:
- deterministic gather as a separately claimed control-plane execution-preparation mechanism,
- a hierarchical metadata structure for logical identity beyond block table indirection,
- representation-aware source binding across multiple tiers,
- epoch-safe publication or rebind of source descriptors during gather validation,
- descriptor-program or DMA-chain style execution output as the central invention theme.

Risk rating:
- Very high against broad claims on blockized logical mapping alone.
- Moderate against claims that require deterministic gather with epoch-safe validation and representation-aware mixed-tier assembly.

### 2. vAttention

Reference:
- Ramya Prabhu et al., “vAttention: Dynamic Memory Management for Serving LLMs without PagedAttention,” arXiv:2405.04437. [arXiv](https://arxiv.org/abs/2405.04437)

Why it matters:
- This is strong art against claims merely covering “dynamic memory management for KV cache.”

What it appears to disclose:
- dynamic memory management for serving LLMs,
- decoupling virtual and physical memory allocation,
- preserving virtual contiguity of KV cache,
- avoiding some overheads of paged attention.

What it does not clearly disclose, based on the current review:
- deterministic gather of logically managed KV objects as the central compute-preparation mechanism,
- representation-aware mixed-tier gather planning,
- prefix-DAG sharing as part of the same gather-centric architecture,
- gather templates and epoch-safe source rebind as a central design.

Risk rating:
- High against claims framed as generalized dynamic KV memory management.
- Lower against claims anchored on deterministic gather with descriptor-form execution outputs and source-validation semantics.

### 3. Prompt Cache

Reference:
- In Gim et al., “Prompt Cache: Modular Attention Reuse for Low-Latency Inference,” arXiv:2311.04934. [arXiv](https://arxiv.org/abs/2311.04934)

Why it matters:
- This is strong art against claims broadly directed to reusing precomputed attention states or prompt prefixes.

What it appears to disclose:
- reuse of attention states across prompts,
- prompt modules and schema-based reusable segments,
- positional correctness during reuse.

What it does not clearly disclose, based on the current review:
- generalized deterministic gather over multi-tier, representation-diverse, relocatable logical objects,
- epoch-safe source validation and rebind,
- unified handling of shared prefixes, branch-local suffixes, compression state, and distributed ownership in one gather path.

Risk rating:
- High against claims broadly reciting prompt reuse or modular prefix reuse.
- Moderate to low against claims requiring the full deterministic gather architecture.

### 4. LMCache

Reference:
- “LMCache: An Efficient KV Cache Layer for Enterprise-Scale LLM Inference.” [LMCache technical report](https://lmcache.ai/tech_report.pdf)

Why it matters:
- This is one of the most important practical-system references for your case.

What it appears to disclose:
- KV cache persistence beyond a single query lifecycle,
- retrieval and metadata preparation,
- asynchronous layer-wise loading and storing,
- controller interfaces for lookup, move, clear, pin, and compress,
- migration across instances,
- pinning in GPU memory,
- tier-aware handling and prefetch to faster storage.

Specific passages observed:
- the report states that KV cache may be reused beyond the lifecycle of a query,
- it describes layer-wise loading and storing,
- it exposes controller interfaces for locating cached entries or migrating them across instances,
- it describes pinning and compression controls.

What it does not clearly disclose, based on the current review:
- deterministic gather as the core execution-preparation invention,
- hierarchical logical object management with explicit source descriptors and representation descriptors,
- epoch-safe source publication or segment-level rebind during gather validation,
- descriptor-program or hardware front-end command output as a central mechanism.

Risk rating:
- High against claims on enterprise KV cache layer, migration, pinning, and tier-aware management by themselves.
- Moderate against properly narrowed deterministic-gather claims.

### 5. PRESERVE

Reference:
- Ahmet Caner Yüzügüler et al., “PRESERVE: Prefetching Model Weights and KV-Cache in Distributed LLM Serving,” arXiv:2501.08192. [arXiv](https://arxiv.org/abs/2501.08192)

Why it matters:
- This is directly relevant to prefetch and overlap arguments.

What it appears to disclose:
- prefetching model weights and KV-cache,
- overlapping movement with communication,
- distributed serving context.

What it does not clearly disclose, based on the current review:
- a logical KV-object substrate with deterministic gather output generation,
- hierarchical object mapping and source descriptor validation,
- descriptor-oriented gather plans spanning shared, tiered, and format-diverse objects.

Risk rating:
- Moderate against claims emphasizing prefetch/overlap alone.
- Lower if overlap appears only as a dependent feature under deterministic gather.

## Closest Patent Publications Identified

### 6. US12182028B1

Reference:
- “Method and apparatus to cache key-value data in low-precision numerics for efficient generative transformer execution.” [Google Patents](https://patents.google.com/patent/US12182028B1/en)

Why it matters:
- Relevant to low-precision KV storage, transposition, blocking, and compute usage of cached KV state.

What it appears to disclose:
- storing and converting projection tokens,
- low-precision or block-form storage,
- transposed key forms,
- compute using converted/transposed cached data.

What it does not clearly disclose, based on the current review:
- deterministic gather from a logically managed multi-tier object space,
- hierarchical metadata with source publication epochs,
- prefix-sharing DAGs and distributed ownership in the same architecture.

Risk rating:
- Moderate for claims emphasizing representation conversion only.
- Lower for deterministic-gather claims.

### 7. US20250061316

Reference:
- “Dynamic quantization and memory management of key-value cache for serving large language models.” [Justia summary](https://patents.justia.com/patent/20250061316)

Observed passage:
- describes fetching desired cached key tensors and value tensors from global shared memory into GPU worker local memory during autoregressive decoding.

Why it matters:
- Relevant to memory management, quantization, and movement of KV state into local compute memory.

What it appears to disclose:
- dynamic quantization and memory management,
- local-memory fetching for attention.

What it does not clearly disclose, based on the current review:
- deterministic gather plan compilation over logically managed objects,
- hierarchical source descriptors with epoch-safe rebinding,
- shared prefix and branch-aware object ownership integrated with descriptor-form execution outputs.

Risk rating:
- Moderate.

### 8. CN120851217A

Reference:
- “Large model reasoning method, system, electronic device and storage medium based on multi-level cache mechanism.” [Google Patents](https://patents.google.com/patent/CN120851217A/en)

Why it matters:
- Highly relevant as worldwide art against broad multi-tier KV claims.

What it appears to disclose:
- a three-level caching mechanism using GPU memory, system memory, and disk,
- KV cache blocks,
- matching based on block identifiers,
- determining prefill and decode instances based on KV distribution and load,
- multi-instance routing and prefix matching.

What it does not clearly disclose, based on the current review:
- deterministic gather as a discrete execution-preparation mechanism,
- representation-aware transform chains attached to logical KV objects,
- epoch-safe gather validation and source rebinding,
- descriptor-program output.

Risk rating:
- High against claims merely reciting multi-level KV caching with block identifiers.
- Moderate against properly narrowed gather-centric claims.

### 9. CN120338090A

Reference:
- “A method and device for reasoning of a large language model.” [Google Patents](https://patents.google.com/patent/CN120338090A/en)

Why it matters:
- Relevant to a shared memory pool of historical KV cache data and reuse by matching segmented input data.

What it appears to disclose:
- matching input segments against historical KV cache data in a shared memory pool,
- bypassing recomputation when matching cached KV state exists,
- generating and storing KV cache data corresponding to segmented input data.

What it does not clearly disclose, based on the current review:
- deterministic gather from logical objects across multiple representations and tiers,
- epoch-safe relocation-aware binding,
- descriptor-form execution output.

Risk rating:
- Moderate.

### 10. US20250094712A1

Reference:
- “Multi-granular clustering-based solution for key-value cache compression.” [Google Patents](https://patents.google.com/patent/US20250094712A1/en)

Why it matters:
- Relevant to compression, multi-level memory, and distributed environments.

What it appears to disclose:
- KV cache compression,
- multi-level memory or hierarchical cache discussion,
- distributed-computing applicability.

What it does not clearly disclose, based on the current review:
- deterministic gather with source descriptors and output descriptors,
- epoch-safe source publication and rebind,
- gather templates.

Risk rating:
- Moderate for compression-focused claims.

## Prior-Art Themes Most Likely to Be Used Against Claims

The following themes should be treated as already crowded:

- fixed-size or variable-size KV blocks,
- logical block lookup tables,
- non-contiguous KV allocation,
- block-based copy-on-write,
- prefix reuse and prompt reuse,
- KV offloading to CPU or lower tiers,
- KV quantization and compression,
- asynchronous load/store of KV by layer,
- distributed KV migration and replication,
- prefetching of KV state.

These features can still appear in dependent claims, but they should not carry the novelty burden of the broadest independent claims.

## Features That Still Look Most Defensible

On the current record, the following combination still appears the most defensible:

- logical management of KV state as named objects with source descriptors and representation descriptors,
- deterministic gather as a separately defined control-plane execution-preparation mechanism,
- gather-plan compilation that binds source objects from shared, tiered, and format-diverse storage,
- validation of source bindings against metadata epochs before launch,
- segment-level rebind or plan regeneration when an epoch change is detected,
- execution-ready output that may be descriptor-form rather than only materialized buffers,
- optional gather-template reuse under source and epoch compatibility rules.

The strongest system claim should therefore require a deterministic gather engine configured to:

- resolve logically managed KV objects,
- bind and validate sources,
- determine transform paths,
- assemble execution-ready output in a predetermined order,
- and provide that output to compute without direct metadata traversal by the compute kernel.

## Claim-Drafting Consequences

### Features that should be in the broadest independent claims

- deterministic gather engine,
- logically managed KV objects,
- hierarchical metadata manager,
- source binding from logical-to-physical mappings,
- transform-path determination based on representation state,
- execution-ready output for compute without direct metadata traversal by compute.

### Features that should likely appear in at least one independent claim or a strong dependent claim set

- metadata epoch validation before gather launch,
- segment-level rebind or plan regeneration after epoch change,
- multi-tier source binding for the same inference step,
- descriptor-form output, not only materialized tiles,
- gather-template reuse.

### Features that should generally be dependent claims unless specifically needed

- fixed block size,
- particular tree type,
- particular compression algorithm,
- specific promotion heuristic,
- specific replication heuristic,
- specific pointer-swap or version-table implementation.

## Recommended Claim Strategy After This Search

Use a three-tier claim strategy:

### Tier 1: Primary independent claims

Focus on deterministic gather and source validation.

### Tier 2: Strong fallback dependents

Add:
- descriptor programs / DMA chains / stream descriptors,
- epoch-safe rebind,
- representation-aware mixed-tier assembly,
- gather templates.

### Tier 3: Secondary dependents

Add:
- prefix DAG,
- branch-aware ownership,
- copy-on-write divergence,
- distributed replica refresh,
- compressed-cold-to-warm promotion,
- popular-prefix store.

## References Used

- PagedAttention paper: [arXiv 2309.06180](https://arxiv.org/abs/2309.06180)
- vLLM paged attention docs: [vLLM docs](https://docs.vllm.ai/en/stable/design/paged_attention/)
- vAttention paper: [arXiv 2405.04437](https://arxiv.org/abs/2405.04437)
- Prompt Cache paper: [arXiv 2311.04934](https://arxiv.org/abs/2311.04934)
- PRESERVE paper: [arXiv 2501.08192](https://arxiv.org/abs/2501.08192)
- LMCache report: [LMCache technical report](https://lmcache.ai/tech_report.pdf)
- US12182028B1: [Google Patents](https://patents.google.com/patent/US12182028B1/en)
- US20250061316: [Justia summary](https://patents.justia.com/patent/20250061316)
- CN120851217A: [Google Patents](https://patents.google.com/patent/CN120851217A/en)
- CN120338090A: [Google Patents](https://patents.google.com/patent/CN120338090A/en)
- US20250094712A1: [Google Patents](https://patents.google.com/patent/US20250094712A1/en)

## Recommended Next Step

Revise the draft claims into three explicit sets:

- Broad set centered on deterministic gather plus source validation.
- Medium set adding epoch-safe rebind and descriptor-form output.
- Fallback set adding gather templates, multi-tier representation transforms, and distributed replica controls.
