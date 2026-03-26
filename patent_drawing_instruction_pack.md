# Patent Drawing Instruction Pack

Project: Deterministic Gather of Hierarchically Managed Key-Value State for Neural Network Inference

Purpose: This document is intended for a patent illustrator or draftsperson preparing formal black-and-white patent drawings corresponding to the provisional specification.

General drafting instructions:

- Use conventional patent drawing style only.
- No color, shading only if required for clarity and acceptable under patent drawing practice.
- Use consistent reference numerals across figures.
- Use simple block-diagram, flowchart, timing-diagram, and structural-diagram conventions.
- Use clean labels and avoid marketing language.
- Use the same reference numeral for the same element wherever repeated.
- Show functional relationships clearly, but do not overload a figure with unnecessary implementation detail.
- Where possible, group related control-plane elements separately from compute-plane elements.
- Depict optional elements with dashed boxes or dashed lines.

Recommended core reference numerals:

- 100 overall inference system
- 102 runtime scheduler
- 104 deterministic gather engine
- 106 hierarchical metadata manager
- 108 prefix sharing / DAG manager
- 110 residency controller
- 112 compression and representation manager
- 114 allocator / compaction manager
- 116 distributed ownership manager
- 118 compute interface
- 120 attention / compute unit
- 122 hot memory tier
- 124 warm memory tier
- 126 cold memory tier
- 128 remote device or remote node
- 130 logical KV object
- 132 source descriptor
- 134 representation descriptor
- 136 gather plan
- 138 gather template
- 140 epoch transition record
- 142 distributed ownership record
- 144 shared prefix node
- 146 branch-local suffix node
- 148 execution-ready tile
- 150 descriptor-form execution artifact
- 152 DMA chain / command chain
- 154 stream descriptor set
- 156 hardware front-end command structure

## Figure 1: Deterministic Gather-Centric System Architecture

Figure type:
- High-level system block diagram.

Required content:
- Show overall system 100.
- Show scheduler 102, gather engine 104, metadata manager 106, prefix sharing manager 108, residency controller 110, representation manager 112, allocator / compaction manager 114, optional distributed ownership manager 116, compute interface 118, and compute unit 120.
- Show hot tier 122, warm tier 124, and cold tier 126 coupled to gather engine 104.
- Show control-plane relationship from 102/106/108/110/112/114/116 to gather engine 104.
- Show data-plane output from gather engine 104 to compute interface 118 and compute unit 120.

Drafting note:
- Visually separate control plane and compute plane, preferably by dotted boundary or labeled regions.

## Figure 2: Logical Namespace and Shared Lineage Structure

Figure type:
- Structural lineage diagram.

Required content:
- Show model root and one or more shared prefix nodes 144.
- Show descendant session branches and branch-local suffix nodes 146.
- Show that multiple sessions reference the same shared prefix node.
- Show logical KV objects 130 attached to shared and branch-local nodes.

Drafting note:
- This figure should communicate logical sharing, not physical placement.

## Figure 3: Metadata Mapping Entry and Epoch Fields

Figure type:
- Structured record / table diagram.

Required content:
- Show a representative metadata mapping entry.
- Include logical object identifier, current epoch, prior draining epoch, source descriptor pointer, representation descriptor pointer, tier bitmap, pin count, reservation count, migration state, and ownership state.
- Show relation from metadata mapping entry to source descriptor 132, representation descriptor 134, and epoch transition record 140.

Drafting note:
- This should read like an architectural data structure figure, not source code.

## Figure 4: Epoch Commit and Reader Drain Timeline

Figure type:
- Timing diagram or sequence diagram.

Required content:
- Show old source, new source, metadata manager, and gather reader activity.
- Show preparation of new source, publication of next source descriptor, epoch commit, use of old source by prior readers, use of new source by later readers, drain condition, and reclamation.

Drafting note:
- Make clear that logical identity is preserved while physical source changes.

## Figure 5: Gather-Plan Compilation Pipeline

Figure type:
- Functional flow diagram.

Required content:
- Show input logical access request.
- Show logical expansion, source binding, transform-path selection, template match or synthesis, overlap scheduling, and execution-ready emission.
- Show output as either execution-ready tile 148 or descriptor-form artifact 150.

Drafting note:
- This is one of the central figures and should highlight deterministic gather as the main operational path.

## Figure 6: Reusable Gather Template Structure

Figure type:
- Structural block diagram.

Required content:
- Show gather template 138 with abstract segment slots.
- Show associated transform classes, output layout contract, and source rebinding table.
- Show validation against epoch compatibility rules.

Drafting note:
- Make clear that the template is reusable across recurrent logical access shapes.

## Figure 7: Mixed-Tier Overlap Scheduling

Figure type:
- Time-lane or scheduling diagram.

Required content:
- Separate lanes for hot tier 122, warm tier 124, cold tier 126, optional remote node 128, transform stage, and compute launch.
- Show overlapped SRAM read, HBM transfer, host or cold-tier fetch, remote transfer, decompression, reformatting, and output assembly.

Drafting note:
- Emphasize overlapping operations rather than serial execution.

## Figure 8: Single Accelerator Embodiment

Figure type:
- System architecture embodiment diagram.

Required content:
- Show single accelerator boundary.
- Show on-chip SRAM as hot tier 122, HBM as warm tier 124, host DRAM as cold tier 126.
- Show metadata control adjacent to accelerator.
- Show deterministic gather engine 104 assembling execution-ready output from all tiers.

Drafting note:
- This figure should support the best-mode style embodiment described in the text.

## Figure 9: Distributed Popular-Prefix Store Embodiment

Figure type:
- Distributed system block diagram.

Required content:
- Show multiple accelerators or nodes.
- Show distributed ownership manager 116 and shared prefix store.
- Show shared prefix nodes 144 and local suffix nodes 146.
- Show gather engine 104 on at least one node binding local and remote sources.

Drafting note:
- Distinguish shared logical ownership from physical replica placement.

## Figure 10: Remote Fetch Versus Local Replication Decision

Figure type:
- Decision flowchart.

Required content:
- Show remote-owned object request.
- Show decision criteria such as expected reuse, local capacity, service class, and link cost.
- Show branch to direct remote fetch or local replica creation.
- Show metadata update after local replica admission.

Drafting note:
- This figure should be policy-oriented and should not require any specific mathematical formula.

## Figure 11: Resumable Session with Compressed Cold Tier

Figure type:
- Lifecycle / state-transition diagram.

Required content:
- Show active session state in hot and warm tiers.
- Show suspend event, compression, archival to cold tier, resume event, prewarm, deterministic gather, and resumed compute.
- Show compact resume index as a retained warm-tier structure if space allows.

Drafting note:
- Make clear that full logical identity is preserved through suspend and resume.

## Figure 12: Compute Interface Normalization

Figure type:
- Output-format architecture diagram.

Required content:
- Show gather engine 104 generating either execution-ready tile 148 or descriptor-form artifact 150.
- Show descriptor-form artifact branching into DMA chain 152, stream descriptor set 154, and hardware front-end command structure 156.
- Show compute interface 118 feeding compute unit 120.
- Show metadata manager 106 and related control-plane components outside the compute-consumption path.

Drafting note:
- This figure should directly support claim coverage for non-materialized outputs.

## Priority Additional Figures for Complete-Specification Preparation

### Figure 13: Gather-Plan Dependency DAG

Figure type:
- Directed acyclic graph / execution dependency diagram.

Required content:
- Show gather plan 136 represented as dependency nodes and edges.
- Include node types for source fetch, remote fetch, decompression, reformatting, staging, tile write, descriptor emission, and completion fence.
- Show that the deterministic gather engine schedules execution according to dependency order rather than simple linear traversal.
- Show at least one branch in the DAG corresponding to mixed-tier overlap.

Drafting note:
- This figure should directly support the dependent claim directed to a gather-plan DAG structure and should be treated as a priority figure.

### Figure 14: Copy-on-Write Divergence Handling

Figure type:
- Branch lineage / state-transition diagram.

Required content:
- Show a shared prefix lineage, an open tail object, a seal boundary at divergence, and resulting branch-local suffix objects.
- Show which logical objects remain shared and which objects become branch-local.

Drafting note:
- This figure should support branch-aware ownership and divergence handling.

### Figure 15: Segment-Level Epoch-Change Rebind During Gather Validation

Figure type:
- Validation and rebind flow diagram.

Required content:
- Show a compiled gather plan validated against current epoch state.
- Show detection of a stale source descriptor for one or more segments.
- Show segment-level rebind to a replacement source descriptor or fallback plan regeneration.
- Show preservation of deterministic segment ordering after rebind.

Drafting note:
- This figure should be treated as a priority figure because it supports the claim emphasis on epoch validation and rebind during deterministic gather.

### Figure 16: Compressed-Cold-to-Warm Promotion Path

Figure type:
- Promotion / transform pipeline diagram.

Required content:
- Show a compressed cold-tier object, transfer to a warm-tier path, decompression or reconstruction, validation, and publication of the warm-tier source descriptor.
- Show subsequent binding by the deterministic gather engine.

Drafting note:
- This figure should support the staged promotion embodiment used during resume and mixed-tier execution preparation.

### Figure 17: Distributed Replica Invalidation and Refresh Workflow

Figure type:
- Distributed consistency and refresh flow diagram.

Required content:
- Show owner epoch change, stale replica detection, invalidation of stale replicas for new gather bindings, refresh or discard decision, and publication of refreshed replica metadata.

Drafting note:
- This figure should support distributed freshness and replica-update embodiments.

## Figure Commissioning Priority

- Priority 1: Figure 13 (Gather-Plan Dependency DAG)
- Priority 1: Figure 15 (Segment-Level Epoch-Change Rebind During Gather Validation)
- Priority 2: Figure 14 (Copy-on-Write Divergence Handling)
- Priority 2: Figure 16 (Compressed-Cold-to-Warm Promotion Path)
- Priority 3: Figure 17 (Distributed Replica Invalidation and Refresh Workflow)

## Illustrator Checklist

- Ensure every figure is monochrome and formal.
- Ensure all arrows are unambiguous.
- Ensure reference numerals are reused consistently.
- Avoid long text inside the figures.
- Prefer short labels with longer meaning reserved for the specification body.
