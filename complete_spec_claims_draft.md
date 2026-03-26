# Draft Claims for Later Complete Specification

Suggested title:

Systems and Methods for Deterministic Gather of Hierarchically Managed Key-Value State for Neural Network Inference

The following draft claims are intended for later complete-specification development. They are not intended to be inserted into the provisional specification.

## Claim Set A: Broad

1. A computer-implemented system for autoregressive neural network inference, the system comprising:
   a memory arrangement storing key-value state as a plurality of logically managed key-value objects associated with prior token positions of one or more inference sequences;
   a hierarchical metadata manager configured to maintain logical-to-physical mappings for the logically managed key-value objects independently of current physical placement, the mappings indicating, for at least a subset of the logically managed key-value objects, one or more of a memory tier, a physical location, a representation state, an ownership state, and a version state;
   a deterministic gather engine configured, for an inference step, to resolve a logical access requirement for historical key-value state by traversal of the hierarchical metadata manager, to bind source key-value objects from one or more physical placements based on the logical-to-physical mappings, to determine any required transform path associated with the bound source key-value objects, to validate the bound source key-value objects against a current version state before execution, and to generate an execution-ready output for a compute unit by assembling the bound source key-value objects in a predetermined order; and
   a compute interface configured to provide the execution-ready output to an attention operator or related neural network compute operator without requiring the compute operator to traverse the hierarchical metadata manager directly.

2. The system as claimed in claim 1, wherein the execution-ready output comprises one or more materialized tiles or buffers arranged for direct consumption by the compute unit.

3. The system as claimed in claim 1, wherein the execution-ready output comprises a descriptor-form execution artifact selected from descriptor programs, DMA chains, ordered stream descriptors, hardware front-end command structures, and combinations thereof.

4. The system as claimed in claim 1, wherein the hierarchical metadata manager is configured to maintain the logical-to-physical mappings in a structure selected from a radix tree, a B+-tree, a page-table-like hierarchy, an extent map, and a hybrid thereof.

5. The system as claimed in claim 1, further comprising a prefix-sharing manager configured to maintain a shared lineage structure for common prompt prefixes used by a plurality of sessions, wherein the deterministic gather engine is configured to assemble a gather result from shared prefix objects and branch-local suffix objects without duplication of all underlying physical storage.

6. The system as claimed in claim 1, further comprising a residency controller configured to control placement of the logically managed key-value objects across a plurality of memory tiers including at least a first tier and a second tier having different access characteristics, wherein the deterministic gather engine is configured to gather source key-value objects from a plurality of such memory tiers for the same inference step.

7. The system as claimed in claim 1, further comprising a representation manager configured to maintain, for at least one logically managed key-value object, a representation descriptor indicating a representation class and a transform path, wherein the deterministic gather engine is configured to apply decompression, dequantization, reformatting, or transposition according to the representation descriptor before or during generation of the execution-ready output.

8. The system as claimed in claim 1, wherein the deterministic gather engine is configured to compile a gather plan specifying an ordered set of gather segments, corresponding source descriptors, corresponding transform actions, and corresponding output placements for the inference step.

9. The system as claimed in claim 8, wherein the gather plan further specifies overlap scheduling for at least two among remote transfer, cold-tier fetch, warm-tier transfer, decompression, reformatting, staging, and tile assembly.

10. The system as claimed in claim 8, wherein the deterministic gather engine is configured to reuse a gather template for recurrent logical access shapes, subject to validation of source compatibility and version compatibility.

11. The system as claimed in claim 8, wherein the gather plan comprises a directed acyclic structure representing dependencies among source fetches, transfers, transform operations, staging operations, or output-placement operations.

12. The system as claimed in claim 1, further comprising a resume index identifying one or more likely first-window logical segments for a suspended session, wherein the deterministic gather engine is configured to perform staged prewarm using the resume index before or during resumption of the suspended session.

13. The system as claimed in claim 1, wherein a residency controller and a representation manager jointly provide placement information and transform information to the deterministic gather engine for compilation of a single gather plan for the inference step.

## Claim Set B: Medium Fallback

14. A computer-implemented system for autoregressive neural network inference, the system comprising:
   a plurality of logically managed key-value objects representing historical inference state;
   a metadata manager storing, for each of at least a subset of the logically managed key-value objects, a current source descriptor identifying a currently valid physical source and an associated version state;
   a deterministic gather engine configured to compile a gather plan for an inference step by selecting source descriptors for a plurality of the logically managed key-value objects, validating the selected source descriptors against a current metadata version state before launch, and, upon detecting that at least one selected source descriptor has become stale, rebinding an affected gather segment to a replacement source descriptor or regenerating the gather plan; and
   a compute interface configured to supply an execution-ready output generated according to the validated gather plan to a compute operator.

15. The system as claimed in claim 14, wherein the metadata manager is configured to publish a next source descriptor for a logical key-value object while retaining a prior source descriptor for draining readers.

16. The system as claimed in claim 14, wherein the version state is managed by a mechanism selected from pointer swap, versioned page publication, epoch table publication, shadow mapping publication, and combinations thereof.

17. The system as claimed in claim 14, wherein the deterministic gather engine preserves deterministic segment ordering when rebinding an affected gather segment.

18. The system as claimed in claim 14, wherein the execution-ready output comprises a descriptor-form execution artifact rather than a fully materialized execution buffer.

19. The system as claimed in claim 14, wherein the deterministic gather engine validates a reusable gather template before source rebinding.

20. The system as claimed in claim 14, wherein the logically managed key-value objects are obtained from a plurality of memory tiers and at least two different representation classes.

## Claim Set C: Narrower Prosecution-Safe Fallback

21. A computer-implemented method for autoregressive neural network inference, the method comprising:
   maintaining historical key-value state as a plurality of logical key-value objects addressable independently of physical placement;
   storing, for at least one logical key-value object, a plurality of descriptor records comprising a source descriptor identifying a current source location and a representation descriptor identifying a representation class and transform requirement;
   receiving a logical access request for an inference step;
   compiling, by a deterministic gather engine, a gather plan identifying an ordered set of gather segments corresponding to the logical access request;
   binding, for the ordered set of gather segments, source descriptors across a plurality of memory tiers;
   validating the bound source descriptors against a current metadata epoch before execution;
   upon detecting an epoch change affecting at least one bound source descriptor, rebinding an affected gather segment to a current valid source descriptor or regenerating the gather plan;
   performing one or more transfers or transforms according to the gather plan, including at least one of decompression, dequantization, reformatting, transposition, or staging; and
   outputting an execution-ready artifact comprising either a materialized tile or a descriptor-form command structure for use by a neural network compute unit.

22. The method as claimed in claim 21, further comprising gathering shared prefix objects and branch-local suffix objects into the execution-ready artifact.

23. The method as claimed in claim 21, further comprising handling branch divergence by copy-on-write divergence handling or seal-and-fork divergence handling.

24. The method as claimed in claim 21, further comprising promoting a compressed cold-tier source object to a warm-tier representation before a subsequent deterministic gather.

25. The method as claimed in claim 21, further comprising selecting between remote fetch and local replication for a remotely owned source object in dependence upon predicted reuse, communication cost, or latency class.

26. The method as claimed in claim 21, further comprising invalidating or refreshing a distributed replica in response to an owner-epoch change or representation-version change.

27. The method as claimed in claim 21, wherein the descriptor-form command structure comprises a DMA chain, a stream descriptor set, or a hardware front-end command structure.

28. A non-transitory computer-readable medium storing instructions which, when executed by one or more processors, cause performance of the method as claimed in any one of claims 21 to 27.

29. A distributed inference system comprising:
   a plurality of accelerators or nodes;
   a distributed ownership manager configured to maintain a global logical namespace for key-value objects across the plurality of accelerators or nodes; and
   at least one deterministic gather engine configured to resolve, for an inference step, a set of shared prefix objects and branch-local suffix objects from local or remote sources, to validate source descriptors against current version information, and to generate an execution-ready output therefrom.

30. The distributed inference system as claimed in claim 29, wherein the distributed ownership manager maintains freshness information for an owner copy and one or more replica copies of a key-value object.

31. The distributed inference system as claimed in claim 29, wherein a shared prefix object is selectively replicated across a subset of the plurality of accelerators or nodes according to predicted reuse.

## Claim Strategy Notes

- Claim Set A is intended to preserve broader coverage centered on deterministic gather as the main invention.
- Claim Set B is intended as a stronger fallback set emphasizing epoch validation, source rebinding, and execution-plan integrity.
- Claim Set C is intended as a narrower fallback set tied to descriptor records, epoch change handling, representation transforms, and distributed replica control.
- Claims directed only to blockized KV storage, paging, prompt reuse, tiering, or compression without the deterministic gather path should not bear the novelty burden.
