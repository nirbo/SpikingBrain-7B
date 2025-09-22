# WORKPLAN

## Immediate Objectives
- Adopt the HF hybrid attention architecture as the primary fine-tuning target (compatible with Unsloth workflows).
- Increase parameter count to roughly 14B by widening hidden width, expanding depth, and raising KV head granularity while keeping the interleaved SWA/GLA layout intact.
- Track efficiency impacts (VRAM, TTFT, throughput) after the size increase to confirm linear-attention advantages remain.

## In-Progress Tasks
- Update HF configuration defaults to the ~14B specification (wider hidden size, more layers, higher KV head fan-out).
- Prepare documentation describing how the wider model maps "intelligence" (width), "accuracy" (KV head count), and "depth" (layer count) adjustments.

## Optional / TBD Enhancements
- Sparse MoE expansion for HybridBlock MLPs (target 14–20B total params with 3–5B active via top-1 routing); requires router design, load-balancing loss, and checkpoint conversion.
- Griffin-style gated SSM branch alongside existing attention for richer temporal dynamics.
- Memory-router / retrieval augmentation for ultra-long factual recall.
- Integrate spike-coding + quantization path directly into the HF runtime to unify deployment pipelines.

## Notes
- Revisit head allocation once initial scaling experiments complete; KV fan-out may be tuned (e.g., 6–8) to balance memory and accuracy.
- Advanced features (MoE, SSM, retrieval, spike pipeline) remain optional until the 14B baseline stabilizes.
