# szl-receipt-attn domain leader study r1 (propose-only)

Date: 2026-11-25. Method: web-verified leader landscape + owner-direct execution under standing authority GREENLIGHT-RULING-2026-11-24-K1 rank5 OUROBOROS_NEXT_TRANCHE. The Kimi governing seat remains stood down per explicit owner directive (disclosed); owner-direct nonce T19-OWNER-DIRECT-20261125-K1. Study pinned to HEAD e606b0147f4b8626c888bc5da300ef7f00fce716, fetched 2026-11-25. This card is documentation only; it changes no code, CI, config, or existing PR status. All absorb actions are FUTURE PROPOSALS (non-authorizing); implementation and merges remain separately reviewed and gated. L Conjecture 1: OPEN (advisory, never a closed theorem). No speedup, energy, or proven-trust claims; upstream performance claims are labeled as upstream claims only. No unverified signing, attestation, or provenance claims; this study verifies nothing about this repo's artifacts.

## Domain: original Triton tiled fused attention (FlashAttention silhouette, not a rehost) with per-call SHA3-256 receipt chain emission, Kernel Hub source-only distribution.

## Leaders studied (web-verified 2026-11-25)
1. FlashAttention — https://github.com/Dao-AILab/flash-attention — the FA1/FA2/FA3 lineage this repo explicitly names as inspiration and explicitly NOT a rehost of; moat is the IO-aware tiling discipline the silhouette borrows while adding receipt-chain emission as the original construction.
2. Triton — https://github.com/openai/triton — the language the kernel's original SZL tiles are written in; moat is the GPU-kernel-in-Python ecosystem with mask/preference semantics this repo honestly limits (attn_mask always selects the torch reference path).
3. huggingface/kernels — https://github.com/huggingface/kernels — the Kernel Hub loader ecosystem whose get_kernel versioned-revision pinning this repo teaches (v1 branch commit, not branch name); moat is immutable-revision loading discipline.

## Absorb actions (FUTURE PROPOSALS, non-authorizing)
1. (FlashAttention) Propose a documentation-only tiling-idiom comparison note that keeps the silhouette/originality boundary sharp (never a rehost of flash-attn2/3/4, hopper/, cute/ or any .cu).
2. (Triton) Propose a doc review of mask/preference edge semantics so prefer="triton" can never silently discard a supplied attn_mask; the torch_reference_fallback path stays labeled in every receipt.
3. (huggingface/kernels) Propose documenting the narrow trust_remote_code form (per-repo list) and the revision=<commit> pinning convention alongside the existing three-identity separation (GitHub commit / model-repo commit / kernel-repo revision).
4. (baseline) Preserve the honesty baseline: import-LIVE on CPU only (torch 2.14.1, CUDA false, max_abs_vs_sdpa 0.0), Triton/CUDA timing UNAVAILABLE, no cubin, CI has no GPU, no speedup claims, no fabricated CUDA benches, correctness band atol/rtol 1e-5 documented not benched; receipts name the backend actually used and snapshot payloads at emit time.


## Do not do (per governing seat)

Do not merge, bypass the signature wall, or claim implementation beyond this card.
