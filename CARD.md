# szl-receipt-attn

Original SZL construction in the fused-attention category.
Inspired by Dao et al. FA1/FA2/FA3. **Not** a rehost of Dao-AILab or kernels-community/flash-attn2/3/4.

GitHub is source of truth. Hub is the publish mirror. ATELIER owns Hub cards.
Doctrine v11. Λ = Conjecture 1 (never a theorem). No speedup claim.
CPU: torch reference. CUDA: original Triton tiles + online softmax (head dim <= 32 in v0).
Sage INT8/FP8 is a fourth kernel, ROADMAP, not this package.

Apache-2.0. Copyright 2026 SZL Holdings.
