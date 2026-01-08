# Problem: Transformer LM Resource Accounting (5 points)

## (a) GPT-2 XL Parameter Count & Memory

Consider GPT-2 XL with the following configuration:

| Parameter | Value |
|-----------|-------|
| vocab_size | 50,257 |
| context_length | 1,024 |
| num_layers | 48 |
| d_model | 1,600 |
| num_heads | 25 |
| d_ff | 6,400 |

**Question:** How many trainable parameters would our model have? Assuming each parameter is represented using single-precision floating point, how much memory is required to just load this model?

**Deliverable:** Using the parameterization implied by your earlier spec (no biases; RoPE has no trainable params; **separate** token embedding and LM head matrices):

### Parameter count

**1 Token embedding**

* (W_{\text{embed}} \in \mathbb{R}^{\text{vocab}\times d_{\text{model}}})
* (50{,}257 \times 1{,}600 = 80{,}411{,}200)

**2 Per Transformer block (× 48 layers)**

**Attention (Q, K, V, output projections)**
Each is (d_{\text{model}}\times d_{\text{model}}):

* (4 \times 1{,}600 \times 1{,}600 = 10{,}240{,}000)

**FFN (SwiGLU: (w1, w3: d_{\text{model}}\times d_{ff}), (w2: d_{ff}\times d_{\text{model}}))**

* (2 \times (1{,}600 \times 6{,}400) + (6{,}400 \times 1{,}600))
* (= 3 \times 1{,}600 \times 6{,}400 = 30{,}720{,}000)

**RMSNorm weights (ln1, ln2)**
Each is ((d_{\text{model}},)):

* (2 \times 1{,}600 = 3{,}200)

So **per layer**:

* (10{,}240{,}000 + 30{,}720{,}000 + 3{,}200 = 40{,}963{,}200)

All layers:

* (48 \times 40{,}963{,}200 = 1{,}966{,}233{,}600)

**3 Final RMSNorm**

* (1{,}600)

**4 LM head**

* (W_{\text{lm}} \in \mathbb{R}^{\text{vocab}\times d_{\text{model}}})
* (50{,}257 \times 1{,}600 = 80{,}411{,}200)

---

### Total trainable parameters

[
80{,}411{,}200 + 1{,}966{,}233{,}600 + 1{,}600 + 80{,}411{,}200
= \boxed{2{,}127{,}057{,}600\text{ parameters}}
]
About **2.13B** parameters.

### Memory to load (FP32, 4 bytes/param)

[
2{,}127{,}057{,}600 \times 4 = 8{,}508{,}230{,}400\ \text{bytes}
]

* (\boxed{8.51\ \text{GB}}) (decimal GB, (10^9) bytes)
* (\boxed{7.92\ \text{GiB}}) (binary GiB, (2^{30}) bytes)

**Note:** This is just raw weights. Training typically needs much more (gradients + optimizer states + activations).

If you **tie** `lm_head.weight` to `token_embeddings.weight` (common in GPT-like LMs), subtract (80{,}411{,}200) params → about **0.32 GB** less in FP32.


---

Below I count FLOPs using the common convention **1 multiply = 1 FLOP and 1 add = 1 FLOP**, so a dense matmul ( (m\times k)\cdot(k\times n)) costs **(\approx 2mkn)** FLOPs. I assume **batch size = 1** and sequence length **(S=\text{context_length})**, and I ignore non-matmul ops (LayerNorm/RMSNorm, GELU, softmax, masking, residual adds).

---

## (b) Matrix multiplies in one forward pass (GPT-2 XL-shaped, (L=48, D=1600, H=25, d_k=D/H=64, d_{ff}=6400, S=1024))

### Per Transformer layer (repeated (L) times)

Let (X\in\mathbb{R}^{S\times D}).

1. **Q projection**: (X (S\times D)\cdot W_Q(D\times D)\rightarrow (S\times D))
   FLOPs: (2SD^2)

2. **K projection**: (X\cdot W_K)
   FLOPs: (2SD^2)

3. **V projection**: (X\cdot W_V)
   FLOPs: (2SD^2)

4. **Attention scores** (per head): (Q_h (S\times d_k)\cdot K_h^\top(d_k\times S)\rightarrow (S\times S))
   All heads together: FLOPs (=2HS^2d_k = 2S^2D)

5. **Attention weighted sum** (per head): (P_h (S\times S)\cdot V_h(S\times d_k)\rightarrow (S\times d_k))
   All heads: FLOPs (=2HS^2d_k = 2S^2D)

6. **Output projection**: (\text{Concat}(S\times D)\cdot W_O(D\times D)\rightarrow (S\times D))
   FLOPs: (2SD^2)

7. **FFN expand**: (X(S\times D)\cdot W_1(D\times d_{ff})\rightarrow (S\times d_{ff}))
   FLOPs: (2SDd_{ff})

8. **FFN contract**: (\text{FF}(S\times d_{ff})\cdot W_2(d_{ff}\times D)\rightarrow (S\times D))
   FLOPs: (2SDd_{ff})

**Per-layer total (matmuls only):**

* Attention projections (Q,K,V,O): (4\cdot 2SD^2 = 8SD^2)
* Attention matmuls (QKᵀ + PV): (2S^2D + 2S^2D = 4S^2D)
* FFN matmuls: (2SDd_{ff}+2SDd_{ff}=4SDd_{ff})

So
[
\text{FLOPs/layer} = 8SD^2 + 4SDd_{ff} + 4S^2D.
]

### Final LM head (once)

9. **Vocabulary projection**: (X(S\times D)\cdot W_{\text{lm}}(D\times V)\rightarrow (S\times V))
   FLOPs: (2SDV)

---

### Numeric totals for GPT-2 XL-shaped with (S=1024)

* Per-layer FLOPs:
  [
  8SD^2 + 4SDd_{ff} + 4S^2D
  = 69{,}625{,}446{,}400
  ]
* All 48 layers:
  [
  3{,}342{,}021{,}427{,}200
  ]
* LM head:
  [
  2SDV = 164{,}682{,}137{,}600
  ]
* **Total forward FLOPs:**
  [
  \boxed{3{,}506{,}703{,}564{,}800\ \text{FLOPs} ;\approx; 3.51\ \text{TFLOPs}}
  ]

---

## (c) Which parts require the most FLOPs?

For (S=1024) and large (D), the **FFN (MLP) matmuls dominate**, followed by the **Q/K/V/O projection matmuls**; the (S^2) attention matmuls are a smaller share at this context length.

(For XL at (S=1024): FFN ≈ 57.4%, attn projections ≈ 28.7%, attn (S^2) matmuls ≈ 9.2%, LM head ≈ 4.7%.)

---

## (d) FLOPs breakdown (as proportion of total) for GPT-2 small/medium/large/XL (all with (S=1024), (d_{ff}=4D), (V=50{,}257))

Percentages below are **of total forward FLOPs** (matmuls only, including LM head).

| Model                                | Total FLOPs (TFLOPs) | FFN matmuls | Attn proj (Q,K,V,O) | Attn matmuls (QKᵀ+PV) | LM head |
| ------------------------------------ | -------------------: | ----------: | ------------------: | --------------------: | ------: |
| GPT-2 small (L=12, D=768, H=12)      |                0.292 |      39.76% |              19.88% |                13.25% |  27.10% |
| GPT-2 medium (L=24, D=1024, H=16)    |                0.827 |      49.86% |              24.93% |                12.46% |  12.75% |
| GPT-2 large (L=36, D=1280, H=20)     |                1.775 |      54.46% |              27.23% |                10.89% |   7.42% |
| GPT-2 XL-shaped (L=48, D=1600, H=25) |                3.507 |      57.41% |              28.71% |                 9.19% |   4.70% |

**Trend (1–2 sentences):** As model size increases (larger (D) and (L) at fixed (S)), compute shifts toward the **per-token dense matmuls** (FFN and projection layers, scaling like (O(LSD^2))), while the **LM head** ((O(SDV))) and the **(S^2) attention matmuls** ((O(LS^2D))) become proportionally smaller.

---

## (e) GPT-2 XL-shaped with context length (S=16{,}384)

Total forward FLOPs become
[
\boxed{133{,}416{,}668{,}364{,}800\ \text{FLOPs} \approx 133.4\ \text{TFLOPs}}
]
which is about **38.0×** higher than at (S=1024) (not 16×) because attention includes **(S^2)** matmuls. The relative FLOPs shift heavily toward attention matmuls: **QKᵀ+PV rises to ~61.8%**, while FFN drops to ~24.1%, attention projections to ~12.1%, and LM head to ~2.0%.

If you want, I can also include the (non-matmul) softmax/GELU/normalization FLOPs, but they typically do not change the “dominant terms” conclusions above.
