# Affinity Audit (Refactor Branch)

Date: 2026-04-01  
Scope: `vv=Avv`, `pp=App`, `pv=Apv`, `vs=Avs`, `ps=Aps`

## A) Existence

- `vv (Avv)`: exists
- `pp (App)`: exists
- `pv (Apv)`: exists (requires `return_cross=True`)
- `vs (Avs)`: exists
- `ps (Aps)`: exists

## B) Where Computed

- `App/Avv/Apv`:
  - file: `src/models/vit_backbones/vit.py`
  - function: `Attention.compute_affinity(...)`
  - core ops:
    - `App = prompt_tokens @ prompt_tokens^T`
    - `Avv = patch_tokens @ patch_tokens^T`
    - `Apv = prompt_tokens @ patch_tokens^T`

- `Avs/Aps`:
  - file: `src/models/vit_backbones/vit.py`
  - function: `Block._compute_semantic_affinity(...)`
  - core ops:
    - `Avs = patch_query @ sem_key^T`
    - `Aps = prompt_query @ sem_key^T`

Notes:
- `src/models/vit_prompt/vit.py` also computes side-branch `Aps/Avs` (semantic-token branch), but requested `vv/pp/pv/vs/ps` are already available in backbone affinity path above.

## C) Shapes

Before head-average:
- `App`: `[B, H, Lp, Lp]`
- `Avv`: `[B, H, Lv, Lv]`
- `Apv`: `[B, H, Lp, Lv]`
- `Avs`: `[B, H, Lv, Ls]`
- `Aps`: `[B, H, Lp, Ls]`

After head-average (trainer/loss side commonly used):
- `App`: `[B, Lp, Lp]`
- `Avv`: `[B, Lv, Lv]`
- `Apv`: `[B, Lp, Lv]`
- `Avs`: `[B, Lv, Ls]`
- `Aps`: `[B, Lp, Ls]`

## D) Query / Key Direction

- `vv (Avv)`: query=patch, key=patch
- `pp (App)`: query=prompt, key=prompt
- `pv (Apv)`: query=prompt, key=patch
- `vs (Avs)`: query=patch, key=semantic
- `ps (Aps)`: query=prompt, key=semantic

## E) Softmax Dimension

All above use `softmax(dim=-1)` in current code:
- `Avv/App/Apv`: normalize along **key axis** (patch/prompt key)
- `Avs/Aps`: normalize along **semantic key axis**

## F) Whether Trainer/Loss Can Consume Them

- Trainer receives layer-wise affinity list from `forward_with_affinity`.
- `trainer._extract_alignment_aux(...)` currently extracts and head-averages:
  - `attn_pv <- Apv`
  - `attn_vs <- Avs`
  - `attn_ps <- Aps`
- Existing alignment-related loss path uses `attn_pv/attn_vs` (and optionally `attn_ps` in new paths).
- `App/Avv` existed in affinity dict but were not previously summarized by monitor/loss mainline.

## G) Config/Path Dependency (Exists vs Actually Used)

- To produce `App/Avv/Apv/Avs/Aps` in runtime:
  - affinity path must be enabled (`forward_with_affinity` route active)
  - `Apv` additionally needs `return_cross=True` (trainer already forces true when aux needed)
  - `Avs/Aps` need semantics branch input available
- If a run does not enter affinity path or semantics path, some matrices can exist in code but be absent in that run output.
