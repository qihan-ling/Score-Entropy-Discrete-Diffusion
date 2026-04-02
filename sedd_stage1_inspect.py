"""
Stage 1: SEDD Model Architecture Inspection

Loads the pretrained SEDD model and runs a single forward pass,
printing shapes and intermediate values at each layer to understand
exactly what the model computes.

Usage:
    python sedd_stage1_inspect.py --model_path louaaron/sedd-medium
"""

import argparse
import torch
import torch.nn.functional as F
import numpy as np

from sedd_helpers import load_sedd_model


def inspect_architecture(model_path, device):
    print(f"{'='*70}")
    print(f"SEDD Architecture Inspection")
    print(f"{'='*70}\n")

    model, graph, noise, tokenizer = load_sedd_model(model_path, device)
    model.eval()

    cfg = model.config
    print(f"Model config:")
    print(f"  hidden_size  = {cfg.model.hidden_size}")
    print(f"  n_heads      = {cfg.model.n_heads}")
    print(f"  n_blocks     = {cfg.model.n_blocks}")
    print(f"  cond_dim     = {cfg.model.cond_dim}")
    print(f"  length       = {cfg.model.length}")
    print(f"  dropout      = {cfg.model.dropout}")
    print(f"  scale_by_sigma = {cfg.model.scale_by_sigma}")
    print(f"  graph type   = {cfg.graph.type}")
    print(f"  absorb       = {model.absorb}")
    vocab_size = cfg.tokens + (1 if model.absorb else 0)
    print(f"  vocab_size   = {vocab_size} (tokens={cfg.tokens} + absorb={1 if model.absorb else 0})")
    print(f"  MASK token   = {graph.dim - 1}")
    print()

    # --- Create dummy input ---
    print(f"{'='*70}")
    print("Creating dummy input (all MASK tokens)")
    print(f"{'='*70}\n")

    x = graph.sample_limit(1, 1024).to(device)
    print(f"  x shape       = {x.shape}")
    print(f"  x dtype       = {x.dtype}")
    print(f"  x[0, :5]      = {x[0, :5].tolist()}")
    print(f"  all MASK?      = {(x == graph.dim - 1).all().item()}")
    print()

    t_val = torch.tensor([0.5], device=device)
    sigma_val = noise(t_val)[0]
    print(f"  t              = {t_val.item()}")
    print(f"  sigma          = {sigma_val.item():.6f}")
    print()

    # --- Layer-by-layer forward pass ---
    print(f"{'='*70}")
    print("Layer-by-layer forward pass")
    print(f"{'='*70}\n")

    with torch.no_grad():
        # 1. Token embedding
        h = model.vocab_embed(x)
        print(f"[vocab_embed]")
        print(f"  input:  x          {x.shape} (int64)")
        print(f"  output: h          {h.shape}")
        print(f"  embed weight shape {model.vocab_embed.embedding.shape}")
        print()

        # 2. Timestep embedding + SiLU
        c = F.silu(model.sigma_map(sigma_val))
        print(f"[sigma_map + SiLU]")
        print(f"  input:  sigma      {sigma_val.shape}")
        print(f"  output: c          {c.shape}")
        print(f"  c range            [{c.min().item():.4f}, {c.max().item():.4f}]")
        print()

        # 3. Rotary embeddings
        rotary_cos_sin = model.rotary_emb(h)
        cos, sin = rotary_cos_sin
        print(f"[rotary_emb]")
        print(f"  input:  h          {h.shape}")
        print(f"  cos shape          {cos.shape}")
        print(f"  sin shape          {sin.shape}")
        print()

        # 4. DDiT blocks
        block_input = h
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            for idx, block in enumerate(model.blocks):
                block_output = block(block_input, rotary_cos_sin, c, seqlens=None)

                if idx == 0 or idx == len(model.blocks) - 1:
                    label = "first" if idx == 0 else "last"
                    print(f"[DDiTBlock {idx} ({label})]")
                    print(f"  input shape        {block_input.shape}")
                    print(f"  output shape       {block_output.shape}")
                    print(f"  output dtype       {block_output.dtype}")
                    print(f"  output range       [{block_output.float().min().item():.4f}, {block_output.float().max().item():.4f}]")

                    adaln_out = block.adaLN_modulation(c)
                    print(f"  adaLN_modulation   {adaln_out.shape} -> 6 chunks of {adaln_out.shape[-1]//6}")
                    print()
                elif idx == 1:
                    print(f"  ... ({len(model.blocks) - 2} more blocks) ...\n")

                block_input = block_output

            # 5. Final layer
            logits = model.output_layer(block_output, c)

        print(f"[output_layer (DDitFinalLayer)]")
        print(f"  input shape        {block_output.shape}")
        print(f"  output shape       {logits.shape}")
        print(f"  output range       [{logits.float().min().item():.6f}, {logits.float().max().item():.6f}]")
        print()

        # 6. scale_by_sigma correction
        if model.scale_by_sigma:
            esigm1_log = torch.where(
                sigma_val < 0.5,
                torch.expm1(sigma_val),
                sigma_val.exp() - 1,
            ).log().to(logits.dtype)[:, None, None]
            logits_scaled = logits - esigm1_log - np.log(logits.shape[-1] - 1)
            print(f"[scale_by_sigma]")
            print(f"  esigm1_log         {esigm1_log.item():.6f}")
            print(f"  log(V-1)           {np.log(logits.shape[-1] - 1):.6f}")
            print(f"  scaled range       [{logits_scaled.float().min().item():.6f}, {logits_scaled.float().max().item():.6f}]")
            print()
        else:
            logits_scaled = logits

        # 7. Diagonal masking (scatter)
        output = torch.scatter(
            logits_scaled, -1, x[..., None], torch.zeros_like(logits_scaled[..., :1])
        )
        print(f"[diagonal masking (scatter)]")
        print(f"  output shape       {output.shape}")
        print()

    # --- Verify against full forward pass ---
    print(f"{'='*70}")
    print("Verification: full model.forward() vs manual computation")
    print(f"{'='*70}\n")

    with torch.no_grad():
        full_output = model(x, sigma_val)

    print(f"  full_output shape  {full_output.shape}")
    diff = (full_output.float() - output.float()).abs().max().item()
    print(f"  max abs diff       {diff:.2e}")
    match = diff < 1e-2
    print(f"  match?             {'YES' if match else 'NO (may differ due to bf16 accumulation order)'}")
    print()

    # --- Verify diagonal masking ---
    print(f"{'='*70}")
    print("Verification: diagonal masking")
    print(f"{'='*70}\n")

    mask_token = graph.dim - 1
    diag_values = full_output[0, torch.arange(1024), x[0]].float()
    all_zero = (diag_values == 0).all().item()
    print(f"  For each position i, output[0, i, x[0,i]] should be 0 (diagonal masking)")
    print(f"  All diagonal entries zero? {all_zero}")
    if not all_zero:
        nonzero = (diag_values != 0).sum().item()
        print(f"  Non-zero diagonal entries: {nonzero} / 1024")
    print()

    # --- Score interpretation ---
    print(f"{'='*70}")
    print("Score interpretation at position 0")
    print(f"{'='*70}\n")

    scores_pos0 = full_output[0, 0].float()
    probs_pos0 = scores_pos0.exp()
    probs_pos0_norm = probs_pos0 / probs_pos0.sum()

    top5_vals, top5_ids = probs_pos0_norm.topk(5)
    print(f"  Top 5 tokens (normalized probability):")
    for val, tid in zip(top5_vals, top5_ids):
        token_str = tokenizer.decode([tid.item()])
        print(f"    {tid.item():6d}  {repr(token_str):20s}  p={val.item():.6f}")
    print()

    print(f"  Score at MASK token (should be 0 due to diagonal masking):")
    print(f"    score[MASK={mask_token}] = {scores_pos0[mask_token].item():.6f}")
    print()

    print("Done.")


def main():
    parser = argparse.ArgumentParser(description="SEDD Stage 1: Architecture Inspection")
    parser.add_argument("--model_path", type=str, default="louaaron/sedd-medium")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    inspect_architecture(args.model_path, device)


if __name__ == "__main__":
    main()
