from qgfd import StableQGFDLayer as QGFDLayer
import torch
import torch.nn.functional as F
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Verification script for Theorems 1-3
def verify_theorems():
    torch.manual_seed(42)
    B, N, D = 1, 3, 4  # Small scale for readable outputs
    hidden = torch.randn(B, N, D)

    print("=== Theorem 1: Equivalence to Softmax Attention ===")
    layer = QGFDLayer(embed_dim=D, num_heads=1, diffusion_steps=1, alpha=0.0)
    attn_output, p = layer(hidden)

    # Standard attention
    Q = layer.q_proj(hidden).view(B, N, 1, D).permute(0, 2, 1, 3)
    K = layer.k_proj(hidden).view(B, N, 1, D).permute(0, 2, 1, 3)
    V = layer.v_proj(hidden).view(B, N, 1, D).permute(0, 2, 1, 3)
    scores = torch.einsum("bhqd,bhkd->bhqk", Q, K) / (math.sqrt(D) + 1e-8)
    p_std = F.softmax(scores, dim=-1)
    attn_std = torch.einsum("bhqk,bhkd->bhqd", p_std, V)
    attn_std = attn_std.permute(0, 2, 1, 3).contiguous().view(B, N, D)
    attn_std = layer.out_proj(attn_std)

    diff = torch.max(torch.abs(attn_output - attn_std))
    print(f"Max difference (QGFD vs. standard): {diff:.8f}")
    assert diff < 1e-7, "Theorem 1 failed"
    print("Theorem 1 verified: QGFD matches standard attention for alpha=0, T=1")

    print("\n=== Theorem 2: Geometric Convergence ===")
    layer = QGFDLayer(embed_dim=D, num_heads=1, diffusion_steps=20, alpha=0.5)
    _, p = layer(hidden)
    Q = layer.q_proj(hidden).view(B, N, 1, D).permute(0, 2, 1, 3)
    K = layer.k_proj(hidden).view(B, N, 1, D).permute(0, 2, 1, 3)
    scores = torch.einsum("bhqd,bhkd->bhqk", Q, K) / (math.sqrt(D) + 1e-8)
    p0 = F.softmax(scores, dim=-1)
    P = layer.build_transition_from_keys(K)

    # Compute fixed point: p_inf = (1-alpha) * p0 @ inv(I - alpha P)
    I = torch.eye(N, device=P.device).unsqueeze(0).unsqueeze(0)
    alpha = layer.alpha
    p_inf = (1 - alpha) * torch.einsum("bhij,bhjk->bhik", p0, torch.linalg.inv(I - alpha * P))

    diff = torch.max(torch.abs(p - p_inf))
    print(f"Max difference (p_T vs. p_inf): {diff:.8f}")
    assert diff < 1e-5, "Theorem 2 failed"

    # Check geometric convergence
    p_t = p0.clone()
    errors = []
    for t in range(5):
        p_t = (1 - alpha) * p0 + alpha * torch.einsum("bhqn,bhnm->bhqm", p_t, P)
        error = torch.norm(p_t - p_inf, p=2)
        errors.append(error.item())

    ratios = [errors[t] / errors[t-1] for t in range(1, len(errors))]
    print("Convergence errors:", [f"{e:.8f}" for e in errors])
    print("Convergence ratios:", [f"{r:.6f}" for r in ratios])
    assert all(r < alpha for r in ratios), "Theorem 2 failed: not geometric"
    print("Theorem 2 verified: Converges to fixed point geometrically")

    print("\n=== Theorem 3: Multi-Hop Expansion (Corrected) ===")
    T = 3
    layer = QGFDLayer(embed_dim=D, num_heads=1, diffusion_steps=T, alpha=0.5)
    _, p_iter = layer(hidden)

    # Extract p0 and P (same as above)
    Q = layer.q_proj(hidden).view(B, N, 1, D).permute(0, 2, 1, 3)
    K = layer.k_proj(hidden).view(B, N, 1, D).permute(0, 2, 1, 3)
    scores = torch.einsum("bhqd,bhkd->bhqk", Q, K) / (math.sqrt(D) + 1e-8)
    p0 = F.softmax(scores, dim=-1)
    P = layer.build_transition_from_keys(K)
    alpha = layer.alpha

    # Corrected: No transpose, use consistent einsum "bhij,bhjk->bhik" for @
    p_explicit = torch.zeros_like(p0)
    P_k = torch.eye(N, device=P.device).unsqueeze(0).unsqueeze(0)  # P^0 = I
    for k in range(T):
        term = torch.einsum("bhij,bhjk->bhik", p0, P_k)
        p_explicit += (1 - alpha) * (alpha ** k) * term
        P_k = torch.einsum("bhij,bhjk->bhik", P_k, P)
    # Remainder term
    term_T = torch.einsum("bhij,bhjk->bhik", p0, P_k)
    p_explicit += (alpha ** T) * term_T

    diff = torch.max(torch.abs(p_iter - p_explicit))
    print(f"Max difference (iterative vs. explicit): {diff:.8f}")
    assert diff < 1e-7, "Theorem 3 failed"
    print("p^(T) (iterative):\n", p_iter.squeeze().detach().numpy())
    print("Theorem 3 verified: Matches corrected Neumann series expansion")

# Verification script for Theorem 4
def verify_theorem4():
    torch.manual_seed(42)
    B, N, D = 1, 3, 4
    hidden = torch.randn(B, N, D, requires_grad=True)

    print("=== Theorem 4: Differentiability and Gradient Flow ===")

    # Case 1: alpha=0, T=1 (should match standard attention gradients)
    layer_qgfd = QGFDLayer(embed_dim=D, num_heads=1, diffusion_steps=1, alpha=0.0)
    out_qgfd, _ = layer_qgfd(hidden)
    out_qgfd.sum().backward(retain_graph=True)
    grad_qgfd = hidden.grad.clone()

    # Standard attention gradients (reuse projections)
    Q = layer_qgfd.q_proj(hidden).view(B, N, 1, D).permute(0, 2, 1, 3)
    K = layer_qgfd.k_proj(hidden).view(B, N, 1, D).permute(0, 2, 1, 3)
    V = layer_qgfd.v_proj(hidden).view(B, N, 1, D).permute(0, 2, 1, 3)
    hidden.grad.zero_()  # Reset
    scores = torch.einsum("bhqd,bhkd->bhqk", Q, K) / (math.sqrt(D) + 1e-8)
    p_std = F.softmax(scores, dim=-1)
    attn_std = torch.einsum("bhqk,bhkd->bhqd", p_std, V)
    attn_std = attn_std.permute(0, 2, 1, 3).contiguous().view(B, N, D)
    attn_std = layer_qgfd.out_proj(attn_std)
    attn_std.sum().backward()
    grad_std = hidden.grad.clone()

    diff_grad = torch.max(torch.abs(grad_qgfd - grad_std))
    print(f"Max gradient difference (QGFD vs. standard, alpha=0): {diff_grad:.8f}")
    assert diff_grad < 1e-7, "Theorem 4 failed: gradients mismatch for alpha=0"

    # Case 2: alpha=0.5, T=3 (check finite norm, no NaN/Inf)
    hidden.grad.zero_()
    layer_diff = QGFDLayer(embed_dim=D, num_heads=1, diffusion_steps=3, alpha=0.5)
    out_diff, _ = layer_diff(hidden)
    out_diff.sum().backward()
    grad_diff = hidden.grad.norm().item()
    print(f"Gradient norm (QGFD, alpha=0.5, T=3): {grad_diff:.6f}")
    assert torch.isfinite(torch.tensor(grad_diff)) and grad_diff > 0.1, "Theorem 4 failed: invalid gradients"

    # Theoretical expectation: scaled by contraction (rough check: norm < 2.0 for stability)
    assert grad_diff < 2.0, "Theorem 4 failed: gradient explosion"
    print("Theorem 4 verified: Gradients flow smoothly and match baseline")

# Verification script for Theorem 5 (fixed)
def verify_theorem5():
    torch.manual_seed(42)
    B, N, D = 1, 3, 4
    hidden = torch.randn(B, N, D, requires_grad=False)

    print("=== Theorem 5: Oversmoothing Bounds (verification) ===")
    alpha = 0.8  # you can tune; very large alpha may cause non-monotonic short-term behavior
    layer = QGFDLayer(embed_dim=D, num_heads=1, diffusion_steps=50, alpha=alpha)
    _, p_final = layer(hidden)

    # Extract p0, P, V
    Q = layer.q_proj(hidden).view(B, N, 1, D).permute(0, 2, 1, 3)
    K = layer.k_proj(hidden).view(B, N, 1, D).permute(0, 2, 1, 3)
    V = layer.v_proj(hidden).view(B, N, 1, D).permute(0, 2, 1, 3)

    scores = torch.einsum("bhqd,bhkd->bhqk", Q, K) / (math.sqrt(D) + 1e-8)
    p0 = F.softmax(scores, dim=-1)
    P = layer.build_transition_from_keys(K)

    # Compute stationary pi properly: pi_{t+1} = pi_t @ P (pi is row vector)
    pi = torch.ones(B, 1, N, device=P.device) / N  # shape [B, H, N]
    for _ in range(500):  # power iter
        pi = torch.einsum("bhn,bhnm->bhm", pi, P)
        pi = pi / (pi.sum(dim=-1, keepdim=True) + 1e-12)
    pi_final = pi  # [B, H, N]

    # Simulate p^(T) for consecutive Ts and compute ||p^T - pi||_2
    Ts = list(range(1, 10))  # 1..9
    errors = []
    for T in Ts:
        p_t = p0.clone()
        for _ in range(T):
            p_t = (1 - alpha) * p0 + alpha * torch.einsum("bhqn,bhnm->bhqm", p_t, P)
        # broadcast pi to match p_t shape [B,H,Q,K] -> pi.unsqueeze(2)
        pi_broad = pi_final.unsqueeze(2)  # [B, H, 1, N]
        error = torch.norm(p_t - pi_broad, p=2).item()
        errors.append(error)
        print(f"T={T}: ||p^(T) - pi||_2 = {error:.6f}")

    # Check geometric decay: ratios ~ gamma <1
    ratios = [errors[i] / errors[i - 1] for i in range(1, len(errors))]
    print("Oversmoothing ratios:", [f"{r:.6f}" for r in ratios])
    assert all(0.0 < r < 1.0 for r in ratios), "Theorem 5 failed: not geometric (ratios not in (0,1))"

    # Oversmoothing in representations: pairwise distances -> 0 (use p_t at final T)
    # Using final T from above (last p_t)
    h_t = torch.einsum("bhqk,bhkd->bhqd", p_t, V).squeeze(1)  # [B, N, D]
    dists = torch.norm(h_t.unsqueeze(1) - h_t.unsqueeze(0), dim=-1)  # [B, N, N]
    dists_sq = dists.squeeze(0)
    trace_dists = torch.trace(dists_sq)
    avg_dist = (dists_sq.sum() - trace_dists) / (N * (N - 1))  # exclude self
    print(f"Avg pairwise dist in h^(T={Ts[-1]}): {avg_dist.item():.6f}")
    assert avg_dist < 0.5, "Theorem 5 failed: no homogenization (avg pairwise distance too large)"

    print("Theorem 5 verified: Exponential convergence to stationary pi with oversmoothing")

if __name__ == "__main__":
    verify_theorems()
    verify_theorem4()
    verify_theorem5()
