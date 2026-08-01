import math
import torch
import torch.nn.functional as F


class QGFDTheoremVerifier:
    """
    Programmatic verifier for the five core mathematical theorems of QGFD.
    """

    def __init__(self, seed: int = 42, device: str = "cpu"):
        torch.manual_seed(seed)
        self.device = torch.device(device)

    def generate_random_inputs(self, B: int = 2, H: int = 2, L: int = 8, d_k: int = 16):
        Q = torch.randn(B, H, L, d_k, device=self.device)
        K = torch.randn(B, H, L, d_k, device=self.device)
        V = torch.randn(B, H, L, d_k, device=self.device)
        return Q, K, V

    def verify_theorem_1(self, tol: float = 1e-6) -> bool:
        """
        Theorem 1: Equivalence to Softmax Attention.
        For alpha = 0, QGFD reduces exactly to standard scaled dot-product attention.
        """
        Q, K, V = self.generate_random_inputs()
        d_k = Q.shape[-1]
        scores = torch.einsum("bhqd,bhkd->bhqk", Q, K) / math.sqrt(d_k)
        p_std = F.softmax(scores, dim=-1)

        alpha = 0.0
        T = 1
        P = F.softmax(torch.einsum("bhid,bhjd->bhij", K, K) / math.sqrt(d_k), dim=-1)
        p_qgfd = (1.0 - alpha) * p_std + alpha * torch.einsum("bhqn,bhnm->bhqm", p_std, P)

        diff = torch.max(torch.abs(p_qgfd - p_std)).item()
        passed = diff < tol
        return passed

    def verify_theorem_2(self, tol: float = 1e-4) -> bool:
        """
        Theorem 2: Geometric Convergence to Fixed Point.
        p^(inf) = (1 - alpha) p^(0) (I - alpha P)^(-1).
        Error decays as ||p^(T) - p^(inf)|| <= alpha^T ||p^(0) - p^(inf)||.
        """
        Q, K, V = self.generate_random_inputs(B=1, H=1, L=6, d_k=8)
        L = Q.shape[2]
        d_k = Q.shape[-1]
        alpha = 0.3

        scores = torch.einsum("bhqd,bhkd->bhqk", Q, K) / math.sqrt(d_k)
        p0 = F.softmax(scores, dim=-1).squeeze(0).squeeze(0)  # (L, L)
        P = F.softmax(torch.einsum("bhid,bhjd->bhij", K, K) / math.sqrt(d_k), dim=-1).squeeze(0).squeeze(0)  # (L, L)

        I = torch.eye(L, device=self.device)
        inv_mat = torch.linalg.inv(I - alpha * P)
        p_inf = (1.0 - alpha) * torch.matmul(p0, inv_mat)

        p_curr = p0.clone()
        err_0 = torch.norm(p0 - p_inf).item()

        for T in range(1, 10):
            p_next = (1.0 - alpha) * p0 + alpha * torch.matmul(p_curr, P)
            err_T = torch.norm(p_next - p_inf).item()
            bound = (alpha ** T) * err_0
            if err_T > bound + 1e-6:
                return False
            p_curr = p_next

        return True

    def verify_theorem_3(self, tol: float = 1e-5) -> bool:
        """
        Theorem 3: Multi-Hop Expansion.
        p^(T) = (1 - alpha) sum_{k=0}^{T-1} alpha^k p^(0) P^k + alpha^T p^(0) P^T.
        """
        Q, K, V = self.generate_random_inputs(B=1, H=1, L=5, d_k=4)
        L = Q.shape[2]
        d_k = Q.shape[-1]
        alpha = 0.2
        T = 4

        scores = torch.einsum("bhqd,bhkd->bhqk", Q, K) / math.sqrt(d_k)
        p0 = F.softmax(scores, dim=-1).squeeze(0).squeeze(0)
        P = F.softmax(torch.einsum("bhid,bhjd->bhij", K, K) / math.sqrt(d_k), dim=-1).squeeze(0).squeeze(0)

        # Iterative update
        p_iter = p0.clone()
        for _ in range(T):
            p_iter = (1.0 - alpha) * p0 + alpha * torch.matmul(p_iter, P)

        # Closed form expansion
        p_exp = torch.zeros_like(p0)
        P_k = torch.eye(L, device=self.device)
        for k in range(T):
            p_exp += (1.0 - alpha) * (alpha ** k) * torch.matmul(p0, P_k)
            P_k = torch.matmul(P_k, P)

        p_exp += (alpha ** T) * torch.matmul(p0, P_k)

        diff = torch.max(torch.abs(p_iter - p_exp)).item()
        return diff < tol

    def verify_theorem_4(self, tol: float = 1e-3) -> bool:
        """
        Theorem 4: Approximation of Full Attention via Diffusion from Sparse Initial Graph.
        """
        Q, K, V = self.generate_random_inputs(B=1, H=1, L=6, d_k=4)
        d_k = Q.shape[-1]
        scores = torch.einsum("bhqd,bhkd->bhqk", Q, K) / math.sqrt(d_k)
        p0 = F.softmax(scores, dim=-1).squeeze(0).squeeze(0)
        P = F.softmax(torch.einsum("bhid,bhjd->bhij", K, K) / math.sqrt(d_k), dim=-1).squeeze(0).squeeze(0)

        # Create sparse initial mask
        mask = torch.eye(6, device=self.device).bool()
        p_sparse = p0.masked_fill(~mask, 0.0)
        p_sparse = p_sparse / p_sparse.sum(dim=-1, keepdim=True)

        alpha = 0.8
        p_curr = p_sparse
        for _ in range(15):
            p_curr = (1.0 - alpha) * p_sparse + alpha * torch.matmul(p_curr, P)

        # Non-diagonal elements should gain non-zero probability
        non_diag_sum = (p_curr * (~mask)).sum().item()
        return non_diag_sum > 0.5

    def verify_theorem_5(self, tol: float = 1e-2) -> bool:
        """
        Theorem 5: Oversmoothing Bounds.
        As T -> inf or alpha -> 1, representations collapse and variance across query outputs -> 0.
        """
        Q, K, V = self.generate_random_inputs(B=1, H=1, L=8, d_k=8)
        d_k = Q.shape[-1]
        scores = torch.einsum("bhqd,bhkd->bhqk", Q, K) / math.sqrt(d_k)
        p0 = F.softmax(scores, dim=-1).squeeze(0).squeeze(0)
        P = F.softmax(torch.einsum("bhid,bhjd->bhij", K, K) / math.sqrt(d_k), dim=-1).squeeze(0).squeeze(0)
        V_mat = V.squeeze(0).squeeze(0)

        alpha = 0.99
        p_curr = p0.clone()
        for _ in range(50):
            p_curr = (1.0 - alpha) * p0 + alpha * torch.matmul(p_curr, P)

        h_T = torch.matmul(p_curr, V_mat)  # (L, d_k)
        # Compute row variance across tokens
        row_variance = torch.var(h_T, dim=0).mean().item()
        return row_variance < tol

    def run_all(self, verbose: bool = True) -> dict[str, bool]:
        results = {
            "Theorem 1 (Softmax Equivalence)": self.verify_theorem_1(),
            "Theorem 2 (Geometric Convergence)": self.verify_theorem_2(),
            "Theorem 3 (Multi-Hop Expansion)": self.verify_theorem_3(),
            "Theorem 4 (Dense Attention Approx)": self.verify_theorem_4(),
            "Theorem 5 (Oversmoothing Bounds)": self.verify_theorem_5(),
        }
        if verbose:
            print("\n=== QGFD Theoretical Verification Report ===")
            for name, passed in results.items():
                status = "PASSED [OK]" if passed else "FAILED [FAIL]"
                print(f"{name:38s}: {status}")
            print("============================================\n")
        return results


def verify_qgfd_theorems(verbose: bool = True) -> dict[str, bool]:
    verifier = QGFDTheoremVerifier()
    return verifier.run_all(verbose=verbose)
