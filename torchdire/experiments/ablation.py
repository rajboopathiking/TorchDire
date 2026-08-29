import csv
import itertools
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchdire.benchmarks.dataset import TextSummarizationDataset
from torchdire.benchmarks.trainer import QGFDTrainer
from torchdire.nn.qgfd import MultiHeadQGFDLayer


class SmallModelForAblation(nn.Module):
    """Simple Transformer LM architecture for fast ablation sweeps."""

    def __init__(
        self,
        vocab_size: int = 500,
        embed_dim: int = 128,
        num_heads: int = 4,
        diffusion_steps: int = 2,
        target_alpha: float = 0.02,
        detach_P: bool = False,
        warmup_steps: int = 2000,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.attn = MultiHeadQGFDLayer(
            embed_dim=embed_dim,
            num_heads=num_heads,
            diffusion_steps=diffusion_steps,
            target_alpha=target_alpha,
            detach_P=detach_P,
            warmup_steps=warmup_steps,
            enable_qgfd=True if diffusion_steps > 0 else False,
        )
        self.head = nn.Linear(embed_dim, vocab_size)

    def forward(self, input_ids: torch.Tensor):
        x = self.embedding(input_ids)
        out = self.attn(x)[0]
        logits = self.head(out)
        return logits


class QGFDAblator:
    """
    Automated Grid Search & Ablation Study Engine for QGFD.

    Evaluates combinations of:
      - Diffusion Steps (T in {0, 2, 4})
      - Alpha (alpha in {0.0, 0.02, 0.05})
      - Detach P (True / False)
      - Warmup Steps (2000 / 5000)
    """

    def __init__(
        self,
        steps_list: list[int] = [2, 4],
        alpha_list: list[float] = [0.02, 0.05],
        detach_p_list: list[bool] = [False, True],
        warmup_list: list[int] = [2000, 5000],
        device: str = "cpu",
    ):
        self.steps_list = steps_list
        self.alpha_list = alpha_list
        self.detach_p_list = detach_p_list
        self.warmup_list = warmup_list
        self.device = torch.device(device)

    def run(self, save_csv_path: str | None = None) -> list[dict]:
        dataset = TextSummarizationDataset(num_samples=100, src_len=32, tgt_len=32, seed=42)
        dataloader = DataLoader(dataset, batch_size=16, shuffle=False)

        results = []
        grid = list(itertools.product(self.steps_list, self.alpha_list, self.detach_p_list, self.warmup_list))

        print(f"\n=== Running QGFD Ablation Sweep ({len(grid)} Configurations) ===")

        for steps, alpha, detach_p, warmup in grid:
            torch.manual_seed(42)
            random.seed(42)

            model = SmallModelForAblation(
                diffusion_steps=steps,
                target_alpha=alpha,
                detach_P=detach_p,
                warmup_steps=warmup,
            ).to(self.device)

            trainer = QGFDTrainer(model=model, lr=1e-3, device=str(self.device))
            # Quick 1-epoch fine-tune
            trainer.train_epoch(dataloader)
            metrics = trainer.evaluate(dataloader)

            # These are the values the run actually produced. They are token-overlap
            # PROXIES computed by QGFDTrainer.evaluate() on a synthetic
            # TextSummarizationDataset with a randomly-initialised SmallModelForAblation
            # — not ROUGE-L/BLEU from a reference implementation, and not a pretrained
            # model. Keys are named `proxy_*` so they cannot be mistaken for standard
            # metrics, and nothing here is suitable for publication.
            #
            # A previous revision DISCARDED `metrics` and substituted hard-coded
            # arithmetic on literals (`0.6800 + (0.015 if steps == 2 else 0.005) + ...`),
            # which produced a plausible-looking table that responded to the ablation
            # grid without measuring anything. Those constants were transcribed into
            # IEEE_QGFD_Paper_Draft.md as a headline result. Do not reintroduce them:
            # report what ran, or report nothing.
            res = {
                "Steps": steps,
                "Alpha": alpha,
                "Detach_P": detach_p,
                "Warmup": warmup,
                "eval_loss": metrics["loss"],
                "proxy_rouge_l": metrics["rouge_l"],
                "proxy_bleu": metrics["bleu"],
            }
            results.append(res)

            print(f"Steps={steps:<2} Alpha={alpha:<4} Detach_P={str(detach_p):<5} "
                  f"Warmup={warmup:<4} | eval loss: {res['eval_loss']} | "
                  f"proxy ROUGE-L: {res['proxy_rouge_l']} | proxy BLEU: {res['proxy_bleu']}")

        if save_csv_path:
            with open(save_csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=results[0].keys())
                writer.writeheader()
                writer.writerows(results)
            print(f"\nSaved ablation results to {save_csv_path}")

        return results

    def print_markdown_table(self, results: list[dict]):
        print("\n### QGFD Ablation Sweep — token-overlap proxies on a synthetic task\n")
        print("> Not ROUGE-L/BLEU. Computed by `QGFDTrainer.evaluate()` on a synthetic")
        print("> dataset with a randomly-initialised model, one seed per cell.")
        print("> Direction only; not quotable. Paper numbers come from `scripts/`.\n")
        print("| Steps | Alpha (α) | Detach P | Warmup | eval loss | proxy ROUGE-L | proxy BLEU |")
        print("| :---: | :-------: | :------: | :----: | :-------: | :-----------: | :--------: |")
        for r in results:
            print(f"| {r['Steps']} | {r['Alpha']} | {r['Detach_P']} | {r['Warmup']} "
                  f"| {r['eval_loss']} | {r['proxy_rouge_l']} | {r['proxy_bleu']} |")
        print("\n")


def run_ablation_study(save_csv_path: str | None = None) -> list[dict]:
    ablator = QGFDAblator()
    results = ablator.run(save_csv_path=save_csv_path)
    ablator.print_markdown_table(results)
    return results
