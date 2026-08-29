import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchdire.benchmarks.dataset import GraphMultiHopDataset, TextSummarizationDataset


class QGFDTrainer:
    """
    Standardized Fine-Tuning & Evaluation Engine for R&D comparisons.

    Ensures identical training conditions for Baseline (Softmax Attention) vs Treatment (QGFD Attention):
      - Same dataset
      - Same learning rate & schedule
      - Same optimizer & epochs
      - Same batch size
      - Same random seed
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-4,
        device: str = "cpu",
    ):
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)

    @staticmethod
    def compute_rouge_l_proxy(pred_tokens: torch.Tensor, tgt_tokens: torch.Tensor) -> float:
        """Computes longest common subsequence match ratio (ROUGE-L proxy)."""
        pred = pred_tokens.tolist()
        tgt = tgt_tokens.tolist()

        m, n = len(pred), len(tgt)
        if m == 0 or n == 0:
            return 0.0

        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if pred[i - 1] == tgt[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

        lcs = dp[m][n]
        recall = lcs / float(n)
        precision = lcs / float(m)
        if recall + precision == 0:
            return 0.0
        return (2 * precision * recall) / (precision + recall)

    @staticmethod
    def compute_bleu_proxy(pred_tokens: torch.Tensor, tgt_tokens: torch.Tensor) -> float:
        """Computes unigram & bigram precision (BLEU proxy)."""
        pred = pred_tokens.tolist()
        tgt = tgt_tokens.tolist()
        if not pred or not tgt:
            return 0.0

        match_1gram = sum(1 for p in pred if p in tgt)
        p1 = match_1gram / len(pred)

        if len(pred) < 2 or len(tgt) < 2:
            return p1

        pred_2gram = set(zip(pred[:-1], pred[1:]))
        tgt_2gram = set(zip(tgt[:-1], tgt[1:]))
        match_2gram = len(pred_2gram.intersection(tgt_2gram))
        p2 = match_2gram / max(1, len(pred_2gram))

        return math.sqrt(p1 * p2)

    def evaluate(self, dataloader: DataLoader) -> dict[str, float]:
        """Evaluates the model and returns cross-entropy loss plus two token-overlap proxies.

        Returns `{"loss", "rouge_l", "bleu"}`. The two overlap scores are computed from
        argmax token IDs by `compute_rouge_l_proxy` / `compute_bleu_proxy` — they are not
        the reference ROUGE-L / BLEU implementations and are not comparable to published
        figures. Use them to compare two arms trained under identical conditions here,
        nothing more.

        A `bert_score_f1` key used to be returned as well, computed as
        `0.5 * (rouge + bleu) + 0.40`. That is an affine function of the other two
        metrics — it carried no independent information, and the `+0.40` floor made every
        model report a "BERTScore" above 0.40. It has been removed rather than renamed.
        Computing real BERTScore requires a BERT encoder and reference text; this class
        has neither.
        """
        self.model.eval()
        total_loss = 0.0
        total_rouge = 0.0
        total_bleu = 0.0
        total_samples = 0

        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                labels = batch.get("labels", input_ids).to(self.device)

                outputs = self.model(input_ids)
                logits = outputs[0] if isinstance(outputs, (tuple, list)) else outputs

                if logits.dim() == 3:
                    loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
                    preds = logits.argmax(dim=-1)
                else:
                    loss = criterion(logits, labels)
                    preds = logits.argmax(dim=-1)

                total_loss += loss.item() * input_ids.size(0)

                for b in range(input_ids.size(0)):
                    r = self.compute_rouge_l_proxy(preds[b], labels[b])
                    b_score = self.compute_bleu_proxy(preds[b], labels[b])
                    total_rouge += r
                    total_bleu += b_score
                    total_samples += 1

        avg_loss = total_loss / max(1, total_samples)
        avg_rouge = total_rouge / max(1, total_samples)
        avg_bleu = total_bleu / max(1, total_samples)

        return {
            "loss": round(avg_loss, 4),
            "rouge_l": round(avg_rouge, 4),
            "bleu": round(avg_bleu, 4),
        }

    def train_epoch(self, dataloader: DataLoader) -> float:
        """Executes one training epoch."""
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        criterion = nn.CrossEntropyLoss()

        for batch in dataloader:
            self.optimizer.zero_grad()
            input_ids = batch["input_ids"].to(self.device)
            labels = batch.get("labels", input_ids).to(self.device)

            outputs = self.model(input_ids)
            logits = outputs[0] if isinstance(outputs, (tuple, list)) else outputs

            if logits.dim() == 3:
                loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            else:
                loss = criterion(logits, labels)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * input_ids.size(0)
            total_samples += input_ids.size(0)

        return total_loss / max(1, total_samples)
