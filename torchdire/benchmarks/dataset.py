import random
import torch
from torch.utils.data import Dataset


class GraphMultiHopDataset(Dataset):
    """
    Synthetic multi-hop graph path retrieval benchmark.
    Tests ability to navigate multi-step path dependencies (A -> B -> C -> D).
    """

    def __init__(self, num_samples: int = 500, seq_len: int = 64, num_nodes: int = 16, num_hops: int = 3, seed: int = 42):
        super().__init__()
        random.seed(seed)
        torch.manual_seed(seed)
        self.samples = []

        for _ in range(num_samples):
            # Generate random adjacency paths
            nodes = list(range(1, num_nodes + 1))
            path = random.sample(nodes, num_hops + 1)

            # Construct token sequence containing graph edges
            seq = [0] * seq_len
            for i in range(num_hops):
                seq[i * 2] = path[i]
                seq[i * 2 + 1] = path[i + 1]

            # Query token asks for start node's target after num_hops
            query_token = path[0]
            target_token = path[-1]

            self.samples.append({
                "input_ids": torch.tensor(seq, dtype=torch.long),
                "query": torch.tensor(query_token, dtype=torch.long),
                "target": torch.tensor(target_token, dtype=torch.long),
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class PasskeyRetrievalDataset(Dataset):
    """
    Passkey retrieval task for testing long-context recall as sequence length grows.
    """

    def __init__(self, num_samples: int = 200, seq_len: int = 256, vocab_size: int = 1000, seed: int = 42):
        super().__init__()
        random.seed(seed)
        torch.manual_seed(seed)
        self.samples = []

        for _ in range(num_samples):
            seq = torch.randint(10, vocab_size, (seq_len,))
            passkey = random.randint(100, vocab_size - 1)
            passkey_pos = random.randint(10, seq_len - 10)

            seq[passkey_pos] = passkey

            self.samples.append({
                "input_ids": seq,
                "passkey": torch.tensor(passkey, dtype=torch.long),
                "passkey_pos": torch.tensor(passkey_pos, dtype=torch.long),
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class TextSummarizationDataset(Dataset):
    """
    Synthetic text summarization sequence dataset for evaluating ROUGE-L and BLEU metrics.
    """

    def __init__(self, num_samples: int = 300, src_len: int = 128, tgt_len: int = 32, vocab_size: int = 500, seed: int = 42):
        super().__init__()
        random.seed(seed)
        torch.manual_seed(seed)
        self.samples = []

        for _ in range(num_samples):
            src = torch.randint(5, vocab_size, (src_len,))
            tgt = src[:tgt_len].clone()  # Target is key sub-sequence summary

            self.samples.append({
                "input_ids": src,
                "labels": tgt,
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
