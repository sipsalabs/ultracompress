"""Fast local data loading from pre-tokenized files.

Usage:
    from lib.local_data import LocalTokenDataset
    dataset = LocalTokenDataset('fineweb_edu_100M_tokens.pt', device='cuda:0')
    batch = dataset.get_batch(batch_size=4, seq_len=64)
"""
import torch


class LocalTokenDataset:
    """Random-access dataset from pre-tokenized .pt file."""

    def __init__(self, path: str, device: str = 'cuda:0'):
        self.tokens = torch.load(path, weights_only=True).to(torch.long)
        self.device = device
        print(f"  LocalTokenDataset: {self.tokens.numel():,} tokens from {path}")

    def get_batch(self, batch_size: int = 4, seq_len: int = 64) -> torch.Tensor:
        starts = torch.randint(0, self.tokens.numel() - seq_len, (batch_size,))
        batch = torch.stack([self.tokens[s:s + seq_len] for s in starts])
        return batch.to(self.device)

    def __len__(self) -> int:
        return self.tokens.numel()
