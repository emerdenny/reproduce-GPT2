from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    block_size: int = 256
    vocab_size: int = 65
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # GPT2 is a decoder-only model, no encoder w/ cross attn like in AIAYN
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(
                    config.vocab_size, config.n_embd
                ),  # weights of token embeddings
                wpe=nn.Embedding(
                    config.block_size, config.n_embd
                ),  # weights of position embeddings
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=nn.LayerNorm(
                    config.n_embd
                ),  # from GPT2 paper, not shown in AIAYN transformer figure
            )
        )
        self.lm_head = nn.Linear(
            config.n_embd, config.vocab_size, bias=False
        )  # projects from embedding dim to token dim for output


def main():
    print("Hello from reproduce-gpt2!")


if __name__ == "__main__":
    main()
