import torch
import torch.nn as nn
import math

class HarmonicEmbedding(nn.Module):
    """
    等效替代 pytorch3d.renderer.implicit.harmonic_embedding
    """
    def __init__(self, n_harmonic_functions: int = 6, append_input: bool = True):
        super().__init__()
        self.append_input = append_input
        self.n_harmonic_functions = n_harmonic_functions
        frequencies = 2.0 ** torch.arange(n_harmonic_functions, dtype=torch.float32)
        self.register_buffer("_frequencies", frequencies * math.pi, persistent=False)

    def forward(self, x: torch.Tensor, **kwargs):
        embed = (x[..., None] * self._frequencies).view(*x.shape[:-1], -1)
        embed = torch.cat((embed.sin(), embed.cos()), dim=-1)
        if self.append_input:
            return torch.cat([x, embed], dim=-1)
        return embed

    def get_output_dim(self, input_dim: int = 3) -> int:
        """
        返回经过位置编码后的特征维度。
        公式：2 * input_dim * n_harmonic_functions + (input_dim if append_input else 0)
        """
        dim = input_dim * 2 * self.n_harmonic_functions
        if self.append_input:
            dim += input_dim
        return dim