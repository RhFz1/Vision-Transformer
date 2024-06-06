import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class ModelArgs:
    pass

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int, embd_dim: int, patch_size: int = 16) -> None:
        super().__init__()

        # Here we are trying to convert image to patches then patches to latent space vectors.
        # A smart way is to use Convolution operation
        # Choose a kernel size of patchsize and same stride to ensure patch generation
        # let the depth or number of filters be the latent dim.
        # If you don't understand, take a pen and paper and try to draw.

        self.patch_size = patch_size

        self.cnvblck = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=embd_dim,
                      kernel_size=patch_size,
                      stride=patch_size,
                      padding = 0),
            nn.Flatten(start_dim=2, end_dim=3), # This flattens the 2D map to 1D. (batch, embd_dim, (h * w)/K^2)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        img_dim = x.shape[-1]

        assert img_dim % self.patch_size == 0, "Image shape should be divisible by the patchsize"

        x = self.cnvblck(x)
        x = x.permute(0, 2, 1) # this is to ensure embd dim is the last one (b, embd, (h * w)/k^2) -> (b, (h * w)/k^2, embd)
        
        return x.type_as(x) # to ensure type remains the same after operations.
    
class Head(nn.Module):
    def __init__(self, head_size: int, n_embd: int) -> None:
        super().__init__()
        self.head_size = head_size
        self.n_embd = n_embd
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # if unclear revise self attention
        B, T, C = x.shape

        k = self.key(x) # (B, T, embd) -> (B, T, headsize)
        q = self.query(x) # (B, T, embd) -> (B, T, headsize)

        att = q @ k.transpose(-2, -1) * (torch.rsqrt(k.size(-1))) # (B, T, T), sclaing to ensure gaussian properties.
        mat = torch.tril(torch.ones((T, T)))
        att = torch.masked_fill(mat == 0, float('-inf'))
        att = F.softmax(att, dim = -1) # (b, t, t)

        v = self.value(x)

        out = att @ v # (B, T, T) (B, T, headsize) -> (B, T, headsize)
        return out

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()

        self.rmgain = nn.Parameter(torch.ones((dim)))
        self.eps = eps
    def _norm(self, x: torch.Tensor):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
    def forward(self, x:torch.Tensor):
        return self.rmgain * self._norm(x.float()).type_as(x)
    
class MultiHeadedAttention(nn.Module):
    def __init__(self, n_heads: int, head_size: int, n_embd: int) -> None:
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, n_embd) for _ in range(n_heads)])
        self.proj = nn.Linear(n_heads * head_size, n_embd, bias= False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat([h(x) for h in self.heads], dim = -1)
        out = self.proj(out)
        return out

class Block(nn.Module):
    def __init__(self, n_layers: int, n_heads: int, n_embd: int) -> None:
        super().__init__()
        self.head_size = n_embd // n_heads
        self.mha = MultiHeadedAttention(n_heads=n_heads, head_size=self.head_size, n_embd=n_embd)
        self.norm = RMSNorm(n_embd)
        self.ffwd = 




class TransformerEncoder(nn.Module):

    def __init__(self, n_embd: int, head_size):
        pass