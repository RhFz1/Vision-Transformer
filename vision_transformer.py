import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class ModelArgs:
    n_embd: int = 768
    n_layers: int = 2
    n_heads: int = 24
    max_batch_size: int = 8
    dropout: float = 0.3
    classes: int = 3
    patch_size: int = 16
    block_size: int = 196
    in_channels: int = 3


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int, n_embd: int, batch_size: int, patch_size: int = 16) -> None:
        super().__init__()

        # Here we are trying to convert image to patches then patches to latent space vectors.
        # A smart way is to use Convolution operation
        # Choose a kernel size of patchsize and same stride to ensure patch generation
        # let the depth or number of filters be the latent dim.
        # If you don't understand, take a pen and paper and try to draw.

        self.patch_size = patch_size

        self.cnvblck = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=n_embd,
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

        att = q @ k.transpose(-2, -1) * (k.size(-1) ** -0.5) # (B, T, T), sclaing to ensure gaussian properties retention.
        # mat = torch.tril(torch.ones((T, T)))
        # att = att.masked_fill(mat[:T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim = -1) # (b, t, t)

        v = self.value(x)

        out = att @ v # (B, T, T) (B, T, headsize) -> (B, T, headsize)
        return out

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.rmgain = nn.Parameter(torch.ones((dim)))
    def _norm(self, x: torch.Tensor):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps) # (B, Block, embd)
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

class FeedForward(nn.Module):
    def __init__(self, n_embd: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )
    def forward(self, x: torch.Tensor):
        x = self.net(x)
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, n_embd: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.head_size = n_embd // n_heads
        self.mha = MultiHeadedAttention(n_heads=n_heads, head_size=self.head_size, n_embd=n_embd)
        self.norm = RMSNorm(n_embd)
        self.ffwd = FeedForward(n_embd=n_embd, dropout=dropout)
    def forward(self, x: torch.Tensor):
        x = x + self.mha(self.norm(x))
        x = x + self.ffwd(self.norm(x))
        return x

def ComputePositionalEmbeddings(args: ModelArgs):

    epow = 10000 ** (torch.arange(0, args.n_embd, 2) / args.n_embd) # basically (n_embd / 2)
    opow = 10000 ** (torch.arange(1, args.n_embd, 2) / args.n_embd) # basically (n_embd / 2)
    
    pos = torch.arange(0, args.block_size + 1)
    
    eratio = pos.unsqueeze(1) / epow.unsqueeze(0) # (blocksize + 1, embd / 2)
    oratio = pos.unsqueeze(1) / opow.unsqueeze(0) # (blocksize + 1, embd / 2)

    sine = torch.sin(eratio)
    cosine = torch.cos(oratio)

    # Trying to alternate the values.
    comb = torch.stack((sine, cosine), dim = 2) # this is done to ensure sine values correspong to even and cosine to odd positions.
    out = comb.flatten(start_dim=-2, end_dim=-1) # (blocksize, embd)
    return out

class FullNetwork(nn.Module):
    
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.args = args
        self.patchembd = PatchEmbedding(args.in_channels, args.n_embd, args.max_batch_size, args.patch_size)
        self.eblocks = nn.Sequential(*[TransformerEncoder(args.n_embd, args.n_heads, args.dropout) for _ in range(args.n_layers)])
        self.norm = RMSNorm(args.n_embd)
        self.emdrop = nn.Dropout(args.dropout)
        self.lm_head = nn.Sequential(
            nn.LayerNorm(normalized_shape=args.n_embd),
            nn.Linear(args.n_embd, 4 * args.classes),
            nn.GELU(),
            nn.Linear(4 * args.classes, args.classes),
            nn.Dropout(args.dropout)
        )
        self.pos_embd = nn.Parameter(ComputePositionalEmbeddings(args))
        self.class_token = nn.Parameter(torch.randn(1, 1, args.n_embd),
                                    requires_grad=True)

    def forward(self, x: torch.Tensor, y: torch.Tensor = None):
        
        # Getting the batch size
        B = x.shape[0]

        # Creating patches for each image and converting each patch to latent space vector.
        x = self.patchembd(x) # (B , block, nembd)
        # Adding special token, as per paper.
        x = torch.cat((x, self.class_token.expand(B, -1, -1)), dim = 1)

        # Adding position embeds.
        x = self.pos_embd + x # (B, block, nembd)

        x = self.emdrop(x)

        # Passing the block of embd vectors through enc. transformer layer.
        x = self.eblocks(x)
        
        # Feedforward for classification.
        logits = self.lm_head(x)
        B, T, cls = logits.shape


        # This block is ambiguous to me
        # As I produce 197 patches for each image, each one of them providing values for the 3 classes.
        # Should I calculate softmax for each then average or I take the average of logits along 197 dim then proceed with softmax.
        # Still have to work it through
        # But this works fine!!

        # Manual implementation of cross entropy
        probsn = F.softmax(logits, dim = -1) # (B, T, cls)
        probs = torch.mean(probsn, dim = 1) # (B, cls)
        
        return logits, probs