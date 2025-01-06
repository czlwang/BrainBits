import torch
from torch import nn
from torch.nn import functional as F


class LayerVectorQuantizer(nn.Module):
    # modified from https://github.com/zalandoresearch/pytorch-vq-vae
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(LayerVectorQuantizer, self).__init__()
        
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1/self._num_embeddings, 1/self._num_embeddings)
        self._commitment_cost = commitment_cost

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        # inputs = inputs.permute(0, 2, 3, 1).contiguous()
        # convert inputs from BCT -> BTC
        inputs = inputs.permute(0, 2, 1).contiguous()
        input_shape = inputs.shape
        
        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)
        
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.weight.t()))
            
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss
        
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # convert quantized from BHWC -> BCHW
        # return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings
        # convert quantized from BTC -> BCT
        return loss, quantized.permute(0, 2, 1).contiguous(), perplexity, encodings

class LayerVectorQuantizerEMA(nn.Module):
    # modified from https://github.com/zalandoresearch/pytorch-vq-vae
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, decay, epsilon=1e-5):
        super(LayerVectorQuantizerEMA, self).__init__()
        
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.normal_()
        self._commitment_cost = commitment_cost
        
        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()
        
        self._decay = decay
        self._epsilon = epsilon

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        # inputs = inputs.permute(0, 2, 3, 1).contiguous()
        # convert inputs from BCT -> BTC
        inputs = inputs.permute(0, 2, 1).contiguous()
        input_shape = inputs.shape
        
        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)
        
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.weight.t()))
            
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        
        # Use EMA to update the embedding vectors
        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + \
                                     (1 - self._decay) * torch.sum(encodings, 0)
            
            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                (self._ema_cluster_size + self._epsilon)
                / (n + self._num_embeddings * self._epsilon) * n)
            
            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w = nn.Parameter(self._ema_w * self._decay + (1 - self._decay) * dw)
            
            self._embedding.weight = nn.Parameter(self._ema_w / self._ema_cluster_size.unsqueeze(1))
        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self._commitment_cost * e_latent_loss
        
        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # convert quantized from BHWC -> BCHW
        # return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings
        # convert quantized from BTC -> BCT
        return loss, quantized.permute(0, 2, 1).contiguous(), perplexity, encodings


class VQVAE(nn.Module):
    def __init__(self, inp_dim, mlp_dim, out_dim, codebook_size, tokens, bottleneck_dim, commitment_cost, decay=0):
        super(VQVAE, self).__init__()
        self.num_tokens = tokens
        self.bottleneck_dim = bottleneck_dim
        # 2 layer mlp for encoder with relu
        self._encoder = torch.nn.Sequential(
            torch.nn.Linear(inp_dim, mlp_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(mlp_dim, bottleneck_dim*self.num_tokens),
        )
        if decay > 0.0:
            self._vq_vae = LayerVectorQuantizerEMA(codebook_size, bottleneck_dim, commitment_cost, decay)
        else:
            self._vq_vae = LayerVectorQuantizer(codebook_size, bottleneck_dim, commitment_cost)

        self._decoder = torch.nn.Sequential(
            torch.nn.Linear(bottleneck_dim*self.num_tokens, mlp_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(mlp_dim, out_dim),
        )

    def forward(self, x):
        z = self._encoder(x)
        z = z.view(z.shape[0], self.num_tokens, self.bottleneck_dim)
        loss, quantized, perplexity, _ = self._vq_vae(z)
        quantized = quantized.view(z.shape[0], self.num_tokens*self.bottleneck_dim)
        x_recon = self._decoder(quantized)

        return x_recon, loss, perplexity
