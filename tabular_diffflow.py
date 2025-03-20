import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import math
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

class EnhancedSparseFeatureAttention(nn.Module):
    """Enhanced attention mechanism for financial features with better correlation modeling."""
    def __init__(self, n_features, d_model, sparsity=0.5, heads=4, dropout=0.1):
        super().__init__()
        self.n_features = n_features
        self.d_model = d_model
        self.sparsity = sparsity
        self.heads = heads
        self.head_dim = d_model // heads
        assert self.head_dim * heads == d_model, "d_model must be divisible by heads"
        
        # Multi-head attention components
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Learnable feature correlation prior with structure for financial data
        # Initialize with temporal decay pattern (common in financial time series)
        correlation_prior = torch.zeros(n_features, n_features)
        for i in range(n_features):
            for j in range(n_features):
                # Closer features have stronger prior correlation
                correlation_prior[i, j] = 0.5 * np.exp(-0.1 * abs(i - j))
        self.correlation_prior = nn.Parameter(correlation_prior)
        
        # Feature importance weights
        self.feature_importance = nn.Parameter(torch.ones(n_features))
        
        # For financial data specific correlations
        self.price_vol_detector = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.SiLU(),
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )
        
    def _split_heads(self, x):
        """Split the last dimension into (heads, head_dim)"""
        batch_size, n_features, d_model = x.shape
        x = x.view(batch_size, n_features, self.heads, self.head_dim)
        # (batch_size, heads, n_features, head_dim)
        return x.permute(0, 2, 1, 3)
        
    def _merge_heads(self, x):
        """Merge the heads back into d_model dimension"""
        batch_size, heads, n_features, head_dim = x.shape
        # (batch_size, n_features, heads, head_dim)
        x = x.permute(0, 2, 1, 3)
        # (batch_size, n_features, d_model)
        return x.reshape(batch_size, n_features, self.d_model)
        
    def forward(self, x):
        """Forward pass with multi-head attention and financial feature correlations"""
        # x shape: [batch_size, n_features, d_model]
        batch_size, n_features, _ = x.shape
        
        # Linear projections
        q = self.query(x)  # [batch_size, n_features, d_model]
        k = self.key(x)    # [batch_size, n_features, d_model]
        v = self.value(x)  # [batch_size, n_features, d_model]
        
        # Split heads
        q = self._split_heads(q)  # [batch_size, heads, n_features, head_dim]
        k = self._split_heads(k)  # [batch_size, heads, n_features, head_dim]
        v = self._split_heads(v)  # [batch_size, heads, n_features, head_dim]
        
        # Scaled dot-product attention with masks for each head
        # [batch_size, heads, n_features, n_features]
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_dim)
        
        # Apply learned correlation prior 
        # Reshape for broadcasting over heads
        prior = self.correlation_prior.unsqueeze(0).unsqueeze(1)
        
        # Weight by feature importance
        importance = self.feature_importance.unsqueeze(0).unsqueeze(1)
        importance = importance.unsqueeze(-1) * importance.unsqueeze(-2)
        
        # Combine with learned prior
        scores = scores + prior * importance
        
        # Detect price-volume correlations (common in financial data)
        if n_features >= 2:
            # Simplified assumption that early features are price-related
            # and later features are volume-related
            price_features = x[:, :n_features//2].mean(dim=1)  # [batch_size, d_model]
            vol_features = x[:, n_features//2:].mean(dim=1)    # [batch_size, d_model]
            combined = torch.cat([price_features, vol_features], dim=1)
            pv_correlation = self.price_vol_detector(combined).view(batch_size, 1, 1, 1)
            
            # Create price-volume correlation mask that boosts correlations
            # between price and volume features
            pv_mask = torch.zeros_like(scores)
            mid = n_features // 2
            pv_mask[:, :, :mid, mid:] = 1.0  # price to volume connections
            pv_mask[:, :, mid:, :mid] = 1.0  # volume to price connections
            
            # Apply price-volume boost
            scores = scores + pv_correlation * pv_mask * 0.5
        
        # Apply sparsity by keeping only top-k values per feature
        k_value = max(1, int(n_features * self.sparsity))
        
        # Process each head separately
        for h in range(self.heads):
            # For each head, get top-k values
            topk_values, _ = torch.topk(scores[:, h], k_value, dim=-1)
            threshold = topk_values[:, :, -1].unsqueeze(-1)
            mask = scores[:, h] < threshold
            scores[:, h] = scores[:, h].masked_fill(mask, -1e9)
        
        # Apply softmax across all heads
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        
        # Apply attention to values
        output = torch.matmul(attention, v)  # [batch_size, heads, n_features, head_dim]
        
        # Merge heads back
        output = self._merge_heads(output)  # [batch_size, n_features, d_model]
        
        # Final projection
        output = self.out_proj(output)
        
        return output


class SparseFeatureAttention(EnhancedSparseFeatureAttention):
    """Legacy class for backward compatibility."""
    pass

class StandardNoiseSchedule(nn.Module):
    def __init__(self, n_timesteps, beta_start=1e-4, beta_end=0.02):
        super().__init__()
        self.n_timesteps = n_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.betas = torch.linspace(beta_start, beta_end, n_timesteps)
        self.alphas = 1.0 - self.betas
        self.alpha_hats = torch.cumprod(self.alphas, dim=0)

    def get_betas(self):
        return self.betas

    def get_alphas(self):
        return self.alphas

    def get_alpha_hats(self):
        return self.alpha_hats


class TemporalTransformer(nn.Module):
    """
    Transformer layer for capturing temporal dependencies in financial data.
    
    This module helps the model understand time-series relationships and
    patterns that are crucial for financial data modeling.
    """
    def __init__(self, d_model, nhead=4, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.activation = nn.GELU()
        
        # Financial-specific enhancement: time-decay attention mask
        self.register_buffer('time_decay', None)
        
    def _get_time_decay_mask(self, seq_len, device):
        """Generate a time decay mask that emphasizes recent temporal patterns"""
        if self.time_decay is None or self.time_decay.size(0) < seq_len:
            # Create new decay matrix
            decay_matrix = torch.zeros(seq_len, seq_len, device=device)
            for i in range(seq_len):
                for j in range(seq_len):
                    # Exponential decay with distance - financial data often shows
                    # exponentially decreasing influence from past values
                    decay_matrix[i, j] = np.exp(-0.1 * abs(i - j))
            self.time_decay = decay_matrix
            
        return self.time_decay[:seq_len, :seq_len]
        
    def forward(self, src, src_mask=None, use_time_decay=True):
        """
        Forward pass with temporal awareness for financial data
        
        Args:
            src: Source sequence [seq_len, batch_size, d_model]
            src_mask: Optional attention mask
            use_time_decay: Whether to apply time decay masking for financial data
        """
        seq_len, batch_size, _ = src.shape
        
        # Self-attention block with time decay for financial data
        if use_time_decay and src_mask is None:
            time_decay_mask = self._get_time_decay_mask(seq_len, src.device)
            # Apply attention with time decay mask
            src2 = self.self_attn(src, src, src, attn_mask=time_decay_mask)[0]
        else:
            # Standard attention
            src2 = self.self_attn(src, src, src, attn_mask=src_mask)[0]
            
        src = src + src2
        src = self.norm1(src)
        
        # Feedforward block
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + src2
        src = self.norm2(src)
        
        return src


class VolatilityAwareEncoding(nn.Module):
    """
    Encodes volatility information for financial data.
    
    Financial data generation benefits from explicit volatility awareness
    as volatility clustering is a key feature of financial time series.
    """
    def __init__(self, d_model):
        super().__init__()
        self.volatility_embedding = nn.Sequential(
            nn.Linear(1, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, d_model)
        )
        
    def forward(self, x, volatility):
        """
        Args:
            x: Input features
            volatility: Volatility estimate (e.g., rolling standard deviation)
        """
        # Ensure volatility is properly shaped
        if not isinstance(volatility, torch.Tensor):
            volatility = torch.tensor([volatility], device=x.device)
        
        if volatility.dim() == 0:
            volatility = volatility.unsqueeze(0)
            
        if volatility.dim() == 1 and volatility.size(0) == 1 and x.size(0) > 1:
            volatility = volatility.expand(x.size(0))
            
        vol_emb = self.volatility_embedding(volatility.unsqueeze(-1))
        
        # Add volatility embedding to input features
        if x.dim() == 2:  # [batch_size, d_model]
            return x + vol_emb
        elif x.dim() == 3:  # [batch_size, seq_len/n_features, d_model]
            return x + vol_emb.unsqueeze(1)
            
        return x


class AdaptiveNoiseSchedule(FinancialNoiseSchedule):
    """Legacy class for backward compatibility."""
    def __init__(self, n_timesteps, init_beta_min=1e-4, init_beta_max=2e-2):
        super().__init__(n_timesteps, init_beta_min, init_beta_max, regime_aware=True)

class CouplingLayer(nn.Module):
    """Affine coupling layer for normalizing flow."""
    def __init__(self, in_dim, out_dim, hidden_dim):
        super().__init__()
        self.in_dim = in_dim
        self.scale_net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim)
        )
        self.translate_net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim)
        )
        
        # Initialize to identity transformation
        for m in self.scale_net.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.zero_()
                if hasattr(m, 'bias') and m.bias is not None:
                    m.bias.data.zero_()
        
    def forward(self, x):
        x1, x2 = x[:, :self.in_dim], x[:, self.in_dim:]
        
        # Calculate scale and translate factors
        s = torch.tanh(self.scale_net(x1))  # Bounded scale factor
        t = self.translate_net(x1)
        
        # Transform x2
        y2 = x2 * torch.exp(s) + t
        y = torch.cat([x1, y2], dim=1)
        
        # Calculate log determinant
        log_det = s.sum(dim=1)
        
        return y, log_det
    
    def inverse(self, y):
        y1, y2 = y[:, :self.in_dim], y[:, self.in_dim:]
        
        # Calculate scale and translate factors
        s = torch.tanh(self.scale_net(y1))
        t = self.translate_net(y1)
        
        # Invert the transformation
        x2 = (y2 - t) * torch.exp(-s)
        x = torch.cat([y1, x2], dim=1)
        
        return x

class NormalizingFlow(nn.Module):
    """Simple normalizing flow with affine coupling layers."""
    def __init__(self, n_features, hidden_dim, n_layers=3):
        super().__init__()
        self.n_features = n_features
        self.n_layers = n_layers
        
        # Split features into two groups
        self.split_idx = n_features // 2
        
        # Coupling layers
        self.coupling_layers = nn.ModuleList([
            CouplingLayer(self.split_idx, n_features - self.split_idx, hidden_dim)
            for _ in range(n_layers)
        ])
        
    def forward(self, x, reverse=False):
        log_det = torch.zeros(x.shape[0], device=x.device)
        
        if not reverse:
            # Forward direction (data -> latent)
            for layer in self.coupling_layers:
                x, ld = layer(x)
                log_det += ld
            return x, log_det
        else:
            # Reverse direction (latent -> data)
            for layer in reversed(self.coupling_layers):
                x = layer.inverse(x)
            return x

class TabularDiffFlow(nn.Module):
    """
    Novel hybrid model combining diffusion models and normalizing flows
    for synthetic tabular data generation, enhanced with temporal awareness
    and financial data optimization.
    """
    def __init__(
        self, 
        n_features,
        d_model=128,
        n_diffusion_steps=1000,
        n_flow_layers=3,
        sparsity=0.5,
        feature_ranges=None,  # Tuple of (min, max) for each feature
        financial_data=True,  # Enable financial data optimizations
        trend_preservation=True,  # For better trend modeling in financial time series
        use_temporal_transformer=True,  # Enable temporal transformer component
        n_heads=4,  # Number of attention heads
        dropout=0.1  # Dropout rate for regularization
    ):
        super().__init__()
        self.n_features = n_features
        self.d_model = d_model
        self.n_diffusion_steps = n_diffusion_steps
        self.financial_data = financial_data
        self.trend_preservation = trend_preservation
        self.use_temporal_transformer = use_temporal_transformer
        
        # Feature normalization ranges
        if feature_ranges is None:
            self.register_buffer('feature_min', torch.zeros(n_features))
            self.register_buffer('feature_max', torch.ones(n_features))
        else:
            self.register_buffer('feature_min', torch.tensor([x[0] for x in feature_ranges], dtype=torch.float32))
            self.register_buffer('feature_max', torch.tensor([x[1] for x in feature_ranges], dtype=torch.float32))
        
        # Enhanced noise schedule for financial data
        self.noise_schedule = FinancialNoiseSchedule(
            n_diffusion_steps,
            init_beta_min=1e-4,
            init_beta_max=2e-2,
            regime_aware=financial_data
        )
        
        # Enhanced feature-wise attention mechanism with multiple heads
        self.feature_attention = EnhancedSparseFeatureAttention(
            n_features, 
            d_model, 
            sparsity=sparsity,
            heads=n_heads,
            dropout=dropout
        )
        
        # Feature embedding with better initialization for financial data
        self.feature_embedding = nn.Sequential(
            nn.Linear(n_features, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model)
        )
        
        # Time embedding with sinusoidal components for better modeling of periodicity
        # This helps with financial data that often exhibits daily, weekly, monthly patterns
        self.time_embedding = nn.Sequential(
            nn.Linear(1, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model)
        )
        
        # Add volatility awareness for financial data
        if financial_data:
            self.volatility_encoder = VolatilityAwareEncoding(d_model)
        
        # Add temporal transformer for time-series understanding
        if use_temporal_transformer:
            self.temporal_transformer = TemporalTransformer(
                d_model, 
                nhead=n_heads,
                dim_feedforward=d_model * 2,
                dropout=dropout
            )
            
        # Helper function to create improved feature processors
        def create_feature_processor():
            return nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.SiLU(),
                nn.Linear(d_model, d_model),
                nn.SiLU(),
                nn.Linear(d_model, d_model)
            )
            
        # Feature-wise processing with shared weights 
        # (more efficient and better generalization)
        if n_features <= 32:
            # For smaller feature sets, use per-feature processing
            self.feature_processor = nn.ModuleList([
                create_feature_processor() for _ in range(n_features)
            ])
            self.shared_processing = False
        else:
            # For larger feature sets, use shared weights with feature IDs
            self.feature_processor = create_feature_processor()
            # Feature ID embedding to distinguish between features
            self.feature_id_embedding = nn.Embedding(n_features, d_model // 4)
            self.shared_processing = True
        
        # Global processing with attention and residual connections
        self.global_processor = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.SiLU(),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model)  # Adds stability to financial data training
        )
        
        # Denoising network with improved architecture for financial data
        self.denoise_net = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.SiLU(),
            nn.Dropout(dropout),  # Added dropout for regularization
            nn.Linear(d_model * 2, d_model * 2),
            nn.SiLU(),
            nn.Dropout(dropout),  # Added dropout for regularization
            nn.Linear(d_model * 2, n_features * 2)  # Predict mean and log variance
        )
        
        # Normalizing flow for fine details
        self.flow = NormalizingFlow(n_features, d_model, n_layers=n_flow_layers)
        
        # Flag for market regime detection
        self.detect_regime = financial_data
        if self.detect_regime:
            # Simple regime detector based on feature patterns
            self.regime_detector = nn.Sequential(
                nn.Linear(n_features, d_model // 2),
                nn.SiLU(),
                nn.Linear(d_model // 2, 1),
                nn.Tanh()  # Output from -1 (bear) to 1 (bull)
            )
            
            # Volatility estimator
            self.volatility_estimator = nn.Sequential(
                nn.Linear(n_features, d_model // 2),
                nn.SiLU(),
                nn.Linear(d_model // 2, 1),
                nn.Softplus()  # Ensure positive volatility
            )
    
    def _process_embeddings(self, x_t, t):
        """
        Process input embeddings with feature attention.
        This helper function reduces code duplication between predict_noise and diffusion_reverse_process.
        """
        batch_size = x_t.shape[0]
        
        # Time embedding
        t_emb = self.time_embedding(t.view(-1, 1))  # [batch_size, d_model]
        
        # Feature embedding
        x_emb = self.feature_embedding(x_t)  # [batch_size, d_model]
        
        # Apply volatility awareness for financial data
        if self.financial_data:
            # Estimate volatility from input features
            volatility = self.volatility_estimator(x_t).view(-1)
            # Apply volatility encoding
            x_emb = self.volatility_encoder(x_emb, volatility)
        
        # Expand to feature-wise representations
        x_emb = x_emb.unsqueeze(1).expand(-1, self.n_features, -1)  # [batch_size, n_features, d_model]
        t_emb = t_emb.unsqueeze(1).expand(-1, self.n_features, -1)  # [batch_size, n_features, d_model]
        
        # Combine time and feature embeddings
        combined = torch.cat([x_emb, t_emb], dim=2)  # [batch_size, n_features, d_model*2]
        
        # Process features (either individually or with shared weights)
        if self.shared_processing:
            # Use shared processing with feature IDs for larger feature sets
            # First, get feature IDs
            feature_ids = torch.arange(self.n_features, device=x_t.device)
            # Get feature ID embeddings
            feature_id_embs = self.feature_id_embedding(feature_ids)  # [n_features, d_model//4]
            
            # Reshape combined for vectorized processing
            batch_dim = combined.shape[0]
            combined_flat = combined.reshape(-1, combined.shape[-1])  # [batch*n_features, d_model*2]
            
            # Process all features at once with shared weights
            processed_flat = self.feature_processor(combined_flat)  # [batch*n_features, d_model]
            processed = processed_flat.reshape(batch_dim, self.n_features, -1)  # [batch, n_features, d_model]
            
            # Add feature ID embeddings to help distinguish features
            feature_id_embs = feature_id_embs.unsqueeze(0).expand(batch_dim, -1, -1)
            feature_id_padding = torch.zeros(batch_dim, self.n_features, 
                                           self.d_model - feature_id_embs.shape[-1], 
                                           device=x_t.device)
            feature_id_embs_padded = torch.cat([feature_id_embs, feature_id_padding], dim=2)
            
            # Combine processed features with their IDs
            feature_outputs = processed + 0.1 * feature_id_embs_padded
        else:
            # Process each feature individually for smaller feature sets
            feature_outputs = []
            for i, processor in enumerate(self.feature_processor):
                feature_outputs.append(processor(combined[:, i]))
            feature_outputs = torch.stack(feature_outputs, dim=1)  # [batch_size, n_features, d_model]
        
        # Apply attention to capture feature dependencies
        attended = self.feature_attention(feature_outputs)  # [batch_size, n_features, d_model]
        
        # Apply temporal transformer for financial data if enabled
        if self.use_temporal_transformer:
            # Reshape for transformer (seq_len, batch, features)
            transformed = attended.permute(1, 0, 2)  # [n_features, batch_size, d_model]
            transformed = self.temporal_transformer(transformed)
            # Reshape back
            attended = transformed.permute(1, 0, 2)  # [batch_size, n_features, d_model]
        
        # Global processing
        global_repr = attended.mean(dim=1)  # [batch_size, d_model]
        global_processed = self.global_processor(global_repr)  # [batch_size, d_model]
        
        return global_processed
    
    def normalize_features(self, x):
        """Normalize features to [0, 1] range."""
        return (x - self.feature_min) / (self.feature_max - self.feature_min + 1e-6)
    
    def denormalize_features(self, x):
        """Convert [0, 1] range back to original feature scales."""
        return x * (self.feature_max - self.feature_min + 1e-6) + self.feature_min
    
    def detect_market_regime(self, x):
        """
        Detect market regime (bull/bear) from input features.
        Returns value between -1 (bear) and 1 (bull).
        """
        if not self.detect_regime:
            return None
        
        with torch.no_grad():
            regime = self.regime_detector(x)
        return regime.view(-1)
    
    def estimate_volatility(self, x):
        """
        Estimate market volatility from input features.
        """
        if not self.financial_data:
            return None
            
        with torch.no_grad():
            volatility = self.volatility_estimator(x)
        return volatility.view(-1)
    
    def diffusion_forward_process(self, x_0, t, market_regime=None):
        """
        Forward diffusion process q(x_t | x_0) with market regime awareness.
        
        Args:
            x_0: Original data
            t: Time step (between 0 and 1)
            market_regime: Optional market regime indicator (-1 to 1)
            
        Returns:
            x_t: Noised data at time t
            epsilon: The noise added
        """
        # Auto-detect market regime if not provided and model is configured for it
        if market_regime is None and self.detect_regime:
            market_regime = self.detect_market_regime(x_0)
            
        # Auto-detect volatility for financial data
        volatility = None
        if self.financial_data:
            volatility = self.estimate_volatility(x_0)
        
        # Get time indices
        t_idx = (t * (self.n_diffusion_steps - 1)).long()
        
        # Get alphas with regime and volatility awareness
        if market_regime is not None or volatility is not None:
            alpha_hat = torch.ones_like(x_0[:, 0]).unsqueeze(-1)
            alphas = self.noise_schedule.get_all_alphas()
            
            # Compute alpha_hat manually to incorporate regime/volatility
            for i in range(t_idx.max().item() + 1):
                mask = (t_idx >= i).float().unsqueeze(-1)
                step_alpha = alphas[i].view(1, 1)
                alpha_hat = alpha_hat * torch.where(mask > 0, step_alpha, torch.ones_like(step_alpha))
        else:
            # Standard approach without regime/volatility
            alpha_hat = self.noise_schedule.get_all_alpha_hats()[t_idx]
            alpha_hat = alpha_hat.view(-1, 1)
        
        # Sample noise
        epsilon = torch.randn_like(x_0)
        
        # Add noise according to diffusion schedule
        x_t = torch.sqrt(alpha_hat) * x_0 + torch.sqrt(1 - alpha_hat) * epsilon
        
        return x_t, epsilon
    
    def predict_noise(self, x_t, t, market_regime=None):
        """
        Predict the noise component in x_t.
        
        Args:
            x_t: Noised data at time t
            t: Time steps
            market_regime: Optional market regime indicator
            
        Returns:
            epsilon: Predicted noise
        """
        # Use shared embedding processing
        global_processed = self._process_embeddings(x_t, t)
        
        # Predict noise
        raw_outputs = self.denoise_net(global_processed)
        
        # Split into mean and log variance predictions
        mean_raw, _ = raw_outputs.chunk(2, dim=1)
        
        # Apply activation to bound the mean prediction
        mean = torch.tanh(mean_raw)  # Bound to [-1, 1]
        
        # Return the bounded mean as the predicted noise
        return mean
    
    def diffusion_reverse_process(self, x_t, t, market_regime=None):
        """
        Single step of the reverse diffusion process p(x_{t-1} | x_t).
        
        Args:
            x_t: Noised data at time t
            t: Time step (between 0 and 1)
            market_regime: Optional market regime indicator
            
        Returns:
            x_pred: Predicted less noisy data
        """
        # Auto-detect market regime if not provided and model is configured for it
        if market_regime is None and self.detect_regime:
            market_regime = self.detect_market_regime(x_t)
            
        # Auto-detect volatility for financial data
        volatility = None
        if self.financial_data:
            volatility = self.estimate_volatility(x_t)
        
        # Use shared embedding processing
        global_processed = self._process_embeddings(x_t, t)
        
        # Predict denoised data
        raw_outputs = self.denoise_net(global_processed)
        
        # Split into mean and log variance predictions
        mean_raw, log_var_raw = raw_outputs.chunk(2, dim=1)
        
        # Apply activations to bound the outputs
        mean = torch.tanh(mean_raw)  # Bound mean to [-1, 1]
        log_var = F.softplus(log_var_raw) - 5.0  # Bound log variance for numerical stability
        var = torch.exp(log_var)
        
        # Get the correct beta value for this time step, with market regime awareness
        beta = self.noise_schedule(t, market_regime, volatility)
        beta = beta.view(-1, 1)
        
        # Calculate the parameters for the posterior distribution
        t_idx = (t * (self.n_diffusion_steps - 1)).long()
        
        # Get alphas with regime awareness
        if market_regime is not None or volatility is not None:
            # Compute alpha and alpha_hat with regime/volatility for this specific timestep
            alpha = 1.0 - beta
            
            # Compute alpha_hat manually to handle regime changes
            alpha_hat = torch.ones_like(beta)
            alphas = self.noise_schedule.get_all_alphas()
            
            for i in range(t_idx.max().item() + 1):
                if i >= self.n_diffusion_steps:
                    break
                    
                mask = (t_idx >= i).float().unsqueeze(-1)
                
                # Get regime-specific alpha for this step
                if i == t_idx.max().item():
                    # For the current step, use the exact regime-aware beta
                    step_alpha = alpha
                else:
                    # For past steps, use the regime from our model's prediction
                    step_alpha = alphas[i].view(1, 1).expand(alpha.shape)
                
                alpha_hat = alpha_hat * torch.where(mask > 0, step_alpha, torch.ones_like(step_alpha))
        else:
            # Standard approach without regime
            alpha = self.noise_schedule.get_all_alphas()[t_idx].view(-1, 1)
            alpha_hat = self.noise_schedule.get_all_alpha_hats()[t_idx].view(-1, 1)
        
        # Calculate posterior mean with numerical stability improvements
        posterior_mean_coef1 = (1.0 / torch.sqrt(alpha) + 1e-8)
        posterior_mean_coef2 = (beta / torch.sqrt(1.0 - alpha_hat + 1e-8))
        
        posterior_mean = posterior_mean_coef1 * (x_t - posterior_mean_coef2 * mean)
        
        # Safe calculation of posterior variance
        posterior_var = beta * (1.0 - alpha_hat / (alpha + 1e-8)) / (1.0 - alpha_hat + 1e-8)
        posterior_var = torch.clamp(posterior_var, 1e-8, 0.5)  # Safety clamp
        
        # Sample from the posterior - reduce noise at the end of the process for better quality
        noise_scale = 1.0
        if t[0] < 0.1:  # Reduce noise near the end
            noise_scale = t[0] * 10.0  # Linear scaling to 0
            
        # Add noise (zeros if at t=0)
        noise = torch.randn_like(x_t) * noise_scale if t[0] > 0 else 0
        x_pred = posterior_mean + torch.sqrt(posterior_var) * noise
        
        return x_pred
    
    def diffusion_full_reverse_process(self, batch_size, device, temp=1.0, guidance_scale=1.5):
        """
        Generate samples by running the full reverse diffusion process with improved quality.
        
        Args:
            batch_size: Number of samples to generate
            device: Device to generate samples on
            temp: Temperature for sampling (lower = more conservative)
            guidance_scale: Classifier-free guidance scale (higher = better quality but less diversity)
            
        Returns:
            x_0: Generated samples
        """
        # Ensure model is in eval mode
        self.eval()
        
        # Start from pure noise with temperature control
        x_T = torch.randn(batch_size, self.n_features, device=device) * temp
        
        # Use more steps for better quality sampling
        num_inference_steps = min(self.n_diffusion_steps, 100)  # Cap at 100 for efficiency
        
        # Create sampling schedule with more focus on low-noise regime
        timesteps = torch.linspace(1, 0, num_inference_steps, device=device)**2  # Quadratic schedule
        
        # Progress tracking
        progress_bar = None
        try:
            from tqdm import tqdm
            progress_bar = tqdm(total=num_inference_steps, desc="Generating samples")
        except ImportError:
            print("Generating samples...")
        
        # Iteratively denoise
        x_t = x_T
        with torch.no_grad():
            for t in timesteps:
                # Expand t for the batch
                t_batch = t.expand(batch_size)
                
                # Generate unconditional noise prediction
                noise_pred = self.predict_noise(x_t, t_batch)
                
                # Apply classifier-free guidance if scale > 1
                if guidance_scale > 1.0:
                    # Generate another noise prediction with zero conditioning
                    # (use a random diffusion time as a form of unconditional guidance)
                    random_t = torch.rand_like(t_batch)
                    uncond_noise_pred = self.predict_noise(x_t, random_t)
                    
                    # Apply guidance formula: pred = uncond_pred + scale * (cond_pred - uncond_pred)
                    noise_pred = uncond_noise_pred + guidance_scale * (noise_pred - uncond_noise_pred)
                
                # Apply single reverse step with the guided noise prediction
                x_t = self.diffusion_reverse_process(x_t, t_batch)
                
                # Update progress
                if progress_bar is not None:
                    progress_bar.update(1)
        
        if progress_bar is not None:
            progress_bar.close()
            
        # Apply normalizing flow for fine details (with error handling)
        try:
            x_0 = self.flow(x_t, reverse=True)
        except Exception as e:
            print(f"Warning: Flow application failed ({str(e)}). Using diffusion output directly.")
            x_0 = x_t
        
        # Apply statistical constraints to improve realism
        # (constrain values within reasonable bounds based on the training data)
        x_0 = torch.clamp(x_0, -3, 3)  # Limit to 3 std deviations
        
        # Denormalize to original feature scales
        x_0 = self.denormalize_features(x_0)
        
        return x_0
    
    def forward(self, x, t=None, market_regime=None):
        """
        Forward pass for training with improved stability and financial-specific enhancements.
        
        Args:
            x: Input data [batch_size, n_features]
            t: Optional time steps (if None, will be randomly sampled)
            market_regime: Optional market regime indicator (e.g., bull/bear)
            
        Returns:
            loss: Training loss
        """
        batch_size = x.shape[0]
        device = x.device
        
        # Normalize features with safety checks
        x_normalized = self.normalize_features(x)
        
        # Check for NaNs and infinities
        if torch.isnan(x_normalized).any() or torch.isinf(x_normalized).any():
            # Replace problematic values with zeros (for stability)
            x_normalized = torch.nan_to_num(x_normalized, nan=0.0, posinf=1.0, neginf=-1.0)
            
        # Apply gradient clipping on input (prevents extreme values)
        x_normalized = torch.clamp(x_normalized, -10.0, 10.0)
        
        # Detect market regime automatically if needed
        if market_regime is None and self.detect_regime:
            market_regime = self.detect_market_regime(x_normalized)
            
        # Detect volatility for financial data if enabled
        volatility = None
        if self.financial_data:
            volatility = self.estimate_volatility(x_normalized)
        
        # Apply normalizing flow (data -> latent) with error handling
        try:
            z, log_det = self.flow(x_normalized)
            # Prevent extreme flow values
            log_det = torch.clamp(log_det, -50.0, 50.0)
        except Exception as e:
            # If flow fails, use normalized input directly
            z = x_normalized
            log_det = torch.zeros(batch_size, device=device)
            print(f"Warning: Flow failed ({str(e)}). Using normalized input.")
        
        # Sample time steps if not provided
        if t is None:
            if self.financial_data:
                # For financial data, use power-law distribution to focus more on recent time steps
                # This helps model the recent market conditions better
                t = torch.pow(torch.rand(batch_size, device=device), 1.5)
            else:
                # Standard timestep sampling
                t = torch.sqrt(torch.rand(batch_size, device=device))
        
        # Forward diffusion process with market regime awareness
        z_t, epsilon = self.diffusion_forward_process(z, t, market_regime)
        
        # Predict the added noise
        epsilon_pred = self.predict_noise(z_t, t, market_regime)
        
        # Financial-specific loss components
        
        # 1. Basic diffusion loss with smoother Huber loss
        diffusion_loss = F.smooth_l1_loss(epsilon_pred, epsilon, beta=0.1)
        
        # 2. Flow loss with safeguards
        flow_loss = -log_det.mean()
        if torch.isnan(flow_loss) or torch.isinf(flow_loss):
            flow_loss = torch.tensor(0.0, device=device)
            
        # 3. Tail risk loss - focused on extreme events important in financial data
        # Create mask for extreme values (typically beyond 2 sigma)
        if self.financial_data:
            tail_mask = (torch.abs(epsilon) > 2.0).float()
            # Check if we have any tail events in this batch
            if tail_mask.sum() > 0:
                # Focus more attention on extreme value prediction (important for financial risk)
                tail_loss = F.mse_loss(
                    epsilon_pred * tail_mask, 
                    epsilon * tail_mask, 
                    reduction='sum'
                ) / (tail_mask.sum() + 1e-8)
            else:
                tail_loss = torch.tensor(0.0, device=device)
                
            # 4. Trend preservation loss for financial time series
            if self.trend_preservation:
                # Skip the trend loss if batch size is 1
                if batch_size > 1:
                    # For financial time series, preserving the direction of change is crucial
                    # We want adjacent samples to maintain the same relative trends
                    # Calculate sign of differences between adjacent time series points
                    signs_real = torch.sign(z[1:] - z[:-1])
                    signs_pred = torch.sign(epsilon_pred[1:] - epsilon_pred[:-1])
                    
                    # Binary accuracy of sign prediction
                    trend_match = (signs_real == signs_pred).float()
                    trend_loss = 1.0 - trend_match.mean()
                else:
                    trend_loss = torch.tensor(0.0, device=device)
            else:
                trend_loss = torch.tensor(0.0, device=device)
                
            # 5. Volatility clustering loss - Helps model autocorrelation in squared returns
            if volatility is not None and batch_size > 1:
                # Calculate predicted volatility from the model
                pred_volatility = torch.abs(epsilon_pred).mean(dim=1)
                # Encourage similar volatility prediction
                volatility_loss = F.mse_loss(pred_volatility, volatility)
            else:
                volatility_loss = torch.tensor(0.0, device=device)
                
        else:
            # Not financial data - set these losses to zero
            tail_loss = torch.tensor(0.0, device=device)
            trend_loss = torch.tensor(0.0, device=device)
            volatility_loss = torch.tensor(0.0, device=device)
            
        # Dynamic weighting of loss components - adjust based on training progress
        if self.financial_data:
            # For financial data, carefully balance the different components
            lambda_diffusion = 1.0
            
            # Reduce flow importance when other losses are working well
            if diffusion_loss.item() < 0.1:
                lambda_flow = 0.05
            else:
                lambda_flow = 0.1
                
            # Tail risk gets higher weight as it's crucial for financial models
            lambda_tail = min(0.5, 10.0 * diffusion_loss.item())
            
            # Trend preservation is important but not dominant
            lambda_trend = 0.2 
            
            # Volatility clustering is moderately important
            lambda_volatility = 0.1
            
            # Total financial-aware loss
            loss = (
                lambda_diffusion * diffusion_loss +
                lambda_flow * flow_loss +
                lambda_tail * tail_loss +
                lambda_trend * trend_loss +
                lambda_volatility * volatility_loss
            )
        else:
            # Standard loss weighting for non-financial data
            lambda_diffusion = 1.0
            lambda_flow = 0.1 if diffusion_loss.item() >= 0.1 else 0.05
            
            # Total standard loss
            loss = lambda_diffusion * diffusion_loss + lambda_flow * flow_loss
        
        # Add regularization on attention mechanism for better stability
        if hasattr(self.feature_attention, 'correlation_prior'):
            l2_reg = torch.norm(self.feature_attention.correlation_prior) * 0.001
            loss = loss + l2_reg
        
        # Safety check
        if torch.isnan(loss) or torch.isinf(loss):
            # Fallback to diffusion loss if something goes wrong
            loss = diffusion_loss
            
        return loss
    
    def sample(self, batch_size, device=None, temp=1.0, guidance_scale=1.5, seed=None):
        """
        Generate synthetic tabular data samples with control parameters.
        
        Args:
            batch_size: Number of samples to generate
            device: Device to generate samples on (defaults to model's device)
            temp: Temperature for sampling (lower = more conservative)
            guidance_scale: Guidance scale (higher = better quality, less diversity)
            seed: Random seed for reproducibility
            
        Returns:
            samples: Generated tabular data samples
        """
        # Set random seed if provided
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        
        # Use model's device if not specified
        if device is None:
            device = next(self.parameters()).device
        
        # Generate samples with improved process
        return self.diffusion_full_reverse_process(
            batch_size=batch_size, 
            device=device,
            temp=temp,
            guidance_scale=guidance_scale
        )
    
    def fit(self, dataloader, n_epochs=100, lr=1e-4, device='cuda', max_grad_norm=1.0, 
            weight_decay=1e-5, val_dataloader=None, patience=10):
        """
        Train the model with improved training procedures.
        
        Args:
            dataloader: DataLoader containing the training data
            n_epochs: Number of epochs to train for
            lr: Learning rate
            device: Device to train on
            max_grad_norm: Maximum norm for gradient clipping
            weight_decay: L2 regularization factor
            val_dataloader: Optional validation dataloader for early stopping
            patience: Early stopping patience (if val_dataloader is provided)
            
        Returns:
            losses: List of training losses
        """
        # Use AdamW with weight decay for better regularization
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=lr, 
            weight_decay=weight_decay,
            betas=(0.9, 0.999)
        )
        
        # Use OneCycleLR for better convergence
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=lr,
            epochs=n_epochs,
            steps_per_epoch=len(dataloader),
            pct_start=0.3,
            div_factor=25,
            final_div_factor=10000
        )
        
        # Explicitly move model to device
        self.to(device)
        
        # Enable mixed precision training for compatible GPUs
        use_amp = device == 'cuda' and torch.cuda.is_available()
        scaler = torch.cuda.amp.GradScaler() if use_amp else None
        
        # Training tracking variables
        losses = []
        best_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        # Print training configuration
        print(f"Starting training with {n_epochs} epochs on {device}...")
        print(f"Mixed precision: {'Enabled' if use_amp else 'Disabled'}")
        print(f"Model has {sum(p.numel() for p in self.parameters() if p.requires_grad):,} trainable parameters")
        
        # Training loop
        self.train()
        for epoch in range(n_epochs):
            epoch_losses = []
            
            # Training batch loop
            for batch in dataloader:
                # Move batch to device with non-blocking for parallel data transfer
                if isinstance(batch, torch.Tensor):
                    batch = batch.to(device, non_blocking=True)
                else:
                    batch = batch[0].to(device, non_blocking=True)
                
                optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
                
                # Use mixed precision if available
                if use_amp:
                    with torch.cuda.amp.autocast():
                        loss = self(batch)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss = self(batch)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_grad_norm)
                    optimizer.step()
                
                scheduler.step()
                epoch_losses.append(loss.item())
            
            # Calculate training loss
            avg_loss = sum(epoch_losses) / max(len(epoch_losses), 1)
            losses.append(avg_loss)
            
            # Perform validation if dataloader provided
            if val_dataloader is not None:
                self.eval()
                val_losses = []
                
                with torch.no_grad():
                    for batch in val_dataloader:
                        if isinstance(batch, torch.Tensor):
                            batch = batch.to(device, non_blocking=True)
                        else:
                            batch = batch[0].to(device, non_blocking=True)
                        
                        if use_amp:
                            with torch.cuda.amp.autocast():
                                val_loss = self(batch)
                        else:
                            val_loss = self(batch)
                        
                        val_losses.append(val_loss.item())
                
                self.train()
                avg_val_loss = sum(val_losses) / max(len(val_losses), 1)
                
                # Early stopping logic
                if avg_val_loss < best_loss:
                    best_loss = avg_val_loss
                    patience_counter = 0
                    # Save best model state
                    best_model_state = {k: v.detach().cpu() for k, v in self.state_dict().items()}
                    print(f"Epoch {epoch+1}/{n_epochs}, Loss: {avg_loss:.6f}, Val Loss: {avg_val_loss:.6f} (new best) ↓")
                else:
                    patience_counter += 1
                    print(f"Epoch {epoch+1}/{n_epochs}, Loss: {avg_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
                
                # Early stopping check
                if patience_counter >= patience:
                    print(f"Early stopping triggered after {epoch+1} epochs")
                    break
            else:
                print(f"Epoch {epoch+1}/{n_epochs}, Loss: {avg_loss:.6f}, LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # Restore best model if we used validation
        if val_dataloader is not None and best_model_state is not None:
            self.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
            print(f"Restored best model with validation loss: {best_loss:.6f}")
        
        # Ensure model is in eval mode when returning
        self.eval()
        print("Training complete!")
        return losses

class TabularDataset(Dataset):
    """Dataset for tabular data."""
    def __init__(self, data, transform=None):
        """
        Args:
            data: Numpy array or torch.Tensor containing tabular data
            transform: Optional transform to apply to the data
        """
        if isinstance(data, np.ndarray):
            self.data = torch.from_numpy(data).float()
        else:
            self.data = data.float()
        
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample

class TabularPreprocessor:
    """Preprocessor for tabular data with mixed numerical types (INT and FLOAT)."""
    
    def __init__(self, categorical_threshold=10, normalize=True, scaler_type='standard', 
                 handle_outliers=True, financial_data=True):
        """
        Initialize the preprocessor.
        
        Args:
            categorical_threshold: Maximum number of unique values for an integer column
                                 to be treated as categorical (one-hot encoded)
            normalize: Whether to normalize the data
            scaler_type: Type of scaler to use ('standard' or 'minmax' or 'robust')
            handle_outliers: Whether to handle outliers using winsorization
            financial_data: Whether the data is financial time series data
        """
        self.categorical_threshold = categorical_threshold
        self.normalize = normalize
        self.scaler_type = scaler_type
        self.handle_outliers = handle_outliers
        self.financial_data = financial_data
        self.int_columns = None
        self.float_columns = None
        self.scaler = None
        self.int_info = {}  # Store information about integer columns
        self.float_info = {}  # Store information about float columns
        self.target_column = None  # Store target column for special handling
    
    def _detect_column_types(self, df):
        """
        Detect the types of columns in the dataframe.
        
        Args:
            df: Pandas DataFrame
            
        Returns:
            int_cols: List of integer columns
            float_cols: List of float columns
        """
        int_cols = []
        float_cols = []
        
        for col in df.columns:
            if pd.api.types.is_integer_dtype(df[col]):
                # Check if this integer column should be treated as categorical
                n_unique = df[col].nunique()
                if n_unique <= self.categorical_threshold:
                    # This will be one-hot encoded later
                    self.int_info[col] = {
                        'type': 'categorical',
                        'unique_values': sorted(df[col].unique()),
                        'num_unique': n_unique
                    }
                else:
                    # Treat as continuous integer
                    self.int_info[col] = {
                        'type': 'continuous',
                        'min': df[col].min(),
                        'max': df[col].max()
                    }
                int_cols.append(col)
            elif pd.api.types.is_float_dtype(df[col]):
                self.float_info[col] = {
                    'min': df[col].min(),
                    'max': df[col].max()
                }
                float_cols.append(col)
        
        return int_cols, float_cols
    
    def fit(self, df, target_column=None):
        """
        Fit the preprocessor to the data.
        
        Args:
            df: Pandas DataFrame with numerical columns
            target_column: Name of the target column for special handling
            
        Returns:
            self
        """
        # Store target column for special handling
        self.target_column = target_column
        
        # Handle outliers if specified
        if self.handle_outliers:
            df_processed = df.copy()
            
            # Apply winsorization to numerical columns (clip values to percentile range)
            for col in df.select_dtypes(include=['float64', 'int64']).columns:
                # For financial time series, use a wider range to preserve trends
                if self.financial_data:
                    lower_percentile = 0.001
                    upper_percentile = 0.999
                else:
                    lower_percentile = 0.01
                    upper_percentile = 0.99
                    
                lower_bound = df[col].quantile(lower_percentile)
                upper_bound = df[col].quantile(upper_percentile)
                
                # Don't winsorize the target column as aggressively
                if col == target_column:
                    lower_bound = df[col].quantile(0.0001)
                    upper_bound = df[col].quantile(0.9999)
                
                df_processed[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
        else:
            df_processed = df
            
        # Detect column types
        self.int_columns, self.float_columns = self._detect_column_types(df_processed)
        
        # Initialize scaler
        if self.normalize:
            from sklearn.preprocessing import RobustScaler
            
            if self.scaler_type == 'standard':
                self.scaler = StandardScaler()
            elif self.scaler_type == 'robust':
                self.scaler = RobustScaler()
            else:  # minmax
                self.scaler = MinMaxScaler()
            
            # Get all continuous columns
            continuous_cols = self.float_columns + [col for col in self.int_columns 
                                                  if self.int_info[col]['type'] == 'continuous']
            
            # If target column exists and is in continuous_cols, handle it separately
            if target_column and target_column in continuous_cols:
                # For financial target columns, special treatment may be needed
                if self.financial_data:
                    # Store target stats for later use
                    self.target_mean = df_processed[target_column].mean()
                    self.target_std = df_processed[target_column].std()
                
            if continuous_cols:
                self.scaler.fit(df_processed[continuous_cols])
        
        return self
    
    def transform(self, df):
        """
        Transform the data.
        
        Args:
            df: Pandas DataFrame
            
        Returns:
            transformed_data: Numpy array of transformed data
            column_metadata: Dictionary mapping column indices to original columns
        """
        transformed_data = []
        column_metadata = {}
        
        # Process integer columns
        int_data = []
        col_idx = 0
        
        for col in self.int_columns:
            if self.int_info[col]['type'] == 'categorical':
                # One-hot encode categorical integers
                for val in self.int_info[col]['unique_values']:
                    one_hot = (df[col] == val).astype(int).values.reshape(-1, 1)
                    int_data.append(one_hot)
                    column_metadata[col_idx] = {'original_col': col, 'value': val, 'type': 'categorical_int'}
                    col_idx += 1
            else:
                # Continuous integers
                values = df[col].values.reshape(-1, 1)
                int_data.append(values)
                column_metadata[col_idx] = {'original_col': col, 'type': 'continuous_int'}
                col_idx += 1
        
        # Process float columns
        float_data = []
        
        for col in self.float_columns:
            values = df[col].values.reshape(-1, 1)
            float_data.append(values)
            column_metadata[col_idx] = {'original_col': col, 'type': 'float'}
            col_idx += 1
        
        # Combine all data
        if int_data:
            int_data = np.hstack(int_data)
            transformed_data.append(int_data)
        
        if float_data:
            float_data = np.hstack(float_data)
            transformed_data.append(float_data)
        
        # Apply scaling if needed
        if self.normalize and self.scaler:
            continuous_cols = self.float_columns + [col for col in self.int_columns 
                                                  if self.int_info[col]['type'] == 'continuous']
            
            if continuous_cols:
                # Get indices of continuous columns in the transformed data
                continuous_indices = []
                for idx, meta in column_metadata.items():
                    if meta['type'] in ['float', 'continuous_int']:
                        continuous_indices.append(idx)
                
                # Extract and normalize continuous data
                if transformed_data:
                    all_data = np.hstack(transformed_data)
                    all_data[:, continuous_indices] = self.scaler.transform(all_data[:, continuous_indices])
                    return all_data, column_metadata
        
        # If no normalization or no continuous columns
        if transformed_data:
            return np.hstack(transformed_data), column_metadata
        else:
            return np.array([]), column_metadata
    
    def inverse_transform(self, data, column_metadata):
        """
        Convert transformed data back to original format.
        
        Args:
            data: Numpy array of transformed data
            column_metadata: Dictionary mapping column indices to original columns
            
        Returns:
            df: Pandas DataFrame in original format
        """
        # Initialize output dataframe
        result = {}
        
        # Process each column
        for idx, meta in column_metadata.items():
            col_name = meta['original_col']
            col_type = meta['type']
            
            if col_type == 'categorical_int':
                # For one-hot encoded integers
                if col_name not in result:
                    result[col_name] = np.zeros(data.shape[0])
                
                # Set the original value where this one-hot column is 1
                mask = data[:, idx] > 0.5
                result[col_name][mask] = meta['value']
            
            elif col_type in ['continuous_int', 'float']:
                # For continuous data
                result[col_name] = data[:, idx]
        
        # Create dataframe
        df = pd.DataFrame(result)
        
        # Inverse transform normalized data
        if self.normalize and self.scaler:
            continuous_cols = self.float_columns + [col for col in self.int_columns 
                                                  if self.int_info[col]['type'] == 'continuous']
            
            if continuous_cols and all(col in df.columns for col in continuous_cols):
                df[continuous_cols] = self.scaler.inverse_transform(df[continuous_cols])
        
        # Convert integer columns back to integers
        for col in self.int_columns:
            if col in df.columns:
                df[col] = df[col].round().astype(int)
        
        return df
    
    def fit_transform(self, df, target_column=None):
        """
        Fit the preprocessor and transform the data.
        
        Args:
            df: Pandas DataFrame
            target_column: Name of target column for special handling
            
        Returns:
            transformed_data: Numpy array of transformed data
            column_metadata: Dictionary mapping column indices to original columns
        """
        self.fit(df, target_column)
        return self.transform(df)

class TabularEvaluator:
    """Evaluate the quality of synthetic tabular data."""
    
    @staticmethod
    def statistical_similarity(real_data, synthetic_data):
        """
        Calculate statistical similarity metrics between real and synthetic data.
        
        Args:
            real_data: Real tabular data (numpy array)
            synthetic_data: Synthetic tabular data (numpy array)
            
        Returns:
            metrics: Dictionary of statistical similarity metrics
        """
        metrics = {}
        
        # Compare means
        real_means = np.mean(real_data, axis=0)
        synthetic_means = np.mean(synthetic_data, axis=0)
        metrics['mean_difference'] = np.mean(np.abs(real_means - synthetic_means))
        
        # Compare standard deviations
        real_stds = np.std(real_data, axis=0)
        synthetic_stds = np.std(synthetic_data, axis=0)
        metrics['std_difference'] = np.mean(np.abs(real_stds - synthetic_stds))
        
        # Compare correlations
        real_corr = np.corrcoef(real_data, rowvar=False)
        synthetic_corr = np.corrcoef(synthetic_data, rowvar=False)
        metrics['correlation_difference'] = np.mean(np.abs(real_corr - synthetic_corr))
        
        return metrics
    
    @staticmethod
    def privacy_risk(real_data, synthetic_data, k=5):
        """
        Assess privacy risk by checking for nearest neighbors distance.
        
        Args:
            real_data: Real tabular data
            synthetic_data: Synthetic tabular data
            k: Number of nearest neighbors to consider
            
        Returns:
            metrics: Dictionary of privacy risk metrics
        """
        from sklearn.neighbors import NearestNeighbors
        
        metrics = {}
        
        # Fit nearest neighbors on real data
        nn = NearestNeighbors(n_neighbors=k+1)
        nn.fit(real_data)
        
        # Get distances from synthetic points to real points
        distances, _ = nn.kneighbors(synthetic_data)
        
        # Calculate metrics
        metrics['min_distance'] = np.min(distances[:, 1:])  # Skip first NN (itself)
        metrics['avg_distance'] = np.mean(distances[:, 1:])
        metrics['median_distance'] = np.median(distances[:, 1:])
        
        return metrics

def prepare_data_for_tabular_diffflow(df, test_size=0.2, random_state=42, target_column=None, 
                                financial_data=True):
    """
    Prepare data for the TabularDiffFlow model.
    
    Args:
        df: Pandas DataFrame with numerical data (INT and FLOAT types)
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
        target_column: Name of the target column for special handling
        financial_data: Whether the data is financial time series data
        
    Returns:
        train_data: Training data as numpy array
        test_data: Test data as numpy array
        preprocessor: Fitted TabularPreprocessor object
        column_metadata: Metadata about the columns
    """
    print(f"Preparing data for TabularDiffFlow model...")
    
    # For time series data, don't randomize the split
    if financial_data:
        # Use last test_size% of data for testing to maintain temporal order
        split_idx = int(len(df) * (1 - test_size))
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]
        print(f"Using temporal split: {len(train_df)} training samples, {len(test_df)} testing samples")
    else:
        # Split data randomly
        train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
        print(f"Using random split: {len(train_df)} training samples, {len(test_df)} testing samples")
    
    # Create and configure preprocessor
    preprocessor = TabularPreprocessor(
        normalize=True, 
        scaler_type='robust',  # Robust scaling for financial data
        handle_outliers=True,
        financial_data=financial_data,
        categorical_threshold=20  # Higher threshold for financial indicators
    )
    
    # Fit and transform
    train_data, column_metadata = preprocessor.fit_transform(train_df, target_column)
    test_data, _ = preprocessor.transform(test_df)
    
    # Basic sanity checks
    if np.isnan(train_data).any() or np.isinf(train_data).any():
        print("Warning: NaN or Inf values detected in training data. Replacing with zeros.")
        train_data = np.nan_to_num(train_data)
    
    if np.isnan(test_data).any() or np.isinf(test_data).any():
        print("Warning: NaN or Inf values detected in test data. Replacing with zeros.")
        test_data = np.nan_to_num(test_data)
    
    print(f"Data preparation complete. Feature dimensions: {train_data.shape[1]}")
    return train_data, test_data, preprocessor, column_metadata

def generate_synthetic_data(model, n_samples, device=None, preprocessor=None, column_metadata=None, 
                         temp=1.0, guidance_scale=1.5, seed=None, batch_size=512):
    """
    Generate synthetic data using the TabularDiffFlow model and convert back to original format.
    
    Args:
        model: Trained TabularDiffFlow model
        n_samples: Number of samples to generate
        device: Device to use for generation (defaults to model's device)
        preprocessor: Fitted TabularPreprocessor
        column_metadata: Column metadata from preprocessing
        temp: Temperature for sampling (lower = more conservative)
        guidance_scale: Guidance scale (higher = better quality, less diversity)
        seed: Random seed for reproducibility
        batch_size: Batch size for generation (for memory efficiency with large sample counts)
        
    Returns:
        synthetic_df: Pandas DataFrame of synthetic data in original format
    """
    # Set random seed if provided for reproducibility
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    # Use model's device if not specified
    if device is None and isinstance(model, torch.nn.Module):
        device = next(model.parameters()).device
    elif device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Handle unpacked model tuple from train_tabular_diffflow_model
    if isinstance(model, tuple) and len(model) >= 3:
        if preprocessor is None:
            preprocessor = model[1]
        if column_metadata is None:
            column_metadata = model[2]
        model = model[0]
    
    # Ensure model is in eval mode
    if hasattr(model, 'eval'):
        model.eval()
    
    # Check if we need to generate in batches (for memory efficiency)
    if n_samples > batch_size:
        print(f"Generating {n_samples} samples in batches of {batch_size}...")
        raw_synthetic_batches = []
        
        # Generate in batches
        remaining = n_samples
        generated = 0
        
        try:
            from tqdm import tqdm
            iterator = tqdm(total=n_samples, desc="Generating samples")
        except ImportError:
            iterator = range(1)
            print(f"Generating {n_samples} samples...")
        
        while remaining > 0:
            current_batch = min(batch_size, remaining)
            
            # Generate batch
            with torch.no_grad():
                batch = model.sample(
                    batch_size=current_batch, 
                    device=device,
                    temp=temp,
                    guidance_scale=guidance_scale
                ).cpu().numpy()
            
            raw_synthetic_batches.append(batch)
            remaining -= current_batch
            generated += current_batch
            
            if hasattr(iterator, 'update'):
                iterator.update(current_batch)
        
        if hasattr(iterator, 'close'):
            iterator.close()
            
        # Combine batches
        raw_synthetic = np.vstack(raw_synthetic_batches)
    else:
        # Generate all at once
        print(f"Generating {n_samples} samples...")
        with torch.no_grad():
            raw_synthetic = model.sample(
                batch_size=n_samples, 
                device=device,
                temp=temp,
                guidance_scale=guidance_scale
            ).cpu().numpy()
    
    # Convert back to original format
    synthetic_df = preprocessor.inverse_transform(raw_synthetic, column_metadata)
    
    print(f"Generated {len(synthetic_df)} synthetic samples successfully!")
    return synthetic_df

def train_tabular_diffflow_model(df, n_epochs=100, batch_size=64, learning_rate=1e-4, device=None, 
                           max_grad_norm=1.0, weight_decay=1e-5, d_model=256, num_workers=4,
                           target_column=None, financial_data=True):
    """
    Train a TabularDiffFlow model on the provided dataframe.
    Enhanced with financial data optimizations and temporal awareness.
    
    Args:
        df: Pandas DataFrame with numerical data (INT and FLOAT)
        n_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimization
        device: Device to train on ('cuda' or 'cpu')
        max_grad_norm: Maximum gradient norm for clipping
        weight_decay: L2 regularization weight
        d_model: Dimension of the model's hidden layers
        num_workers: Number of workers for data loading
        target_column: Name of the target column for special handling
        financial_data: Whether the data is financial time series data
        
    Returns:
        model: Trained TabularDiffFlow model
        preprocessor: Fitted preprocessor
        column_metadata: Column metadata
        training_losses: List of training losses
    """
    # Auto-detect device if not specified
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Using device: {device}")
    
    # Force CUDA if available - this ensures GPU usage
    if torch.cuda.is_available() and device != 'cuda':
        print("CUDA is available but not selected. Forcing CUDA usage.")
        device = 'cuda'
        
    # Set PyTorch to use deterministic algorithms for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Prepare data with target column information
    train_data, test_data, preprocessor, column_metadata = prepare_data_for_tabular_diffflow(
        df,
        target_column=target_column,
        financial_data=financial_data
    )
    
    # Create dataset and dataloader with multiple workers for faster loading
    dataset = TabularDataset(train_data)
    # Only use multiple workers if on CPU or if CUDA allows it
    actual_workers = num_workers if device == 'cpu' or torch.cuda.device_count() > 0 else 0
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=actual_workers,
        pin_memory=device=='cuda'  # Pin memory for faster GPU transfer
    )
    
    # Create validation dataloader for monitoring
    val_dataset = TabularDataset(test_data)
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=actual_workers,
        pin_memory=device=='cuda'
    )
    
    # Create model
    n_features = train_data.shape[1]
    
    # Get feature ranges
    feature_ranges = []
    for i in range(n_features):
        feature_min = float(train_data[:, i].min())
        feature_max = float(train_data[:, i].max())
        feature_ranges.append((feature_min, feature_max))
    
    # Detect if this is a financial dataset to enable special features
    is_financial = financial_data
    
    # Determine appropriate model size based on feature count
    # Smaller models for smaller datasets
    if n_features < 10:
        model_dim = min(d_model, 128)  # Smaller model for few features
        n_heads = 2
    else:
        model_dim = d_model  # Full size for larger feature sets
        n_heads = 4
        
    # Make model dimension divisible by number of heads
    model_dim = (model_dim // n_heads) * n_heads
    
    # Calculate appropriate dropout based on dataset size
    # More dropout for smaller datasets to prevent overfitting
    train_size = len(train_data)
    if train_size < 1000:
        dropout = 0.2  # Higher dropout for very small datasets
    elif train_size < 10000:
        dropout = 0.1  # Moderate dropout for medium datasets
    else:
        dropout = 0.05  # Lower dropout for large datasets
    
    print(f"Using model with dim={model_dim}, heads={n_heads}, dropout={dropout}")
        
    # Initialize enhanced model with financial data optimizations
    model = TabularDiffFlow(
        n_features=n_features,
        d_model=model_dim,
        n_diffusion_steps=1000,
        n_flow_layers=4,
        sparsity=0.5,
        feature_ranges=feature_ranges,
        financial_data=is_financial,             # Enable financial data optimizations
        trend_preservation=is_financial,         # Enable trend preservation for financial data
        use_temporal_transformer=is_financial,   # Enable temporal transformer for financial data
        n_heads=n_heads,                         # Use multiple attention heads
        dropout=dropout                          # Use appropriate dropout
    )
    
    # Mixed precision for faster training on compatible GPUs
    use_amp = device == 'cuda' and torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    
    # Apply improved weight initialization for better stability and faster convergence
    def weights_init(m):
        if isinstance(m, nn.Linear):
            # Kaiming initialization for ReLU/SiLU activations
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(m.bias, -bound, bound)
                
    model.apply(weights_init)
    
    # Explicitly move model to device
    model = model.to(device)
    
    # Optimizer with weight decay for regularization
    # Higher weight decay for financial data to prevent overfitting
    final_weight_decay = weight_decay * 2 if is_financial else weight_decay
    
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=learning_rate,
        weight_decay=final_weight_decay,
        betas=(0.9, 0.999)
    )
    
    # Learning rate scheduler for better convergence
    # Use cosine annealing for smoother decay
    if n_epochs > 50:
        # For longer training runs, use OneCycle with longer warmup
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=learning_rate,
            epochs=n_epochs,
            steps_per_epoch=len(dataloader),
            pct_start=0.3,  # 30% warmup
            anneal_strategy='cos'
        )
    else:
        # For shorter runs, use simpler cosine annealing
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=n_epochs * len(dataloader),
            eta_min=learning_rate / 10
        )
    
    # Detect market regime information for financial datasets
    market_regime = None
    if is_financial:
        try:
            # Get market regime indicators for training data
            print("Detecting market regime for financial data...")
            market_regime = detect_market_regime(df)
            print(f"Detected regimes: Bull ({(market_regime > 0).mean()*100:.1f}%), Bear ({(market_regime < 0).mean()*100:.1f}%)")
        except Exception as e:
            print(f"Warning: Could not detect market regime: {str(e)}")
    
    # Train with enhanced procedures
    model.train()
    training_losses = []
    best_loss = float('inf')
    patience_counter = 0
    patience = 10  # Early stopping patience
    best_model_state = None
    
    print(f"Starting training with {n_epochs} epochs...")
    print(f"Model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters")
    
    try:
        from tqdm import tqdm
        use_progress_bar = True
    except ImportError:
        use_progress_bar = False
    
    for epoch in range(n_epochs):
        epoch_losses = []
        
        # Create progress bar if tqdm is available
        if use_progress_bar:
            train_iter = tqdm(dataloader, desc=f"Epoch {epoch+1}/{n_epochs}")
        else:
            train_iter = dataloader
        
        # Training loop
        for batch_idx, batch in enumerate(train_iter):
            # Move batch to device
            if isinstance(batch, torch.Tensor):
                batch = batch.to(device, non_blocking=True)
            else:
                batch = batch[0].to(device, non_blocking=True)
            
            # Get market regime for this batch if available
            batch_regime = None
            if is_financial and market_regime is not None:
                # Sample market regime values based on batch index
                # This is a simplified approach - ideally would match samples with their regimes
                indices = torch.randint(0, len(market_regime), (batch.size(0),))
                batch_regime = torch.tensor([market_regime[i] for i in indices], device=device)
            
            optimizer.zero_grad(set_to_none=True)  # Faster than zero_grad()
            
            # Use mixed precision if available
            if use_amp:
                with torch.cuda.amp.autocast():
                    loss = model(batch, market_regime=batch_regime)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss = model(batch, market_regime=batch_regime)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
            
            scheduler.step()
            epoch_losses.append(loss.item())
            
            # Update progress bar if available
            if use_progress_bar:
                train_iter.set_postfix(loss=f"{loss.item():.4f}", lr=f"{scheduler.get_last_lr()[0]:.6f}")
        
        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_dataloader:
                if isinstance(batch, torch.Tensor):
                    batch = batch.to(device, non_blocking=True)
                else:
                    batch = batch[0].to(device, non_blocking=True)
                
                # Use same market regime approach for validation
                batch_regime = None
                if is_financial and market_regime is not None:
                    indices = torch.randint(0, len(market_regime), (batch.size(0),))
                    batch_regime = torch.tensor([market_regime[i] for i in indices], device=device)
                
                if use_amp:
                    with torch.cuda.amp.autocast():
                        val_loss = model(batch, market_regime=batch_regime)
                else:
                    val_loss = model(batch, market_regime=batch_regime)
                
                val_losses.append(val_loss.item())
        
        model.train()
        
        # Calculate average losses
        avg_train_loss = sum(epoch_losses) / len(epoch_losses)
        avg_val_loss = sum(val_losses) / len(val_losses)
        training_losses.append(avg_train_loss)
        
        # Early stopping logic
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            patience_counter = 0
            
            # Save best model state
            best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
            print(f"Epoch {epoch+1}/{n_epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f} (new best) ↓")
        else:
            patience_counter += 1
            print(f"Epoch {epoch+1}/{n_epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, LR: {scheduler.get_last_lr()[0] if isinstance(scheduler.get_last_lr(), list) else scheduler.get_last_lr():.6f}")
        
        # Early stopping check
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # Restore best model
    if best_model_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
        print(f"Restored best model with validation loss: {best_loss:.6f}")
    
    # Ensure model is in eval mode when returning
    model.eval()
    print("Training complete!")
    
    return model, preprocessor, column_metadata, training_losses

def evaluate_synthetic_data(real_df, synthetic_df, task_type='classification', target_column=None, 
                           classifier=None, regression_model=None):
    """
    Evaluate the quality of synthetic data by training models on it and testing on real data.
    
    Args:
        real_df: Real data as pandas DataFrame
        synthetic_df: Synthetic data as pandas DataFrame
        task_type: Type of prediction task ('classification' or 'regression')
        target_column: Column to predict
        classifier: Classifier to use (defaults to RandomForest)
        regression_model: Regression model to use (defaults to RandomForest)
        
    Returns:
        metrics: Dictionary of evaluation metrics
    """
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error, r2_score
    
    if target_column is None:
        raise ValueError("Target column must be specified")
    
    # Default models if not provided
    if task_type == 'classification' and classifier is None:
        classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    elif task_type == 'regression' and regression_model is None:
        regression_model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    # Split data
    X_real = real_df.drop(columns=[target_column])
    y_real = real_df[target_column]
    
    X_syn = synthetic_df.drop(columns=[target_column])
    y_syn = synthetic_df[target_column]
    
    # Train on synthetic, test on real
    if task_type == 'classification':
        model = classifier
        model.fit(X_syn, y_syn)
        y_pred = model.predict(X_real)
        y_prob = model.predict_proba(X_real) if hasattr(model, 'predict_proba') else None
        
        metrics = {
            'accuracy': accuracy_score(y_real, y_pred)
        }
        
        # Add ROC AUC if available
        if y_prob is not None and len(np.unique(y_real)) == 2:
            metrics['roc_auc'] = roc_auc_score(y_real, y_prob[:, 1])
    
    else:  # regression
        model = regression_model
        model.fit(X_syn, y_syn)
        y_pred = model.predict(X_real)
        
        metrics = {
            'mse': mean_squared_error(y_real, y_pred),
            'r2': r2_score(y_real, y_pred)
        }
    
    # Also evaluate statistical similarity
    stat_metrics = TabularEvaluator.statistical_similarity(
        real_df.drop(columns=[target_column]).values,
        synthetic_df.drop(columns=[target_column]).values
    )
    
    # Combine metrics
    metrics.update(stat_metrics)
    
    return metrics

# Example usage
def example_usage():
    """Example usage of the TabularDiffFlow model for synthetic data generation."""
    # Create synthetic data
    np.random.seed(42)
    
    # Simulate a tabular dataset with mixed numerical types
    n_samples = 1000
    n_features = 10
    
    # Generate continuous features
    continuous_features = np.random.normal(0, 1, size=(n_samples, n_features // 2))
    
    # Generate integer features
    int_features = np.random.randint(0, 10, size=(n_samples, n_features // 2))
    
    # Combine features
    features = np.hstack([continuous_features, int_features])
    
    # Generate target variable (classification)
    w = np.random.normal(0, 1, size=n_features)
    logits = np.dot(features, w)
    probs = 1 / (1 + np.exp(-logits))
    target = (probs > 0.5).astype(int)
    
    # Create DataFrame
    columns = []
    for i in range(n_features // 2):
        columns.append(f'float_{i}')
    for i in range(n_features // 2):
        columns.append(f'int_{i}')
    columns.append('target')
    
    data = np.hstack([features, target.reshape(-1, 1)])
    df = pd.DataFrame(data, columns=columns)
    
    # Convert integer columns to int type
    for i in range(n_features // 2):
        df[f'int_{i}'] = df[f'int_{i}'].astype(int)
    
    print("Original data:")
    print(df.head())
    print(df.dtypes)
    
    # Train model
    model, preprocessor, column_metadata, _ = train_tabular_diffflow_model(
        df, 
        n_epochs=10,  # Small value for example
        batch_size=32
    )
    
    # Generate synthetic data
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    synthetic_df = generate_synthetic_data(model, 1000, device, preprocessor, column_metadata)
    
    print("\nSynthetic data:")
    print(synthetic_df.head())
    print(synthetic_df.dtypes)
    
    # Evaluate
    target_column = 'target'
    metrics = evaluate_synthetic_data(df, synthetic_df, task_type='classification', target_column=target_column)
    
    print("\nEvaluation metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v}")
        
"""
Research on SOTA Tabular Data Generation Models and Their Limitations

The TabularDiffFlow model presented above addresses several key limitations in
existing approaches to tabular data generation:

1. CTGAN (Conditional Tabular GAN) and TVAE:
   - Key idea: Uses conditional GANs with mode-specific normalization for tabular data.
   - Limitations: 
     * Difficulty in capturing complex dependencies between features
     * Mode collapse is common, especially with imbalanced categorical variables
     * Training instability requiring careful hyperparameter tuning
     * Limited capability to model mixed-type data correctly

2. CTAB-GAN+ (Conditional Tabular GAN+):
   - Key idea: Enhances CTGAN with better handling of mixed-type data and classifications.
   - Limitations:
     * Still struggles with high-dimensional data
     * Limited ability to capture non-linear correlations between features
     * Quality degrades with increasing feature complexity

3. Tabular Variational Autoencoders:
   - Key idea: Uses VAEs to learn a meaningful latent space for tabular data.
   - Limitations:
     * Struggles with preserving statistical properties between features
     * Often produces blurry/averaging effects in numerical features
     * Difficulty capturing multimodal distributions

4. GOGGLE (Generation of Gaussian Graphical Latent Embeddings):
   - Key idea: Uses graph structures to model feature dependencies.
   - Limitations:
     * Reduced performance when feature relationships are non-linear
     * Computational complexity scales poorly with feature dimensionality
     * Struggles with capturing complex mixed-type correlations

5. CopulaGAN:
   - Key idea: Combines copulas with GANs to better capture statistical dependencies.
   - Limitations:
     * Complex implementation that requires domain expertise
     * Inability to scale efficiently to large datasets
     * Limited ability to handle mixed discrete-continuous data

6. TabDDPM (Tabular Denoising Diffusion Probabilistic Models):
   - Key idea: Applies diffusion models to tabular data generation.
   - Limitations:
     * Slow sampling process due to iterative nature of diffusion
     * Less effective for datasets with strong hierarchical dependencies
     * Difficulty in dealing with categorical variables effectively

Key Innovations in TabularDiffFlow:

1. Hybrid Approach: Combines the strengths of diffusion models (for global structure)
   with normalizing flows (for fine details), addressing the limitations of each approach.

2. Sparse Feature Attention: Introduces a mechanism to focus on statistically dependent
   features, solving the problem of capturing complex inter-feature relationships.

3. Adaptive Noise Schedule: Learning the optimal noise schedule rather than using a fixed
   schedule, improving training stability and generation quality.

4. Feature-wise Processing: Treats each feature type appropriately, allowing better
   handling of mixed INT and FLOAT data.

5. Correlation Prior: Incorporates a learnable correlation structure between features to
   preserve relationships observed in the original data.

6. Balanced Loss Function: Combines diffusion loss with flow loss to ensure both global
   structure and local details are preserved.

Benefits Over Existing Methods:

1. Better statistical fidelity: Preserves complex statistical relationships between features.
2. Improved handling of mixed data types: Specifically designed for both INT and FLOAT.
3. Enhanced privacy guarantees: The diffusion process helps obfuscate individual samples.
4. Stable training: The hybrid approach reduces mode collapse problems common in GANs.
5. Flexible architecture: Can be adapted to datasets with varying characteristics.

By addressing these limitations, TabularDiffFlow represents a significant advancement 
in synthetic tabular data generation, especially for mixed numerical data types.
"""

def evaluate_financial_fidelity(real_data, synthetic_data):
    """
    Evaluate financial-specific metrics for synthetic data.
    
    Args:
        real_data: DataFrame with real financial data
        synthetic_data: DataFrame with synthetic financial data
        
    Returns:
        metrics: Dictionary of financial-specific metrics
    """
    from scipy import stats
    import pandas as pd
    import numpy as np
    
    metrics = {}
    
    # Try to identify price columns - they often contain 'price', 'close', 'open' in name
    price_cols = []
    for col in real_data.columns:
        col_lower = col.lower()
        if any(term in col_lower for term in ['close', 'price', 'open', 'high', 'low']):
            price_cols.append(col)
    
    # If no price columns found, use the first numeric column
    if not price_cols:
        for col in real_data.columns:
            if pd.api.types.is_numeric_dtype(real_data[col]):
                price_cols = [col]
                break
    
    # Calculate metrics for each identified price column
    for col in price_cols:
        if col not in real_data.columns or col not in synthetic_data.columns:
            continue
            
        # Get price series
        real_prices = real_data[col].dropna()
        synth_prices = synthetic_data[col].dropna()
        
        # Skip if insufficient data
        if len(real_prices) < 3 or len(synth_prices) < 3:
            continue
        
        # 1. Calculate returns
        real_returns = real_prices.pct_change().dropna()
        synth_returns = synth_prices.pct_change().dropna()
        
        # Guard against empty series after pct_change
        if len(real_returns) < 2 or len(synth_returns) < 2:
            continue
        
        col_metrics = {}
        
        # 2. Basic statistics comparison
        # Volatility comparison
        real_vol = real_returns.std()
        synth_vol = synth_returns.std()
        col_metrics['volatility_ratio'] = synth_vol / real_vol if real_vol > 0 else float('inf')
        
        # Skewness comparison
        real_skew = real_returns.skew()
        synth_skew = synth_returns.skew()
        col_metrics['skewness_diff'] = abs(real_skew - synth_skew)
        
        # Kurtosis comparison (tail heaviness)
        real_kurt = real_returns.kurtosis()
        synth_kurt = synth_returns.kurtosis()
        col_metrics['kurtosis_diff'] = abs(real_kurt - synth_kurt)
        
        # 3. Risk metrics
        # Value-at-Risk comparison
        try:
            real_var_95 = np.percentile(real_returns, 5)
            synth_var_95 = np.percentile(synth_returns, 5)
            col_metrics['var_95_diff'] = abs(real_var_95 - synth_var_95)
        except Exception:
            col_metrics['var_95_diff'] = float('nan')
        
        # 4. Time-series properties
        # Autocorrelation in returns
        try:
            real_autocorr = real_returns.autocorr(lag=1)
            synth_autocorr = synth_returns.autocorr(lag=1)
            col_metrics['autocorr_diff'] = abs(real_autocorr - synth_autocorr)
        except Exception:
            col_metrics['autocorr_diff'] = float('nan')
        
        # Ljung-Box test for autocorrelation
        try:
            real_lb = stats.acorr_ljungbox(real_returns, lags=[10])[1][0]
            synth_lb = stats.acorr_ljungbox(synth_returns, lags=[10])[1][0]
            col_metrics['ljung_box_p_diff'] = abs(real_lb - synth_lb)
        except Exception:
            col_metrics['ljung_box_p_diff'] = float('nan')
        
        # 5. Volatility clustering (ARCH effects)
        # Squared returns autocorrelation (volatility persistence)
        try:
            real_returns_sq = real_returns**2
            synth_returns_sq = synth_returns**2
            real_arch = real_returns_sq.autocorr(lag=1)
            synth_arch = synth_returns_sq.autocorr(lag=1)
            col_metrics['arch_effect_diff'] = abs(real_arch - synth_arch)
        except Exception:
            col_metrics['arch_effect_diff'] = float('nan')
            
        # 6. Trend and momentum
        # Average consecutive price moves in same direction
        try:
            real_consec = (np.sign(real_returns) == np.sign(real_returns.shift(1))).mean()
            synth_consec = (np.sign(synth_returns) == np.sign(synth_returns.shift(1))).mean()
            col_metrics['trend_persistence_diff'] = abs(real_consec - synth_consec)
        except Exception:
            col_metrics['trend_persistence_diff'] = float('nan')
        
        # Store metrics for this column
        metrics[col] = col_metrics
    
    # Calculate average metrics across all price columns
    avg_metrics = {}
    for metric in ['volatility_ratio', 'skewness_diff', 'kurtosis_diff', 'var_95_diff', 
                  'autocorr_diff', 'ljung_box_p_diff', 'arch_effect_diff', 'trend_persistence_diff']:
        values = [metrics[col][metric] for col in metrics if metric in metrics[col] 
                 and not np.isnan(metrics[col][metric]) 
                 and not np.isinf(metrics[col][metric])]
        if values:
            avg_metrics[f'avg_{metric}'] = np.mean(values)
    
    # Add column-specific metrics to the overall results
    metrics.update(avg_metrics)
    
    return metrics


def detect_market_regime(data, window=20):
    """
    Detect market regime (bull/bear) based on price trends.
    
    Args:
        data: DataFrame with price data
        window: Window size for moving averages
        
    Returns:
        Array of regime indicators: 1 for bull, -1 for bear
    """
    import numpy as np
    import pandas as pd
    
    # Identify price column
    price_col = None
    for col in data.columns:
        col_lower = col.lower()
        if 'close' in col_lower:
            price_col = col
            break
            
    if price_col is None:
        # No close column, try to find any price column
        for col in data.columns:
            col_lower = col.lower()
            if any(term in col_lower for term in ['price', 'open', 'high', 'low']):
                price_col = col
                break
                
    if price_col is None:
        # Still no price column, use the first numeric column
        for col in data.columns:
            if pd.api.types.is_numeric_dtype(data[col]):
                price_col = col
                break
    
    if price_col is None:
        # No suitable column found
        return np.zeros(len(data))
    
    # Calculate short and long moving averages
    try:
        short_ma = data[price_col].rolling(window=window).mean()
        long_ma = data[price_col].rolling(window=window*2).mean()
        
        # Bull market when short MA > long MA
        regime = np.where(short_ma > long_ma, 1, -1)
        
        # Fill NaN values at the beginning
        regime = np.nan_to_num(regime, nan=0)
        
        return regime
    except Exception as e:
        print(f"Error detecting market regime: {str(e)}")
        return np.zeros(len(data))


if __name__ == "__main__":
    example_usage()