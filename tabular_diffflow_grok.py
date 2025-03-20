import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split


# Noise Schedule Class
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


# Sparse Feature Attention Mechanism
class SparseFeatureAttention(nn.Module):
    def __init__(self, n_features, d_model, sparsity=0.5, heads=4, dropout=0.1):
        super().__init__()
        self.n_features = n_features
        self.d_model = d_model
        self.sparsity = sparsity
        self.heads = heads
        self.head_dim = d_model // heads
        assert self.head_dim * heads == d_model, "d_model must be divisible by heads"

        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x):
        batch_size, n_features, d_model = x.shape
        x = x.view(batch_size, n_features, self.heads, self.head_dim)
        return x.permute(0, 2, 1, 3)

    def _merge_heads(self, x):
        batch_size, heads, n_features, head_dim = x.shape
        x = x.permute(0, 2, 1, 3)
        return x.reshape(batch_size, n_features, self.d_model)

    def forward(self, x):
        batch_size, n_features, _ = x.shape

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        k_value = max(1, int(n_features * self.sparsity))
        for h in range(self.heads):
            topk_values, _ = torch.topk(scores[:, h], k_value, dim=-1)
            threshold = topk_values[:, :, -1].unsqueeze(-1)
            mask = scores[:, h] < threshold
            scores[:, h] = scores[:, h].masked_fill(mask, -1e9)

        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)

        output = torch.matmul(attention, v)
        output = self._merge_heads(output)
        output = self.out_proj(output)

        return output


# Coupling Layer for Normalizing Flow
class CouplingLayer(nn.Module):
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

        for m in self.scale_net.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.zero_()
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        x1, x2 = x[:, :self.in_dim], x[:, self.in_dim:]
        s = torch.tanh(self.scale_net(x1))
        t = self.translate_net(x1)
        y2 = x2 * torch.exp(s) + t
        y = torch.cat([x1, y2], dim=1)
        log_det = s.sum(dim=1)
        return y, log_det

    def inverse(self, y):
        y1, y2 = y[:, :self.in_dim], y[:, self.in_dim:]
        s = torch.tanh(self.scale_net(y1))
        t = self.translate_net(y1)
        x2 = (y2 - t) * torch.exp(-s)
        x = torch.cat([y1, x2], dim=1)
        return x


# Normalizing Flow Model
class NormalizingFlow(nn.Module):
    def __init__(self, n_features, hidden_dim, n_layers=3):
        super().__init__()
        self.n_features = n_features
        self.n_layers = n_layers
        self.split_idx = n_features // 2
        self.coupling_layers = nn.ModuleList([
            CouplingLayer(self.split_idx, n_features - self.split_idx, hidden_dim)
            for _ in range(n_layers)
        ])

    def forward(self, x, reverse=False):
        log_det = torch.zeros(x.shape[0], device=x.device)
        if not reverse:
            for layer in self.coupling_layers:
                x, ld = layer(x)
                log_det += ld
            return x, log_det
        else:
            for layer in reversed(self.coupling_layers):
                x = layer.inverse(x)
            return x


# Main TabularDiffFlow Model
class TabularDiffFlow(nn.Module):
    def __init__(
            self,
            n_features,
            d_model=128,
            n_diffusion_steps=1000,
            n_flow_layers=3,
            sparsity=0.5,
            feature_ranges=None,
            use_temporal_transformer=False,
            n_heads=4,
            dropout=0.1
    ):
        super().__init__()
        self.n_features = n_features
        self.d_model = d_model
        self.n_diffusion_steps = n_diffusion_steps

        if feature_ranges is None:
            self.register_buffer('feature_min', torch.zeros(n_features))
            self.register_buffer('feature_max', torch.ones(n_features))
        else:
            self.register_buffer('feature_min', torch.tensor([x[0] for x in feature_ranges], dtype=torch.float32))
            self.register_buffer('feature_max', torch.tensor([x[1] for x in feature_ranges], dtype=torch.float32))

        self.noise_schedule = StandardNoiseSchedule(n_diffusion_steps)

        self.feature_attention = SparseFeatureAttention(
            n_features,
            d_model,
            sparsity=sparsity,
            heads=n_heads,
            dropout=dropout
        )

        self.feature_embedding = nn.Sequential(
            nn.Linear(n_features, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model)
        )

        self.time_embedding = nn.Sequential(
            nn.Linear(1, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model)
        )

        if use_temporal_transformer:
            from torch.nn import TransformerEncoder, TransformerEncoderLayer
            encoder_layers = TransformerEncoderLayer(d_model, n_heads, d_model * 2, dropout)
            self.temporal_transformer = TransformerEncoder(encoder_layers, num_layers=1)
        else:
            self.temporal_transformer = None

        def create_feature_processor():
            return nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.SiLU(),
                nn.Linear(d_model, d_model),
                nn.SiLU(),
                nn.Linear(d_model, d_model)
            )

        if n_features <= 32:
            self.feature_processor = nn.ModuleList([create_feature_processor() for _ in range(n_features)])
            self.shared_processing = False
        else:
            self.feature_processor = create_feature_processor()
            self.feature_id_embedding = nn.Embedding(n_features, d_model // 4)
            self.shared_processing = True

        self.global_processor = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.SiLU(),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model)
        )

        self.denoise_net = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, n_features * 2)
        )

        self.flow = NormalizingFlow(n_features, d_model, n_layers=n_flow_layers)

    def _process_embeddings(self, x_t, t):
        batch_size = x_t.shape[0]

        t_emb = self.time_embedding(t.view(-1, 1))
        x_emb = self.feature_embedding(x_t)

        x_emb = x_emb.unsqueeze(1).expand(-1, self.n_features, -1)
        t_emb = t_emb.unsqueeze(1).expand(-1, self.n_features, -1)

        combined = torch.cat([x_emb, t_emb], dim=2)

        if self.shared_processing:
            feature_ids = torch.arange(self.n_features, device=x_t.device)
            feature_id_embs = self.feature_id_embedding(feature_ids)
            batch_dim = combined.shape[0]
            combined_flat = combined.reshape(-1, combined.shape[-1])
            processed_flat = self.feature_processor(combined_flat)
            processed = processed_flat.reshape(batch_dim, self.n_features, -1)
            feature_id_embs = feature_id_embs.unsqueeze(0).expand(batch_dim, -1, -1)
            feature_id_padding = torch.zeros(batch_dim, self.n_features,
                                             self.d_model - feature_id_embs.shape[-1],
                                             device=x_t.device)
            feature_id_embs_padded = torch.cat([feature_id_embs, feature_id_padding], dim=2)
            feature_outputs = processed + 0.1 * feature_id_embs_padded
        else:
            feature_outputs = [processor(combined[:, i]) for i, processor in enumerate(self.feature_processor)]
            feature_outputs = torch.stack(feature_outputs, dim=1)

        attended = self.feature_attention(feature_outputs)

        if self.temporal_transformer:
            transformed = attended.permute(1, 0, 2)
            transformed = self.temporal_transformer(transformed)
            attended = transformed.permute(1, 0, 2)

        global_repr = attended.mean(dim=1)
        global_processed = self.global_processor(global_repr)

        return global_processed

    def normalize_features(self, x):
        return (x - self.feature_min) / (self.feature_max - self.feature_min + 1e-6)

    def denormalize_features(self, x):
        return x * (self.feature_max - self.feature_min + 1e-6) + self.feature_min

    def diffusion_forward_process(self, x_0, t):
        t_idx = (t * (self.n_diffusion_steps - 1)).long()
        alpha_hat = self.noise_schedule.get_alpha_hats()[t_idx].view(-1, 1)
        epsilon = torch.randn_like(x_0)
        x_t = torch.sqrt(alpha_hat) * x_0 + torch.sqrt(1 - alpha_hat) * epsilon
        return x_t, epsilon

    def predict_noise(self, x_t, t):
        global_processed = self._process_embeddings(x_t, t)
        raw_outputs = self.denoise_net(global_processed)
        mean_raw, _ = raw_outputs.chunk(2, dim=1)
        mean = torch.tanh(mean_raw)
        return mean

    def diffusion_reverse_process(self, x_t, t):
        global_processed = self._process_embeddings(x_t, t)
        raw_outputs = self.denoise_net(global_processed)
        mean_raw, log_var_raw = raw_outputs.chunk(2, dim=1)
        mean = torch.tanh(mean_raw)
        log_var = F.softplus(log_var_raw) - 5.0
        var = torch.exp(log_var)

        t_idx = (t * (self.n_diffusion_steps - 1)).long()
        beta = self.noise_schedule.get_betas()[t_idx].view(-1, 1)
        alpha = self.noise_schedule.get_alphas()[t_idx].view(-1, 1)
        alpha_hat = self.noise_schedule.get_alpha_hats()[t_idx].view(-1, 1)

        posterior_mean_coef1 = (1.0 / torch.sqrt(alpha) + 1e-8)
        posterior_mean_coef2 = (beta / torch.sqrt(1.0 - alpha_hat + 1e-8))
        posterior_mean = posterior_mean_coef1 * (x_t - posterior_mean_coef2 * mean)

        posterior_var = beta * (1.0 - alpha_hat / (alpha + 1e-8)) / (1.0 - alpha_hat + 1e-8)
        posterior_var = torch.clamp(posterior_var, 1e-8, 0.5)

        noise_scale = 1.0 if t[0] > 0 else 0
        noise = torch.randn_like(x_t) * noise_scale
        x_pred = posterior_mean + torch.sqrt(posterior_var) * noise

        return x_pred

    def diffusion_full_reverse_process(self, batch_size, device, temp=1.0, guidance_scale=1.5):
        self.eval()
        x_T = torch.randn(batch_size, self.n_features, device=device) * temp
        num_inference_steps = min(self.n_diffusion_steps, 100)
        timesteps = torch.linspace(1, 0, num_inference_steps, device=device) ** 2
        x_t = x_T
        with torch.no_grad():
            for t in timesteps:
                t_batch = t.expand(batch_size)
                noise_pred = self.predict_noise(x_t, t_batch)
                if guidance_scale > 1.0:
                    random_t = torch.rand_like(t_batch)
                    uncond_noise_pred = self.predict_noise(x_t, random_t)
                    noise_pred = uncond_noise_pred + guidance_scale * (noise_pred - uncond_noise_pred)
                x_t = self.diffusion_reverse_process(x_t, t_batch)
        try:
            x_0 = self.flow(x_t, reverse=True)
        except Exception as e:
            print(f"Warning: Flow application failed ({str(e)}). Using diffusion output directly.")
            x_0 = x_t
        x_0 = torch.clamp(x_0, -3, 3)
        x_0 = self.denormalize_features(x_0)
        return x_0

    def forward(self, x, t=None):
        batch_size = x.shape[0]
        device = x.device
        x_normalized = self.normalize_features(x)
        if torch.isnan(x_normalized).any() or torch.isinf(x_normalized).any():
            x_normalized = torch.nan_to_num(x_normalized, nan=0.0, posinf=1.0, neginf=-1.0)
        x_normalized = torch.clamp(x_normalized, -10.0, 10.0)
        try:
            z, log_det = self.flow(x_normalized)
            log_det = torch.clamp(log_det, -50.0, 50.0)
        except Exception as e:
            z = x_normalized
            log_det = torch.zeros(batch_size, device=device)
            print(f"Warning: Flow failed ({str(e)}). Using normalized input.")
        if t is None:
            t = torch.sqrt(torch.rand(batch_size, device=device))
        z_t, epsilon = self.diffusion_forward_process(z, t)
        epsilon_pred = self.predict_noise(z_t, t)
        diffusion_loss = F.smooth_l1_loss(epsilon_pred, epsilon, beta=0.1)
        flow_loss = -log_det.mean()
        if torch.isnan(flow_loss) or torch.isinf(flow_loss):
            flow_loss = torch.tensor(0.0, device=device)
        lambda_diffusion = 1.0
        lambda_flow = 0.1 if diffusion_loss.item() >= 0.1 else 0.05
        loss = lambda_diffusion * diffusion_loss + lambda_flow * flow_loss
        if torch.isnan(loss) or torch.isinf(loss):
            loss = diffusion_loss
        return loss


# Dataset Class
class TabularDataset(Dataset):
    def __init__(self, data, transform=None):
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


# Data Preprocessing Class
class TabularPreprocessor:
    def __init__(self, categorical_threshold=10, normalize=True, scaler_type='standard',
                 handle_outliers=True, financial_data=False):
        self.categorical_threshold = categorical_threshold
        self.normalize = normalize
        self.scaler_type = scaler_type
        self.handle_outliers = handle_outliers
        self.financial_data = financial_data
        self.int_columns = None
        self.float_columns = None
        self.scaler = None
        self.int_info = {}
        self.float_info = {}
        self.target_column = None

    def _detect_column_types(self, df):
        int_cols = []
        float_cols = []
        for col in df.columns:
            if pd.api.types.is_integer_dtype(df[col]):
                n_unique = df[col].nunique()
                if n_unique <= self.categorical_threshold:
                    self.int_info[col] = {
                        'type': 'categorical',
                        'unique_values': sorted(df[col].unique()),
                        'num_unique': n_unique
                    }
                else:
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
        self.target_column = target_column
        if self.handle_outliers:
            df_processed = df.copy()
            for col in df.select_dtypes(include=['float64', 'int64']).columns:
                if self.financial_data:
                    lower_percentile = 0.001
                    upper_percentile = 0.999
                else:
                    lower_percentile = 0.01
                    upper_percentile = 0.99
                lower_bound = df[col].quantile(lower_percentile)
                upper_bound = df[col].quantile(upper_percentile)
                if col == target_column:
                    lower_bound = df[col].quantile(0.0001)
                    upper_bound = df[col].quantile(0.9999)
                df_processed[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
        else:
            df_processed = df
        self.int_columns, self.float_columns = self._detect_column_types(df_processed)
        if self.normalize:
            continuous_cols = self.float_columns + [col for col in self.int_columns
                                                    if self.int_info[col]['type'] == 'continuous']
            if continuous_cols:
                if self.scaler_type == 'standard':
                    self.scaler = StandardScaler()
                elif self.scaler_type == 'robust':
                    self.scaler = RobustScaler()
                else:
                    self.scaler = MinMaxScaler()
                self.scaler.fit(df_processed[continuous_cols])
        return self

    def transform(self, df):
        transformed_data = []
        column_metadata = {}
        int_data = []
        col_idx = 0
        for col in self.int_columns:
            if self.int_info[col]['type'] == 'categorical':
                for val in self.int_info[col]['unique_values']:
                    one_hot = (df[col] == val).astype(int).values.reshape(-1, 1)
                    int_data.append(one_hot)
                    column_metadata[col_idx] = {'original_col': col, 'value': val, 'type': 'categorical_int'}
                    col_idx += 1
            else:
                values = df[col].values.reshape(-1, 1)
                int_data.append(values)
                column_metadata[col_idx] = {'original_col': col, 'type': 'continuous_int'}
                col_idx += 1
        float_data = []
        for col in self.float_columns:
            values = df[col].values.reshape(-1, 1)
            float_data.append(values)
            column_metadata[col_idx] = {'original_col': col, 'type': 'float'}
            col_idx += 1
        if int_data:
            int_data = np.hstack(int_data)
            transformed_data.append(int_data)
        if float_data:
            float_data = np.hstack(float_data)
            transformed_data.append(float_data)
        if self.normalize and self.scaler:
            continuous_cols = self.float_columns + [col for col in self.int_columns
                                                    if self.int_info[col]['type'] == 'continuous']
            if continuous_cols:
                continuous_indices = [idx for idx, meta in column_metadata.items()
                                      if meta['type'] in ['float', 'continuous_int']]
                if transformed_data:
                    all_data = np.hstack(transformed_data)
                    all_data[:, continuous_indices] = self.scaler.transform(all_data[:, continuous_indices])
                    return all_data, column_metadata
        if transformed_data:
            return np.hstack(transformed_data), column_metadata
        else:
            return np.array([]), column_metadata

    def inverse_transform(self, data, column_metadata):
        result = {}
        for idx, meta in column_metadata.items():
            col_name = meta['original_col']
            col_type = meta['type']
            if col_type == 'categorical_int':
                if col_name not in result:
                    result[col_name] = np.zeros(data.shape[0])
                mask = data[:, idx] > 0.5
                result[col_name][mask] = meta['value']
            elif col_type in ['continuous_int', 'float']:
                result[col_name] = data[:, idx]
        df = pd.DataFrame(result)
        if self.normalize and self.scaler:
            continuous_cols = self.float_columns + [col for col in self.int_columns
                                                    if self.int_info[col]['type'] == 'continuous']
            if continuous_cols and all(col in df.columns for col in continuous_cols):
                df[continuous_cols] = self.scaler.inverse_transform(df[continuous_cols])
        for col in self.int_columns:
            if col in df.columns:
                df[col] = df[col].round().astype(int)
        return df

    def fit_transform(self, df, target_column=None):
        self.fit(df, target_column)
        return self.transform(df)


# Data Preparation Function
def prepare_data_for_tabular_diffflow(df, test_size=0.2, random_state=42, target_column=None, financial_data=False):
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
    preprocessor = TabularPreprocessor(normalize=True, scaler_type='robust', handle_outliers=True,
                                       financial_data=financial_data)
    train_data, column_metadata = preprocessor.fit_transform(train_df, target_column)
    test_data, _ = preprocessor.transform(test_df)
    return train_data, test_data, preprocessor, column_metadata


# Training Function
def train_tabular_diffflow_model(df, n_epochs=100, batch_size=64, learning_rate=1e-4, device=None,
                                 max_grad_norm=1.0, weight_decay=1e-5, d_model=256, num_workers=4,
                                 target_column=None, financial_data=False):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    train_data, test_data, preprocessor, column_metadata = prepare_data_for_tabular_diffflow(
        df,
        target_column=target_column,
        financial_data=financial_data
    )
    dataset = TabularDataset(train_data)
    actual_workers = num_workers if device == 'cpu' or torch.cuda.device_count() > 0 else 0
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=actual_workers,
        pin_memory=device == 'cuda'
    )
    val_dataset = TabularDataset(test_data)
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=actual_workers,
        pin_memory=device == 'cuda'
    )
    n_features = train_data.shape[1]
    feature_ranges = [(float(train_data[:, i].min()), float(train_data[:, i].max())) for i in range(n_features)]
    model = TabularDiffFlow(
        n_features=n_features,
        d_model=d_model,
        n_diffusion_steps=1000,
        n_flow_layers=4,
        sparsity=0.5,
        feature_ranges=feature_ranges,
        use_temporal_transformer=financial_data,
        n_heads=4,
        dropout=0.1
    )
    use_amp = device == 'cuda' and torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.999)
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        epochs=n_epochs,
        steps_per_epoch=len(dataloader),
        pct_start=0.3,
        anneal_strategy='cos'
    )
    training_losses = []
    best_loss = float('inf')
    patience_counter = 0
    patience = 10
    best_model_state = None
    for epoch in range(n_epochs):
        epoch_losses = []
        for batch in dataloader:
            batch = batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            if use_amp:
                with torch.cuda.amp.autocast():
                    loss = model(batch)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss = model(batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
            scheduler.step()
            epoch_losses.append(loss.item())
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_dataloader:
                batch = batch.to(device, non_blocking=True)
                if use_amp:
                    with torch.cuda.amp.autocast():
                        val_loss = model(batch)
                else:
                    val_loss = model(batch)
                val_losses.append(val_loss.item())
        model.train()
        avg_train_loss = sum(epoch_losses) / len(epoch_losses)
        avg_val_loss = sum(val_losses) / len(val_losses)
        training_losses.append(avg_train_loss)
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            patience_counter = 0
            best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
            print(
                f"Epoch {epoch + 1}/{n_epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f} (new best) ↓")
        else:
            patience_counter += 1
            print(f"Epoch {epoch + 1}/{n_epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break
    if best_model_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
        print(f"Restored best model with validation loss: {best_loss:.6f}")
    model.eval()
    return model, preprocessor, column_metadata, training_losses


# Synthetic Data Generation Function
def generate_synthetic_data(model, n_samples, device=None, preprocessor=None, column_metadata=None,
                            temp=1.0, guidance_scale=1.5, seed=None, batch_size=512):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    if device is None:
        device = next(model.parameters()).device
    if isinstance(model, tuple) and len(model) >= 3:
        if preprocessor is None:
            preprocessor = model[1]
        if column_metadata is None:
            column_metadata = model[2]
        model = model[0]
    model.eval()
    if n_samples > batch_size:
        raw_synthetic_batches = []
        remaining = n_samples
        while remaining > 0:
            current_batch = min(batch_size, remaining)
            with torch.no_grad():
                batch = model.diffusion_full_reverse_process(
                    batch_size=current_batch,
                    device=device,
                    temp=temp,
                    guidance_scale=guidance_scale
                ).cpu().numpy()
            raw_synthetic_batches.append(batch)
            remaining -= current_batch
        raw_synthetic = np.vstack(raw_synthetic_batches)
    else:
        with torch.no_grad():
            raw_synthetic = model.diffusion_full_reverse_process(
                batch_size=n_samples,
                device=device,
                temp=temp,
                guidance_scale=guidance_scale
            ).cpu().numpy()
    synthetic_df = preprocessor.inverse_transform(raw_synthetic, column_metadata)
    return synthetic_df


# Evaluation Function
def evaluate_synthetic_data(real_df, synthetic_df, task_type='classification', target_column=None,
                            classifier=None, regression_model=None):
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error, r2_score
    if target_column is None:
        raise ValueError("Target column must be specified")
    if task_type == 'classification' and classifier is None:
        classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    elif task_type == 'regression' and regression_model is None:
        regression_model = RandomForestRegressor(n_estimators=100, random_state=42)
    X_real = real_df.drop(columns=[target_column])
    y_real = real_df[target_column]
    X_syn = synthetic_df.drop(columns=[target_column])
    y_syn = synthetic_df[target_column]
    if task_type == 'classification':
        model = classifier
        model.fit(X_syn, y_syn)
        y_pred = model.predict(X_real)
        y_prob = model.predict_proba(X_real) if hasattr(model, 'predict_proba') else None
        metrics = {
            'accuracy': accuracy_score(y_real, y_pred)
        }
        if y_prob is not None and len(np.unique(y_real)) == 2:
            metrics['roc_auc'] = roc_auc_score(y_real, y_prob[:, 1])
    else:
        model = regression_model
        model.fit(X_syn, y_syn)
        y_pred = model.predict(X_real)
        metrics = {
            'mse': mean_squared_error(y_real, y_pred),
            'r2': r2_score(y_real, y_pred)
        }
    return metrics


# Example Usage
if __name__ == "__main__":
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    continuous_features = np.random.normal(0, 1, size=(n_samples, n_features // 2))
    int_features = np.random.randint(0, 10, size=(n_samples, n_features // 2))
    features = np.hstack([continuous_features, int_features])
    w = np.random.normal(0, 1, size=n_features)
    logits = np.dot(features, w)
    probs = 1 / (1 + np.exp(-logits))
    target = (probs > 0.5).astype(int)
    columns = [f'float_{i}' for i in range(n_features // 2)] + [f'int_{i}' for i in range(n_features // 2)] + ['target']
    data = np.hstack([features, target.reshape(-1, 1)])
    df = pd.DataFrame(data, columns=columns)
    for i in range(n_features // 2):
        df[f'int_{i}'] = df[f'int_{i}'].astype(int)
    print("Original data:")
    print(df.head())
    model, preprocessor, column_metadata, _ = train_tabular_diffflow_model(
        df,
        n_epochs=10,
        batch_size=32
    )
    synthetic_df = generate_synthetic_data(model, 1000, preprocessor=preprocessor, column_metadata=column_metadata)
    print("\nSynthetic data:")
    print(synthetic_df.head())
    metrics = evaluate_synthetic_data(df, synthetic_df, task_type='classification', target_column='target')
    print("\nEvaluation metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v}")