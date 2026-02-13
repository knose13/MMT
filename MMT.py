import os
import random
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 全局变量
_SHAP_AVAILABLE = False
try:
    import shap

    _SHAP_AVAILABLE = True
except ImportError:
    pass


class PositionalEncoding(nn.Module):
    """位置编码：为时序特征添加位置信息"""

    def __init__(self, d_model: int, max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        # 生成位置编码矩阵 (max_seq_len, d_model)
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_seq_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class CrossModalAttention(nn.Module):
    """动态跨模态注意力：传感器特征（Query）与视觉特征（Key/Value）融合"""

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.2):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        # q: 传感器特征 (batch, seq_len, d_model)
        # kv: 视觉特征 (batch, seq_len, d_model)
        attn_output, _ = self.multihead_attn(q, kv, kv)
        q = self.norm(q + self.dropout(attn_output))
        return q


class TransformerRegressor(nn.Module):
    """核心多模态Transformer回归模型（传感器+视觉特征融合）"""

    def __init__(
            self,
            sensor_dim: int,  # 传感器特征维度（通常为1，即dust历史值）
            visual_dim: int,  # 视觉特征维度（颜色+纹理特征数量）
            d_model: int = 64,
            nhead: int = 4,
            num_decoder_layers: int = 2,
            dim_feedforward: int = 256,
            dropout: float = 0.2,
            max_seq_len: int = 5000,
    ):
        super().__init__()
        self.d_model = d_model

        # 模态投影：将传感器/视觉特征投影到d_model维度
        self.sensor_proj = nn.Linear(sensor_dim, d_model)
        self.visual_proj = nn.Linear(visual_dim, d_model)

        # 位置编码
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)

        # 编码器：分别编码两个模态
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, activation='relu'
        )
        self.sensor_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.visual_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

        # 解码器：跨模态注意力融合（多层）
        self.decoder_layers = nn.ModuleList([
            CrossModalAttention(d_model, nhead, dropout) for _ in range(num_decoder_layers)
        ])

        # 输出头：预测最终结果
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, sensor_x: torch.Tensor, visual_x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sensor_x: (batch, seq_len, sensor_dim) 传感器特征（dust历史值）
            visual_x: (batch, seq_len, visual_dim) 视觉特征（颜色和纹理）
        Returns:
            pred: (batch,) 预测值
        """
        # 投影到d_model维度
        sensor_emb = self.sensor_proj(sensor_x)
        visual_emb = self.visual_proj(visual_x)

        # 添加位置编码
        sensor_emb = self.pos_encoding(sensor_emb)
        visual_emb = self.pos_encoding(visual_emb)

        # 编码器处理各模态
        sensor_encoded = self.sensor_encoder(sensor_emb)
        visual_encoded = self.visual_encoder(visual_emb)

        # 解码器融合模态特征
        sensor_features = sensor_encoded
        for decoder_layer in self.decoder_layers:
            sensor_features = decoder_layer(sensor_features, visual_encoded)

        # 用最后一个时间步特征预测
        last_hidden = sensor_features[:, -1, :]
        pred = self.head(last_hidden)
        return pred.squeeze(-1)


# 损失函数（保留所有类型，适配不同需求）
class QuantileLoss(nn.Module):
    def __init__(self, quantile: float = 0.5):
        super().__init__()
        self.quantile = quantile

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        error = target - pred
        loss = torch.max(self.quantile * error, (self.quantile - 1) * error)
        return torch.mean(loss)


class WeightedMSELoss(nn.Module):
    def __init__(self, extreme_weight: float = 2.0, threshold_percentile: float = 0.9):
        super().__init__()
        self.extreme_weight = extreme_weight
        self.threshold_percentile = threshold_percentile

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target_abs = torch.abs(target)
        threshold = torch.quantile(target_abs, self.threshold_percentile)
        weights = torch.where(target_abs > threshold, self.extreme_weight, 1.0)
        mse = (pred - target) ** 2
        return torch.mean(weights * mse)


class AdaptiveLoss(nn.Module):
    def __init__(self, variance_penalty: float = 0.1, extreme_weight: float = 1.5):
        super().__init__()
        self.variance_penalty = variance_penalty
        self.extreme_weight = extreme_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target_mean = torch.mean(target)
        target_std = torch.std(target) + 1e-8
        normalized_target = torch.abs((target - target_mean) / target_std)
        weights = torch.where(normalized_target > 1.5, self.extreme_weight, 1.0)
        weighted_mse = torch.mean(weights * (pred - target) ** 2)

        pred_var = torch.var(pred) + 1e-8
        target_var = torch.var(target) + 1e-8
        variance_ratio = pred_var / target_var
        variance_loss = self.variance_penalty * torch.relu(0.5 - variance_ratio) * target_var

        return weighted_mse + variance_loss


class AsymmetricLoss(nn.Module):
    """不对称损失：对低估预测给予更大惩罚"""

    def __init__(self, underestimate_penalty: float = 2.0, high_value_weight: float = 3.0,
                 high_threshold_percentile: float = 0.8):
        super().__init__()
        self.underestimate_penalty = underestimate_penalty
        self.high_value_weight = high_value_weight
        self.high_threshold_percentile = high_threshold_percentile

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        error = pred - target
        target_abs = torch.abs(target)
        high_threshold = torch.quantile(target_abs, self.high_threshold_percentile)

        base_weights = torch.where(target_abs > high_threshold, self.high_value_weight, 1.0)
        underestimate_mask = error < 0
        penalty_weights = torch.where(underestimate_mask, self.underestimate_penalty, 1.0)

        combined_weights = base_weights * penalty_weights
        mse = (pred - target) ** 2
        return torch.mean(combined_weights * mse)


class EncouragingHighLoss(nn.Module):
    """鼓励预测高值的损失函数"""

    def __init__(self, underestimate_penalty: float = 2.5, high_value_weight: float = 4.0,
                 variance_boost: float = 0.2, high_threshold_percentile: float = 0.75):
        super().__init__()
        self.underestimate_penalty = underestimate_penalty
        self.high_value_weight = high_value_weight
        self.variance_boost = variance_boost
        self.high_threshold_percentile = high_threshold_percentile

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        error = pred - target
        target_abs = torch.abs(target)
        high_threshold = torch.quantile(target_abs, self.high_threshold_percentile)
        high_value_mask = target_abs > high_threshold

        base_weights = torch.where(high_value_mask, self.high_value_weight, 1.0)
        underestimate_mask = error < 0
        penalty_weights = torch.where(underestimate_mask, self.underestimate_penalty, 1.0)

        combined_weights = base_weights * penalty_weights
        weighted_mse = torch.mean(combined_weights * (pred - target) ** 2)

        pred_var = torch.var(pred) + 1e-8
        target_var = torch.var(target) + 1e-8
        variance_ratio = pred_var / target_var
        variance_loss = self.variance_boost * torch.relu(0.3 - variance_ratio) * target_var

        return weighted_mse + variance_loss


# 数据集类
class MultiModalTimeSeriesDataset(Dataset):
    """多模态时序数据集：传感器特征 + 视觉特征"""

    def __init__(self, sensor_X: np.ndarray, visual_X: np.ndarray, y: np.ndarray):
        self.sensor_X = sensor_X.astype(np.float32)
        self.visual_X = visual_X.astype(np.float32)
        self.y = y.astype(np.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.sensor_X[idx], self.visual_X[idx], self.y[idx]


# 核心工具函数
def separate_sensor_visual_features(feature_cols: List[str], target_col: str) -> Tuple[List[str], List[str]]:
    """分离传感器特征（dust历史值）和视觉特征（颜色/纹理）"""
    visual_keywords = ['color', 'texture', 'Color', 'Texture']
    visual_features = [f for f in feature_cols if any(kw in f for kw in visual_keywords)]
    sensor_features = [target_col]  # 传感器特征固定为目标值历史值
    return sensor_features, visual_features


def build_multimodal_sequences(
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str,
        lookback: int,
        horizon: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """构建多模态时序序列：传感器序列 + 视觉序列 + 目标值"""
    sensor_feat_cols, visual_feat_cols = separate_sensor_visual_features(feature_cols, target_col)

    # 提取并处理传感器、视觉特征
    data_sensor = df[target_col].values.reshape(-1, 1)
    data_visual = df[visual_feat_cols].values if len(visual_feat_cols) > 0 else np.zeros((len(df), 1), dtype=np.float32)
    data_tgt = df[target_col].values
    N = len(df)

    sensor_samples, visual_samples, labels = [], [], []
    for t in range(lookback - 1, N - horizon):
        start, end = t - (lookback - 1), t + 1
        sensor_x = data_sensor[start:end, :]
        visual_x = data_visual[start:end, :]
        y = data_tgt[t + horizon]

        # 跳过含缺失值的样本
        if np.any(np.isnan(sensor_x)) or np.any(np.isnan(visual_x)) or np.isnan(y):
            continue

        sensor_samples.append(sensor_x)
        visual_samples.append(visual_x)
        labels.append(y)

    # 处理空样本情况
    if len(sensor_samples) == 0:
        return (np.empty((0, lookback, 1)),
                np.empty((0, lookback, len(visual_feat_cols) if len(visual_feat_cols) > 0 else 1)),
                np.empty((0,)))

    return np.stack(sensor_samples, axis=0), np.stack(visual_samples, axis=0), np.array(labels)


def prepare_multimodal_dataloaders(
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str,
        lookback: int,
        horizon: int,
        train_ratio: float,
        batch_size: int,
) -> Tuple[DataLoader, DataLoader, DataLoader, StandardScaler, StandardScaler]:
    """准备多模态数据加载器（训练/验证/测试）"""
    N = len(df)
    train_end = int(N * train_ratio)
    df_train = df.iloc[:train_end].copy()
    df_valtest = df.iloc[train_end:].copy()

    # 标准化目标值（传感器特征）
    tgt_scaler = StandardScaler()
    df_train[[target_col]] = tgt_scaler.fit_transform(df_train[[target_col]].values)
    df_valtest[[target_col]] = tgt_scaler.transform(df_valtest[[target_col]].values)

    # 标准化视觉特征（如有）
    _, visual_feat_cols = separate_sensor_visual_features(feature_cols, target_col)
    if len(visual_feat_cols) > 0:
        visual_scaler = StandardScaler()
        df_train[visual_feat_cols] = visual_scaler.fit_transform(df_train[visual_feat_cols].values)
        df_valtest[visual_feat_cols] = visual_scaler.transform(df_valtest[visual_feat_cols].values)

    # 构建多模态序列并分割数据集
    df_scaled = pd.concat([df_train, df_valtest], axis=0)
    sensor_X_all, visual_X_all, y_all = build_multimodal_sequences(df_scaled, feature_cols, target_col, lookback,
                                                                   horizon)

    if sensor_X_all.shape[0] == 0:
        raise ValueError("窗口划分后无有效样本，请调整lookback/horizon或检查数据长度")

    # 分割训练/验证/测试集
    split_idx = int(sensor_X_all.shape[0] * train_ratio)
    sensor_X_train, visual_X_train, y_train = sensor_X_all[:split_idx], visual_X_all[:split_idx], y_all[:split_idx]
    sensor_X_tmp, visual_X_tmp, y_tmp = sensor_X_all[split_idx:], visual_X_all[split_idx:], y_all[split_idx:]
    val_split = int(0.5 * sensor_X_tmp.shape[0])
    sensor_X_val, visual_X_val, y_val = sensor_X_tmp[:val_split], visual_X_tmp[:val_split], y_tmp[:val_split]
    sensor_X_test, visual_X_test, y_test = sensor_X_tmp[val_split:], visual_X_tmp[val_split:], y_tmp[val_split:]

    # 创建数据加载器
    train_loader = DataLoader(MultiModalTimeSeriesDataset(sensor_X_train, visual_X_train, y_train),
                              batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(MultiModalTimeSeriesDataset(sensor_X_val, visual_X_val, y_val),
                            batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(MultiModalTimeSeriesDataset(sensor_X_test, visual_X_test, y_test),
                             batch_size=batch_size, shuffle=False, drop_last=False)

    return train_loader, val_loader, test_loader, StandardScaler(), tgt_scaler  # 虚拟feat_scaler适配接口


# 模型训练与评估
def train_transformer_model(
        train_loader: DataLoader,
        val_loader: DataLoader,
        sensor_dim: int,
        visual_dim: int,
        device: torch.device,
        d_model: int,
        nhead: int,
        num_decoder_layers: int,
        dim_feedforward: int,
        dropout: float,
        epochs: int,
        lr: float,
        weight_decay: float,
        loss_type: str,
        feature_set_name: str,
        horizon: int
) -> TransformerRegressor:
    """训练多模态Transformer模型"""
    # 初始化模型、优化器、损失函数
    model = TransformerRegressor(
        sensor_dim=sensor_dim, visual_dim=visual_dim, d_model=d_model, nhead=nhead,
        num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward, dropout=dropout
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # 选择损失函数
    if loss_type == "mse":
        criterion = nn.MSELoss()
    elif loss_type == "smooth_l1":
        criterion = nn.SmoothL1Loss()
    elif loss_type == "mae":
        criterion = nn.L1Loss()
    elif loss_type == "quantile":
        criterion = QuantileLoss(quantile=0.5)
    elif loss_type == "weighted_mse":
        criterion = WeightedMSELoss(extreme_weight=2.0, threshold_percentile=0.9)
    elif loss_type == "adaptive":
        criterion = AdaptiveLoss(variance_penalty=0.1, extreme_weight=1.5)
    elif loss_type == "asymmetric":
        criterion = AsymmetricLoss(underestimate_penalty=2.0, high_value_weight=3.0, high_threshold_percentile=0.8)
    elif loss_type == "encouraging_high":
        criterion = EncouragingHighLoss(underestimate_penalty=2.5, high_value_weight=4.0, variance_boost=0.2,
                                        high_threshold_percentile=0.75)
    else:
        raise ValueError(f"未知损失函数类型: {loss_type}")

    # 学习率调度与早停设置
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-5,
                                                           verbose=False)
    best_val = float("inf")
    best_state = {}
    patience = 15
    wait = 0

    # 训练日志表头
    print(f"\n{'=' * 60}")
    print(f"[当前任务] 特征集: {feature_set_name:15s} | Horizon: {horizon:2d} | 总轮次: {epochs}")
    print(f"{'=' * 60}")
    print(
        f"{'Epoch':>5s} {'Train Loss':>12s} {'Val Loss':>12s} {'Best Val Loss':>15s} {'EarlyStop Wait':>12s} {'LR':>10s}")
    print(f"{'-' * 60}")

    # 训练循环
    for ep in range(1, epochs + 1):
        model.train()
        train_losses = []
        for sensor_xb, visual_xb, yb in train_loader:
            sensor_xb, visual_xb, yb = sensor_xb.to(device), visual_xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(sensor_xb, visual_xb)
            loss = criterion(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        # 验证
        model.eval()
        val_losses = []
        with torch.no_grad():
            for sensor_xb, visual_xb, yb in val_loader:
                sensor_xb, visual_xb, yb = sensor_xb.to(device), visual_xb.to(device), yb.to(device)
                pred = model(sensor_xb, visual_xb)
                loss = criterion(pred, yb)
                val_losses.append(loss.item())

        # 计算平均损失
        avg_train = float(np.mean(train_losses)) if train_losses else float("nan")
        avg_val = float(np.mean(val_losses)) if val_losses else float("inf")

        # 更新最佳模型与早停计数
        if avg_val < best_val - 1e-6:
            best_val = avg_val
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"\n[早停触发] 连续{patience}轮无改善，停止训练")
                break

        # 更新学习率并打印日志
        scheduler.step(avg_val)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"{ep:5d} {avg_train:12.6f} {avg_val:12.6f} {best_val:15.6f} {wait:12d} {current_lr:10.6f}")

    # 加载最佳模型
    if best_state:
        model.load_state_dict(best_state)
        print(f"\n[训练完成] 最佳验证损失: {best_val:.6f}")
    else:
        print("\n[训练完成] 未找到最佳模型（无有效验证损失）")
    return model


def evaluate_transformer_model(
        model: TransformerRegressor,
        data_loader: DataLoader,
        device: torch.device,
        tgt_scaler: StandardScaler
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """评估多模态Transformer模型，返回真实值、预测值及评估指标"""
    model.eval()
    all_preds, all_trues = [], []
    with torch.no_grad():
        for sensor_xb, visual_xb, yb in data_loader:
            sensor_xb, visual_xb = sensor_xb.to(device), visual_xb.to(device)
            pred = model(sensor_xb, visual_xb).detach().cpu().numpy()
            all_preds.append(pred)
            all_trues.append(yb.numpy())

    # 拼接并反标准化（基于原始值计算指标）
    y_true_scaled = np.concatenate(all_trues, axis=0)
    y_pred_scaled = np.concatenate(all_preds, axis=0)
    y_true = tgt_scaler.inverse_transform(y_true_scaled.reshape(-1, 1)).ravel()
    y_pred = tgt_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

    # 计算评估指标
    metrics = {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
        "MAPE": float(np.mean(np.abs((y_true - y_pred) / (np.clip(np.abs(y_true), 1e-8, None)))) * 100.0)
    }
    return y_true_scaled, y_pred_scaled, metrics


# 核心实验函数
def run_transformer_experiment(
        df: pd.DataFrame,
        all_feature_cols: List[str],
        pareto_features: List[str],
        target_col: str,
        exclude_cols: Tuple[str, ...],
        lookback: int,
        horizons: List[int],
        output_prefix: str,
        train_ratio: float,
        batch_size: int,
        d_model: int,
        nhead: int,
        num_decoder_layers: int,
        dim_feedforward: int,
        dropout: float,
        epochs: int,
        lr: float,
        weight_decay: float,
        loss_type: str,
        device: torch.device,
):
    """运行多模态Transformer实验"""
    results_rows, predictions_data = [], []
    os.makedirs(".", exist_ok=True)

    # 定义特征集
    feature_sets = {
        "sensor_only": [target_col],  # 单一传感器模态（对比用）
        "target_plus_pareto": sorted(list(set([target_col] + pareto_features))),  # 传感器+Pareto视觉特征
        "all_features": [c for c in all_feature_cols if c not in exclude_cols],  # 传感器+所有视觉特征
    }

    # 任务进度初始化
    total_tasks = len(feature_sets) * len(horizons)
    current_task = 0
    print(f"[总任务统计] 特征集数: {len(feature_sets)} | Horizons: {horizons} | 总任务数: {total_tasks}")

    # 遍历所有特征集和预测步长
    for set_name, feats in feature_sets.items():
        if target_col not in feats:
            feats.append(target_col)

        # 分离传感器和视觉特征，获取维度
        sensor_feat_cols, visual_feat_cols = separate_sensor_visual_features(feats, target_col)
        sensor_dim = len(sensor_feat_cols)
        visual_dim = len(visual_feat_cols) if len(visual_feat_cols) > 0 else 1

        for H in horizons:
            current_task += 1
            print(f"\n{'=' * 80}")
            print(f"[任务进度] {current_task}/{total_tasks} | 特征集: {set_name} | Horizon: {H}")
            print(f"[模态信息] 传感器维度: {sensor_dim} | 视觉维度: {visual_dim}")
            print(f"{'=' * 80}")

            # 准备数据加载器
            print(
                f"[数据加载] 传感器特征: {sensor_feat_cols} | 视觉特征数: {visual_dim} | Lookback: {lookback} | 批次大小: {batch_size}")
            train_loader, val_loader, test_loader, _, tgt_scaler = prepare_multimodal_dataloaders(
                df=df, feature_cols=feats, target_col=target_col, lookback=lookback,
                horizon=H, train_ratio=train_ratio, batch_size=batch_size
            )
            print(
                f"[数据加载完成] 训练样本数: {len(train_loader.dataset)} | 验证样本数: {len(val_loader.dataset)} | 测试样本数: {len(test_loader.dataset)}")

            # 训练模型
            model = train_transformer_model(
                train_loader=train_loader, val_loader=val_loader, sensor_dim=sensor_dim, visual_dim=visual_dim,
                device=device, d_model=d_model, nhead=nhead, num_decoder_layers=num_decoder_layers,
                dim_feedforward=dim_feedforward, dropout=dropout, epochs=epochs, lr=lr,
                weight_decay=weight_decay, loss_type=loss_type, feature_set_name=set_name, horizon=H
            )

            # 评估模型
            print(f"\n[开始评估] 特征集: {set_name} | Horizon: {H}")
            y_true_scaled, y_pred_scaled, metrics = evaluate_transformer_model(model, test_loader, device, tgt_scaler)
            print(
                f"[评估完成] MAE: {metrics['MAE']:.4f} | RMSE: {metrics['RMSE']:.4f} | R2: {metrics['R2']:.4f} | MAPE: {metrics['MAPE']:.2f}%")

            # 保存预测结果和评估指标
            y_true = tgt_scaler.inverse_transform(y_true_scaled.reshape(-1, 1)).ravel()
            y_pred = tgt_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

            # 保存详细预测数据
            for idx, (true_val, pred_val) in enumerate(zip(y_true, y_pred)):
                predictions_data.append({
                    "model_type": "multimodal_transformer",
                    "feature_set": set_name,
                    "horizon": H,
                    "lookback": lookback,
                    "sample_index": idx,
                    "true_value": float(true_val),
                    "true_value_scaled": float(y_true_scaled[idx]),
                    "predicted_value": float(pred_val),
                    "predicted_value_scaled": float(y_pred_scaled[idx]),
                    "residual": float(pred_val - true_val),
                })

            # 保存评估指标汇总
            results_rows.append({
                "model_type": "multimodal_transformer",
                "feature_set": set_name,
                "horizon": H,
                "lookback": lookback,
                "sensor_dim": sensor_dim,
                "visual_dim": visual_dim,
                "MAE": metrics["MAE"],
                "RMSE": metrics["RMSE"],
                "R2": metrics["R2"],
                "MAPE_%": metrics["MAPE"],
            })

    # 保存结果文件
    results_df = pd.DataFrame(results_rows)
    summary_csv = f"{output_prefix}_metrics_summary.csv"
    results_df.to_csv(summary_csv, index=False)
    print(f"\n[文件保存] 汇总指标已保存到: {os.path.abspath(summary_csv)}")

    predictions_df = pd.DataFrame(predictions_data)
    predictions_csv = f"{output_prefix}_predictions_data.csv"
    predictions_df.to_csv(predictions_csv, index=False)
    print(f"[文件保存] 详细预测数据已保存到: {os.path.abspath(predictions_csv)}")

    return results_df, predictions_df


# 辅助工具函数
def load_pareto_features(path_txt: str) -> List[str]:
    """加载Pareto筛选的视觉特征列表"""
    with open(path_txt, "r", encoding="utf-8") as f:
        feats = [line.strip() for line in f if line.strip()]
    return feats


# 主函数（相关参数）
def main():
    parser = argparse.ArgumentParser(description="多模态Transformer时序预测模型")
    parser.add_argument("--input", type=str, default="non_concentration_features_split_fixed_standard.csv",
                        help="输入CSV路径")
    parser.add_argument("--target", type=str, default="dust", help="目标列名（dust）")
    parser.add_argument("--exclude", type=str, default="time", help="需要排除的列（逗号分隔）")
    parser.add_argument("--pareto_txt", type=str, default="pareto_selected_features.txt", help="Pareto特征列表路径")
    parser.add_argument("--lookback", type=int, default=20, help="历史窗口长度")
    parser.add_argument("--horizons", type=str, default="1,5,10,15,20", help="预测步长（逗号分隔）")
    parser.add_argument("--train_ratio", type=float, default=0.4, help="训练集比例（时间有序划分）")
    parser.add_argument("--batch_size", type=int, default=64, help="批次大小")

    parser.add_argument("--d_model", type=int, default=64, help="Transformer特征维度")
    parser.add_argument("--nhead", type=int, default=4, help="注意力头数量")
    parser.add_argument("--layers", type=int, default=2, help="解码器层数")
    parser.add_argument("--dim_feedforward", type=int, default=256, help="前馈网络维度")
    parser.add_argument("--dropout", type=float, default=0.2, help="dropout概率")

    parser.add_argument("--epochs", type=int, default=60, help="最大训练轮次（启用早停）")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="L2正则化系数")
    parser.add_argument("--loss_type", type=str, default="smooth_l1",
                        choices=["mse", "smooth_l1", "mae", "quantile", "weighted_mse", "adaptive", "asymmetric",
                                 "encouraging_high"],
                        help="损失函数类型")

    parser.add_argument("--seed", type=int, default=42, help="随机种子（保证可复现）")
    parser.add_argument("--out", type=str, default="transformer", help="输出文件前缀")
    parser.add_argument("--enable_shap", action="store_true", default=False, help="启用SHAP特征解释（可选）")
    parser.add_argument("--shap_max_background", type=int, default=200, help="SHAP背景样本上限")
    parser.add_argument("--shap_max_test", type=int, default=1000, help="SHAP测试样本上限")

    args = parser.parse_args()

    print(f"[程序启动] 多模态Transformer粉尘预测模型 | 随机种子: {args.seed} | 设备: 正在检测...")
    df = pd.read_csv(args.input)
    if args.target not in df.columns:
        raise ValueError(f"目标列 '{args.target}' 不存在，可用列: {list(df.columns)}")

    exclude_cols = tuple([c.strip() for c in args.exclude.split(",") if c.strip()]) if args.exclude else tuple()
    all_feature_cols = [c for c in df.columns if c != args.target and c not in exclude_cols]
    pareto_features = load_pareto_features(args.pareto_txt)
    pareto_features = [c for c in pareto_features if c in df.columns and c not in exclude_cols]
    horizons = [int(x.strip()) for x in args.horizons.split(",") if x.strip()]

    # 固定随机种子（可复现）
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # 设备检测
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[设备检测完成] 使用设备: {device}")
    print(
        f"[参数汇总] 损失函数: {args.loss_type} | 学习率: {args.lr} | 批次大小: {args.batch_size} | 最大轮次: {args.epochs}")

    results_df, predictions_df = run_transformer_experiment(
        df=df, all_feature_cols=all_feature_cols, pareto_features=pareto_features,
        target_col=args.target, exclude_cols=exclude_cols, lookback=args.lookback,
        horizons=horizons, output_prefix=args.out, train_ratio=args.train_ratio,
        batch_size=args.batch_size, d_model=args.d_model, nhead=args.nhead,
        num_decoder_layers=args.layers, dim_feedforward=args.dim_feedforward,
        dropout=args.dropout, epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
        loss_type=args.loss_type, device=device
    )

    print(f"\n[全部完成] 所有多模态Transformer任务训练+评估结束！可查看生成的CSV结果文件。")


if __name__ == "__main__":
    main()

