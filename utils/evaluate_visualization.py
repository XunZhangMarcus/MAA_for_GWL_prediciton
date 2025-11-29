import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ------------ Hydrology-focused metrics ------------ #
def _flatten(arr):
    return np.array(arr, dtype=float).flatten()


def nse(obs, sim):
    """Nash–Sutcliffe Efficiency."""
    obs_f = _flatten(obs)
    sim_f = _flatten(sim)
    denom = np.sum((obs_f - np.mean(obs_f)) ** 2)
    if denom == 0:
        return np.nan
    return 1.0 - np.sum((sim_f - obs_f) ** 2) / denom


def kge(obs, sim):
    """Kling–Gupta Efficiency."""
    obs_f = _flatten(obs)
    sim_f = _flatten(sim)
    if obs_f.std() == 0:
        return np.nan
    r = np.corrcoef(obs_f, sim_f)[0, 1] if len(obs_f) > 1 else np.nan
    alpha = sim_f.std() / obs_f.std() if obs_f.std() != 0 else np.nan
    beta = sim_f.mean() / obs_f.mean() if obs_f.mean() != 0 else np.nan
    components = [(r - 1) ** 2, (alpha - 1) ** 2, (beta - 1) ** 2]
    return 1 - np.sqrt(np.nansum(components))


def bias(obs, sim):
    obs_f = _flatten(obs)
    sim_f = _flatten(sim)
    return float(np.mean(sim_f - obs_f))


# ------------ Core helpers ------------ #
def inverse_transform(predictions, scaler):
    """逆转换预测结果，保持原始形状。"""
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    original_shape = predictions.shape
    restored = scaler.inverse_transform(predictions.reshape(-1, original_shape[-1]))
    return restored.reshape(original_shape)


def compute_metrics(true_values, predicted_values):
    """计算回归指标（MSE/MAE/RMSE/MAPE/NSE/KGE/R2/Bias）。"""
    true_arr = np.array(true_values)
    pred_arr = np.array(predicted_values)

    mse = mean_squared_error(true_arr, pred_arr)
    mae = mean_absolute_error(true_arr, pred_arr)
    rmse = float(np.sqrt(mse))
    with np.errstate(divide="ignore", invalid="ignore"):
        mape = float(np.mean(np.abs((true_arr - pred_arr) / np.maximum(np.abs(true_arr), 1e-8))) * 100)
    per_target_mse = np.mean((true_arr - pred_arr) ** 2, axis=0)
    metric_nse = nse(true_arr, pred_arr)
    metric_kge = kge(true_arr, pred_arr)
    metric_r2 = r2_score(true_arr.flatten(), pred_arr.flatten())
    metric_bias = bias(true_arr, pred_arr)

    return {
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "per_target_mse": per_target_mse,
        "nse": metric_nse,
        "kge": metric_kge,
        "r2": metric_r2,
        "bias": metric_bias,
    }


def validate(model, val_x, val_y, val_label_y, predict_step=1, device="cuda"):
    """验证阶段的 MSE 与分类精度。"""
    model.eval()
    model = model.to(device)
    with torch.no_grad():
        val_x = val_x.clone().detach().float().to(device)
        if isinstance(val_y, np.ndarray):
            val_y = torch.tensor(val_y).float()
        val_y = val_y.clone().detach().float().to(device)

        if isinstance(val_label_y, np.ndarray):
            val_label_y = torch.tensor(val_label_y).long()
        val_label_y = val_label_y.clone().detach().long().to(device)

        reg_preds, cls_preds = model(val_x)
        reg_preds = reg_preds[:, -predict_step:]

        val_y = val_y[:, -predict_step:].squeeze(-1)
        if predict_step > 1:
            val_y = val_y.squeeze(dim=1)
        mse_loss = F.mse_loss(reg_preds, val_y)

        cls_preds = cls_preds[:, -predict_step:, :]
        cls_targets = val_label_y[:, -predict_step:]

        pred_labels = cls_preds.argmax(dim=-1).long()
        acc = (pred_labels == cls_targets).sum() / cls_targets.numel()

    return mse_loss.item(), acc.item()


def validate_with_label(model, val_x, val_y, val_labels):
    """仅依赖标签的验证（用于多 GAN 训练阶段）。"""
    model.eval()
    with torch.no_grad():
        val_x = val_x.clone().detach().float()
        val_y_t = torch.tensor(val_y).float() if isinstance(val_y, np.ndarray) else val_y.clone().detach().float()

        if isinstance(val_labels, np.ndarray):
            val_lbl_t = torch.tensor(val_labels).long().to(val_x.device)
        else:
            val_lbl_t = val_labels.clone().detach().long().to(val_x.device)

        predictions, logits = model(val_x)
        predictions = predictions.cpu().numpy()
        val_y_np = val_y_t.cpu().numpy()

        mse_loss = F.mse_loss(torch.tensor(predictions).float().squeeze(), torch.tensor(val_y_np).float().squeeze())

        true_cls = val_lbl_t[:, -1].squeeze()
        pred_cls = logits.argmax(dim=1)
        acc = (pred_cls == true_cls).float().mean()

        return mse_loss, acc


# ------------ Visualization helpers ------------ #
def plot_generator_losses(data_G, output_dir):
    plt.rcParams.update({"font.size": 12})
    all_data = data_G
    N = len(all_data)
    plt.figure(figsize=(6 * N, 5))

    for i, data in enumerate(all_data):
        plt.subplot(1, N, i + 1)
        for j, acc in enumerate(data):
            label = f"G{i + 1} vs D{j + 1}" if j < len(data) - 1 else f"G{i + 1} Combined"
            plt.plot(acc, label=label, linewidth=2)

        plt.xlabel("Epoch", fontsize=14)
        plt.ylabel("Loss", fontsize=14)
        plt.title(f"G{i + 1} Loss", fontsize=16)
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "generator_losses.png"), dpi=500)
    plt.close()


def plot_discriminator_losses(data_D, output_dir):
    plt.rcParams.update({"font.size": 12})
    N = len(data_D)
    plt.figure(figsize=(6 * N, 5))

    for i, data in enumerate(data_D):
        plt.subplot(1, N, i + 1)
        for j, acc in enumerate(data):
            label = f"D{i + 1} vs G{j + 1}" if j < len(data) - 1 else f"D{i + 1} Combined"
            plt.plot(acc, label=label, linewidth=2)

        plt.xlabel("Epoch", fontsize=14)
        plt.ylabel("Loss", fontsize=14)
        plt.title(f"D{i + 1} Loss", fontsize=16)
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "discriminator_losses.png"), dpi=500)
    plt.close()


def visualize_overall_loss(histG, histD, output_dir):
    plt.rcParams.update({"font.size": 12})
    N = len(histG)
    plt.figure(figsize=(5 * N, 4))

    for i, (g, d) in enumerate(zip(histG, histD)):
        plt.plot(g, label=f"G{i + 1} Loss", linewidth=2)
        plt.plot(d, label=f"D{i + 1} Loss", linewidth=2)

    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    plt.title("Generator & Discriminator Loss", fontsize=16)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "overall_losses.png"), dpi=500)
    plt.close()


def plot_mse_loss(hist_MSE_G, hist_val_loss, num_epochs, output_dir):
    plt.rcParams.update({"font.size": 12})
    N = len(hist_MSE_G)
    plt.figure(figsize=(5 * N, 4))

    for i, (MSE, val_loss) in enumerate(zip(hist_MSE_G, hist_val_loss)):
        plt.plot(range(num_epochs), MSE, label=f"Train MSE G{i + 1}", linewidth=2)
        plt.plot(range(num_epochs), val_loss, label=f"Val MSE G{i + 1}", linewidth=2, linestyle="--")

    plt.title("MSE Loss (Train vs Val)", fontsize=16)
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("MSE", fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "mse_losses.png"), dpi=500)
    plt.close()


def plot_fitting_curve(true_values, predicted_values, output_dir, model_name):
    plt.rcParams.update({"font.size": 12})
    plt.figure(figsize=(10, 6))
    plt.plot(true_values, label="Observed GWL", linewidth=2)
    plt.plot(predicted_values, label="Predicted GWL", linewidth=2, linestyle="--")
    plt.title(f"{model_name} Groundwater Level Fit", fontsize=16)
    plt.xlabel("Time", fontsize=14)
    plt.ylabel("Groundwater Level", fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{model_name}_fitting_curve.png", dpi=500)
    plt.close()


def plot_residuals(true_values, predicted_values, output_dir, model_name):
    plt.rcParams.update({"font.size": 12})
    residuals = np.array(true_values).flatten() - np.array(predicted_values).flatten()
    plt.figure(figsize=(10, 4))
    plt.plot(residuals, label="Residual (Obs - Pred)", linewidth=1.5)
    plt.axhline(0, color="black", linestyle="--", linewidth=1)
    plt.title(f"{model_name} Residuals", fontsize=16)
    plt.xlabel("Time", fontsize=14)
    plt.ylabel("Residual", fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{model_name}_residuals.png", dpi=500)
    plt.close()


# ------------ Evaluation wrappers ------------ #
def _append_metric(result_dict, prefix, metrics):
    for key, val in metrics.items():
        if key == "per_target_mse":
            continue
        result_dict.setdefault(f"{prefix}_{key}", []).append(val)
    result_dict.setdefault(f"{prefix}_mse_per_target", []).append(metrics["per_target_mse"])


def evaluate_best_models(generators, best_model_state, train_xes, train_y, test_xes, test_y, y_scaler, output_dir):
    N = len(generators)

    for i in range(N):
        generators[i].load_state_dict(best_model_state[i])
        generators[i].eval()

    train_y_inv = inverse_transform(train_y, y_scaler)
    test_y_inv = inverse_transform(test_y, y_scaler)

    result = {}

    with torch.no_grad():
        for i in range(N):
            train_pred, _ = generators[i](train_xes[i])
            train_pred_inv = inverse_transform(train_pred.cpu().numpy(), y_scaler)
            train_metrics = compute_metrics(train_y_inv, train_pred_inv)
            plot_fitting_curve(train_y_inv, train_pred_inv, output_dir, f"G{i + 1}_Train")
            plot_residuals(train_y_inv, train_pred_inv, output_dir, f"G{i + 1}_Train")
            logging.info(
                "Train G%d | MSE %.4f | MAE %.4f | RMSE %.4f | NSE %.4f | KGE %.4f | R2 %.4f",
                i + 1,
                train_metrics["mse"],
                train_metrics["mae"],
                train_metrics["rmse"],
                train_metrics["nse"],
                train_metrics["kge"],
                train_metrics["r2"],
            )
            _append_metric(result, "train", train_metrics)

        for i in range(N):
            test_pred, _ = generators[i](test_xes[i])
            test_pred_inv = inverse_transform(test_pred.cpu().numpy(), y_scaler)
            test_metrics = compute_metrics(test_y_inv, test_pred_inv)
            plot_fitting_curve(test_y_inv, test_pred_inv, output_dir, f"G{i + 1}_Test")
            plot_residuals(test_y_inv, test_pred_inv, output_dir, f"G{i + 1}_Test")
            logging.info(
                "Test G%d | MSE %.4f | MAE %.4f | RMSE %.4f | NSE %.4f | KGE %.4f | R2 %.4f",
                i + 1,
                test_metrics["mse"],
                test_metrics["mae"],
                test_metrics["rmse"],
                test_metrics["nse"],
                test_metrics["kge"],
                test_metrics["r2"],
            )
            _append_metric(result, "test", test_metrics)

    return result


def evaluate_best_models_for_backtrader(generators, best_model_state, train_xes, train_y, test_xes, test_y, y_scaler, output_dir):
    # 保留接口以兼容旧调用，内部复用 evaluate_best_models 的结果并额外返回预测
    result = evaluate_best_models(generators, best_model_state, train_xes, train_y, test_xes, test_y, y_scaler, output_dir)
    test_preds_inv = []
    for i in range(len(generators)):
        generators[i].eval()
        with torch.no_grad():
            test_pred, _ = generators[i](test_xes[i])
            test_pred_inv = inverse_transform(test_pred.cpu().numpy(), y_scaler)
            test_preds_inv.append(test_pred_inv)
    result["test_preds_inv"] = test_preds_inv
    return result
