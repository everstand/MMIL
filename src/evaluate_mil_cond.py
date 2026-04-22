import numpy as np
import torch

from helpers import vsumm_helper


def evaluate_mil_cond(model,
                      val_loader,
                      device: str):
    model.eval()

    fscore_list = []
    spearman_list = []

    with torch.no_grad():
        for (
            key,
            seq,
            soft_label,
            text_cond,
            text_target,
            all_text_features,
            caption_spans_idx,
            caption_valid_mask,
            gtscore,
            user_summary,
            cps,
            n_frames,
            nfps,
            picks,
        ) in val_loader:
            seq_tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)
            text_cond_tensor = torch.tensor(text_cond, dtype=torch.float32).to(device)

            _, _, attn_weights, _, _, _ = model(seq_tensor, text_cond_tensor)
            summary_scores = attn_weights.detach().cpu().numpy().astype(np.float32)

            if not np.isfinite(summary_scores).all():
                num_nan = int(np.isnan(summary_scores).sum())
                num_inf = int(np.isinf(summary_scores).sum())
                raise ValueError(
                    f'Non-finite summary_scores for sample {key}: '
                    f'nan={num_nan}, inf={num_inf}, '
                    f'seq_shape={seq.shape}, text_cond_shape={text_cond.shape}'
                )

            picks_np = np.asarray(picks, dtype=np.int32)
            if summary_scores.shape[0] != picks_np.shape[0]:
                raise ValueError(
                    f'Summary score length mismatch for sample {key}: '
                    f'scores={summary_scores.shape[0]} vs picks={picks_np.shape[0]}'
                )

            pred_summ = vsumm_helper.get_keyshot_summ(
                summary_scores,
                cps,
                int(np.asarray(n_frames).item()),
                nfps,
                picks_np,
            )

            if user_summary is None:
                raise ValueError(f'Missing user_summary for evaluation sample: {key}')

            eval_metric = infer_eval_metric_from_key(key)
            fscore = vsumm_helper.get_summ_f1score(
                pred_summ=pred_summ,
                test_summ=user_summary,
                eval_metric=eval_metric,
            )
            fscore_list.append(float(fscore))

            if gtscore is None:
                continue

            gtscore_np = np.asarray(gtscore, dtype=np.float32)
            if gtscore_np.shape[0] != summary_scores.shape[0]:
                min_len = min(gtscore_np.shape[0], summary_scores.shape[0])
                gtscore_np = gtscore_np[:min_len]
                summary_scores_eval = summary_scores[:min_len]
            else:
                summary_scores_eval = summary_scores

            spearman = compute_spearman(summary_scores_eval, gtscore_np)
            if not np.isnan(spearman):
                spearman_list.append(float(spearman))

    mean_fscore = float(np.mean(fscore_list)) if fscore_list else 0.0
    mean_spearman = float(np.mean(spearman_list)) if spearman_list else 0.0
    return mean_fscore, mean_spearman


def infer_eval_metric_from_key(key: str) -> str:
    key_lower = str(key).lower()
    if 'tvsum' in key_lower:
        return 'avg'
    if 'summe' in key_lower:
        return 'max'
    raise ValueError(f'Cannot infer dataset name from key: {key}')


def compute_spearman(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)

    if x.shape != y.shape:
        raise ValueError(f'Shape mismatch in compute_spearman: {x.shape} vs {y.shape}')
    if x.size < 2:
        return float('nan')

    rx = rankdata_average(x)
    ry = rankdata_average(y)

    rx = rx - rx.mean()
    ry = ry - ry.mean()

    denom = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    if denom <= 0:
        return float('nan')

    return float((rx * ry).sum() / denom)


def rankdata_average(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind='mergesort')
    sorted_values = values[order]

    ranks = np.zeros(values.shape[0], dtype=np.float64)

    start = 0
    while start < values.shape[0]:
        end = start + 1
        while end < values.shape[0] and sorted_values[end] == sorted_values[start]:
            end += 1

        avg_rank = 0.5 * (start + end - 1) + 1.0
        ranks[order[start:end]] = avg_rank
        start = end

    return ranks