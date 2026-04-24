import numpy as np
import torch

from helpers import vsumm_helper
from helpers.eval_protocol_helper import (
    compute_rank_metrics_from_gtscore,
    infer_f1_metric_from_key,
    safe_nanmean,
)


def evaluate_mil(model,
                 val_loader,
                 device: str):
    model.eval()

    fscore_list = []
    kendall_list = []
    spearman_list = []

    with torch.no_grad():
        for key, seq, soft_label, gtscore, user_summary, cps, n_frames, nfps, picks in val_loader:
            seq_tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)

            _, _, attn_weights, _ = model(seq_tensor)
            summary_scores = attn_weights.detach().cpu().numpy().astype(np.float32)

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

            eval_metric = infer_f1_metric_from_key(key)
            fscore = vsumm_helper.get_summ_f1score(
                pred_summ=pred_summ,
                test_summ=user_summary,
                eval_metric=eval_metric,
            )
            fscore_list.append(float(fscore))

            if gtscore is None:
                raise ValueError(f'Missing gtscore for rank evaluation sample: {key}')

            rank_metrics = compute_rank_metrics_from_gtscore(
                pred_scores=summary_scores,
                gtscore=np.asarray(gtscore, dtype=np.float32),
                key=str(key),
            )
            kendall_list.append(rank_metrics['kendall'])
            spearman_list.append(rank_metrics['spearman'])

    return {
        'fscore': float(np.mean(fscore_list)) if fscore_list else 0.0,
        'kendall': safe_nanmean(kendall_list),
        'spearman': safe_nanmean(spearman_list),
        'num_videos': int(len(fscore_list)),
        'num_rank_videos': int(sum(np.isfinite(v) for v in kendall_list)),
    }
