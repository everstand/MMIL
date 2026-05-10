import numpy as np
import torch

from modules.shot_context_encoder import (
    build_context_shot_text_stats,
    build_shot_time_features,
)
from helpers import vsumm_helper
from helpers.eval_protocol_helper import (
    compute_rank_metrics_from_gtscore,
    infer_f1_metric_from_key,
    safe_nanmean,
)


def evaluate_mil_cond(model,
                      val_loader,
                      device: str,
                      selection_score_source: str = 'frame',
                      shot_eval_head: str = 'selection',
                      shot_head_mode: str = 'single',
                      summary_budget: float = 0.15):
    if selection_score_source not in ('frame', 'shot_head'):
        raise ValueError(
            f'Invalid selection_score_source={selection_score_source}; expected frame or shot_head.'
        )
    if shot_eval_head not in ('selection', 'rank'):
        raise ValueError(f'Invalid shot_eval_head={shot_eval_head}; expected selection or rank.')
    if shot_head_mode not in ('single', 'dual', 'context'):
        raise ValueError(f'Invalid shot_head_mode={shot_head_mode}; expected single, dual, or context.')
    if selection_score_source == 'shot_head' and shot_eval_head == 'rank' and shot_head_mode not in ('dual', 'context'):
        raise ValueError('--shot-eval-head rank requires --shot-head-mode dual or context.')
    model.eval()

    fscore_list = []
    kendall_list = []
    spearman_list = []
    caption_coverage_list = []

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
            text_cond_mask,
            caption_coverage_ratio,
        ) in val_loader:
            seq_tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)
            text_cond_tensor = torch.tensor(text_cond, dtype=torch.float32).to(device)
            text_cond_mask_tensor = torch.tensor(
                text_cond_mask,
                dtype=torch.float32,
                device=device,
            )
            caption_coverage_list.append(float(np.asarray(caption_coverage_ratio).item()))

            if selection_score_source == 'shot_head':
                (
                    _instance_logits,
                    _pool_logits,
                    _frame_summary_scores,
                    _bag_logits,
                    _summary_feat,
                    _pre_cross_frame_repr,
                    cond_frame_repr,
                    _hidden_frame_repr,
                ) = model(
                    seq_tensor,
                    text_cond_tensor,
                    text_cond_mask_tensor,
                )
                overlaps, shot_lengths = build_sampled_to_shot_overlap_eval(
                    picks=torch.tensor(picks, dtype=torch.long, device=device),
                    cps=torch.tensor(cps, dtype=torch.long, device=device),
                    n_frames=int(np.asarray(n_frames).item()),
                )
                if shot_head_mode == 'context':
                    all_text_features_tensor = torch.tensor(
                        all_text_features, dtype=torch.float32, device=device
                    )
                    caption_spans_idx_tensor = torch.tensor(
                        caption_spans_idx, dtype=torch.long, device=device
                    )
                    caption_valid_mask_tensor = torch.tensor(
                        caption_valid_mask, dtype=torch.float32, device=device
                    )
                    shot_text_feat, _shot_mass_density, _valid_shots = build_context_shot_text_stats(
                        caption_spans_idx=caption_spans_idx_tensor,
                        caption_valid_mask=caption_valid_mask_tensor,
                        all_text_features=all_text_features_tensor,
                        overlaps=overlaps,
                        shot_lengths=shot_lengths,
                    )
                    shot_time_feat = build_shot_time_features(
                        cps=torch.tensor(cps, dtype=torch.long, device=device),
                        nfps=torch.tensor(nfps, dtype=torch.float32, device=device),
                        n_frames=int(np.asarray(n_frames).item()),
                    )
                else:
                    shot_text_feat = None
                    shot_time_feat = None
                shot_scores = model.predict_shot_scores(
                    frame_repr=cond_frame_repr,
                    overlaps=overlaps,
                    shot_lengths=shot_lengths,
                    head=shot_eval_head,
                    shot_text_feat=shot_text_feat,
                    shot_time_feat=shot_time_feat,
                )
                shot_scores_np = shot_scores.detach().cpu().numpy().astype(np.float32)
                pred_summ = get_keyshot_summ_from_shot_scores(
                    shot_scores=shot_scores_np,
                    cps=cps,
                    nfps=nfps,
                    n_frames=int(np.asarray(n_frames).item()),
                    summary_budget=summary_budget,
                )
                summary_scores = project_shot_scores_to_sampled_scores(
                    shot_scores=shot_scores,
                    overlaps=overlaps,
                ).detach().cpu().numpy().astype(np.float32)
            else:
                summary_scores = model.predict_summary_scores(
                    seq_tensor,
                    text_cond_tensor,
                    text_cond_mask_tensor,
                ).detach().cpu().numpy().astype(np.float32)

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

            if selection_score_source != 'shot_head':
                pred_summ = vsumm_helper.get_keyshot_summ(
                    summary_scores,
                    cps,
                    int(np.asarray(n_frames).item()),
                    nfps,
                    picks_np,
                    proportion=summary_budget,
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
        'caption_coverage': float(np.mean(caption_coverage_list)) if caption_coverage_list else 0.0,
    }



def get_keyshot_summ_from_shot_scores(shot_scores,
                                      cps,
                                      nfps,
                                      n_frames: int,
                                      summary_budget: float = 0.15) -> np.ndarray:
    """Generate a keyshot summary directly from shot-level scores.

    Shape contract:
        shot_scores: [S]
        cps: [S, 2]
        nfps: [S]
        return: [n_frames] bool summary
    """
    scores = np.asarray(shot_scores, dtype=np.float32).reshape(-1)
    cps_np = np.asarray(cps, dtype=np.int32)
    nfps_np = np.asarray(nfps, dtype=np.int32).reshape(-1)
    n_frames_int = int(n_frames)
    if scores.ndim != 1:
        raise ValueError(f'Expected shot_scores shape [S], got {scores.shape}')
    if cps_np.ndim != 2 or cps_np.shape[1] != 2:
        raise ValueError(f'Expected cps shape [S, 2], got {cps_np.shape}')
    if scores.shape[0] != cps_np.shape[0]:
        raise ValueError(f'shot_scores/cps length mismatch: {scores.shape[0]} vs {cps_np.shape[0]}')
    if scores.shape[0] != nfps_np.shape[0]:
        raise ValueError(f'shot_scores/nfps length mismatch: {scores.shape[0]} vs {nfps_np.shape[0]}')
    if n_frames_int <= 0:
        raise ValueError(f'Invalid n_frames={n_frames}')
    if not (0.0 < float(summary_budget) < 1.0):
        raise ValueError(f'Invalid summary_budget={summary_budget}; expected 0 < budget < 1.')
    if not np.isfinite(scores).all():
        raise ValueError('Non-finite shot_scores in direct shot summary generation.')

    values = np.round(np.clip(scores, 0.0, 1.0) * 1000.0).astype(np.int32)
    capacity = int(n_frames_int * float(summary_budget))
    packed = vsumm_helper.knapsack(values.tolist(), nfps_np.tolist(), capacity)

    summary = np.zeros(n_frames_int, dtype=bool)
    for seg_idx in packed:
        first, last = cps_np[int(seg_idx)]
        first = int(max(0, min(first, n_frames_int - 1)))
        last = int(max(first, min(last, n_frames_int - 1)))
        summary[first:last + 1] = True
    return summary


def build_sampled_to_shot_overlap_eval(picks: torch.Tensor,
                                       cps: torch.Tensor,
                                       n_frames: int):
    if picks.ndim != 1:
        raise ValueError(f'Expected picks shape [T], got {tuple(picks.shape)}')
    if cps.ndim != 2 or cps.shape[1] != 2:
        raise ValueError(f'Expected cps shape [S, 2], got {tuple(cps.shape)}')
    if int(n_frames) <= 0:
        raise ValueError(f'Invalid n_frames={n_frames}')

    picks = picks.to(torch.long)
    cps = cps.to(torch.long)
    lo = picks
    hi = torch.empty_like(lo)
    hi[:-1] = picks[1:]
    hi[-1] = int(n_frames)

    overlaps = []
    for shot_idx in range(cps.shape[0]):
        first = int(cps[shot_idx, 0].item())
        last_exclusive = int(cps[shot_idx, 1].item()) + 1
        inter = torch.minimum(hi, lo.new_tensor(last_exclusive)) - torch.maximum(
            lo, lo.new_tensor(first)
        )
        overlaps.append(torch.clamp(inter, min=0).to(torch.float32))
    overlaps = torch.stack(overlaps, dim=0)
    shot_lengths = overlaps.sum(dim=1)
    if not torch.all(shot_lengths > 0):
        raise ValueError('Detected non-positive shot length in evaluation overlap.')
    return overlaps, shot_lengths


def project_shot_scores_to_sampled_scores(shot_scores: torch.Tensor,
                                          overlaps: torch.Tensor) -> torch.Tensor:
    if shot_scores.ndim != 1:
        raise ValueError(f'Expected shot_scores shape [S], got {tuple(shot_scores.shape)}')
    if overlaps.ndim != 2:
        raise ValueError(f'Expected overlaps shape [S, T], got {tuple(overlaps.shape)}')
    if overlaps.shape[0] != shot_scores.shape[0]:
        raise ValueError('Shot count mismatch in project_shot_scores_to_sampled_scores.')
    sample_lengths = overlaps.sum(dim=0).clamp_min(1.0)
    return torch.matmul(overlaps.transpose(0, 1), shot_scores) / sample_lengths
