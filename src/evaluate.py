import logging
from pathlib import Path

import numpy as np
import torch

from helpers import init_helper, data_helper, vsumm_helper, bbox_helper
from helpers.eval_protocol_helper import compute_rank_metrics_from_gtscore, infer_f1_metric_from_key
from modules.model_zoo import get_model

logger = logging.getLogger()


def evaluate(model, val_loader, nms_thresh, device, return_rank: bool = False):
    model.eval()
    stats = data_helper.AverageMeter('fscore', 'diversity', 'kendall', 'spearman')

    with torch.no_grad():
        for test_key, seq, gtscore, cps, n_frames, nfps, picks, user_summary in val_loader:
            seq_len = len(seq)
            seq_torch = torch.from_numpy(seq).unsqueeze(0).to(device)

            pred_cls, pred_bboxes = model.predict(seq_torch)

            pred_bboxes = np.clip(pred_bboxes, 0, seq_len).round().astype(np.int32)
            pred_cls, pred_bboxes = bbox_helper.nms(pred_cls, pred_bboxes, nms_thresh)

            sample_scores = bbox2sample_scores(seq_len, pred_cls, pred_bboxes)
            pred_summ = vsumm_helper.get_keyshot_summ(
                sample_scores,
                cps,
                int(np.asarray(n_frames).item()),
                nfps,
                np.asarray(picks, dtype=np.int32),
            )

            eval_metric = infer_f1_metric_from_key(test_key)
            fscore = vsumm_helper.get_summ_f1score(pred_summ, user_summary, eval_metric)

            pred_summ_downsampled = vsumm_helper.downsample_summ(pred_summ)
            diversity = vsumm_helper.get_summ_diversity(pred_summ_downsampled, seq)

            rank_metrics = compute_rank_metrics_from_gtscore(
                pred_scores=sample_scores,
                gtscore=np.asarray(gtscore, dtype=np.float32),
                key=str(test_key),
            )

            stats.update(
                fscore=fscore,
                diversity=diversity,
                kendall=rank_metrics['kendall'],
                spearman=rank_metrics['spearman'],
            )

    if return_rank:
        return {
            'fscore': stats.fscore,
            'diversity': stats.diversity,
            'kendall': stats.kendall,
            'spearman': stats.spearman,
        }

    return stats.fscore, stats.diversity


def bbox2sample_scores(seq_len: int, pred_cls: np.ndarray, pred_bboxes: np.ndarray) -> np.ndarray:
    score = np.zeros(seq_len, dtype=np.float32)
    for bbox_idx in range(len(pred_bboxes)):
        lo, hi = pred_bboxes[bbox_idx, 0], pred_bboxes[bbox_idx, 1]
        score[lo:hi] = np.maximum(score[lo:hi], pred_cls[bbox_idx])
    return score


def main():
    args = init_helper.get_arguments()

    init_helper.init_logger(args.model_dir, args.log_file)
    init_helper.set_random_seed(args.seed)

    logger.debug('Arguments: %s', vars(args))
    model = get_model(args.model, **vars(args))
    model = model.eval().to(args.device)

    for split_path in args.splits:
        split_path = Path(split_path)
        splits = data_helper.load_yaml(split_path)

        stats = data_helper.AverageMeter('fscore', 'diversity', 'kendall', 'spearman')

        for split_idx, split in enumerate(splits):
            ckpt_path = data_helper.get_ckpt_path(args.model_dir, split_path, split_idx)
            state_dict = torch.load(str(ckpt_path), map_location=lambda storage, loc: storage)
            model.load_state_dict(state_dict)

            val_set = data_helper.VideoDataset(split['test_keys'])
            val_loader = data_helper.DataLoader(val_set, shuffle=False)

            metrics = evaluate(
                model,
                val_loader,
                args.nms_thresh,
                args.device,
                return_rank=True,
            )
            stats.update(
                fscore=metrics['fscore'],
                diversity=metrics['diversity'],
                kendall=metrics['kendall'],
                spearman=metrics['spearman'],
            )

            logger.info(
                '%s split %d | F1=%.4f',
                split_path.stem,
                split_idx + 1,
                fscore,
            )
            logger.debug(
                '%s split %d detail | diversity=%.4f',
                split_path.stem,
                split_idx + 1,
                diversity,
            )

        logger.info(
            '%s | F1=%.4f',
            split_path.stem,
            stats.fscore,
        )
        logger.debug(
            '%s detail | diversity=%.4f',
            split_path.stem,
            stats.diversity,
        )


if __name__ == '__main__':
    main()
