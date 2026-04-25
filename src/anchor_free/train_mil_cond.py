import logging
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from anchor_free.dsnet_af_mil_cond import DSNetAFMILCond
from evaluate_mil_cond import evaluate_mil_cond
from helpers import data_helper, mil_data_helper_cond
from helpers.shot_utility_helper import (
    ShotUtilityStore,
    resolve_shot_utility_path,
)

logger = logging.getLogger(__name__)


def train(args, split, save_path):
    logger.debug(
        'Loss weights | score_head=%s | rank_loss=%s | lambda_pair=%s | pair_margin=%s | '
        'lambda_listwise=%s | listwise_temperature=%s | '
        'lambda_select=%s | lambda_budget=%s | summary_budget=%s | neg_q=%s | '
        'utility_formula=%s | lambda_align=%s | lambda_aux=%s',
        args.score_head,
        args.rank_loss,
        args.lambda_pair,
        args.pair_margin,
        args.lambda_listwise,
        args.listwise_temperature,
        args.lambda_select,
        args.lambda_budget,
        args.summary_budget,
        args.negative_quantile,
        args.utility_formula,
        args.lambda_align,
        args.lambda_aux,
    )

    if 'val_keys' not in split:
        raise ValueError(
            'train_mil_cond requires split["val_keys"]. '
            'Do not use test_keys for checkpoint selection.'
        )

    validate_rank_loss_args(args)

    train_keys = split['train_keys']
    val_keys = split['val_keys']
    test_keys = split['test_keys']

    train_dataset_name = infer_single_dataset_name(train_keys)
    val_dataset_name = infer_single_dataset_name(val_keys)
    test_dataset_name = infer_single_dataset_name(test_keys)

    if len({train_dataset_name, val_dataset_name, test_dataset_name}) != 1:
        raise ValueError(
            f'Mixed dataset split is not allowed: '
            f'train={train_dataset_name}, val={val_dataset_name}, test={test_dataset_name}'
        )

    dataset_name = train_dataset_name

    shot_utility_store = None
    if args.rank_loss in ('listwise_utility', 'budgeted_pseudo_summary'):
        utility_path = resolve_shot_utility_path(
            dataset_name=dataset_name,
            explicit_path=args.shot_utility_path,
        )
        shot_utility_store = ShotUtilityStore(utility_path)
        logger.info(
            'Using utility-based rank loss | rank_loss=%s | path=%s | formula=%s',
            args.rank_loss,
            utility_path,
            args.utility_formula,
        )

    train_set = mil_data_helper_cond.VideoDatasetMILCond(
        train_keys,
        text_cond_num=args.text_cond_num,
        random_text_sampling=True,
    )
    val_set = mil_data_helper_cond.VideoDatasetMILCond(
        val_keys,
        text_cond_num=args.text_cond_num,
        random_text_sampling=False,
    )
    test_set = mil_data_helper_cond.VideoDatasetMILCond(
        test_keys,
        text_cond_num=args.text_cond_num,
        random_text_sampling=False,
    )

    num_classes = infer_num_classes(train_set)
    model = DSNetAFMILCond(
        base_model=args.base_model,
        num_feature=args.num_feature,
        num_hidden=args.num_hidden,
        num_head=args.num_head,
        num_classes=num_classes,
        score_head=args.score_head,
    ).to(args.device)

    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    train_loader = data_helper.DataLoader(train_set, shuffle=True)
    val_loader = data_helper.DataLoader(val_set, shuffle=False)
    test_loader = data_helper.DataLoader(test_set, shuffle=False)

    save_path = str(save_path)
    if save_path.endswith('.pth'):
        kendall_save_path = save_path[:-4] + '_max_kendall.pth'
        spearman_save_path = save_path[:-4] + '_max_spearman.pth'
    else:
        kendall_save_path = save_path + '_max_kendall.pth'
        spearman_save_path = save_path + '_max_spearman.pth'

    best_val_fscore = -1.0
    kendall_at_best_fscore = 0.0
    spearman_at_best_fscore = 0.0

    max_val_kendall = -1.0
    fscore_at_max_kendall = 0.0
    spearman_at_max_kendall = 0.0

    max_val_spearman = -1.0
    fscore_at_max_spearman = 0.0
    kendall_at_max_spearman = 0.0

    for epoch in range(args.max_epoch):
        model.train()
        stats = data_helper.AverageMeter(
            'loss',
            'rank_loss',
            'weighted_rank_loss',
            'pair_loss',
            'weighted_pair_loss',
            'listwise_loss',
            'weighted_listwise_loss',
            'selection_loss',
            'weighted_selection_loss',
            'budget_loss',
            'weighted_budget_loss',
            'align_loss',
            'weighted_align_loss',
            'bag_loss',
            'weighted_bag_loss',
            'num_aux_active',
            'num_aux_skipped',
            'num_supervised_shots',
            'num_positive_shots',
            'num_negative_shots',
        )

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
        ) in train_loader:
            seq_tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(args.device)
            text_cond_tensor = torch.tensor(text_cond, dtype=torch.float32).to(args.device)
            text_target_tensor = torch.tensor(text_target, dtype=torch.float32).to(args.device)
            all_text_features_tensor = torch.tensor(
                all_text_features, dtype=torch.float32, device=args.device
            )
            caption_spans_idx_tensor = torch.tensor(
                caption_spans_idx, dtype=torch.long, device=args.device
            )
            caption_valid_mask_tensor = torch.tensor(
                caption_valid_mask, dtype=torch.float32, device=args.device
            )
            soft_label_tensor = torch.tensor(soft_label, dtype=torch.float32).to(args.device)

            cps_np = np.asarray(cps, dtype=np.int32)
            nfps_np = np.asarray(nfps, dtype=np.int32)
            cps_tensor = torch.tensor(cps_np, dtype=torch.long, device=args.device)
            picks_tensor = torch.tensor(picks, dtype=torch.long, device=args.device)
            n_frames_int = int(np.asarray(n_frames).item())

            assert_finite_tensor('seq_tensor', seq_tensor, key)
            assert_finite_tensor('text_cond_tensor', text_cond_tensor, key)
            assert_finite_tensor('text_target_tensor', text_target_tensor, key)
            assert_finite_tensor('all_text_features_tensor', all_text_features_tensor, key)

            normalized_target, is_effective = normalize_soft_label(soft_label_tensor)

            (
                instance_logits,
                pool_logits,
                summary_scores,
                bag_logits,
                summary_feat,
                _,
            ) = model(seq_tensor, text_cond_tensor)

            overlaps, shot_lengths = build_sampled_to_shot_overlap(
                picks=picks_tensor,
                cps=cps_tensor,
                n_frames=n_frames_int,
            )

            pred_shot_scores = aggregate_attn_to_shot_scores(
                attn_weights=summary_scores,
                overlaps=overlaps,
                shot_lengths=shot_lengths,
            )

            selection_shot_scores = aggregate_frame_scores_to_shot_scores(
                frame_scores=summary_scores,
                overlaps=overlaps,
                shot_lengths=shot_lengths,
            )

            pair_loss = seq_tensor.new_zeros(())
            weighted_pair_loss = seq_tensor.new_zeros(())
            listwise_loss = seq_tensor.new_zeros(())
            weighted_listwise_loss = seq_tensor.new_zeros(())
            selection_loss = seq_tensor.new_zeros(())
            weighted_selection_loss = seq_tensor.new_zeros(())
            budget_loss = seq_tensor.new_zeros(())
            weighted_budget_loss = seq_tensor.new_zeros(())
            num_supervised_shots = 0.0
            num_positive_shots = 0.0
            num_negative_shots = 0.0

            if args.rank_loss == 'sparse_pair':
                shot_text_feat, shot_mass_density, valid_shots = build_shot_text_stats(
                    caption_spans_idx=caption_spans_idx_tensor,
                    caption_valid_mask=caption_valid_mask_tensor,
                    all_text_features=all_text_features_tensor,
                    overlaps=overlaps,
                    shot_lengths=shot_lengths,
                )

                shot_change, change_valid_mask = compute_shot_semantic_change(
                    shot_text_feat=shot_text_feat,
                    valid_shots=valid_shots,
                )

                pos_idx, neg_idx = mine_sparse_shot_pairs(
                    shot_change=shot_change,
                    shot_mass_density=shot_mass_density,
                    change_valid_mask=change_valid_mask,
                    top_ratio=0.2,
                )

                pair_loss = compute_sparse_pair_rank_loss(
                    pred_shot_scores=pred_shot_scores,
                    pos_idx=pos_idx,
                    neg_idx=neg_idx,
                    margin=args.pair_margin,
                )
                weighted_pair_loss = args.lambda_pair * pair_loss

                assert_finite_tensor('shot_change', shot_change, key)
                assert_finite_tensor('pair_loss', pair_loss.unsqueeze(0), key)
                assert_finite_tensor('weighted_pair_loss', weighted_pair_loss.unsqueeze(0), key)

            elif args.rank_loss == 'listwise_utility':
                if shot_utility_store is None:
                    raise RuntimeError('shot_utility_store is required for listwise_utility rank loss.')

                h5_key = Path(str(key)).name
                teacher_utility_np = shot_utility_store.get(
                    h5_key=h5_key,
                    formula_name=args.utility_formula,
                )
                teacher_utility = torch.tensor(
                    teacher_utility_np,
                    dtype=torch.float32,
                    device=args.device,
                )

                if teacher_utility.shape[0] != pred_shot_scores.shape[0]:
                    raise ValueError(
                        f'Shot utility length mismatch for sample {key}: '
                        f'utility={teacher_utility.shape[0]} vs pred_shot_scores={pred_shot_scores.shape[0]}'
                    )

                listwise_loss = compute_listwise_utility_loss(
                    pred_shot_scores=pred_shot_scores,
                    teacher_utility=teacher_utility,
                    temperature=args.listwise_temperature,
                )
                weighted_listwise_loss = args.lambda_listwise * listwise_loss

                assert_finite_tensor('teacher_utility', teacher_utility, key)
                assert_finite_tensor('listwise_loss', listwise_loss.unsqueeze(0), key)
                assert_finite_tensor('weighted_listwise_loss', weighted_listwise_loss.unsqueeze(0), key)

            elif args.rank_loss == 'budgeted_pseudo_summary':
                if shot_utility_store is None:
                    raise RuntimeError('shot_utility_store is required for budgeted_pseudo_summary.')

                h5_key = Path(str(key)).name
                masks = shot_utility_store.get_budgeted_masks(
                    h5_key=h5_key,
                    formula_name=args.utility_formula,
                    cps=cps_np,
                    nfps=nfps_np,
                    n_frames=n_frames_int,
                    summary_budget=args.summary_budget,
                    negative_quantile=args.negative_quantile,
                )

                target = torch.tensor(masks['target'], dtype=torch.float32, device=args.device)
                supervised_mask = torch.tensor(masks['supervised_mask'], dtype=torch.bool, device=args.device)
                selected_mask = torch.tensor(masks['selected_mask'], dtype=torch.bool, device=args.device)
                negative_mask = torch.tensor(masks['negative_mask'], dtype=torch.bool, device=args.device)

                if target.shape[0] != selection_shot_scores.shape[0]:
                    raise ValueError(
                        f'Pseudo-summary target length mismatch for sample {key}: '
                        f'target={target.shape[0]} vs pred={selection_shot_scores.shape[0]}'
                    )

                selection_loss = compute_confidence_gated_bce_loss(
                    pred_shot_scores=selection_shot_scores,
                    target=target,
                    supervised_mask=supervised_mask,
                )
                budget_loss = compute_budget_regularizer(
                    selection_shot_scores=selection_shot_scores,
                    shot_lengths=shot_lengths,
                    n_frames=n_frames_int,
                    summary_budget=args.summary_budget,
                )

                weighted_selection_loss = args.lambda_select * selection_loss
                weighted_budget_loss = args.lambda_budget * budget_loss

                num_supervised_shots = float(supervised_mask.sum().item())
                num_positive_shots = float(selected_mask.sum().item())
                num_negative_shots = float(negative_mask.sum().item())

                assert_finite_tensor('selection_shot_scores', selection_shot_scores, key)
                assert_finite_tensor('selection_loss', selection_loss.unsqueeze(0), key)
                assert_finite_tensor('budget_loss', budget_loss.unsqueeze(0), key)

            elif args.rank_loss == 'none':
                pass
            else:
                raise ValueError(f'Unknown rank_loss: {args.rank_loss}')

            align_loss = compute_align_loss(summary_feat, text_target_tensor)
            weighted_align_loss = args.lambda_align * align_loss

            assert_finite_tensor('summary_scores', summary_scores, key)
            assert_finite_tensor('bag_logits', bag_logits, key)
            assert_finite_tensor('summary_feat', summary_feat, key)
            assert_finite_tensor('pred_shot_scores', pred_shot_scores, key)
            assert_finite_tensor('align_loss', align_loss.unsqueeze(0), key)
            assert_finite_tensor('weighted_align_loss', weighted_align_loss.unsqueeze(0), key)

            if is_effective:
                bag_scores = torch.sigmoid(bag_logits)
                bag_loss = F.smooth_l1_loss(bag_scores, normalized_target)
                weighted_bag_loss = args.lambda_aux * bag_loss

                assert_finite_tensor('normalized_target', normalized_target, key)
                assert_finite_tensor('bag_scores', bag_scores, key)
                assert_finite_tensor('bag_loss', bag_loss.unsqueeze(0), key)
                assert_finite_tensor('weighted_bag_loss', weighted_bag_loss.unsqueeze(0), key)

                num_aux_active = 1.0
                num_aux_skipped = 0.0
            else:
                bag_loss = seq_tensor.new_zeros(())
                weighted_bag_loss = seq_tensor.new_zeros(())
                num_aux_active = 0.0
                num_aux_skipped = 1.0

            weighted_rank_loss = (
                weighted_pair_loss
                + weighted_listwise_loss
                + weighted_selection_loss
                + weighted_budget_loss
            )
            rank_loss_value = pair_loss + listwise_loss + selection_loss + budget_loss

            loss = weighted_align_loss + weighted_bag_loss + weighted_rank_loss

            optimizer.zero_grad()
            assert_finite_tensor('loss', loss.unsqueeze(0), key)
            loss.backward()
            optimizer.step()

            stats.update(
                loss=float(loss.item()),
                rank_loss=float(rank_loss_value.item()),
                weighted_rank_loss=float(weighted_rank_loss.item()),
                pair_loss=float(pair_loss.item()),
                weighted_pair_loss=float(weighted_pair_loss.item()),
                listwise_loss=float(listwise_loss.item()),
                weighted_listwise_loss=float(weighted_listwise_loss.item()),
                selection_loss=float(selection_loss.item()),
                weighted_selection_loss=float(weighted_selection_loss.item()),
                budget_loss=float(budget_loss.item()),
                weighted_budget_loss=float(weighted_budget_loss.item()),
                align_loss=float(align_loss.item()),
                weighted_align_loss=float(weighted_align_loss.item()),
                bag_loss=float(bag_loss.item()),
                weighted_bag_loss=float(weighted_bag_loss.item()),
                num_aux_active=num_aux_active,
                num_aux_skipped=num_aux_skipped,
                num_supervised_shots=num_supervised_shots,
                num_positive_shots=num_positive_shots,
                num_negative_shots=num_negative_shots,
            )

        val_metrics = evaluate_mil_cond(
            model=model,
            val_loader=val_loader,
            device=args.device,
        )
        val_fscore = float(val_metrics['fscore'])
        val_kendall = float(val_metrics['kendall'])
        val_spearman = float(val_metrics['spearman'])

        if val_kendall > max_val_kendall:
            max_val_kendall = val_kendall
            fscore_at_max_kendall = val_fscore
            spearman_at_max_kendall = val_spearman
            torch.save(model.state_dict(), kendall_save_path)

        if val_spearman > max_val_spearman:
            max_val_spearman = val_spearman
            fscore_at_max_spearman = val_fscore
            kendall_at_max_spearman = val_kendall
            torch.save(model.state_dict(), spearman_save_path)

        if val_fscore > best_val_fscore:
            best_val_fscore = val_fscore
            kendall_at_best_fscore = val_kendall
            spearman_at_best_fscore = val_spearman
            torch.save(model.state_dict(), save_path)

        logger.info(
            'Epoch %03d/%03d | loss=%.4f | rank=%.4f | val_F1=%.4f | '
            'val_Tau=%.4f | val_Rho=%.4f | best_val_F1=%.4f',
            epoch + 1,
            args.max_epoch,
            stats.loss,
            stats.weighted_rank_loss,
            val_fscore,
            val_kendall,
            val_spearman,
            best_val_fscore,
        )

        logger.debug(
            'Epoch %03d/%03d detail | pair=%.4f/%.4f | listwise=%.4f/%.4f | '
            'select=%.4f/%.4f | budget=%.4f/%.4f | '
            'align=%.4f/%.4f | bag=%.4f/%.4f | '
            'aux_active=%.4f | aux_skipped=%.4f | '
            'sup_shots=%.4f | pos=%.4f | neg=%.4f | '
            'val_max_Tau=%.4f | val_max_Rho=%.4f | '
            'Tau@best_F1=%.4f | Rho@best_F1=%.4f | F1@max_Tau=%.4f | F1@max_Rho=%.4f',
            epoch + 1,
            args.max_epoch,
            stats.pair_loss,
            stats.weighted_pair_loss,
            stats.listwise_loss,
            stats.weighted_listwise_loss,
            stats.selection_loss,
            stats.weighted_selection_loss,
            stats.budget_loss,
            stats.weighted_budget_loss,
            stats.align_loss,
            stats.weighted_align_loss,
            stats.bag_loss,
            stats.weighted_bag_loss,
            stats.num_aux_active,
            stats.num_aux_skipped,
            stats.num_supervised_shots,
            stats.num_positive_shots,
            stats.num_negative_shots,
            max_val_kendall,
            max_val_spearman,
            kendall_at_best_fscore,
            spearman_at_best_fscore,
            fscore_at_max_kendall,
            fscore_at_max_spearman,
        )

    test_at_best_fscore = evaluate_checkpoint(model, save_path, test_loader, args.device)
    test_at_max_kendall = evaluate_checkpoint(model, kendall_save_path, test_loader, args.device)
    test_at_max_spearman = evaluate_checkpoint(model, spearman_save_path, test_loader, args.device)

    return {
        'val_best_fscore': float(best_val_fscore),
        'val_kendall_at_best_fscore': float(kendall_at_best_fscore),
        'val_spearman_at_best_fscore': float(spearman_at_best_fscore),
        'val_max_kendall': float(max_val_kendall),
        'val_fscore_at_max_kendall': float(fscore_at_max_kendall),
        'val_spearman_at_max_kendall': float(spearman_at_max_kendall),
        'val_max_spearman': float(max_val_spearman),
        'val_fscore_at_max_spearman': float(fscore_at_max_spearman),
        'val_kendall_at_max_spearman': float(kendall_at_max_spearman),
        'test_fscore_at_best_fscore': float(test_at_best_fscore['fscore']),
        'test_kendall_at_best_fscore': float(test_at_best_fscore['kendall']),
        'test_spearman_at_best_fscore': float(test_at_best_fscore['spearman']),
        'test_fscore_at_max_kendall': float(test_at_max_kendall['fscore']),
        'test_kendall_at_max_kendall': float(test_at_max_kendall['kendall']),
        'test_spearman_at_max_kendall': float(test_at_max_kendall['spearman']),
        'test_fscore_at_max_spearman': float(test_at_max_spearman['fscore']),
        'test_kendall_at_max_spearman': float(test_at_max_spearman['kendall']),
        'test_spearman_at_max_spearman': float(test_at_max_spearman['spearman']),
    }


def validate_rank_loss_args(args) -> None:
    if args.score_head not in ('single', 'dual'):
        raise ValueError(f'Invalid score_head={args.score_head}; expected single or dual.')

    if args.rank_loss == 'sparse_pair':
        if args.lambda_pair < 0:
            raise ValueError(f'Invalid lambda_pair={args.lambda_pair}; expected >= 0.')
        if args.pair_margin <= 0:
            raise ValueError(f'Invalid pair_margin={args.pair_margin}; expected > 0.')
    elif args.rank_loss == 'listwise_utility':
        if args.lambda_listwise < 0:
            raise ValueError(f'Invalid lambda_listwise={args.lambda_listwise}; expected >= 0.')
        if args.listwise_temperature <= 0:
            raise ValueError(
                f'Invalid listwise_temperature={args.listwise_temperature}; expected > 0.'
            )
    elif args.rank_loss == 'budgeted_pseudo_summary':
        if args.score_head != 'dual':
            raise ValueError(
                '--rank-loss budgeted_pseudo_summary requires --score-head dual.'
            )
        if args.lambda_select < 0:
            raise ValueError(f'Invalid lambda_select={args.lambda_select}; expected >= 0.')
        if args.lambda_budget < 0:
            raise ValueError(f'Invalid lambda_budget={args.lambda_budget}; expected >= 0.')
        if not (0.0 < args.summary_budget < 1.0):
            raise ValueError(f'Invalid summary_budget={args.summary_budget}; expected 0 < budget < 1.')
        if not (0.0 < args.negative_quantile < 1.0):
            raise ValueError(f'Invalid negative_quantile={args.negative_quantile}; expected 0 < q < 1.')
    elif args.rank_loss == 'none':
        pass
    else:
        raise ValueError(
            f'Invalid rank_loss={args.rank_loss}; '
            f'expected sparse_pair, listwise_utility, budgeted_pseudo_summary, or none.'
        )


def compute_listwise_utility_loss(pred_shot_scores: torch.Tensor,
                                  teacher_utility: torch.Tensor,
                                  temperature: float,
                                  eps: float = 1e-8) -> torch.Tensor:
    if pred_shot_scores.ndim != 1:
        raise ValueError(
            f'Expected pred_shot_scores shape [S], got {tuple(pred_shot_scores.shape)}'
        )
    if teacher_utility.ndim != 1:
        raise ValueError(
            f'Expected teacher_utility shape [S], got {tuple(teacher_utility.shape)}'
        )
    if pred_shot_scores.shape[0] != teacher_utility.shape[0]:
        raise ValueError(
            f'Shot count mismatch in compute_listwise_utility_loss: '
            f'pred={tuple(pred_shot_scores.shape)} vs teacher={tuple(teacher_utility.shape)}'
        )
    if pred_shot_scores.shape[0] < 2:
        return pred_shot_scores.new_zeros(())
    if not torch.isfinite(pred_shot_scores).all():
        raise ValueError('Non-finite pred_shot_scores in compute_listwise_utility_loss.')
    if not torch.isfinite(teacher_utility).all():
        raise ValueError('Non-finite teacher_utility in compute_listwise_utility_loss.')
    if temperature <= 0:
        raise ValueError(f'Invalid temperature={temperature}; expected > 0.')

    teacher_range = torch.max(teacher_utility) - torch.min(teacher_utility)
    if float(teacher_range.item()) <= eps:
        return pred_shot_scores.new_zeros(())

    student_prob = pred_shot_scores.clamp_min(eps)
    student_prob = student_prob / student_prob.sum().clamp_min(eps)
    student_log_prob = torch.log(student_prob)

    teacher_prob = torch.softmax(teacher_utility / temperature, dim=0).detach()

    return F.kl_div(
        student_log_prob,
        teacher_prob,
        reduction='sum',
    )


def compute_confidence_gated_bce_loss(pred_shot_scores: torch.Tensor,
                                      target: torch.Tensor,
                                      supervised_mask: torch.Tensor,
                                      eps: float = 1e-6) -> torch.Tensor:
    if pred_shot_scores.ndim != 1:
        raise ValueError(f'Expected pred_shot_scores shape [S], got {tuple(pred_shot_scores.shape)}')
    if target.ndim != 1:
        raise ValueError(f'Expected target shape [S], got {tuple(target.shape)}')
    if supervised_mask.ndim != 1:
        raise ValueError(f'Expected supervised_mask shape [S], got {tuple(supervised_mask.shape)}')
    if pred_shot_scores.shape[0] != target.shape[0] or pred_shot_scores.shape[0] != supervised_mask.shape[0]:
        raise ValueError('Shot count mismatch in compute_confidence_gated_bce_loss.')
    if not torch.isfinite(pred_shot_scores).all():
        raise ValueError('Non-finite pred_shot_scores in compute_confidence_gated_bce_loss.')
    if not torch.isfinite(target).all():
        raise ValueError('Non-finite target in compute_confidence_gated_bce_loss.')

    if int(supervised_mask.sum().item()) == 0:
        return pred_shot_scores.new_zeros(())

    pred = pred_shot_scores[supervised_mask].clamp(min=eps, max=1.0 - eps)
    tgt = target[supervised_mask].clamp(min=0.0, max=1.0)

    return F.binary_cross_entropy(pred, tgt)


def compute_budget_regularizer(selection_shot_scores: torch.Tensor,
                               shot_lengths: torch.Tensor,
                               n_frames: int,
                               summary_budget: float) -> torch.Tensor:
    if selection_shot_scores.ndim != 1:
        raise ValueError(
            f'Expected selection_shot_scores shape [S], got {tuple(selection_shot_scores.shape)}'
        )
    if shot_lengths.ndim != 1:
        raise ValueError(f'Expected shot_lengths shape [S], got {tuple(shot_lengths.shape)}')
    if selection_shot_scores.shape[0] != shot_lengths.shape[0]:
        raise ValueError('Shot count mismatch in compute_budget_regularizer.')
    if n_frames <= 0:
        raise ValueError(f'Invalid n_frames={n_frames}')

    predicted_budget_ratio = torch.sum(selection_shot_scores * shot_lengths) / float(n_frames)
    target_budget = predicted_budget_ratio.new_tensor(float(summary_budget))
    return F.smooth_l1_loss(predicted_budget_ratio, target_budget)


def evaluate_checkpoint(model, ckpt_path, test_loader, device: str):
    state_dict = torch.load(str(ckpt_path), map_location=device)
    model.load_state_dict(state_dict)
    return evaluate_mil_cond(model=model, val_loader=test_loader, device=device)


def infer_single_dataset_name(keys):
    names = {infer_dataset_name_from_key(key) for key in keys}
    if len(names) != 1:
        raise ValueError(f'Expected a single dataset in one run, got: {sorted(names)}')
    return next(iter(names))


def infer_dataset_name_from_key(key: str) -> str:
    key_lower = str(key).lower()
    if 'tvsum' in key_lower:
        return 'tvsum'
    if 'summe' in key_lower:
        return 'summe'
    raise ValueError(f'Cannot infer dataset name from key: {key}')


def infer_num_classes(dataset) -> int:
    if len(dataset) == 0:
        raise ValueError('Cannot infer num_classes from empty dataset.')
    sample = dataset[0]
    soft_label = np.asarray(sample[2], dtype=np.float32)
    if soft_label.ndim != 1:
        raise ValueError(f'Invalid soft_label shape: {soft_label.shape}')
    return int(soft_label.shape[0])


def normalize_soft_label(soft_label: torch.Tensor, eps: float = 1e-8):
    label_min = torch.min(soft_label)
    label_max = torch.max(soft_label)
    label_range = label_max - label_min

    if float(label_range.item()) < eps:
        normalized = torch.zeros_like(soft_label)
        return normalized, False

    normalized = (soft_label - label_min) / (label_range + eps)
    return normalized, True


def assert_finite_tensor(name: str, x: torch.Tensor, key: str) -> None:
    if not torch.isfinite(x).all():
        num_nan = int(torch.isnan(x).sum().item())
        num_inf = int(torch.isinf(x).sum().item())
        raise ValueError(
            f'Non-finite tensor detected: {name} | sample={key} | '
            f'nan={num_nan} | inf={num_inf} | shape={tuple(x.shape)}'
        )


def compute_align_loss(summary_feat: torch.Tensor,
                       text_target: torch.Tensor) -> torch.Tensor:
    if summary_feat.ndim != 1:
        raise ValueError(f'Expected summary_feat shape [D], got {tuple(summary_feat.shape)}')
    if text_target.ndim != 1:
        raise ValueError(f'Expected text_target shape [D], got {tuple(text_target.shape)}')
    if summary_feat.shape[0] != text_target.shape[0]:
        raise ValueError(
            f'Feature dim mismatch in compute_align_loss: '
            f'{summary_feat.shape[0]} vs {text_target.shape[0]}'
        )

    cosine = F.cosine_similarity(
        summary_feat.unsqueeze(0),
        text_target.unsqueeze(0),
        dim=-1,
    ).squeeze(0)

    return 1.0 - cosine


def build_sampled_to_shot_overlap(picks: torch.Tensor,
                                  cps: torch.Tensor,
                                  n_frames: int):
    if picks.ndim != 1:
        raise ValueError(f'Expected picks shape [T], got {tuple(picks.shape)}')
    if cps.ndim != 2 or cps.shape[1] != 2:
        raise ValueError(f'Expected cps shape [S, 2], got {tuple(cps.shape)}')
    if n_frames <= 0:
        raise ValueError(f'Invalid n_frames: {n_frames}')

    picks = picks.to(torch.long)
    cps = cps.to(torch.long)

    lo = picks
    hi = torch.empty_like(lo)
    hi[:-1] = picks[1:]
    hi[-1] = int(n_frames)

    overlaps = []
    for s in range(cps.shape[0]):
        first = int(cps[s, 0].item())
        last_exclusive = int(cps[s, 1].item()) + 1

        first_t = lo.new_tensor(first)
        last_t = lo.new_tensor(last_exclusive)

        inter = torch.minimum(hi, last_t) - torch.maximum(lo, first_t)
        inter = torch.clamp(inter, min=0).to(torch.float32)
        overlaps.append(inter)

    overlaps = torch.stack(overlaps, dim=0)  # [S, T]
    shot_lengths = overlaps.sum(dim=1)       # [S]

    if not torch.all(shot_lengths > 0):
        raise ValueError('Detected non-positive shot length in build_sampled_to_shot_overlap.')

    return overlaps, shot_lengths


def aggregate_attn_to_shot_scores(attn_weights: torch.Tensor,
                                  overlaps: torch.Tensor,
                                  shot_lengths: torch.Tensor,
                                  eps: float = 1e-8) -> torch.Tensor:
    if attn_weights.ndim != 1:
        raise ValueError(f'Expected attn_weights shape [T], got {tuple(attn_weights.shape)}')
    if overlaps.ndim != 2:
        raise ValueError(f'Expected overlaps shape [S, T], got {tuple(overlaps.shape)}')
    if shot_lengths.ndim != 1:
        raise ValueError(f'Expected shot_lengths shape [S], got {tuple(shot_lengths.shape)}')
    if overlaps.shape[1] != attn_weights.shape[0]:
        raise ValueError(
            f'Time-step mismatch in aggregate_attn_to_shot_scores: '
            f'overlaps={tuple(overlaps.shape)} vs attn_weights={tuple(attn_weights.shape)}'
        )
    if overlaps.shape[0] != shot_lengths.shape[0]:
        raise ValueError(
            f'Shot count mismatch in aggregate_attn_to_shot_scores: '
            f'overlaps={tuple(overlaps.shape)} vs shot_lengths={tuple(shot_lengths.shape)}'
        )

    shot_scores = torch.matmul(overlaps, attn_weights) / shot_lengths.clamp_min(1.0)
    shot_scores = shot_scores.clamp_min(0.0)

    total = shot_scores.sum()
    if float(total.item()) <= eps:
        raise ValueError('Predicted shot scores have non-positive sum.')

    return shot_scores / total


def aggregate_frame_scores_to_shot_scores(frame_scores: torch.Tensor,
                                          overlaps: torch.Tensor,
                                          shot_lengths: torch.Tensor) -> torch.Tensor:
    if frame_scores.ndim != 1:
        raise ValueError(f'Expected frame_scores shape [T], got {tuple(frame_scores.shape)}')
    if overlaps.ndim != 2:
        raise ValueError(f'Expected overlaps shape [S, T], got {tuple(overlaps.shape)}')
    if shot_lengths.ndim != 1:
        raise ValueError(f'Expected shot_lengths shape [S], got {tuple(shot_lengths.shape)}')
    if overlaps.shape[1] != frame_scores.shape[0]:
        raise ValueError(
            f'Time-step mismatch in aggregate_frame_scores_to_shot_scores: '
            f'overlaps={tuple(overlaps.shape)} vs frame_scores={tuple(frame_scores.shape)}'
        )

    return torch.matmul(overlaps, frame_scores) / shot_lengths.clamp_min(1.0)


def build_shot_text_stats(caption_spans_idx: torch.Tensor,
                          caption_valid_mask: torch.Tensor,
                          all_text_features: torch.Tensor,
                          overlaps: torch.Tensor,
                          shot_lengths: torch.Tensor,
                          eps: float = 1e-8):
    if caption_spans_idx.ndim != 2 or caption_spans_idx.shape[1] != 2:
        raise ValueError(
            f'Expected caption_spans_idx shape [K, 2], got {tuple(caption_spans_idx.shape)}'
        )
    if caption_valid_mask.ndim != 1:
        raise ValueError(
            f'Expected caption_valid_mask shape [K], got {tuple(caption_valid_mask.shape)}'
        )
    if all_text_features.ndim != 2:
        raise ValueError(
            f'Expected all_text_features shape [K, D], got {tuple(all_text_features.shape)}'
        )
    if overlaps.ndim != 2:
        raise ValueError(f'Expected overlaps shape [S, T], got {tuple(overlaps.shape)}')
    if shot_lengths.ndim != 1:
        raise ValueError(f'Expected shot_lengths shape [S], got {tuple(shot_lengths.shape)}')
    if caption_spans_idx.shape[0] != caption_valid_mask.shape[0]:
        raise ValueError('Caption count mismatch in build_shot_text_stats.')
    if caption_spans_idx.shape[0] != all_text_features.shape[0]:
        raise ValueError('Caption/text count mismatch in build_shot_text_stats.')

    num_shots, num_steps = overlaps.shape
    feat_dim = all_text_features.shape[1]

    shot_text_sum = overlaps.new_zeros((num_shots, feat_dim))
    shot_text_mass = overlaps.new_zeros(num_shots)

    for k in range(caption_spans_idx.shape[0]):
        if float(caption_valid_mask[k].item()) <= 0.5:
            continue

        start_idx = int(caption_spans_idx[k, 0].item())
        end_idx = int(caption_spans_idx[k, 1].item())

        start_idx = max(0, min(start_idx, num_steps - 1))
        end_idx = max(start_idx, min(end_idx, num_steps - 1))

        shot_overlap = overlaps[:, start_idx:end_idx + 1].sum(dim=1)
        overlap_sum = shot_overlap.sum()
        if float(overlap_sum.item()) <= eps:
            continue

        weights = shot_overlap / overlap_sum
        shot_text_sum = shot_text_sum + weights.unsqueeze(1) * all_text_features[k].unsqueeze(0)
        shot_text_mass = shot_text_mass + weights

    valid_shots = shot_text_mass > eps
    shot_text_feat = overlaps.new_zeros((num_shots, feat_dim))

    if valid_shots.any():
        shot_text_avg = shot_text_sum[valid_shots] / shot_text_mass[valid_shots].unsqueeze(1).clamp_min(eps)
        shot_text_feat[valid_shots] = F.normalize(shot_text_avg, p=2, dim=1)

    shot_mass_density = shot_text_mass / shot_lengths.clamp_min(1.0)
    return shot_text_feat, shot_mass_density, valid_shots


def compute_shot_semantic_change(shot_text_feat: torch.Tensor,
                                 valid_shots: torch.Tensor):
    if shot_text_feat.ndim != 2:
        raise ValueError(f'Expected shot_text_feat shape [S, D], got {tuple(shot_text_feat.shape)}')
    if valid_shots.ndim != 1:
        raise ValueError(f'Expected valid_shots shape [S], got {tuple(valid_shots.shape)}')
    if shot_text_feat.shape[0] != valid_shots.shape[0]:
        raise ValueError('Shot count mismatch in compute_shot_semantic_change.')

    num_shots = shot_text_feat.shape[0]
    shot_change = shot_text_feat.new_zeros(num_shots)
    change_valid_mask = torch.zeros_like(valid_shots, dtype=torch.bool)

    for s in range(num_shots):
        if not bool(valid_shots[s].item()):
            continue

        diffs = []

        if s - 1 >= 0 and bool(valid_shots[s - 1].item()):
            cos_prev = torch.sum(shot_text_feat[s] * shot_text_feat[s - 1])
            diffs.append(1.0 - cos_prev)

        if s + 1 < num_shots and bool(valid_shots[s + 1].item()):
            cos_next = torch.sum(shot_text_feat[s] * shot_text_feat[s + 1])
            diffs.append(1.0 - cos_next)

        if diffs:
            shot_change[s] = torch.stack(diffs).mean()
            change_valid_mask[s] = True

    return shot_change, change_valid_mask


def remove_indices(candidates: torch.Tensor,
                   exclude: torch.Tensor) -> torch.Tensor:
    if candidates.ndim != 1:
        raise ValueError(f'Expected candidates shape [N], got {tuple(candidates.shape)}')
    if exclude.ndim != 1:
        raise ValueError(f'Expected exclude shape [M], got {tuple(exclude.shape)}')

    if exclude.numel() == 0:
        return candidates

    keep = torch.ones(candidates.shape[0], dtype=torch.bool, device=candidates.device)
    for idx in exclude:
        keep = keep & (candidates != idx)

    return candidates[keep]


def mine_sparse_shot_pairs(shot_change: torch.Tensor,
                           shot_mass_density: torch.Tensor,
                           change_valid_mask: torch.Tensor,
                           top_ratio: float = 0.2):
    if shot_change.ndim != 1:
        raise ValueError(f'Expected shot_change shape [S], got {tuple(shot_change.shape)}')
    if shot_mass_density.ndim != 1:
        raise ValueError(f'Expected shot_mass_density shape [S], got {tuple(shot_mass_density.shape)}')
    if change_valid_mask.ndim != 1:
        raise ValueError(f'Expected change_valid_mask shape [S], got {tuple(change_valid_mask.shape)}')
    if shot_change.shape[0] != shot_mass_density.shape[0] or shot_change.shape[0] != change_valid_mask.shape[0]:
        raise ValueError('Shot count mismatch in mine_sparse_shot_pairs.')

    candidate_mask = change_valid_mask & (shot_mass_density > 0)
    candidate_idx = torch.where(candidate_mask)[0]

    if candidate_idx.numel() < 2:
        empty = torch.empty(0, dtype=torch.long, device=shot_change.device)
        return empty, empty

    k = max(1, int(round(candidate_idx.numel() * top_ratio)))
    k = min(k, candidate_idx.numel())

    candidate_change = shot_change[candidate_idx]
    pos_local = torch.topk(candidate_change, k=k, largest=True).indices
    pos_idx = candidate_idx[pos_local]

    mass_thr = torch.median(shot_mass_density[candidate_idx])
    neg_candidate = candidate_idx[shot_mass_density[candidate_idx] >= mass_thr]
    neg_candidate = remove_indices(neg_candidate, pos_idx)

    if neg_candidate.numel() == 0:
        neg_candidate = remove_indices(candidate_idx, pos_idx)

    if neg_candidate.numel() == 0:
        empty = torch.empty(0, dtype=torch.long, device=shot_change.device)
        return empty, empty

    neg_k = max(1, min(k, neg_candidate.numel()))
    neg_change = shot_change[neg_candidate]
    neg_local = torch.topk(neg_change, k=neg_k, largest=False).indices
    neg_idx = neg_candidate[neg_local]

    return pos_idx, neg_idx


def compute_sparse_pair_rank_loss(pred_shot_scores: torch.Tensor,
                                  pos_idx: torch.Tensor,
                                  neg_idx: torch.Tensor,
                                  margin: float) -> torch.Tensor:
    if pred_shot_scores.ndim != 1:
        raise ValueError(
            f'Expected pred_shot_scores shape [S], got {tuple(pred_shot_scores.shape)}'
        )
    if pos_idx.ndim != 1:
        raise ValueError(f'Expected pos_idx shape [P], got {tuple(pos_idx.shape)}')
    if neg_idx.ndim != 1:
        raise ValueError(f'Expected neg_idx shape [N], got {tuple(neg_idx.shape)}')

    if pos_idx.numel() == 0 or neg_idx.numel() == 0:
        return pred_shot_scores.new_zeros(())

    pos_scores = pred_shot_scores[pos_idx].unsqueeze(1)
    neg_scores = pred_shot_scores[neg_idx].unsqueeze(0)

    return F.relu(margin - pos_scores + neg_scores).mean()
