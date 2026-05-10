import logging
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from anchor_free.dsnet_af_mil_cond import DSNetAFMILCond
from evaluate_mil_cond import evaluate_mil_cond
from modules.diff_budget import BudgetedSigmoid
from modules.shot_context_encoder import (
    build_context_shot_text_stats,
    build_shot_time_features,
)
from helpers import data_helper, mil_data_helper_cond
from helpers.preference_teacher_helper import PreferenceTeacherStore
from helpers.shot_utility_helper import (
    ShotUtilityStore,
    resolve_shot_utility_path,
)

logger = logging.getLogger(__name__)


def train(args, split, save_path):
    logger.debug(
        'Loss weights | score_head=%s | selection_score_source=%s | '
        'shot_head_mode=%s | shot_eval_head=%s | rank_loss=%s | '
        'lambda_pair=%s | pair_margin=%s | '
        'lambda_listwise=%s | listwise_temperature=%s | '
        'lambda_select=%s | lambda_budget=%s | summary_budget=%s | neg_q=%s | '
        'gate=%s | margin_thr=%s | utility_formula=%s | lambda_align=%s | lambda_aux=%s | '
        'coverage_aware=%s | coverage_min=%s | text_feature_path=%s | '
        'structured_caption_path=%s | shot_utility_path=%s | diff_budget=%s | diff_tau=%s | lambda_delta=%s',
        args.score_head,
        args.selection_score_source,
        args.shot_head_mode,
        args.shot_eval_head,
        args.rank_loss,
        args.lambda_pair,
        args.pair_margin,
        args.lambda_listwise,
        args.listwise_temperature,
        args.lambda_select,
        args.lambda_budget,
        args.summary_budget,
        args.negative_quantile,
        args.teacher_gate_mode,
        args.teacher_margin_threshold,
        args.utility_formula,
        args.lambda_align,
        args.lambda_aux,
        args.caption_coverage_aware,
        args.coverage_loss_min_weight,
        args.text_feature_path,
        args.structured_caption_path,
        args.shot_utility_path,
        args.use_diff_budget_selector,
        args.diff_budget_tau,
        args.lambda_delta,
    )
    logger.debug(
        'Preference distill config | path=%s | lambda_pair=%s | lambda_list=%s | '
        'lambda_inclusion=%s | lambda_budget=%s | conf_thr=%s | margin=%s | tau=%s',
        args.preference_teacher_path,
        args.lambda_pref_pair,
        args.lambda_pref_list,
        args.lambda_pref_inclusion,
        args.lambda_pref_budget,
        args.pref_confidence_threshold,
        args.pref_pair_margin,
        args.pref_list_temperature,
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
    if args.rank_loss in ('listwise_utility', 'budgeted_pseudo_summary', 'hybrid_sparse_budget'):
        utility_path = resolve_shot_utility_path(
            dataset_name=dataset_name,
            explicit_path=args.shot_utility_path,
        )
        shot_utility_store = ShotUtilityStore(utility_path)
        logger.info(
            'Using utility-based supervision | rank_loss=%s | path=%s | formula=%s',
            args.rank_loss,
            utility_path,
            args.utility_formula,
        )

    preference_teacher_store = None
    if args.rank_loss == 'preference_distill':
        if not args.preference_teacher_path:
            raise ValueError('--rank-loss preference_distill requires --preference-teacher-path.')
        preference_teacher_store = PreferenceTeacherStore(Path(args.preference_teacher_path))
        logger.info(
            'Using preference-distilled supervision | path=%s | pair=%.3g | list=%.3g | inclusion=%.3g | budget=%.3g',
            args.preference_teacher_path,
            args.lambda_pref_pair,
            args.lambda_pref_list,
            args.lambda_pref_inclusion,
            args.lambda_pref_budget,
        )

    train_set = mil_data_helper_cond.VideoDatasetMILCond(
        train_keys,
        text_cond_num=args.text_cond_num,
        random_text_sampling=True,
        caption_coverage_aware=args.caption_coverage_aware,
        text_feature_path=args.text_feature_path,
        structured_caption_path=args.structured_caption_path,
    )
    val_set = mil_data_helper_cond.VideoDatasetMILCond(
        val_keys,
        text_cond_num=args.text_cond_num,
        random_text_sampling=False,
        caption_coverage_aware=args.caption_coverage_aware,
        text_feature_path=args.text_feature_path,
        structured_caption_path=args.structured_caption_path,
    )
    test_set = mil_data_helper_cond.VideoDatasetMILCond(
        test_keys,
        text_cond_num=args.text_cond_num,
        random_text_sampling=False,
        caption_coverage_aware=args.caption_coverage_aware,
        text_feature_path=args.text_feature_path,
        structured_caption_path=args.structured_caption_path,
    )

    num_classes = infer_num_classes(train_set)
    model = DSNetAFMILCond(
        base_model=args.base_model,
        num_feature=args.num_feature,
        num_hidden=args.num_hidden,
        num_head=args.num_head,
        num_classes=num_classes,
        score_head=args.score_head,
        use_shot_head=args.selection_score_source == 'shot_head',
        shot_head_mode=args.shot_head_mode,
    ).to(args.device)

    budget_selector = (
        BudgetedSigmoid(tau=args.diff_budget_tau).to(args.device)
        if args.use_diff_budget_selector
        else None
    )

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
            'delta_loss',
            'weighted_delta_loss',
            'align_loss',
            'weighted_align_loss',
            'bag_loss',
            'weighted_bag_loss',
            'num_aux_active',
            'num_aux_skipped',
            'num_supervised_shots',
            'num_positive_shots',
            'num_negative_shots',
            'teacher_gate_weight',
            'teacher_margin',
            'teacher_student_shot_tau',
            'caption_coverage_ratio',
            'coverage_loss_weight',
            'num_valid_text_cond',
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
            text_cond_mask,
            caption_coverage_ratio,
        ) in train_loader:
            seq_tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(args.device)
            text_cond_tensor = torch.tensor(text_cond, dtype=torch.float32).to(args.device)
            text_cond_mask_tensor = torch.tensor(text_cond_mask, dtype=torch.float32, device=args.device)
            text_target_tensor = torch.tensor(text_target, dtype=torch.float32).to(args.device)
            all_text_features_tensor = torch.tensor(all_text_features, dtype=torch.float32, device=args.device)
            caption_spans_idx_tensor = torch.tensor(caption_spans_idx, dtype=torch.long, device=args.device)
            caption_valid_mask_tensor = torch.tensor(caption_valid_mask, dtype=torch.float32, device=args.device)
            soft_label_tensor = torch.tensor(soft_label, dtype=torch.float32).to(args.device)

            cps_np = np.asarray(cps, dtype=np.int32)
            nfps_np = np.asarray(nfps, dtype=np.int32)
            cps_tensor = torch.tensor(cps_np, dtype=torch.long, device=args.device)
            picks_tensor = torch.tensor(picks, dtype=torch.long, device=args.device)
            n_frames_int = int(np.asarray(n_frames).item())

            assert_finite_tensor('seq_tensor', seq_tensor, key)
            assert_finite_tensor('text_cond_tensor', text_cond_tensor, key)
            assert_finite_tensor('text_cond_mask_tensor', text_cond_mask_tensor, key)
            assert_finite_tensor('text_target_tensor', text_target_tensor, key)
            assert_finite_tensor('all_text_features_tensor', all_text_features_tensor, key)

            caption_coverage_ratio_value = float(np.asarray(caption_coverage_ratio).item())
            coverage_loss_weight = compute_coverage_loss_weight(
                seq_tensor=seq_tensor,
                coverage_ratio=caption_coverage_ratio_value,
                enabled=args.caption_coverage_aware,
                min_weight=args.coverage_loss_min_weight,
            )
            num_valid_text_cond = float((text_cond_mask_tensor > 0.5).sum().item())

            normalized_target, is_effective = normalize_soft_label(soft_label_tensor)

            (
                instance_logits,
                pool_logits,
                summary_scores,
                bag_logits,
                summary_feat,
                pre_cross_frame_repr,
                cond_frame_repr,
                hidden_frame_repr,
            ) = model(
                seq_tensor,
                text_cond_tensor,
                text_cond_mask_tensor if args.caption_coverage_aware else None,
            )

            pool_weights = torch.softmax(pool_logits, dim=0)

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
            pool_shot_scores = aggregate_attn_to_shot_scores(
                attn_weights=pool_weights,
                overlaps=overlaps,
                shot_lengths=shot_lengths,
            )
            frame_selection_shot_scores = aggregate_frame_scores_to_shot_scores(
                frame_scores=summary_scores,
                overlaps=overlaps,
                shot_lengths=shot_lengths,
            )
            needs_shot_text_stats = (
                args.rank_loss in ('sparse_pair', 'hybrid_sparse_budget')
                or (args.selection_score_source == 'shot_head' and args.shot_head_mode == 'context')
            )
            shot_text_feat = None
            shot_mass_density = None
            valid_shots = None
            if needs_shot_text_stats:
                shot_text_feat, shot_mass_density, valid_shots = build_context_shot_text_stats(
                    caption_spans_idx=caption_spans_idx_tensor,
                    caption_valid_mask=caption_valid_mask_tensor,
                    all_text_features=all_text_features_tensor,
                    overlaps=overlaps,
                    shot_lengths=shot_lengths,
                )

            nfps_tensor = torch.tensor(nfps_np, dtype=torch.float32, device=args.device)
            shot_time_feat = None
            if args.selection_score_source == 'shot_head':
                if args.shot_head_mode == 'context':
                    shot_time_feat = build_shot_time_features(
                        cps=cps_tensor,
                        nfps=nfps_tensor,
                        n_frames=n_frames_int,
                    )
                    selection_shot_logits, ranking_shot_logits, delta_select = model.predict_shot_logit_pair(
                        frame_repr=cond_frame_repr,
                        overlaps=overlaps,
                        shot_lengths=shot_lengths,
                        shot_text_feat=shot_text_feat,
                        shot_time_feat=shot_time_feat,
                    )
                else:
                    selection_shot_logits, ranking_shot_logits, delta_select = model.predict_shot_logit_pair(
                        frame_repr=cond_frame_repr,
                        overlaps=overlaps,
                        shot_lengths=shot_lengths,
                    )
                selection_shot_scores = torch.sigmoid(selection_shot_logits)
                ranking_shot_scores = torch.sigmoid(ranking_shot_logits)
            else:
                selection_shot_scores = frame_selection_shot_scores
                ranking_shot_scores = selection_shot_scores
                selection_shot_logits = torch.logit(selection_shot_scores.clamp(1e-6, 1.0 - 1e-6))
                ranking_shot_logits = selection_shot_logits
                delta_select = torch.zeros_like(selection_shot_scores)

            # Ranking supervision should always use raw logits when shot_head is enabled.
            rank_supervision_scores = (
                ranking_shot_logits
                if args.selection_score_source == 'shot_head'
                else ranking_shot_scores
            )

            pair_loss = seq_tensor.new_zeros(())
            weighted_pair_loss = seq_tensor.new_zeros(())
            listwise_loss = seq_tensor.new_zeros(())
            weighted_listwise_loss = seq_tensor.new_zeros(())
            selection_loss = seq_tensor.new_zeros(())
            weighted_selection_loss = seq_tensor.new_zeros(())
            budget_loss = seq_tensor.new_zeros(())
            weighted_budget_loss = seq_tensor.new_zeros(())
            delta_loss = seq_tensor.new_zeros(())
            weighted_delta_loss = seq_tensor.new_zeros(())
            num_supervised_shots = 0.0
            num_positive_shots = 0.0
            num_negative_shots = 0.0
            teacher_gate_weight = 1.0
            teacher_margin = 0.0
            teacher_student_shot_tau = 0.0

            if args.rank_loss in ('sparse_pair', 'hybrid_sparse_budget'):
                if args.selection_score_source == 'shot_head':
                    sparse_scores = rank_supervision_scores
                else:
                    sparse_scores = (
                        pool_shot_scores
                        if args.rank_loss == 'hybrid_sparse_budget'
                        else pred_shot_scores
                    )

                if shot_text_feat is None or shot_mass_density is None or valid_shots is None:
                    raise RuntimeError('Sparse-pair supervision requires shot text stats.')

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
                    pred_shot_scores=sparse_scores,
                    pos_idx=pos_idx,
                    neg_idx=neg_idx,
                    margin=args.pair_margin,
                )
                weighted_pair_loss = args.lambda_pair * coverage_loss_weight * pair_loss

                assert_finite_tensor('shot_change', shot_change, key)
                assert_finite_tensor('pair_loss', pair_loss.unsqueeze(0), key)
                assert_finite_tensor('weighted_pair_loss', weighted_pair_loss.unsqueeze(0), key)

            if args.rank_loss == 'listwise_utility':
                if shot_utility_store is None:
                    raise RuntimeError('shot_utility_store is required for listwise_utility rank loss.')

                h5_key = Path(str(key)).name
                teacher_utility_np = shot_utility_store.get(h5_key=h5_key, formula_name=args.utility_formula)
                teacher_utility = torch.tensor(teacher_utility_np, dtype=torch.float32, device=args.device)

                if teacher_utility.shape[0] != pred_shot_scores.shape[0]:
                    raise ValueError(
                        f'Shot utility length mismatch for sample {key}: '
                        f'utility={teacher_utility.shape[0]} vs pred_shot_scores={pred_shot_scores.shape[0]}'
                    )

                listwise_loss = compute_listwise_utility_loss(
                    pred_shot_scores=(
                        rank_supervision_scores
                        if args.selection_score_source == 'shot_head'
                        else pred_shot_scores
                    ),
                    teacher_utility=teacher_utility,
                    temperature=args.listwise_temperature,
                )
                weighted_listwise_loss = args.lambda_listwise * coverage_loss_weight * listwise_loss

                assert_finite_tensor('teacher_utility', teacher_utility, key)
                assert_finite_tensor('listwise_loss', listwise_loss.unsqueeze(0), key)
                assert_finite_tensor('weighted_listwise_loss', weighted_listwise_loss.unsqueeze(0), key)

            if args.rank_loss in ('budgeted_pseudo_summary', 'hybrid_sparse_budget'):
                if shot_utility_store is None:
                    raise RuntimeError('shot_utility_store is required for budgeted pseudo-summary supervision.')

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

                utility = torch.tensor(masks['utility'], dtype=torch.float32, device=args.device)
                target = torch.tensor(masks['target'], dtype=torch.float32, device=args.device)
                supervised_mask = torch.tensor(masks['supervised_mask'], dtype=torch.bool, device=args.device)
                selected_mask = torch.tensor(masks['selected_mask'], dtype=torch.bool, device=args.device)
                negative_mask = torch.tensor(masks['negative_mask'], dtype=torch.bool, device=args.device)

                if target.shape[0] != selection_shot_scores.shape[0]:
                    raise ValueError(
                        f'Pseudo-summary target length mismatch for sample {key}: '
                        f'target={target.shape[0]} vs pred={selection_shot_scores.shape[0]}'
                    )

                teacher_margin_tensor = compute_teacher_margin(
                    utility=utility,
                    selected_mask=selected_mask,
                    negative_mask=negative_mask,
                )
                gate_weight_tensor = compute_teacher_gate_weight(
                    margin=teacher_margin_tensor,
                    threshold=args.teacher_margin_threshold,
                    mode=args.teacher_gate_mode,
                )

                if args.use_diff_budget_selector:
                    if budget_selector is None:
                        raise RuntimeError('budget_selector is not initialized.')
                    soft_selected_mask, _budget_lambda = budget_selector(
                        select_logits=selection_shot_logits,
                        lengths=nfps_tensor,
                        budget_frames=float(n_frames_int) * float(args.summary_budget),
                    )
                    raw_selection_loss = compute_confidence_gated_weighted_bce_loss(
                        pred_shot_scores=soft_selected_mask,
                        target=target,
                        supervised_mask=supervised_mask,
                        shot_lengths=nfps_tensor,
                    )
                    budget_loss = compute_budget_regularizer(
                        selection_shot_scores=soft_selected_mask,
                        shot_lengths=nfps_tensor,
                        n_frames=n_frames_int,
                        summary_budget=args.summary_budget,
                    )
                    delta_loss = compute_delta_regularizer(delta_select)
                    assert_finite_tensor('diff_budget_soft_mask', soft_selected_mask, key)
                else:
                    raw_selection_loss = compute_confidence_gated_weighted_bce_loss(
                        pred_shot_scores=selection_shot_scores,
                        target=target,
                        supervised_mask=supervised_mask,
                        shot_lengths=shot_lengths,
                    )
                    budget_loss = compute_budget_regularizer(
                        selection_shot_scores=selection_shot_scores,
                        shot_lengths=shot_lengths,
                        n_frames=n_frames_int,
                        summary_budget=args.summary_budget,
                    )

                # New: weak dense-order distill for shot-head under budgeted teacher.
                if (
                    args.rank_loss == 'budgeted_pseudo_summary'
                    and args.selection_score_source == 'shot_head'
                    and args.lambda_listwise > 0.0
                ):
                    listwise_loss = compute_listwise_utility_loss(
                        pred_shot_scores=rank_supervision_scores,
                        teacher_utility=utility,
                        temperature=args.listwise_temperature,
                    )
                    weighted_listwise_loss = (
                        args.lambda_listwise * coverage_loss_weight * listwise_loss
                    )
                    teacher_student_shot_tau = compute_teacher_student_shot_tau(
                        student_scores=rank_supervision_scores,
                        teacher_scores=utility,
                    )
                    assert_finite_tensor('utility_listwise_loss', listwise_loss.unsqueeze(0), key)
                    assert_finite_tensor(
                        'weighted_utility_listwise_loss',
                        weighted_listwise_loss.unsqueeze(0),
                        key,
                    )
                elif args.selection_score_source == 'shot_head':
                    teacher_student_shot_tau = compute_teacher_student_shot_tau(
                        student_scores=rank_supervision_scores,
                        teacher_scores=utility,
                    )

                selection_loss = gate_weight_tensor * coverage_loss_weight * raw_selection_loss
                weighted_selection_loss = args.lambda_select * selection_loss
                weighted_budget_loss = args.lambda_budget * budget_loss
                weighted_delta_loss = args.lambda_delta * delta_loss

                num_supervised_shots = float(supervised_mask.sum().item())
                num_positive_shots = float(selected_mask.sum().item())
                num_negative_shots = float(negative_mask.sum().item())
                teacher_gate_weight = float(gate_weight_tensor.detach().item())
                teacher_margin = float(teacher_margin_tensor.detach().item())

                assert_finite_tensor('selection_shot_scores', selection_shot_scores, key)
                assert_finite_tensor('selection_loss', selection_loss.unsqueeze(0), key)
                assert_finite_tensor('budget_loss', budget_loss.unsqueeze(0), key)

            elif args.rank_loss == 'preference_distill':
                if preference_teacher_store is None:
                    raise RuntimeError('preference_teacher_store is required for preference_distill rank loss.')

                h5_key = Path(str(key)).name
                pref_record = preference_teacher_store.get(h5_key=h5_key)
                teacher_scores = torch.tensor(
                    pref_record['shot_scores'], dtype=torch.float32, device=args.device
                )
                inclusion_prob = torch.tensor(
                    pref_record['inclusion_prob'], dtype=torch.float32, device=args.device
                )
                teacher_confidence_tensor = torch.tensor(
                    pref_record['teacher_confidence'], dtype=torch.float32, device=args.device
                )
                pair_i = torch.tensor(pref_record['pair_i'], dtype=torch.long, device=args.device)
                pair_j = torch.tensor(pref_record['pair_j'], dtype=torch.long, device=args.device)
                pair_label = torch.tensor(
                    pref_record['pair_label'], dtype=torch.float32, device=args.device
                )
                pair_confidence = torch.tensor(
                    pref_record['pair_confidence'], dtype=torch.float32, device=args.device
                )

                if teacher_scores.shape[0] != ranking_shot_scores.shape[0]:
                    raise ValueError(
                        f'Preference teacher length mismatch for sample {key}: '
                        f'teacher={teacher_scores.shape[0]} vs pred={ranking_shot_scores.shape[0]}'
                    )
                if inclusion_prob.shape[0] != selection_shot_scores.shape[0]:
                    raise ValueError(
                        f'Preference inclusion length mismatch for sample {key}: '
                        f'inclusion={inclusion_prob.shape[0]} vs pred={selection_shot_scores.shape[0]}'
                    )
                if teacher_confidence_tensor.shape[0] != selection_shot_scores.shape[0]:
                    raise ValueError(
                        f'Preference confidence length mismatch for sample {key}: '
                        f'confidence={teacher_confidence_tensor.shape[0]} vs pred={selection_shot_scores.shape[0]}'
                    )

                pair_loss = compute_preference_pair_rank_loss(
                    selection_shot_scores=(
                        rank_supervision_scores
                        if args.selection_score_source == 'shot_head'
                        else ranking_shot_scores
                    ),
                    pair_i=pair_i,
                    pair_j=pair_j,
                    pair_label=pair_label,
                    pair_confidence=pair_confidence,
                    margin=args.pref_pair_margin,
                )
                listwise_loss = compute_preference_listwise_loss(
                    selection_shot_scores=(
                        rank_supervision_scores
                        if args.selection_score_source == 'shot_head'
                        else ranking_shot_scores
                    ),
                    teacher_scores=teacher_scores,
                    temperature=args.pref_list_temperature,
                )
                if args.use_diff_budget_selector:
                    if budget_selector is None:
                        raise RuntimeError('budget_selector is not initialized.')
                    soft_selected_mask, _budget_lambda = budget_selector(
                        select_logits=selection_shot_logits,
                        lengths=nfps_tensor,
                        budget_frames=float(n_frames_int) * float(args.summary_budget),
                    )
                    selection_loss = compute_preference_inclusion_loss(
                        selection_shot_scores=soft_selected_mask,
                        inclusion_prob=inclusion_prob,
                        teacher_confidence=teacher_confidence_tensor,
                        confidence_threshold=args.pref_confidence_threshold,
                    )
                    budget_loss = compute_budget_regularizer(
                        selection_shot_scores=soft_selected_mask,
                        shot_lengths=nfps_tensor,
                        n_frames=n_frames_int,
                        summary_budget=args.summary_budget,
                    )
                    delta_loss = compute_delta_regularizer(delta_select)
                    assert_finite_tensor('diff_budget_soft_mask', soft_selected_mask, key)
                else:
                    selection_loss = compute_preference_inclusion_loss(
                        selection_shot_scores=selection_shot_scores,
                        inclusion_prob=inclusion_prob,
                        teacher_confidence=teacher_confidence_tensor,
                        confidence_threshold=args.pref_confidence_threshold,
                    )
                    budget_loss = compute_budget_regularizer(
                        selection_shot_scores=selection_shot_scores,
                        shot_lengths=shot_lengths,
                        n_frames=n_frames_int,
                        summary_budget=args.summary_budget,
                    )

                weighted_pair_loss = args.lambda_pref_pair * coverage_loss_weight * pair_loss
                weighted_listwise_loss = args.lambda_pref_list * coverage_loss_weight * listwise_loss
                weighted_selection_loss = args.lambda_pref_inclusion * coverage_loss_weight * selection_loss
                weighted_budget_loss = args.lambda_pref_budget * budget_loss
                weighted_delta_loss = args.lambda_delta * delta_loss

                supervised_pref_mask = teacher_confidence_tensor >= args.pref_confidence_threshold
                num_supervised_shots = float(supervised_pref_mask.sum().item())
                num_positive_shots = float((inclusion_prob >= 0.60).sum().item())
                num_negative_shots = float((inclusion_prob <= 0.20).sum().item())
                teacher_gate_weight = float(teacher_confidence_tensor.mean().detach().item())
                teacher_margin = float(pair_confidence.mean().detach().item()) if pair_confidence.numel() > 0 else 0.0
                teacher_student_shot_tau = compute_teacher_student_shot_tau(
                    student_scores=rank_supervision_scores,
                    teacher_scores=teacher_scores,
                )

                assert_finite_tensor('selection_shot_scores', selection_shot_scores, key)
                assert_finite_tensor('ranking_shot_scores', ranking_shot_scores, key)
                assert_finite_tensor('preference_teacher_scores', teacher_scores, key)
                assert_finite_tensor('preference_inclusion_prob', inclusion_prob, key)
                assert_finite_tensor('preference_pair_loss', pair_loss.unsqueeze(0), key)
                assert_finite_tensor('preference_listwise_loss', listwise_loss.unsqueeze(0), key)
                assert_finite_tensor('preference_inclusion_loss', selection_loss.unsqueeze(0), key)
                assert_finite_tensor('preference_budget_loss', budget_loss.unsqueeze(0), key)

            elif args.rank_loss == 'none':
                pass
            elif args.rank_loss not in ('sparse_pair', 'listwise_utility', 'hybrid_sparse_budget'):
                raise ValueError(f'Unknown rank_loss: {args.rank_loss}')

            align_loss = compute_align_loss(summary_feat, text_target_tensor)
            weighted_align_loss = args.lambda_align * coverage_loss_weight * align_loss

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
                + weighted_delta_loss
            )
            rank_loss_value = pair_loss + listwise_loss + selection_loss + budget_loss + delta_loss

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
                delta_loss=float(delta_loss.item()),
                weighted_delta_loss=float(weighted_delta_loss.item()),
                align_loss=float(align_loss.item()),
                weighted_align_loss=float(weighted_align_loss.item()),
                bag_loss=float(bag_loss.item()),
                weighted_bag_loss=float(weighted_bag_loss.item()),
                num_aux_active=num_aux_active,
                num_aux_skipped=num_aux_skipped,
                num_supervised_shots=num_supervised_shots,
                num_positive_shots=num_positive_shots,
                num_negative_shots=num_negative_shots,
                teacher_gate_weight=teacher_gate_weight,
                teacher_margin=teacher_margin,
                teacher_student_shot_tau=teacher_student_shot_tau,
                caption_coverage_ratio=caption_coverage_ratio_value,
                coverage_loss_weight=float(coverage_loss_weight.detach().item()),
                num_valid_text_cond=num_valid_text_cond,
            )

        val_metrics = evaluate_mil_cond(
            model=model,
            val_loader=val_loader,
            device=args.device,
            selection_score_source=args.selection_score_source,
            shot_eval_head=args.shot_eval_head,
            shot_head_mode=args.shot_head_mode,
            summary_budget=args.summary_budget,
        )
        val_fscore = float(val_metrics['fscore'])
        val_kendall = float(val_metrics['kendall'])
        val_spearman = float(val_metrics['spearman'])
        val_caption_coverage = float(val_metrics['caption_coverage'])
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

        f1_tie_tol = 1e-8
        better_best_f1_ckpt = (
            val_fscore > best_val_fscore + f1_tie_tol
            or (
                abs(val_fscore - best_val_fscore) <= f1_tie_tol
                and (
                    val_kendall > kendall_at_best_fscore + f1_tie_tol
                    or (
                        abs(val_kendall - kendall_at_best_fscore) <= f1_tie_tol
                        and val_spearman > spearman_at_best_fscore + f1_tie_tol
                    )
                )
            )
        )

        if better_best_f1_ckpt:
            best_val_fscore = val_fscore
            kendall_at_best_fscore = val_kendall
            spearman_at_best_fscore = val_spearman
            torch.save(model.state_dict(), save_path)
            logger.info(
                'Update best-F1 checkpoint | epoch=%03d | val_F1=%.4f | val_Tau=%.4f | val_Rho=%.4f',
                epoch + 1,
                val_fscore,
                val_kendall,
                val_spearman,
            )

        logger.info(
            'Epoch %03d/%03d | loss=%.4f | rank=%.4f | val_F1=%.4f | '
            'val_Tau=%.4f | val_Rho=%.4f | val_cov=%.4f | best_val_F1=%.4f | '
            'ts_shot_Tau=%.4f',
            epoch + 1,
            args.max_epoch,
            stats.loss,
            stats.weighted_rank_loss,
            val_fscore,
            val_kendall,
            val_spearman,
            val_caption_coverage,
            best_val_fscore,
            stats.teacher_student_shot_tau,
        )

        logger.debug(
            'Epoch %03d/%03d detail | pair=%.4f/%.4f | listwise=%.4f/%.4f | '
            'select=%.4f/%.4f | budget=%.4f/%.4f | '
            'align=%.4f/%.4f | bag=%.4f/%.4f | '
            'aux_active=%.4f | aux_skipped=%.4f | sup_shots=%.4f | '
            'pos=%.4f | neg=%.4f | gate=%.4f | margin=%.4f | ts_tau=%.4f | '
            'cap_cov=%.4f | cov_w=%.4f | valid_text=%.4f | '
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
            stats.teacher_gate_weight,
            stats.teacher_margin,
            stats.teacher_student_shot_tau,
            stats.caption_coverage_ratio,
            stats.coverage_loss_weight,
            stats.num_valid_text_cond,
            max_val_kendall,
            max_val_spearman,
            kendall_at_best_fscore,
            spearman_at_best_fscore,
            fscore_at_max_kendall,
            fscore_at_max_spearman,
        )

    test_at_best_fscore = evaluate_checkpoint(
        model,
        save_path,
        test_loader,
        args.device,
        args.selection_score_source,
        args.shot_eval_head,
        args.shot_head_mode,
        args.summary_budget,
    )
    test_at_max_kendall = evaluate_checkpoint(
        model,
        kendall_save_path,
        test_loader,
        args.device,
        args.selection_score_source,
        args.shot_eval_head,
        args.shot_head_mode,
        args.summary_budget,
    )
    test_at_max_spearman = evaluate_checkpoint(
        model,
        spearman_save_path,
        test_loader,
        args.device,
        args.selection_score_source,
        args.shot_eval_head,
        args.shot_head_mode,
        args.summary_budget,
    )

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
        'test_caption_coverage_at_best_fscore': float(test_at_best_fscore['caption_coverage']),
        'test_fscore_at_max_kendall': float(test_at_max_kendall['fscore']),
        'test_kendall_at_max_kendall': float(test_at_max_kendall['kendall']),
        'test_spearman_at_max_kendall': float(test_at_max_kendall['spearman']),
        'test_fscore_at_max_spearman': float(test_at_max_spearman['fscore']),
        'test_kendall_at_max_spearman': float(test_at_max_spearman['kendall']),
        'test_spearman_at_max_spearman': float(test_at_max_spearman['spearman']),
    }


def validate_rank_loss_args(args) -> None:
    if args.score_head not in ('single', 'dual', 'residual_dual'):
        raise ValueError(
            f'Invalid score_head={args.score_head}; expected single, dual, or residual_dual.'
        )
    if args.selection_score_source not in ('frame', 'shot_head'):
        raise ValueError(
            f'Invalid selection_score_source={args.selection_score_source}; expected frame or shot_head.'
        )
    if args.shot_head_mode not in ('single', 'dual', 'context'):
        raise ValueError(
            f'Invalid shot_head_mode={args.shot_head_mode}; expected single, dual, or context.'
        )
    if args.shot_eval_head not in ('selection', 'rank'):
        raise ValueError(
            f'Invalid shot_eval_head={args.shot_eval_head}; expected selection or rank.'
        )
    if args.selection_score_source == 'shot_head' and args.shot_eval_head == 'rank' and args.shot_head_mode not in ('dual', 'context'):
        raise ValueError('--shot-eval-head rank requires --shot-head-mode dual or context.')
    if args.selection_score_source == 'shot_head' and args.rank_loss in ('listwise_utility',):
        raise NotImplementedError(
            'shot_head + listwise_utility is not yet protocol-clean; '
            'use preference_distill or budgeted_pseudo_summary first.'
        )
    if args.use_diff_budget_selector:
        if args.selection_score_source != 'shot_head':
            raise ValueError('--use-diff-budget-selector requires --selection-score-source shot_head.')
        if args.diff_budget_tau <= 0:
            raise ValueError(f'Invalid diff_budget_tau={args.diff_budget_tau}; expected > 0.')
        if args.lambda_delta < 0:
            raise ValueError(f'Invalid lambda_delta={args.lambda_delta}; expected >= 0.')
    if not (0.0 <= args.coverage_loss_min_weight <= 1.0):
        raise ValueError(
            f'Invalid coverage_loss_min_weight={args.coverage_loss_min_weight}; '
            'expected 0 <= weight <= 1.'
        )

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
    elif args.rank_loss in ('budgeted_pseudo_summary', 'hybrid_sparse_budget'):
        if args.score_head not in ('dual', 'residual_dual'):
            raise ValueError(
                f'--rank-loss {args.rank_loss} requires --score-head dual or residual_dual.'
            )
        if args.lambda_select < 0:
            raise ValueError(f'Invalid lambda_select={args.lambda_select}; expected >= 0.')
        if args.lambda_budget < 0:
            raise ValueError(f'Invalid lambda_budget={args.lambda_budget}; expected >= 0.')
        if not (0.0 < args.summary_budget < 1.0):
            raise ValueError(f'Invalid summary_budget={args.summary_budget}; expected 0 < budget < 1.')
        if not (0.0 < args.negative_quantile < 1.0):
            raise ValueError(f'Invalid negative_quantile={args.negative_quantile}; expected 0 < q < 1.')
        if args.teacher_gate_mode not in ('none', 'scale', 'skip'):
            raise ValueError(f'Invalid teacher_gate_mode={args.teacher_gate_mode}.')
        if args.teacher_margin_threshold < 0:
            raise ValueError(
                f'Invalid teacher_margin_threshold={args.teacher_margin_threshold}; expected >= 0.'
            )
    elif args.rank_loss == 'preference_distill':
        if args.score_head not in ('dual', 'residual_dual'):
            raise ValueError(
                '--rank-loss preference_distill requires --score-head dual or residual_dual.'
            )
        if not args.preference_teacher_path:
            raise ValueError('--rank-loss preference_distill requires --preference-teacher-path.')
        if args.lambda_pref_pair < 0:
            raise ValueError(f'Invalid lambda_pref_pair={args.lambda_pref_pair}; expected >= 0.')
        if args.lambda_pref_list < 0:
            raise ValueError(f'Invalid lambda_pref_list={args.lambda_pref_list}; expected >= 0.')
        if args.lambda_pref_inclusion < 0:
            raise ValueError(f'Invalid lambda_pref_inclusion={args.lambda_pref_inclusion}; expected >= 0.')
        if args.lambda_pref_budget < 0:
            raise ValueError(f'Invalid lambda_pref_budget={args.lambda_pref_budget}; expected >= 0.')
        if not (0.0 <= args.pref_confidence_threshold <= 1.0):
            raise ValueError(
                f'Invalid pref_confidence_threshold={args.pref_confidence_threshold}; expected 0 <= thr <= 1.'
            )
        if args.pref_pair_margin <= 0:
            raise ValueError(f'Invalid pref_pair_margin={args.pref_pair_margin}; expected > 0.')
        if args.pref_list_temperature <= 0:
            raise ValueError(
                f'Invalid pref_list_temperature={args.pref_list_temperature}; expected > 0.'
            )
        if not (0.0 < args.summary_budget < 1.0):
            raise ValueError(f'Invalid summary_budget={args.summary_budget}; expected 0 < budget < 1.')
    elif args.rank_loss == 'none':
        pass
    else:
        raise ValueError(
            f'Invalid rank_loss={args.rank_loss}; '
            f'expected sparse_pair, listwise_utility, budgeted_pseudo_summary, hybrid_sparse_budget, preference_distill, or none.'
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

    return F.kl_div(student_log_prob, teacher_prob, reduction='sum')



def compute_preference_pair_rank_loss(selection_shot_scores: torch.Tensor,
                                      pair_i: torch.Tensor,
                                      pair_j: torch.Tensor,
                                      pair_label: torch.Tensor,
                                      pair_confidence: torch.Tensor,
                                      margin: float,
                                      eps: float = 1e-8) -> torch.Tensor:
    if selection_shot_scores.ndim != 1:
        raise ValueError(
            f'Expected selection_shot_scores shape [S], got {tuple(selection_shot_scores.shape)}'
        )
    if pair_i.ndim != 1 or pair_j.ndim != 1:
        raise ValueError('pair_i and pair_j must be 1D tensors.')
    if pair_i.shape[0] != pair_j.shape[0]:
        raise ValueError('pair_i/pair_j length mismatch in preference pair loss.')
    if pair_label.shape[0] != pair_i.shape[0] or pair_confidence.shape[0] != pair_i.shape[0]:
        raise ValueError('Preference pair metadata length mismatch.')
    if margin <= 0:
        raise ValueError(f'Invalid preference pair margin={margin}; expected > 0.')
    if pair_i.numel() == 0:
        return selection_shot_scores.new_zeros(())
    if int(pair_i.min().item()) < 0 or int(pair_i.max().item()) >= selection_shot_scores.shape[0]:
        raise ValueError('pair_i index out of range in preference pair loss.')
    if int(pair_j.min().item()) < 0 or int(pair_j.max().item()) >= selection_shot_scores.shape[0]:
        raise ValueError('pair_j index out of range in preference pair loss.')

    score_i = selection_shot_scores[pair_i]
    score_j = selection_shot_scores[pair_j]
    i_preferred = pair_label > 0.5
    preferred = torch.where(i_preferred, score_i, score_j)
    other = torch.where(i_preferred, score_j, score_i)
    confidence = pair_confidence.to(selection_shot_scores.dtype).clamp(min=0.0, max=1.0)
    conf_sum = confidence.sum()
    if float(conf_sum.item()) <= eps:
        return selection_shot_scores.new_zeros(())
    per_pair = F.relu(float(margin) - preferred + other)
    return torch.sum(per_pair * confidence) / conf_sum.clamp_min(eps)


def compute_preference_listwise_loss(selection_shot_scores: torch.Tensor,
                                     teacher_scores: torch.Tensor,
                                     temperature: float,
                                     eps: float = 1e-8) -> torch.Tensor:
    if selection_shot_scores.ndim != 1:
        raise ValueError(
            f'Expected selection_shot_scores shape [S], got {tuple(selection_shot_scores.shape)}'
        )
    if teacher_scores.ndim != 1:
        raise ValueError(f'Expected teacher_scores shape [S], got {tuple(teacher_scores.shape)}')
    if selection_shot_scores.shape[0] != teacher_scores.shape[0]:
        raise ValueError('Shot count mismatch in compute_preference_listwise_loss.')
    if selection_shot_scores.shape[0] < 2:
        return selection_shot_scores.new_zeros(())
    if temperature <= 0:
        raise ValueError(f'Invalid preference list temperature={temperature}; expected > 0.')
    if not torch.isfinite(selection_shot_scores).all():
        raise ValueError('Non-finite selection_shot_scores in preference listwise loss.')
    if not torch.isfinite(teacher_scores).all():
        raise ValueError('Non-finite teacher_scores in preference listwise loss.')
    teacher_range = torch.max(teacher_scores) - torch.min(teacher_scores)
    if float(teacher_range.item()) <= eps:
        return selection_shot_scores.new_zeros(())
    student_log_prob = F.log_softmax(selection_shot_scores / float(temperature), dim=0)
    teacher_prob = F.softmax(teacher_scores / float(temperature), dim=0).detach()
    return F.kl_div(student_log_prob, teacher_prob, reduction='sum')


def compute_preference_inclusion_loss(selection_shot_scores: torch.Tensor,
                                      inclusion_prob: torch.Tensor,
                                      teacher_confidence: torch.Tensor,
                                      confidence_threshold: float,
                                      eps: float = 1e-6) -> torch.Tensor:
    if selection_shot_scores.ndim != 1:
        raise ValueError(
            f'Expected selection_shot_scores shape [S], got {tuple(selection_shot_scores.shape)}'
        )
    if inclusion_prob.ndim != 1 or teacher_confidence.ndim != 1:
        raise ValueError('inclusion_prob and teacher_confidence must be 1D tensors.')
    if selection_shot_scores.shape[0] != inclusion_prob.shape[0]:
        raise ValueError('Shot count mismatch in preference inclusion loss.')
    if selection_shot_scores.shape[0] != teacher_confidence.shape[0]:
        raise ValueError('Confidence length mismatch in preference inclusion loss.')
    if not (0.0 <= confidence_threshold <= 1.0):
        raise ValueError(f'Invalid confidence_threshold={confidence_threshold}; expected 0..1.')

    mask = teacher_confidence >= float(confidence_threshold)
    if int(mask.sum().item()) == 0:
        return selection_shot_scores.new_zeros(())
    pred = selection_shot_scores[mask].clamp(min=eps, max=1.0 - eps)
    target = inclusion_prob[mask].clamp(min=0.0, max=1.0)
    weight = teacher_confidence[mask].to(pred.dtype).clamp(min=0.0, max=1.0)
    weight_sum = weight.sum()
    if float(weight_sum.item()) <= eps:
        return selection_shot_scores.new_zeros(())
    per_shot = F.binary_cross_entropy(pred, target, reduction='none')
    return torch.sum(per_shot * weight) / weight_sum.clamp_min(eps)


def compute_confidence_gated_weighted_bce_loss(pred_shot_scores: torch.Tensor,
                                               target: torch.Tensor,
                                               supervised_mask: torch.Tensor,
                                               shot_lengths: torch.Tensor,
                                               eps: float = 1e-6) -> torch.Tensor:
    if pred_shot_scores.ndim != 1:
        raise ValueError(f'Expected pred_shot_scores shape [S], got {tuple(pred_shot_scores.shape)}')
    if target.ndim != 1:
        raise ValueError(f'Expected target shape [S], got {tuple(target.shape)}')
    if supervised_mask.ndim != 1:
        raise ValueError(f'Expected supervised_mask shape [S], got {tuple(supervised_mask.shape)}')
    if shot_lengths.ndim != 1:
        raise ValueError(f'Expected shot_lengths shape [S], got {tuple(shot_lengths.shape)}')
    if pred_shot_scores.shape[0] != target.shape[0] or pred_shot_scores.shape[0] != supervised_mask.shape[0]:
        raise ValueError('Shot count mismatch in compute_confidence_gated_weighted_bce_loss.')
    if pred_shot_scores.shape[0] != shot_lengths.shape[0]:
        raise ValueError('Shot length mismatch in compute_confidence_gated_weighted_bce_loss.')
    if not torch.isfinite(pred_shot_scores).all():
        raise ValueError('Non-finite pred_shot_scores in compute_confidence_gated_weighted_bce_loss.')
    if not torch.isfinite(target).all():
        raise ValueError('Non-finite target in compute_confidence_gated_weighted_bce_loss.')

    if int(supervised_mask.sum().item()) == 0:
        return pred_shot_scores.new_zeros(())

    pred = pred_shot_scores.clamp(min=eps, max=1.0 - eps)
    tgt = target.clamp(min=0.0, max=1.0)
    per_shot = F.binary_cross_entropy(pred, tgt, reduction='none')

    pos_mask = supervised_mask & (target > 0.5)
    neg_mask = supervised_mask & (target <= 0.5)

    weights = torch.zeros_like(per_shot)

    if int(pos_mask.sum().item()) > 0:
        pos_lengths = shot_lengths[pos_mask].to(per_shot.dtype)
        weights[pos_mask] = 0.5 * pos_lengths / pos_lengths.sum().clamp_min(eps)

    if int(neg_mask.sum().item()) > 0:
        neg_lengths = shot_lengths[neg_mask].to(per_shot.dtype)
        weights[neg_mask] = 0.5 * neg_lengths / neg_lengths.sum().clamp_min(eps)

    weight_sum = weights.sum()
    if float(weight_sum.item()) <= eps:
        return pred_shot_scores.new_zeros(())

    weights = weights / weight_sum
    return torch.sum(per_shot * weights)


def compute_teacher_margin(utility: torch.Tensor,
                           selected_mask: torch.Tensor,
                           negative_mask: torch.Tensor) -> torch.Tensor:
    if selected_mask.any() and negative_mask.any():
        return utility[selected_mask].mean() - utility[negative_mask].mean()
    return utility.new_tensor(0.0)


def compute_teacher_gate_weight(margin: torch.Tensor,
                                threshold: float,
                                mode: str,
                                eps: float = 1e-6) -> torch.Tensor:
    if mode == 'none':
        return margin.new_tensor(1.0)
    if mode == 'skip':
        return (margin >= threshold).to(margin.dtype)
    if mode == 'scale':
        if threshold <= 0:
            return torch.clamp(margin, min=0.0, max=1.0)
        return torch.clamp((margin - threshold) / (1.0 - threshold + eps), min=0.0, max=1.0)
    raise ValueError(f'Invalid teacher gate mode: {mode}')


def compute_delta_regularizer(delta_select: torch.Tensor) -> torch.Tensor:
    if delta_select.ndim != 1:
        raise ValueError(f'Expected delta_select shape [S], got {tuple(delta_select.shape)}')
    if not torch.isfinite(delta_select).all():
        raise ValueError('Non-finite delta_select in delta regularizer.')
    if delta_select.numel() == 0:
        return delta_select.new_zeros(())
    return torch.mean(delta_select ** 2)


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


def compute_teacher_student_shot_tau(student_scores: torch.Tensor,
                                     teacher_scores: torch.Tensor,
                                     eps: float = 1e-8) -> float:
    """Pairwise Kendall-style monitor between predicted and teacher shot scores."""
    if student_scores.ndim != 1 or teacher_scores.ndim != 1:
        raise ValueError('student_scores and teacher_scores must be 1D tensors.')
    if student_scores.shape[0] != teacher_scores.shape[0]:
        raise ValueError('Shot count mismatch in compute_teacher_student_shot_tau.')
    if student_scores.shape[0] < 2:
        return 0.0

    student_detached = student_scores.detach()
    teacher_detached = teacher_scores.detach()
    diff_student = student_detached.unsqueeze(0) - student_detached.unsqueeze(1)
    diff_teacher = teacher_detached.unsqueeze(0) - teacher_detached.unsqueeze(1)
    upper = torch.triu(
        torch.ones_like(diff_student, dtype=torch.bool),
        diagonal=1,
    )
    valid = upper & (diff_student.abs() > eps) & (diff_teacher.abs() > eps)
    num_valid = int(valid.sum().item())
    if num_valid == 0:
        return 0.0

    agreement = torch.sign(diff_student[valid] * diff_teacher[valid])
    tau = agreement.mean()
    return float(tau.item())


def evaluate_checkpoint(model,
                        ckpt_path,
                        test_loader,
                        device: str,
                        selection_score_source: str = 'frame',
                        shot_eval_head: str = 'selection',
                        shot_head_mode: str = 'single',
                        summary_budget: float = 0.15):
    state_dict = torch.load(str(ckpt_path), map_location=device)
    model.load_state_dict(state_dict)
    return evaluate_mil_cond(
        model=model,
        val_loader=test_loader,
        device=device,
        selection_score_source=selection_score_source,
        shot_eval_head=shot_eval_head,
        shot_head_mode=shot_head_mode,
        summary_budget=summary_budget,
    )


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


def compute_coverage_loss_weight(seq_tensor: torch.Tensor,
                                 coverage_ratio: float,
                                 enabled: bool,
                                 min_weight: float) -> torch.Tensor:
    if not np.isfinite(coverage_ratio):
        raise ValueError(f'Non-finite caption coverage ratio: {coverage_ratio}')
    if not (0.0 <= min_weight <= 1.0):
        raise ValueError(f'Invalid coverage min_weight={min_weight}; expected 0 <= weight <= 1.')
    if not enabled:
        return seq_tensor.new_tensor(1.0)
    coverage_ratio = max(0.0, min(float(coverage_ratio), 1.0))
    return seq_tensor.new_tensor(min_weight + (1.0 - min_weight) * coverage_ratio)


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

    cosine = F.cosine_similarity(summary_feat.unsqueeze(0), text_target.unsqueeze(0), dim=-1).squeeze(0)
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

    overlaps = torch.stack(overlaps, dim=0)
    shot_lengths = overlaps.sum(dim=1)

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
        raise ValueError('Time-step mismatch in aggregate_attn_to_shot_scores.')
    if overlaps.shape[0] != shot_lengths.shape[0]:
        raise ValueError('Shot count mismatch in aggregate_attn_to_shot_scores.')

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
        raise ValueError('Time-step mismatch in aggregate_frame_scores_to_shot_scores.')
    return torch.matmul(overlaps, frame_scores) / shot_lengths.clamp_min(1.0)


def build_shot_text_stats(caption_spans_idx: torch.Tensor,
                          caption_valid_mask: torch.Tensor,
                          all_text_features: torch.Tensor,
                          overlaps: torch.Tensor,
                          shot_lengths: torch.Tensor,
                          eps: float = 1e-6):
    return build_context_shot_text_stats(
        caption_spans_idx=caption_spans_idx,
        caption_valid_mask=caption_valid_mask,
        all_text_features=all_text_features,
        overlaps=overlaps,
        shot_lengths=shot_lengths,
        eps=eps,
    )

def compute_shot_semantic_change(shot_text_feat: torch.Tensor, valid_shots: torch.Tensor):
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


def remove_indices(candidates: torch.Tensor, exclude: torch.Tensor) -> torch.Tensor:
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
    if pos_idx.numel() == 0 or neg_idx.numel() == 0:
        return pred_shot_scores.new_zeros(())
    pos_scores = pred_shot_scores[pos_idx].unsqueeze(1)
    neg_scores = pred_shot_scores[neg_idx].unsqueeze(0)
    return F.relu(margin - pos_scores + neg_scores).mean()
