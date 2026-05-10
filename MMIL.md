## workspace: chim_MMIL  chim_mmil

### baseline -------------------------------
## 生成伪标签
src/
├── make_video_pseudo_labels.py
└── helpers/
    ├── dataset_registry.py
    └── dataset_adapters/
        ├── base.py
        ├── tvsum.py
        └── summe.py
## summe
python src/make_video_pseudo_labels.py --dataset summe --video-dir custom_data/videos/SumMe --h5-path datasets/eccv16_dataset_summe_google_pool5.h5 --openclip-model ViT-L-14 --openclip-pretrained dfn2b
## tvsum
python src/make_video_pseudo_labels.py --dataset tvsum --video-dir custom_data/videos/TVSum --openclip-model ViT-L-14 --openclip-pretrained dfn2b

key：后续串联 HDF5 样本、伪标签、日志，必要
seq：MIL 主干唯一时序输入
soft_label：当前唯一训练监督
gtscore：不进入主损失，但保留给诊断/相关性分析是合理的
user_summary：后续 F-score 仍需要
cps / n_frames / nfps / picks：标准摘要协议的结构元数据块

## 先给 shell 脚本执行权限
chmod +x scripts/train_mil_tvsum.sh
chmod +x scripts/train_mil_summe.sh

## 启动脚本
./scripts/train_mil_tvsum.sh
./scripts/train_mil_summe.sh

## 快速看每个 split 的最佳 F-score,Spearman
grep "Finished split\|All splits finished" models/mil/summe_run/log_mil.txt


### 改进 1 ------------------------------- tumx --chim_MMIL --gemini
## 快速看每个 split 的最佳 F-score,Spearman
grep "Finished split\|All splits finished" models/mil_cond/tvsum_cond_attention_lr5e-5_wd1e-5_laux0.1_m7_seed12345/log_mil_cond.txt

## 生成 SumMe-24 特征
python src/make_openclip_features.py --dataset summe --video-dir custom_data/videos/SumMe --h5-path datasets/eccv16_dataset_summe_google_pool5.h5 --openclip-model ViT-L-14 --openclip-pretrained dfn2b --output-h5 features/openclip_summe24.h5
## 生成 TVSum 特征
python src/make_openclip_features.py --dataset tvsum --video-dir custom_data/videos/TVSum --h5-path datasets/eccv16_dataset_tvsum_google_pool5.h5 --openclip-model ViT-L-14 --openclip-pretrained dfn2b

## 生成 SumMe-24 caption
python tools/generate_dense_captions_gemini.py --dataset summe --video-dir custom_data/videos/SumMe --h5-path datasets/eccv16_dataset_summe_google_pool5.h5 --out-structured captions_raw/summe24_dense_captions_structured.json --out-simple captions/summe24_dense_captions.json
## 生成 TVSum caption
python tools/generate_dense_captions_gemini.py --dataset tvsum --video-dir custom_data/videos/TVSum --h5-path datasets/eccv16_dataset_tvsum_google_pool5.h5 --out-structured captions_raw/tvsum_dense_captions_structured.json --out-simple captions/tvsum_dense_captions.json

## 生成text feature
python src/make_text_features.py --dataset summe --device cpu --openclip-model ViT-L-14 --openclip-pretrained dfn2b
python src/make_text_features.py --dataset tvsum --device cpu --openclip-model ViT-L-14 --openclip-pretrained dfn2b


## 训练Summe
CUDA_VISIBLE_DEVICES=7 bash scripts/train_mil_cond_summe_tagr.sh
## 训练TVSum
CUDA_VISIBLE_DEVICES=7 bash scripts/train_mil_cond_tvsum_tagr.sh
# 改变量形式
LAMBDA_AUX=0.05 bash scripts/train_mil_cond_summe.sh


MMIL_D/
├── .github/workflows/unit-test.yml         # GitHub Actions 单元测试配置，原 DSNet CI 流程
├── .gitignore                              # Git 忽略规则
├── README.md                               # 原始 DSNet 项目说明，当前多模态主线不完全以它为准
├── MMIL.md                                 # 当前 MMIL 实验笔记：数据生成、caption、训练命令等
├── requirements.txt                        # Python 依赖版本，偏原 DSNet/旧环境
│
├── captions/                               # 简化版 dense captions，供 make_text_features.py 编码
│   ├── summe_dense_captions.json           # SumMe caption 文本列表
│   ├── summe_dense_captions_v3.json        # SumMe 新版本 caption 文本列表
│   └── tvsum_dense_captions.json           # TVSum caption 文本列表
│
├── captions_raw/                           # Gemini 生成的结构化 caption 原始结果
│   ├── *_structured.json                   # 带时间段、sample_meta、caption objects 的结构化结果
│   └── *_structured.failures.json          # caption 生成失败记录
│
├── custom_data/videos/
│   ├── SumMe/                              # SumMe 原始视频
│   └── TVSum/                              # TVSum 原始视频
│
├── datasets/                               # DSNet 公共 H5 数据，含 picks/change_points/gtscore/user_summary 等
│   ├── eccv16_dataset_summe_google_pool5.h5
│   ├── eccv16_dataset_tvsum_google_pool5.h5
│   ├── eccv16_dataset_ovp_google_pool5.h5
│   └── eccv16_dataset_youtube_google_pool5.h5
│
├── diagnostics/                            # 分析产物和实验诊断结果
│   ├── *_shot_utility*.csv/json            # shot utility 公式/版本分析结果
│   ├── *_teacher_ceiling*.csv/json         # teacher 上限诊断结果
│   └── *.log / *.pid                       # caption 生成或后台任务日志
│
├── docs/                                   # 阶段性实验文档和日志归档
│   ├── PHASE0_CAPTION_PAIR_SIGNAL.md       # caption pair signal 阶段说明
│   ├── PHASE1_SHOT_PSEUDO_UTILITY.md       # shot pseudo utility 阶段说明
│   ├── PHASE1B_FORMULA_ABLATION.md         # utility 公式消融说明
│   └── mil_cond/*/log_mil_cond.txt         # 各次 MIL 条件训练日志归档
│
├── features/                               # 训练直接使用的 OpenCLIP 特征
│   ├── openclip_summe.h5                   # SumMe 视频帧 OpenCLIP 视觉特征
│   ├── openclip_tvsum.h5                   # TVSum 视频帧 OpenCLIP 视觉特征
│   ├── text_summe.h5                       # SumMe caption 文本特征
│   ├── text_summe_v3.h5                    # SumMe v3 caption 文本特征
│   └── text_tvsum.h5                       # TVSum caption 文本特征
│
├── models/                                 # 模型 checkpoint 和训练日志
│   ├── af_basic/                           # 原 DSNet anchor-free baseline 输出
│   └── mil_cond/*/                         # 当前多模态 MIL 条件训练输出
│
├── prompts/                                # prompt vocabulary，用于 frame-text soft label
│   ├── summe_prompt_vocabulary.txt         # SumMe 文本类别/提示词
│   └── tvsum_prompt_vocabulary.txt         # TVSum 文本类别/提示词
│
├── pseudo_labels/                          # 伪监督信号
│   ├── summe/
│   │   ├── frame_text_scores.npy           # 每帧与 prompt 的相似度分数
│   │   ├── soft_labels.npy                 # 视频级/bag 软标签，训练辅助监督
│   │   ├── shot_utility*.npy               # shot-level pseudo utility，不同版本/公式
│   │   └── meta.yaml                       # 伪标签生成元信息
│   └── tvsum/
│       ├── frame_text_scores.npy
│       ├── soft_labels.npy
│       ├── shot_utility.npy
│       └── meta.yaml
│
├── scripts/
│   ├── train_mil_cond_summe_tagr.sh        # SumMe 当前主训练脚本，已默认使用 MMIL_env
│   └── train_mil_cond_tvsum_tagr.sh        # TVSum 当前主训练脚本，已默认使用 MMIL_env
│
├── splits/                                 # 数据集 fold 划分
│   ├── summe.yml                           # SumMe 标准 split
│   ├── summe_aug.yml                       # SumMe augmented split
│   ├── summe_trans.yml                     # SumMe transfer split
│   ├── tvsum.yml                           # TVSum 标准 split
│   ├── tvsum_aug.yml                       # TVSum augmented split
│   └── tvsum_trans.yml                     # TVSum transfer split
│
├── src/
│   ├── run_train_mil_cond.py               # 当前训练入口：解析参数、加载 split、逐 fold 调训练
│   ├── evaluate_mil_cond.py                # 当前评估入口：F1、Kendall、Spearman、caption coverage
│   ├── make_openclip_features.py           # 从原视频按 H5 picks 抽帧并生成 OpenCLIP 视觉特征
│   ├── make_text_features.py               # 把 dense captions 编码成 OpenCLIP 文本特征
│   ├── make_video_pseudo_labels.py         # 生成 frame_text_scores 和 soft_labels 伪标签
│   ├── analyze_shot_utility_formulas.py    # 分析不同 shot utility 公式的相关性/效果
│   ├── analyze_budgeted_pseudo_summary_teacher.py # 分析 budgeted pseudo-summary teacher 的质量
│   │
│   ├── anchor_free/
│   │   ├── dsnet_af_mil_cond.py            # 当前核心模型 DSNetAFMILCond：视频编码 + 文本 cross-attention
│   │   └── train_mil_cond.py               # 当前核心训练循环和 loss 逻辑
│   │
│   ├── modules/
│   │   └── models.py                       # 底层时序编码器：attention/LSTM/BiLSTM/GCN/linear
│   │
│   └── helpers/
│       ├── data_helper.py                  # 通用 Dataset/DataLoader/AverageMeter/YAML/ckpt 工具
│       ├── dataset_registry.py             # 根据 dataset 名选择 SumMe/TVSum adapter
│       ├── eval_protocol_helper.py         # SumMe/TVSum F1 协议和 rank correlation 工具
│       ├── init_helper.py                  # 随机种子、logger、通用参数初始化
│       ├── key_helper.py                   # H5 key/video name 规范化工具
│       ├── mil_data_helper_cond.py         # 当前核心数据加载器，组装视觉特征、文本特征、caption span、伪标签
│       ├── mil_path_helper.py              # 项目路径约定：features/captions/pseudo_labels 等
│       ├── mil_vis_data.py                 # 可视化/对比 baseline 与 ours summary 的数据准备工具
│       ├── openclip_helper.py              # OpenCLIP 模型加载、图像/文本编码、相似度计算
│       ├── prompt_helper.py                # 读取 prompt vocabulary，获取类别数
│       ├── pseudo_label_helper.py          # soft label 聚合、时序平滑、伪标签打包
│       ├── shot_utility_helper.py          # shot utility 公式、budgeted mask、teacher store
│       ├── tvsum_metadata.py               # TVSum 视频名/key 元数据映射
│       ├── video_text_align_helper.py      # 按 picks/采样率从视频读取 RGB 帧
│       ├── vsumm_helper.py                 # DSNet 摘要协议：knapsack、keyshot summary、F1
│       └── dataset_adapters/
│           ├── base.py                     # 数据集 adapter 基类
│           ├── summe.py                    # SumMe 视频/H5 key 对齐逻辑
│           └── tvsum.py                    # TVSum 视频/H5 key 对齐逻辑
│
└── tools/
    ├── generate_dense_captions_gemini.py   # 调 Gemini 生成 dense captions
    ├── mmil-sync.ps1                       # 本地和 server_8 项目同步脚本
    ├── rsync-exclude-pull.txt              # pull 同步排除规则
    └── rsync-exclude-push.txt              # push 同步排除规则