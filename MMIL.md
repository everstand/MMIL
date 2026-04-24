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

## 生成text feature
python src/make_text_features.py --dataset summe --device cpu --openclip-model ViT-L-14 --openclip-pretrained dfn2b
python src/make_text_features.py --dataset tvsum --device cpu --openclip-model ViT-L-14 --openclip-pretrained dfn2b

## 生成 SumMe-24 caption
python tools/generate_dense_captions_gemini.py --dataset summe --video-dir custom_data/videos/SumMe --h5-path datasets/eccv16_dataset_summe_google_pool5.h5 --out-structured captions_raw/summe24_dense_captions_structured.json --out-simple captions/summe24_dense_captions.json
## 生成 TVSum caption
python tools/generate_dense_captions_gemini.py --dataset tvsum --video-dir custom_data/videos/TVSum --h5-path datasets/eccv16_dataset_tvsum_google_pool5.h5 --out-structured captions_raw/tvsum_dense_captions_structured.json --out-simple captions/tvsum_dense_captions.json

## 训练Summe
CUDA_VISIBLE_DEVICES=6 bash scripts/train_mil_cond_summe_tagr.sh
## 训练TVSum
CUDA_VISIBLE_DEVICES=3 bash scripts/train_mil_cond_tvsum_tagr.sh
# 改变量形式
LAMBDA_AUX=0.05 bash scripts/train_mil_cond_summe.sh
