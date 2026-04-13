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
