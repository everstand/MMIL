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