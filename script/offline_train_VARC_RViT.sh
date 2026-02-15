torchrun --nproc_per_node=4 new_train_RARC.py \
  --epochs 100\
  --depth 10\
  --batch-size 24\
  --image-size 64 \
  --patch-size 2 \
  --learning-rate 1e-4 \
  --weight-decay 0 \
  --embed-dim 512 \
  --num-heads 8 \
  --include-rearc \
  --num-colors 12 \
  --data-root "raw_data/ARC-AGI" \
  --train-split "training" \
  --wandb-project "VisionARC-VIT" \
  --wandb-run-name "VIT-ARC" \
  --best-save-path "saves/new_train_RARC/checkpoint_best.pt" \
  --lr-scheduler "cosine" \
  --architecture "rvit" \
  --vis-every 50 \
  --distributed \
  --use-wandb \


  # torchrun --nproc_per_node=4 offline_train_ARC.py \
  # --epochs 100 \
  # --depth 10 \
  # --batch-size 32 \
  # --image-size 64 \
  # --patch-size 2 \
  # --learning-rate 3e-4 \
  # --weight-decay 0 \
  # --embed-dim 512 \
  # --num-heads 8 \
  # --include-rearc \
  # --num-colors 12 \
  # --data-root "raw_data/ARC-AGI" \
  # --train-split "training" \
  # --wandb-project "VisionARC" \
  # --wandb-run-name "offline_train_VARC" \
  # --save-path "saves/offline_train_ViT/checkpoint_final.pt" \
  # --best-save-path "saves/offline_train_ViT/checkpoint_best.pt" \
  # --lr-scheduler "cosine" \
  # --architecture "vit" \
  # --vis-every 50 \
  # --distributed \
  # --use-wandb \
