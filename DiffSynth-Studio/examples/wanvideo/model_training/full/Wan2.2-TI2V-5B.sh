export DIFFSYNTH_SKIP_DOWNLOAD=true
export HF_HUB_OFFLINE=1
export DIFFSYNTH_MODEL_BASE_PATH="/mnt/shared-storage-user/mahaoxiang/code/DiffSynth-Studio"

accelerate launch examples/wanvideo/model_training/train.py \
  --dataset_base_path /mnt/shared-storage-user/internvla/Users/mahaoxiang/LIBERO_collected_dataset \
  --dataset_metadata_path /mnt/shared-storage-user/internvla/Users/mahaoxiang/LIBERO_collected_dataset/metadata.csv \
  --height 224 \
  --width 224 \
  --num_frames 41 \
  --dataset_repeat 50 \
  --data_file_keys "video" \
  --model_paths '[["/mnt/shared-storage-user/mahaoxiang/code/DiffSynth-Studio/pretrained_model/Wan2.2-TI2V-5B/diffusion_pytorch_model-00001-of-00003.safetensors", "/mnt/shared-storage-user/mahaoxiang/code/DiffSynth-Studio/pretrained_model/Wan2.2-TI2V-5B/diffusion_pytorch_model-00002-of-00003.safetensors", "/mnt/shared-storage-user/mahaoxiang/code/DiffSynth-Studio/pretrained_model/Wan2.2-TI2V-5B/diffusion_pytorch_model-00003-of-00003.safetensors"], "/mnt/shared-storage-user/mahaoxiang/code/DiffSynth-Studio/pretrained_model/Wan2.2-TI2V-5B/models_t5_umt5-xxl-enc-bf16.pth", "/mnt/shared-storage-user/mahaoxiang/code/DiffSynth-Studio/pretrained_model/Wan2.2-TI2V-5B/Wan2.2_VAE.pth"]' \
  --learning_rate 1e-5 \
  --num_epochs 2 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "/mnt/shared-storage-user/internvla2/mahaoxiang/models/train/Wan2.2-TI2V-5B_full" \
  --trainable_models "dit" \
  --extra_inputs "input_image" \
  --tokenizer_path "/mnt/shared-storage-user/mahaoxiang/code/DiffSynth-Studio/pretrained_model/google/umt5-xxl"

  # --model_id_with_origin_paths "pretrained_model/Wan2.2-TI2V-5B:Wan2.2_VAE.pth,pretrained_model/Wan2.2-TI2V-5B:models_t5_umt5-xxl-enc-bf16.pth,pretrained_model/Wan2.2-TI2V-5B:diffusion_pytorch_model*.safetensors" \

    # --model_paths '["pretrained_model/Wan2.2-TI2V-5B/Wan2.2_VAE.pth","pretrained_model/Wan2.2-TI2V-5B/models_t5_umt5-xxl-enc-bf16.pth","pretrained_model/Wan2.2-TI2V-5B/diffusion_pytorch_model-00001-of-00003.safetensors","pretrained_model/Wan2.2-TI2V-5B/diffusion_pytorch_model-00002-of-00003.safetensors","pretrained_model/Wan2.2-TI2V-5B/diffusion_pytorch_model-00003-of-00003.safetensors"]' \