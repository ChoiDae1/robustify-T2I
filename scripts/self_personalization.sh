#accelerate config default # set multigpu 
export MODEL_NAME="DeepFloyd/IF-II-L-v1.0"
export INSTANCE_DIR="datasets/generated_DATA/1shot/STL10"
export testdata="STL10"
export resolution=96 # if STL10 -> 96, others -> 224
export clf_loss_type="CE_CLIP" # CE_CLIP for clip cross entropy, CE_resnet for resnet cross entropy
export num_shot=1 # number of shots for personalization
export output_dir="diffusion_exp/STL10/1shot_1e4_500_ckpt" # path for saving the ckpt
export WANDB_DISABLE_SERVICE=True

accelerate launch --num_processes=4 self_personalization.py \
    --report_to wandb \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --instance_data_dir=$INSTANCE_DIR \
    --resolution=$resolution \
    --train_batch_size=6 \
    --gradient_accumulation_steps=1 \
    --learning_rate=1e-4 \
    --max_train_steps=500 \
    --lr_scheduler="cosine" \
    --pre_compute_text_embeddings \
    --tokenizer_max_length=77 \
    --text_encoder_use_attention_mask \
    --class_labels_conditioning=timesteps \
    --seed=0 \
    --with_clfloss \
    --clf_lambda=0.01 \
    --clf_loss_type=$clf_loss_type \
    --testdata=$testdata \
    --target_type="label" \
    --num_shot=$num_shot \
    --output_dir=$output_dir \

