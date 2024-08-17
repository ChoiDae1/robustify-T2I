export generated_root="datasets/generated_DATA/1shot"
export testdata='STL10'
export diffusion_ckpt='diffusion_exp/STL10/1shot_1e4_500_ckpt'
export output_dir='classifier_exp/STL10/clip_1shot_5e7_10_ckpt' # path for saving the ckpt
export classifier_method='clip'

python classifier_finetuning.py \
    --generated_data_root=$generated_root \
    --testdata=$testdata \
    --diffusion_ckpt=$diffusion_ckpt \
    --classifier_method=$classifier_method \
    --num_shot=1  \
    --num_train_epochs=10 \
    --batch_size=256 \
    --lr=5e-7 \
    --min_lr=0 \
    --wd=0.0 \
    --seed=0 \
    --use_scheduler \
    --use_generated_dataset \
    --out_dir=$output_dir \ 

