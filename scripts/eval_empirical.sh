export testdata='STL10'
export diffusion_ckpt='diffusion_exp/STL10/1shot_1e4_500_ckpt'
export classifier_method='clip'
export classifier_ckpt='classifier_exp/STL10/clip_1shot_5e7_10_ckpt/checkpoint-last.pt'
export inference_samples=100 # number of noise samples for inference 
export attack_type='pgd' # if you want to evaluate empirical clean accuracy, set attack_type='clean'
export num_noise_vec=32 
export norm_type='l_2'
export test_eps=0.5
export sigma=0.25 # if test_eps=0.5 -> sigma=0.25, test_eps=1.0 -> sigma=0.5.
export test_numsteps=100
export outfile='output/STL10/empirical/pgd_l_2_1.0_100.txt'

python evaluate.py \
    --method=IFRobustModel \
    --testdata=$testdata \
    --empirical \
    --diffusion_ckpt=$diffusion_ckpt \
    --classifier_method=$classifier_method \
    --classifier_ckpt=$classifier_ckpt \
    --sigma=$sigma \
    --N0=$inference_samples \
    --attack_type=$attack_type \
    --num_noise_vec=$num_noise_vec \
    --norm_type=$norm_type \
    --test_eps=$test_eps \
    --test_numsteps=$test_numsteps \
    --outfile=$outfile \
