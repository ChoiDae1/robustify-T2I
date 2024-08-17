export testdata='STL10'
export diffusion_ckpt='diffusion_exp/STL10/1shot_1e4_500_ckpt'
export classifier_method='clip'
export classifier_ckpt='classifier_exp/STL10/clip_1shot_5e7_10_ckpt/checkpoint-last.pt'
export N0=100 
export N=10000
export sigma=0.25 
export outfile='output/STL10/certified/sigma_025.txt'

python evaluate.py \
    --method=IFRobustModel \
    --testdata=$testdata \
    --diffusion_ckpt=$diffusion_ckpt \
    --classifier_method=$classifier_method \
    --classifier_ckpt=$classifier_ckpt \
    --N0=$N0 \
    --N=$N \
    --sigma=$sigma \
    --outfile=$outfile \
