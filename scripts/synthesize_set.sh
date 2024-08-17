export testdata='STL10' # dataset to synthesize set
export root='./datasets/DATA' # path for saving original dataset. if you already have the dataset, just set the your path.   
export savedir='./datasets/generate_DATA' # path for saving synthesized set.


python generate_dataset.py \
    --testdata=$testdata \
    --root=$root \
    --savedir=$savedir
