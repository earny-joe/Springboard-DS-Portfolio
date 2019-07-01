# import fastai libraries
from fastai.vision import *
from fastai.callbacks.hooks import *
from fastai.utils.mem import *
from fastai.callbacks import EarlyStoppingCallback

def baseline_setup(path, train_df, valid_df, pathology):
    '''set up baseline model that uses default transformations, creates an ImageList from train_df, splits 20% for a validation set then adds the original validation set as the test set'''
    # default transformations
    tfms = get_transforms()
    # create ImageList from train_df
    src = (ImageList.from_df(df=train_df, path=path, folder='.', suffix=''))
    src = src.split_by_rand_pct(0.2)
    print('Created ImageList from train_df and randomly split 20% of data for validation set.')
    print('-' * 30)
    
    # determine batch_size based on GPU memory
    free = gpu_mem_get_free_no_cache()
    # the max size of bs depends on the available GPU RAM
    if free > 8200: 
        bs=32
    else:           
        bs=16
    print(f"using bs={bs}, have {free}MB of GPU RAM free.")
    print('-' * 30)
          
    # create ImageDataBunch from src
    data = (src.label_from_df(cols='Cardiomegaly')
        .transform(tfms, size=64)
        .databunch(bs = bs)
        .normalize(imagenet_stats))
    print('Created ImageDataBunch.')
    print('-' * 30)
    # create test set from our original validation set
    data.add_test(ImageList.from_df(valid_df, path=path, folder='.', suffix=''))
    print('Created test set')
    print('-' * 30)
    print('The following print out contains information regarding training, validation and test data sets for ImageDataBunch.')
    print(data)
    
    return data


def lr_finder_plot(learn, start_lr=1e-8, end_lr=100, suggestion=True):
    learn.lr_find(start_lr=start_lr, end_lr=end_lr)
    learn.recorder.plot(suggestion=suggestion)
    
    
    
    
    
    
    