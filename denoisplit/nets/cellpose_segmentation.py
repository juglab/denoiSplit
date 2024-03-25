from cellpose import models
from czifile import imread as imread_czi
import numpy as np
import os

def load_czi(fpaths):
    imgs = []
    for fpath in fpaths:
        img = imread_czi(fpath)
        assert img.shape[3] == 1
        img = np.swapaxes(img, 0, 3)
        # the first dimension of img stored in imgs will have dim of 1, where the contenation will happen
        imgs.append(img)
    return imgs
def extension(fpath):
    return os.path.basename(fpath).split('.')[-1]

def load_data(fpaths):
    exts = set([ extension(fpath) for fpath in fpaths])
    assert len(exts) ==1, f'In one call, pass only files with one extension. Found:{exts}'
    if extension(fpaths[0]) == 'czi':
        data = load_czi(fpaths)
    return data

def segment(imgs_2D, use_GPU=True, model_type='nuclei'):
    model = models.Cellpose(gpu=use_GPU, model_type='nuclei')

    # define CHANNELS to run segementation on
    # grayscale=0, R=1, G=2, B=3
    # channels = [cytoplasm, nucleus]
    # if NUCLEUS channel does not exist, set the second channel to 0
    # channels = [0,0]
    # IF ALL YOUR IMAGES ARE THE SAME TYPE, you can give a list with 2 elements
    # channels = [0,0] # IF YOU HAVE GRAYSCALE
    # channels = [2,3] # IF YOU HAVE G=cytoplasm and B=nucleus
    # channels = [2,1] # IF YOU HAVE G=cytoplasm and R=nucleus

    # or if you have different types of channels in each image
    # channels = [[2,3], [0,0], [0,0]]
    channels = [0,0]
    
    # sanity checks on the input. Otherwise, one needs to update channels variable.
    assert isinstance(imgs_2D,list)
    assert all([len(x.shape)==2 for x in imgs_2D])

    # if diameter is set to None, the size of the cells is estimated on a per image basis
    # you can set the average cell `diameter` in pixels yourself (recommended) 
    # diameter can be a list or a single number for all images

    masks, flows, styles, diams = model.eval(imgs_2D, diameter=None, flow_threshold=None, channels=channels)
    return masks, flows, styles, diams