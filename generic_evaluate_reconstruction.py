from omegaconf import DictConfig, OmegaConf
import hydra
import os
import sys
import numpy as np
import h5py
import scipy.io as spio
import nibabel as nib
import scipy as sp
from PIL import Image
from pathlib import Path 
from skimage.color import rgb2gray
from skimage.metrics import structural_similarity as ssim
import json

#import argparse
#parser = argparse.ArgumentParser(description='Argument Parser')
#parser.add_argument("-sub", "--sub",help="Subject Number",default=1)
#args = parser.parse_args()
#sub=int(args.sub)

from scipy.stats import pearsonr,binom,linregress
import numpy as np

import logging
logging.basicConfig(level = logging.INFO)

log = logging.getLogger(__name__)


def pairwise_corr_all(ground_truth, predictions):
    r = np.corrcoef(ground_truth, predictions)#cosine_similarity(ground_truth, predictions)#
    r = r[:len(ground_truth), len(ground_truth):]  # rows: groundtruth, columns: predicitons
    #print(r.shape)
    # congruent pairs are on diagonal
    congruents = np.diag(r)
    #print(congruents)
    
    # for each column (predicition) we should count the number of rows (groundtruth) that the value is lower than the congruent (e.g. success).
    success = r < congruents
    success_cnt = np.sum(success, 0)
    
    # note: diagonal of 'success' is always zero so we can discard it. That's why we divide by len-1
    perf = np.mean(success_cnt) / (len(ground_truth)-1)
    p = 1 - binom.cdf(perf*len(ground_truth)*(len(ground_truth)-1), len(ground_truth)*(len(ground_truth)-1), 0.5)
    
    return perf, p


def evaluate_reconstructions(image_dir, feats_dir, out_file_path):

    net_list = [
        ('inceptionv3','avgpool'),
        ('clip','final'),
        ('alexnet',2),
        ('alexnet',5),
        ('efficientnet','avgpool'),
        ('swav','avgpool')
        ]

    results = {}
    root_path = "/storage/czw/BrainBitsWIP/"
    test_dir = os.path.join(root_path, f'data/eval_features/test_images')
    num_test = 982
    distance_fn = sp.spatial.distance.correlation
    pairwise_corrs = []

    net_results = {}
    for (net_name,layer) in net_list:
        file_name = '{}/{}_{}.npy'.format(test_dir,net_name,layer)
        gt_feat = np.load(file_name)
        
        file_name = '{}/{}_{}.npy'.format(feats_dir,net_name,layer)
        eval_feat = np.load(file_name)
        
        gt_feat = gt_feat.reshape((len(gt_feat),-1))
        eval_feat = eval_feat.reshape((len(eval_feat),-1))
        
        print(net_name,layer)
        if net_name not in net_results:
            net_results[net_name] = {}
        if layer not in net_results[net_name]:
            net_results[net_name][layer] = {}

        if net_name in ['efficientnet','swav']:
            distance = np.array([distance_fn(gt_feat[i],eval_feat[i]) for i in range(num_test)]).mean()
            print('distance: ', distance)
            net_results[net_name][layer]["distance"] = distance
        else:
            pairwise_corrs.append(pairwise_corr_all(gt_feat[:num_test],eval_feat[:num_test])[0])
            print('pairwise corr: ',pairwise_corrs[-1])
            net_results[net_name][layer]["distance"] = pairwise_corrs[-1]
            
    ssim_list = []
    pixcorr_list = []
    for i in range(982):
        gen_image = Image.open(os.path.join(image_dir, f'{i}.png')).resize((425,425))
        gt_image = Image.open(f'/storage/czw/BrainBitsWIP/data/nsddata_stimuli/test_images/{i}.png')
        gen_image = np.array(gen_image)/255.0
        gt_image = np.array(gt_image)/255.0
        pixcorr_res = np.corrcoef(gt_image.reshape(1,-1), gen_image.reshape(1,-1))[0,1]
        pixcorr_list.append(pixcorr_res)
        gen_image = rgb2gray(gen_image)
        gt_image = rgb2gray(gt_image)
        ssim_res = ssim(gen_image, gt_image, multichannel=True, gaussian_weights=True, sigma=1.5, use_sample_covariance=False, data_range=1.0)
        ssim_list.append(ssim_res)
        
    ssim_list = np.array(ssim_list)
    pixcorr_list = np.array(pixcorr_list)
    print('PixCorr: {}'.format(pixcorr_list.mean()))
    results["PixCorr"] = pixcorr_list.mean()
    print('SSIM: {}'.format(ssim_list.mean()))
    results["SSIM"] = ssim_list.mean()
    
    results["net_results"] = net_results
    Path(out_file_path).mkdir(exist_ok=True, parents=True)
    out_file_path = os.path.join(out_file_path, "results.json") 
    with open(out_file_path, "w") as f:
        json.dump(results, f)

@hydra.main(config_path="conf")
def main(cfg: DictConfig) -> None:
    log.info(f"Run testing for all electrodes in all test_subjects")
    log.info(OmegaConf.to_yaml(cfg, resolve=True))
    out_dir = os.getcwd()
    log.info(f'Working directory {os.getcwd()}')
    if "out_dir" in cfg.exp:
        out_dir = cfg.exp.out_dir
    log.info(f'Output directory {out_dir}')

    image_dir = cfg.exp.images_dir
    feats_dir = cfg.exp.feats_dir
    out_file_path = cfg.exp.out_file_path
    evaluate_reconstructions(image_dir, feats_dir, out_file_path) 

if __name__=="__main__":
    main()

