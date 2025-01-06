import vdvae_decode_utils
import versatilediffusion_utils
import wandb
import copy
from omegaconf import DictConfig, OmegaConf
import hydra
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
import os
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm as tqdm
import argparse
import logging
import json
import sys
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import models

logging.basicConfig(level = logging.INFO)

log = logging.getLogger(__name__)
USER = os.getenv('USER')
SAVE_ROOT_PATH = Path('./data/predicted_features/')
BD_ROOT_PATH = Path('../brain-diffuser')


class fMRI2latent(Dataset):
    def __init__(self, fmri_data, vdvae_embeds, clip_text_embeds, clip_vision_embeds):
        self.fmri_data = fmri_data
        self.vdvae_embeds = vdvae_embeds
        self.clip_text_embeds = clip_text_embeds
        self.clip_vision_embeds = clip_vision_embeds

    def __len__(self):
        return len(self.fmri_data)

    def __getitem__(self, idx):
        return {"inputs": torch.FloatTensor(self.fmri_data[idx]), 
                "vdvae_targets": torch.FloatTensor(self.vdvae_embeds[idx]),
                "clip_text_targets": torch.FloatTensor(self.clip_text_embeds[idx]),
                "clip_vision_targets": torch.FloatTensor(self.clip_vision_embeds[idx])}

class BottleneckLinear(nn.Module):
    def __init__(self, input_size, bottleneck_size, d_vdvae, d_clip, n_text, n_vision, cfg, embed_w=None, multi_gpu=False):
        super().__init__()

        self.fmri2embed = nn.Sequential(nn.Linear(input_size, bottleneck_size, bias=False),
                                       )
        if 'pca_preload' in cfg and cfg.pca_preload:
            self.fmri2embed[0].weight = torch.nn.Parameter(torch.FloatTensor(embed_w))
        else:
            self.fmri2embed = nn.Sequential(nn.Linear(input_size, bottleneck_size, bias=True),)

        self.vdvae_embed = nn.Linear(bottleneck_size, d_vdvae)
        self.clip_text_embed = nn.Linear(bottleneck_size, n_text*d_clip) 
        self.clip_vision_embed = nn.Linear(bottleneck_size, n_vision*d_clip)

    def forward(self, fmri_inputs):
        bottleneck_mapping = self.fmri2embed(fmri_inputs)
        vdvae_mapping = self.vdvae_embed(bottleneck_mapping)
        clip_text_mapping = self.clip_text_embed(bottleneck_mapping)
        clip_vision_mapping = self.clip_vision_embed(bottleneck_mapping)
        return vdvae_mapping, clip_text_mapping, clip_vision_mapping, bottleneck_mapping

def get_loss(vdvae_preds, clip_text_preds, clip_vision_preds, batch, reg_cfg, n_batch):
    criterion = nn.MSELoss()
    clip_text_targets = batch["clip_text_targets"].to(reg_cfg.device)
    clip_vision_targets = batch["clip_vision_targets"].to(reg_cfg.device)
    vdvae_targets = batch["vdvae_targets"].to(reg_cfg.device)

    vdvae_loss = criterion(vdvae_preds, vdvae_targets)
    clip_text_loss = criterion(clip_text_preds, clip_text_targets.reshape(n_batch,-1))
    clip_vision_loss = criterion(clip_vision_preds, clip_vision_targets.reshape(n_batch,-1))
    vdvae_weight, clip_text_weight, clip_vision_weight = reg_cfg.get("objective_weights", (0,0,0))
    loss = vdvae_weight*vdvae_loss + clip_text_weight*clip_text_loss + clip_vision_weight*clip_vision_loss
    return loss, vdvae_loss, clip_text_loss, clip_vision_loss

def get_eval_loss(model, val_loader, reg_cfg):
    criterion = nn.MSELoss()
    model.eval()
    with torch.no_grad():
        total_loss, total_clip_text_loss, total_clip_vision_loss, total_vdvae_loss = 0, 0, 0, 0
        for batch in tqdm(val_loader):
            inputs = batch["inputs"].to(reg_cfg.device)
            n_batch = inputs.shape[0]
            vdvae_preds, clip_text_preds, clip_vision_preds, _ = model(inputs) 
            loss, vdvae_loss, clip_text_loss, clip_vision_loss = get_loss(vdvae_preds, clip_text_preds, clip_vision_preds, batch, reg_cfg, n_batch)
            total_loss += loss.item()
            total_clip_text_loss += clip_text_loss.item()
            total_clip_vision_loss += clip_vision_loss.item()
            total_vdvae_loss += vdvae_loss.item()
    return total_loss/len(val_loader), total_clip_text_loss/len(val_loader), total_clip_vision_loss/len(val_loader), total_vdvae_loss/len(val_loader)

def scale_preds(vdvae_preds, train_stats):
    train_mean, train_std = train_stats
    vdvae_preds_arr = vdvae_preds.detach()
    epsilon = 0.0001
    std_norm_test_latent = (vdvae_preds - torch.mean(vdvae_preds_arr,axis=0)) / (torch.nan_to_num(torch.std(vdvae_preds_arr,axis=0),nan=epsilon))
    pred_latents = std_norm_test_latent * torch.FloatTensor(train_std).to(vdvae_preds.device) + torch.FloatTensor(train_mean).to(vdvae_preds.device)
    return pred_latents

def train_linear_mapping(model, train_loader, val_loader, reg_cfg, train_stats, save_path_dir):
    weight_decay = reg_cfg.get("weight_decay", 0.001)
    if reg_cfg.optim == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=reg_cfg.lr, momentum=0.0, weight_decay=weight_decay)
    elif reg_cfg.optim == "Adam":
        optimizer = optim.AdamW(model.parameters(), lr=reg_cfg.lr, weight_decay=weight_decay)
    else:
        print("no optim")

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=reg_cfg.n_epochs)
    min_eval_loss = 100
    best_model = copy.deepcopy(model)
    criterion = nn.MSELoss()
    named_parameters = list(model.named_parameters())

    freeze_embed = False
    n_epoch_per_stage = reg_cfg.get("n_epoch_per_stage", 20)
    for epoch in range(reg_cfg.n_epochs):
        if epoch%10==0 and reg_cfg.get("save_checkpoints", False):
            torch.save(model.state_dict(), save_path_dir / f"weights_{epoch}.pth")

        with tqdm(total=len(train_loader)) as bar:
            bar.set_description(f"Epoch {epoch}")
            train_loss, train_vdvae_loss, train_text_loss, train_vision_loss = 0, 0, 0, 0
            for batch in train_loader:
                inputs = batch["inputs"].to(reg_cfg.device)
                n_batch = inputs.shape[0]
                #targets = batch["targets"].to(reg_cfg.device) #TODO
                #targets = batch["targets"].cuda(1)
                optimizer.zero_grad()

                vdvae_preds, clip_text_preds, clip_vision_preds, _ = model(inputs)
                loss, vdvae_loss, clip_text_loss, clip_vision_loss = get_loss(vdvae_preds, clip_text_preds, clip_vision_preds, batch, reg_cfg, n_batch)
                loss.backward()
                optimizer.step()
                bar.set_postfix({"v":float(vdvae_loss)})
                bar.update()
                train_loss += float(loss)
                train_vdvae_loss += float(vdvae_loss)
                train_text_loss += float(clip_text_loss)
                train_vision_loss += float(clip_vision_loss)
            avg_loss = train_loss/len(train_loader)
            avg_vdvae_loss = train_vdvae_loss/len(train_loader)
            avg_vision_loss = train_vision_loss/len(train_loader)
            avg_text_loss = train_text_loss/len(train_loader)

            eval_loss, eval_clip_text_loss, eval_clip_vision_loss, eval_vdvae_loss = get_eval_loss(model, val_loader, reg_cfg)
            bar.set_postfix({"eval": eval_loss, "mse":avg_loss, "v": avg_vdvae_loss})
            wandb.log({"val_loss": eval_loss, 
                       "train_loss": avg_loss,
                       "train_vdvae_loss": avg_vdvae_loss,
                       "train_vision_loss": avg_vision_loss,
                       "train_text_loss": avg_text_loss,
                       "val_clip_text_loss": eval_clip_text_loss,
                       "val_clip_vision_loss": eval_clip_vision_loss,
                       "val_vdvae_loss": eval_vdvae_loss,
                       })
        if eval_loss < min_eval_loss:
            min_eval_loss = eval_loss
            best_model = copy.deepcopy(model)
            arr = model.fmri2embed[0].weight.detach().cpu().numpy()
            print("eval loss is better")
        scheduler.step()

    #return model#TODO
    return best_model, min_eval_loss#TODO

def eval_model(model, test_loader, device): #, test_fmri):#TODO):
    model.eval()
    with torch.no_grad():
        all_vdvae_preds, all_clip_text_preds, all_clip_vision_preds, all_intermediates = [], [], [], []
        for batch in tqdm(test_loader):
            inputs = batch["inputs"].to(device)
            vdvae_preds, clip_text_preds, clip_vision_preds, intermediate = model(inputs) #TODO
            all_vdvae_preds.append(vdvae_preds.cpu().detach().numpy())
            all_clip_text_preds.append(clip_text_preds.cpu().detach().numpy())
            all_clip_vision_preds.append(clip_vision_preds.cpu().detach().numpy())
            all_intermediates.append(intermediate.cpu().detach().numpy())
        all_vdvae_preds = np.concatenate(all_vdvae_preds)
        all_clip_text_preds = np.concatenate(all_clip_text_preds)
        all_clip_vision_preds = np.concatenate(all_clip_vision_preds)
        all_intermediates = np.concatenate(all_intermediates)
    return all_vdvae_preds, all_clip_text_preds, all_clip_vision_preds, all_intermediates

def scale_latents(pred_test_latent, train_latents):
    std_norm_test_latent = (pred_test_latent - np.mean(pred_test_latent,axis=0)) / np.std(pred_test_latent,axis=0)
    pred_latents = std_norm_test_latent * np.std(train_latents,axis=0) + np.mean(train_latents,axis=0)
    return pred_latents

def get_vdvae_targets(sub):
    log.info("Getting VDVAE targets")

    #get latent targets
    nsd_path = 'data/extracted_features/subj{:02d}/nsd_vdvae_features_31l.npz'.format(sub)
    nsd_features = np.load(BD_ROOT_PATH / nsd_path)

    train_latents = nsd_features['train_latents']
    test_latents = nsd_features['test_latents']

    return train_latents, test_latents

def get_fmri_inputs(sub, cfg):
    #get fmri inputs
    log.info("Getting fMRI inputs")
    train_path = 'data/processed_data/subj{:02d}/nsd_train_fmriavg_nsdgeneral_sub{}.npy'.format(sub,sub)
    train_fmri = np.load(BD_ROOT_PATH / train_path)
    test_path = 'data/processed_data/subj{:02d}/nsd_test_fmriavg_nsdgeneral_sub{}.npy'.format(sub,sub)
    test_fmri = np.load(BD_ROOT_PATH / test_path)

    train_fmri = train_fmri/300
    test_fmri = test_fmri/300

    norm_mean_train = np.mean(train_fmri, axis=0)
    norm_scale_train = np.std(train_fmri, axis=0, ddof=1)
    train_fmri = (train_fmri - norm_mean_train) / norm_scale_train
    test_fmri = (test_fmri - norm_mean_train) / norm_scale_train

    if cfg.exp.get("random_inputs", False):
        train_fmri = np.random.random(train_fmri.shape)
        test_fmri = np.random.random(test_fmri.shape)

    return train_fmri, test_fmri

def get_vd_images(vdvae_images, vision_preds, text_preds):
    net, sampler = versatilediffusion_utils.load_net_and_sampler()
    images = []
    text_preds = torch.tensor(text_preds).half().cuda(1)
    vision_preds = torch.tensor(vision_preds).half().cuda(1)
    for i in range(3):
        vdvae_image = vdvae_images[i]
        cim = vision_preds[i].unsqueeze(0)
        ctx = text_preds[i].unsqueeze(0)
        img = versatilediffusion_utils.sample_image(vdvae_image, cim, ctx, net, sampler)[0]
        images.append(img)
    wandb.log({"versatilediffusion images": [wandb.Image(image) for image in images]})
    return images

def get_vdvae_images(arr, sub, bottleneck_size, save_path_dir):
    pred_latents_path = os.path.join(save_path_dir, "vdvae_preds.npy")
    images = []
    for i in range(3):
        img = vdvae_decode_utils.get_image(pred_latents_path, sub, i)
        images.append(img)
    wandb.log({"vdvae images": [wandb.Image(image) for image in images]})
    return images

def save_preds(arr, sub, bottleneck_size, out_name, save_path_dir):
    save_path_dir.mkdir(parents=True, exist_ok=True)
    np.save(save_path_dir / f"{out_name}.npy", arr)

def save_weights(model, sub, bottleneck_size, save_path_dir):
    arr = model.fmri2embed[0].weight.detach().cpu().numpy()
    save_path_dir.mkdir(parents=True, exist_ok=True)
    np.save(save_path_dir / f"compression_weights.npy", arr)
    #torch.save(model.state_dict(), save_path_dir / f"final_weights.pth")

def get_clip_text_latents(sub):
    log.info("Getting CLIP text targets")

    #get latent targets
    train_path = 'data/extracted_features/subj{:02d}/nsd_cliptext_train.npy'.format(sub)
    train_clip = np.load(BD_ROOT_PATH / train_path)
    test_path = 'data/extracted_features/subj{:02d}/nsd_cliptext_test.npy'.format(sub)
    test_clip = np.load(BD_ROOT_PATH / test_path)
    out_name = "clip_text_preds"

    return train_clip, test_clip

def get_clip_vision_latents(sub):
    log.info("Getting CLIP vision targets")

    #get latent targets
    train_path = 'data/extracted_features/subj{:02d}/nsd_clipvision_train.npy'.format(sub)
    train_clip = np.load(BD_ROOT_PATH / train_path)
    test_path = 'data/extracted_features/subj{:02d}/nsd_clipvision_test.npy'.format(sub)
    test_clip = np.load(BD_ROOT_PATH / test_path)
    out_name = "clip_vision_preds"

    return train_clip, test_clip


def train_all(sub, bottleneck_size, train_fmri, test_fmri, reg_cfg, save_path_dir, cfg):
    clip_text_train, clip_text_test = get_clip_text_latents(sub)
    clip_vision_train, clip_vision_test = get_clip_vision_latents(sub)
    vdvae_embeds_train, vdvae_embeds_test = get_vdvae_targets(sub)

    n_train, n_text, d_clip_embed = clip_text_train.shape
    _, n_vision, _ = clip_vision_train.shape

    _, d_vdvae = vdvae_embeds_train.shape
    n_test, _, = vdvae_embeds_test.shape

    val_split = 0.15 #TODO hardcode
    all_train_data = fMRI2latent(train_fmri, vdvae_embeds_train, clip_text_train, clip_vision_train)
    train_idx, val_idx = train_test_split(list(range(len(all_train_data))), test_size=val_split)
    
    train_input_arr = train_fmri[train_idx]
    #pca = PCA(n_components=bottleneck_size)
    #pca.fit(train_input_arr)
    #pca_components = pca.components_

    train_latent_arr = vdvae_embeds_train[train_idx]
    train_std = np.std(train_latent_arr, axis=0)
    train_mean = np.mean(train_latent_arr, axis=0)
    train_stats = (train_mean, train_std)

    train_data = Subset(all_train_data, train_idx)
    val_data = Subset(all_train_data, val_idx)
    train_loader = DataLoader(train_data, batch_size=reg_cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=reg_cfg.batch_size, shuffle=True)

    model = BottleneckLinear(train_fmri.shape[-1], bottleneck_size, d_vdvae, d_clip_embed, n_text, n_vision, reg_cfg)

    model = model.to(reg_cfg.device)
    #if device=="cuda":
    #    model= nn.DataParallel(model)
    log.info("Training fMRI2latent mapping")

    model, min_eval_loss = train_linear_mapping(model, train_loader, val_loader, reg_cfg, train_stats, save_path_dir)
    wandb.log({"min_eval_loss": min_eval_loss})
    
    log.info("fMRI2latent test evaluation")
    test_data = fMRI2latent(test_fmri, vdvae_embeds_test, clip_text_test, clip_vision_test)
    test_loader = DataLoader(test_data, batch_size=reg_cfg.batch_size, shuffle=False)

    vdvae_preds, text_preds, vision_preds, test_intermediates = eval_model(model, test_loader, reg_cfg.device)
    text_preds = text_preds.reshape(n_test, n_text, d_clip_embed)
    vision_preds = vision_preds.reshape(n_test, n_vision, d_clip_embed)

    scaled_vdvae_preds = scale_latents(vdvae_preds, vdvae_embeds_train)
    scaled_vision_preds = scale_latents(vision_preds, clip_vision_train)
    scaled_text_preds = scale_latents(text_preds, clip_text_train)

    save_preds(scaled_vdvae_preds, sub, bottleneck_size, "vdvae_preds", save_path_dir)
    save_preds(scaled_text_preds, sub, bottleneck_size, "clip_text_preds", save_path_dir)
    save_preds(scaled_vision_preds, sub, bottleneck_size, "clip_vision_preds", save_path_dir)

    vdvae_images = get_vdvae_images(scaled_vdvae_preds, sub, bottleneck_size, save_path_dir)
    vd_images = get_vd_images(vdvae_images, scaled_vision_preds, scaled_text_preds)

    save_preds(vdvae_preds, sub, bottleneck_size,   "unscaled_vdvae_preds", save_path_dir)
    save_preds(text_preds, sub, bottleneck_size,   "unscaled_clip_text_preds", save_path_dir)
    save_preds(vision_preds, sub, bottleneck_size, "unscaled_clip_vision_preds", save_path_dir)

    save_preds(test_intermediates, sub, bottleneck_size, "test_intermediates", save_path_dir)

    all_train_loader = DataLoader(all_train_data, batch_size=reg_cfg.batch_size)#TODO really I should use same train and val split

    save_weights(model, sub, bottleneck_size, save_path_dir)

    if reg_cfg.get("save_train_data", False):
        train_vdvae_preds, train_text_preds, train_vision_preds, train_intermediates = eval_model(model, all_train_loader, reg_cfg.device)#, test_fmri)
        save_preds(train_intermediates, sub, bottleneck_size, "train_intermediates", save_path_dir)

        train_text_preds = train_text_preds.reshape(n_train, n_text, d_clip_embed)
        train_vision_preds = train_vision_preds.reshape(n_train, n_vision, d_clip_embed)

        train_scaled_vdvae_preds =  scale_latents(train_vdvae_preds, vdvae_embeds_train)
        train_scaled_vision_preds = scale_latents(train_vision_preds, clip_vision_train)
        train_scaled_text_preds =   scale_latents(train_text_preds, clip_text_train)

        save_preds(train_scaled_vdvae_preds, sub, bottleneck_size,  "train_vdvae_preds", save_path_dir)
        save_preds(train_scaled_text_preds, sub, bottleneck_size,   "train_clip_text_preds", save_path_dir)
        save_preds(train_scaled_vision_preds, sub, bottleneck_size, "train_clip_vision_preds", save_path_dir)

@hydra.main(config_path="conf")
def main(cfg: DictConfig) -> None:
    log.info(f"Run testing for all electrodes in all test_subjects")
    log.info(OmegaConf.to_yaml(cfg, resolve=True))
    out_dir = os.getcwd()
    log.info(f'Working directory {os.getcwd()}')
    if "out_dir" in cfg.exp:
        out_dir = cfg.exp.out_dir
    log.info(f'Output directory {out_dir}')

    wandb.init(
        # set the wandb project where this run will be logged
        project="brainbits",

        # track hyperparameters and run metadata
        config = OmegaConf.to_container(cfg, resolve=True)
    )

    sub = cfg.exp["sub"]

    train_fmri, test_fmri = get_fmri_inputs(sub, cfg)
    
    bottleneck_sizes = cfg.exp["bottlenecks"]
    reg_cfg = cfg.exp.reg

    #images = [PIL.Image.fromarray(image) for image in image_array]

    #wandb.log({"examples": [wandb.Image(image) for image in images]})

    for bottleneck_size in bottleneck_sizes:
        save_path_dir = SAVE_ROOT_PATH / f'subj{sub:02d}/train_single/train_single_{bottleneck_size}/'
        if cfg.exp.get("random_inputs", False):
            save_path_dir = SAVE_ROOT_PATH / f'subj{sub:02d}/train_single/train_single_random_brain_{bottleneck_size}/'
        save_path_dir.mkdir(parents=True, exist_ok=True)
        train_all(sub, bottleneck_size, train_fmri, test_fmri, reg_cfg, save_path_dir, cfg)
    wandb.finish()

if __name__=="__main__":
    # _debug = '''train.py +exp=latent_reg ++exp.bottlenecks=[5] ++exp.reg.batch_size=128 ++exp.reg.n_epochs=1 ++exp.reg.optim="SGD" ++exp.reg.device="cpu"'''
    # sys.argv = _debug.split(" ")
    main()



