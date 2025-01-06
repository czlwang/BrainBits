#CUDA_VISIBLE_DEVICES=1 python3 -m learn_compression +data_exp=learn_compression ++data_exp.in_dir=orig ++data_exp.subject=S1 ++data_exp.n_components=2000 ++reg.batch_size=512 ++reg.optim="Adam" ++reg.lr=0.0005 ++reg.n_epochs=5 ++reg.device="cuda" ++data_exp.out_path=/storage/czw/semantic-decoding-brainbits/compression_models ++reg.use_iterative=False ++reg.pca_preload=False

import copy
from StimulusModel import LMFeatures
from GPT import GPT
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from glob import glob as glob
import h5py
import logging
from omegaconf import DictConfig, OmegaConf
import hydra
import os 
import json
import numpy as np
from sklearn.decomposition import PCA
from pathlib import Path
from utils_stim import get_stim
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from tqdm import tqdm as tqdm

DATA_TRAIN_DIR = "data_train"


class Brain2GPT(Dataset):
    def __init__(self, resp, gpt_embed):
        self.resp = resp
        self.gpt_embed = gpt_embed

    def __len__(self):
        return len(self.resp)

    def __getitem__(self, idx):
        return { "resp": torch.FloatTensor(self.resp[idx]),
                 "gpt_embed": torch.FloatTensor(self.gpt_embed[idx])
        }

class CompressionNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, cfg, embed_w=None):
        super(CompressionNet, self).__init__()
        self.fmri2embed = nn.Linear(input_dim, hidden_dim)  
        self.dense1_bn = nn.LayerNorm(hidden_dim)
        self.embed2gpt = nn.Linear(hidden_dim, output_dim)

        if 'pca_preload' in cfg and cfg.pca_preload:
            self.fmri2embed.weight = torch.nn.Parameter(torch.FloatTensor(embed_w))

    def forward(self, x, return_intermediate=False):
        x = self.fmri2embed(x)
        x = self.dense1_bn(x)
        if return_intermediate:
            return x
        x = self.embed2gpt(x)
        return x

def get_loss(gpt_preds, batch, reg_cfg, n_batch):
    criterion = nn.MSELoss()
    targets = batch["gpt_embed"].to(reg_cfg.device)
    loss = criterion(targets, gpt_preds)
    return loss

def get_eval_loss(model, val_loader, reg_cfg):
    criterion = nn.MSELoss()
    model.eval()
    with torch.no_grad():
        total_loss = 0
        for batch in tqdm(val_loader):
            inputs = batch["resp"].to(reg_cfg.device)
            n_batch = inputs.shape[0]
            gpt_preds = model(inputs) 
            loss = get_loss(gpt_preds, batch, reg_cfg, n_batch)
            total_loss += loss.item()
    return total_loss/len(val_loader)

def train_linear_mapping(model, train_loader, val_loader, reg_cfg):
    if reg_cfg.optim == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=reg_cfg.lr, momentum=0.0, weight_decay=0.01)
    elif reg_cfg.optim == "Adam":
        optimizer = optim.AdamW(model.parameters(), lr=reg_cfg.lr, weight_decay=0.01)
    else:
        print("no optim")

    iterative_on = True
    if 'use_iterative' in reg_cfg:
        iterative_on = reg_cfg.use_iterative

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
    min_eval_loss = 100
    best_model = copy.deepcopy(model)
    criterion = nn.MSELoss()
    named_parameters = list(model.named_parameters())

    freeze_embed = False
    lr_1, lr_2 = reg_cfg.lr, reg_cfg.lr
    for epoch in range(reg_cfg.n_epochs):
        if iterative_on and epoch%10==0: #TODO HARDCODE
            freeze_embed = not freeze_embed
            if freeze_embed:
                lr_1 = optimizer.param_groups[0]['lr']
                optimizer = optim.AdamW(model.parameters(), lr=lr_2, weight_decay=0.001)
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
            else:
                lr_2 = optimizer.param_groups[0]['lr']
                optimizer = optim.AdamW(model.parameters(), lr=lr_1, weight_decay=0.001)
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
        if iterative_on and freeze_embed:
            for name, param in model.named_parameters():
                if 'fmri2embed' in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
        else:
            for name, param in model.named_parameters():
                if 'fmri2embed' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        with tqdm(total=len(train_loader)) as bar:
            bar.set_description(f"Epoch {epoch}")
            train_loss, train_text_loss = 0, 0
            for batch in train_loader:
                inputs = batch["resp"].to(reg_cfg.device)
                n_batch = inputs.shape[0]
                optimizer.zero_grad()

                gpt_preds = model(inputs)
                loss = get_loss(gpt_preds, batch, reg_cfg, n_batch)
                loss.backward()
                optimizer.step()
                bar.set_postfix({"loss":float(loss)})
                bar.update()
                train_loss += float(loss)
            
            avg_loss = train_loss/len(train_loader)

            eval_loss = get_eval_loss(model, val_loader, reg_cfg)
            bar.set_postfix({"loss":float(eval_loss)})

        if eval_loss < min_eval_loss:
            min_eval_loss = eval_loss
            best_model = copy.deepcopy(model)
        scheduler.step(avg_loss)
    return best_model

def transform_data(model, data, device):
    model.eval()
    with torch.no_grad():
        inputs = torch.FloatTensor(data).to(device)
        outputs = model(inputs, return_intermediate=True)
    return outputs.detach().cpu().numpy()
 
def write_all_train_transformed(stories, model, cfg):
    out_dir = f"bbits_{cfg.data_exp.n_components}"
    if 'out_dir' in cfg.data_exp:
        out_dir = cfg.data_exp.out_dir

    subject = cfg.data_exp.subject
    out_dir_path = os.path.join(DATA_TRAIN_DIR, "train_response", subject, out_dir)
    Path(out_dir_path).mkdir(exist_ok=True, parents=True)

    for story in stories:
        #log.info(f"writing data {story}")
        subject_dir = os.path.join(DATA_TRAIN_DIR, "train_response", subject, cfg.data_exp.in_dir)
        resp_path = os.path.join(subject_dir, "%s.hf5" % story)
        hf = h5py.File(resp_path, "r")
        data = np.nan_to_num(hf["data"][:])
        if cfg.data_exp.n_components != 0:
            transformed = transform_data(model, data, cfg.reg.device)
        else:
            transformed = np.random.random(data.shape)
        hf.close()

        out_path = os.path.join(out_dir_path, f"{story}.hf5")
        hf_out = h5py.File(out_path, "w")
        hf_out.create_dataset('data', data=transformed)
        hf_out.close()

def write_all_test_transformed(model, cfg):
    subject = cfg.data_exp.subject

    out_dir = f"bbits_{cfg.data_exp.n_components}"
    if 'out_dir' in cfg.data_exp:
        out_dir = cfg.data_exp.out_dir

    root_path = "/storage/czw/semantic-decoding-brainbits"
    exps = ["imagined_speech",  "perceived_movie",  "perceived_multispeaker",  "perceived_speech"]
    for exp in exps:
        tasks = glob(os.path.join(root_path, "data_test", "test_response", subject, exp, 'orig', "*"))
        out_dir_path = os.path.join(root_path, "data_test", "test_response", subject, exp, out_dir)
        Path(out_dir_path).mkdir(exist_ok=True, parents=True)

        for task_path in tasks:
            task_name = Path(task_path).stem
            #log.info(f"writing {task_name}")
            hf = h5py.File(task_path, "r")
            data = np.nan_to_num(hf["data"][:])
            if cfg.data_exp.n_components != 0:
                transformed = transform_data(model, data, cfg.reg.device)
            else:
                transformed = np.random.random(data.shape)
            hf.close()

            out_path = os.path.join(out_dir_path, f"{task_name}.hf5")
            hf_out = h5py.File(out_path, "w")
            hf_out.create_dataset('data', data=transformed)
            hf_out.close()

log = logging.getLogger(__name__)
@hydra.main(config_path="../conf")
def main(cfg: DictConfig) -> None:
    log.info("Learn compression")
    log.info(OmegaConf.to_yaml(cfg, resolve=True))
    log.info(f'Working directory {os.getcwd()}')

    if "sessions" in cfg.data_exp:
        sessions = cfg.data_exp.sessions
    else:
        sessions = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 18, 20]
    subject = cfg.data_exp.subject


    # training stories
    stories = []
    with open(os.path.join(DATA_TRAIN_DIR, "sess_to_story.json"), "r") as f:
        sess_to_story = json.load(f) 
    for sess in sessions:
        stories.extend(sess_to_story[str(sess)])

    stories = stories[:1] #TODO

    subject_dir = os.path.join(DATA_TRAIN_DIR, "train_response", subject, cfg.data_exp.in_dir)
    resp = {}
    for story in stories:
        log.info(f"reading data {story}")
        resp_path = os.path.join(subject_dir, "%s.hf5" % story)
        hf = h5py.File(resp_path, "r")
        resp[story] = np.nan_to_num(hf["data"][:])
        hf.close()

    all_stack = np.vstack([resp[story] for story in stories]) 

    print('about to load gpt')
    DATA_LM_DIR = "/storage/czw/semantic-decoding-brainbits/data_lm"
    GPT_DEVICE = "cuda"
    GPT_LAYER = 9
    GPT_WORDS = 5

    gpt = "perceived" #this is the case for all train data
    with open(os.path.join(DATA_LM_DIR, gpt, "vocab.json"), "r") as f:
        gpt_vocab = json.load(f)
    gpt = GPT(path = os.path.join(DATA_LM_DIR, gpt, "model"), vocab = gpt_vocab, device = GPT_DEVICE)
    features = LMFeatures(model = gpt, layer = GPT_LAYER, context_words = GPT_WORDS)
    
    rstim, tr_stats, word_stats = get_stim(stories, features)

    hidden_dim = cfg.data_exp.n_components
    input_dim = all_stack.shape[-1]
    output_dim = rstim.shape[-1]

    all_stack = (all_stack - all_stack.mean(axis=0))/all_stack.std(axis=0)

    rstim = (rstim - rstim.mean(axis=0))/rstim.std(axis=0)

    print(all_stack.shape)
    dataset = Brain2GPT(all_stack, rstim)

    val_split = 0.15 #TODO hardcode
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
    
    reg_cfg = cfg.reg

    train_data = Subset(dataset, train_idx)
    val_data = Subset(dataset, val_idx)
    train_loader = DataLoader(train_data, batch_size=reg_cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=reg_cfg.batch_size, shuffle=True)

    best_eval_loss = 100
    for i in range(3):
        model = CompressionNet(input_dim, hidden_dim, output_dim, cfg.reg)#, embed_w=pca_components)
        model = model.to(reg_cfg.device)
        model = train_linear_mapping(model, train_loader, val_loader, reg_cfg)
        eval_loss = get_eval_loss(model, val_loader, reg_cfg)
        print(eval_loss)
        if eval_loss < best_eval_loss:
            best_model = copy.deepcopy(model)

    out_path = cfg.data_exp.out_path
    Path(out_path).mkdir(parents=True, exist_ok=True)

    model_out_path = os.path.join(out_path, f"compression_model_{cfg.data_exp.n_components}.pth")
    torch.save(best_model.state_dict(), model_out_path)

    results_out_path = os.path.join(out_path, f"bbits_{cfg.data_exp.n_components}_results.json")
    results = {"best_val_loss": best_eval_loss}
    with open(results_out_path, "w") as f:
        json.dump(results, f)

    write_all_train_transformed(stories, best_model, cfg)
    write_all_test_transformed(best_model, cfg)

if __name__ == "__main__":
    main()

