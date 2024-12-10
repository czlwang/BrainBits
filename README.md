# BrainDiffuser case study

Much of this code is adapted from the [brain-diffuser](https://github.com/ozcelikfu/brain-diffuser) repo.

## Prerequisites
- Follow the installation and data downloading steps from [brain-diffuser](https://github.com/ozcelikfu/brain-diffuser)
- Also follow the steps to extract the VDVAE and CLIP features
- You should also move over the `versatile_diffusion` directory from `brain-diffuser` to this directory

The code below expects a neighboring directory named `brain-diffuser` that is organized as given in [https://github.com/ozcelikfu/brain-diffuser](brain-diffuser).

First, we need to learn the mappings to VDVAE, CLIP text, and CLIP vision latents. Here's how to do that with a bottleneck of 50 dimensions:
```
BOTTLENECK=50; SUB=1; CUDA_VISIBLE_DEVICES=1 python train_single.py \
+exp=train_single_template ++exp.bottlenecks=[${BOTTLENECK}] ++exp.reg.batch_size=128 \
++exp.reg.n_epochs=1 ++exp.reg.optim="Adam" ++exp.reg.lr=0.01 ++exp.sub=${SUB} \
++exp.reg.save_checkpoints=False ++exp.reg.weight_decay=0.1 \
'++exp.reg.objective_weights=[1,2,4]' hydra.job.chdir=False \
++exp.out_dir=debug_bottleneck_${BOTTLENECK}_sub_${SUB}
```

Next, we need to create the VDVAE images.
```
BOTTLENECK=50; SUB=1; python generic_vdvae_reconstruct_images.py -results_path=./results/vdvae/subj0${SUB}/train_single/train_single_${BOTTLENECK} -pred_latents_path=./data/predicted_features/subj0${SUB}/train_single/train_single_${BOTTLENECK}/vdvae_preds.npy -sub ${SUB}
```
- `pred_latents_path` is set to point to the outputs of the bottlenecking command (above)

Then, we actually reconstruct the images using the diffusion model.
```
BOTTLENECK=50; SUB=1; python generic_versatilediffusion_reconstruct_images.py -sub ${SUB} \
-vdvae_images=./results/vdvae/subj0${SUB}/train_single/train_single_${BOTTLENECK}/ \
-cliptext_preds=./data/predicted_features/subj0${SUB}/train_single/train_single_${BOTTLENECK}/clip_text_preds.npy \
-clipvision_preds=./data/predicted_features/subj0${SUB}/train_single/train_single_${BOTTLENECK}/clip_vision_preds.npy \
-out_path=./results/versatile_diffusion/subj0${SUB}/train_single_${BOTTLENECK}
```
- make sure that `${BOTTLENECK}` is consistent with the previous step

Finally, we evaluate the reconstructions. First, we need to save the test images and extract features for the test images.
```
python save_test_images.py
```
and
```
BOTTLENECK=50; SUB=1; CUDA_VISIBLE_DEVICES=1,2 python generic_eval_extract_features.py \
+exp=eval_extract_features ++exp.feats_dir=./eval_features/subj0${SUB}/train_single_${BOTTLENECK} \
++exp.images_dir=./results/versatile_diffusion/subj0${SUB}/train_single_${BOTTLENECK}/
```
Then we run the evaluation
```
BOTTLENECK=50; SUB=1; CUDA_VISIBLE_DEVICES=0,1 python generic_evaluate_reconstruction.py \
+exp=evaluate_reconstruction \
++exp.feats_dir=./eval_features/subj0${SUB}/train_single${BOTTLENECK} \
++exp.images_dir=./results/versatile_diffusion/subj0${SUB}/train_single_${BOTTLENECK}/ \
++exp.out_file_path=./eval_results/subj0${SUB}/train_single_${BOTTLENECK}
```

# Tang et al 2023 case study

Coming soon!

# Takagi et al 2023 case study

Coming soon!

