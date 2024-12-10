import numpy as np
import os
from PIL import Image
from pathlib import Path

#The same for all subjects
brain_diffuser_root = "../brain-diffuser"
images = np.load(os.path.join(brain_diffuser_root, 'data/processed_data/subj01/nsd_test_stim_sub1.npy'))

test_images_dir = 'data/nsddata_stimuli/test_images/'
Path(test_images_dir).mkdir(exist_ok=True, parents=True)

if not os.path.exists(test_images_dir):
   os.makedirs(test_images_dir)
for i in range(len(images)):
    im = Image.fromarray(images[i].astype(np.uint8))
    im.save('{}/{}.png'.format(test_images_dir,i))
