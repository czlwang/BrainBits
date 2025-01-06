from glob import glob as glob
import os
from pathlib import Path
import sys

subject = sys.argv[1]
#data_version = "bbits_5000"
data_version = sys.argv[2]
#subject = "S2"

root_path = "/storage/czw/semantic-decoding-brainbits"
exps = ["perceived_speech", "imagined_speech",  "perceived_movie",  "perceived_multispeaker"]
for exp in exps:
    tasks = glob(os.path.join(root_path, "data_test", "test_response", subject, exp, data_version, "*"))
    for task_path in tasks:
        task = Path(task_path).stem
        os.system(f"python3 decoding/run_decoder.py --subject {subject} --experiment {exp} --task {task} --data_version={data_version}")
        os.system(f"python3 decoding/evaluate_predictions.py --subject {subject} --experiment {exp} --task {task} --data_version={data_version}")
        break #TODO
    break #TODO
