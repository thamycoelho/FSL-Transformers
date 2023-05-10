
import wandb
import json
import sys

# Using readlines()
folder = sys.argv[1]
file1 = open(folder + 'log.txt', 'r')
lines = file1.readlines()


# init wandb config
config = lines[0].split()
idx = config.index("--experiment_name")


wandb.init(project='FSL-Transformers', name=config[idx + 1])


eval = False
resume = False
columns = ["experiment name", "finetuning", "accuracy", "confidence interval", "confision_matrix"]
data = []
for line in lines:
    if not "main.py" in line:
        log_dict =  json.loads(line)
        if len(log_dict) > 2: 
            wandb.log(log_dict)
        if eval:
            confusion_matrix = 'confusion_matrix_finetuning.png' if resume else 'confusion_matrix_no_finetuning.png'
            data.append([config[idx + 1], resume, log_dict['acc'], log_dict['confidence_interval'], wandb.Image(folder + confusion_matrix)])
            eval = resume = False

    elif "eval" in line:
        if "resume" in line:
            resume = True
        eval = True
       
table = wandb.Table(columns=columns, data=data)
wandb.log({"Testing dataset": table})
wandb.finish()