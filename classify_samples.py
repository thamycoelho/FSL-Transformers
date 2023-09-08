#  python classify_samples.py --output output/ --dataset test --backbone dino --aggregator average --experiment_name test_aggragator --nQuery 3 --nEpisode 2 --sched None --epochs 3 --eval
import sys
import torch
import training

from pathlib import Path

from dataset import get_loaders
from utils import get_args_parser
from model import ProtoNet

def main(args):

   # Deal with output dir
    output_dir = Path(args.output_dir + "/results/" + args.dataset + "/" + args.experiment_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "log.txt").open("a") as f:
            f.write(" ".join(sys.argv) + "\n")
    # Set device
    device = torch.device(args.device)

    # Get data loaders
    print('Getting dataset')   
    data_loader_train, data_loader_val, global_labels_val = get_loaders(args)    

    # Create model from pretrained backbone
    print('Uploading model')
    model = ProtoNet(args)
    model = model.to(device)

    # Define Trainer 
    trainer = training.Trainer(model=model, lr_scheduler=None, optimizer=None, data_loader_train=data_loader_train,
                               data_loader_val=data_loader_val, global_labels_val=global_labels_val, device=device, output_dir=output_dir,
                               experiment_name=args.experiment_name)

    # Classify
    df = trainer.classify_from_features(args)

    df.to_csv(output_dir / 'result.csv', index=False, sep=',')
    print("output:", output_dir)

   
if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    args.classify = True
    args.extract_features = False

    main(args)