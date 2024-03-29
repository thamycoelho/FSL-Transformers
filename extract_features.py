#  python test_inference_dataloader.py --output output/ --dataset test --backbone dino --aggregator average --experiment_name test_aggragator --nQuery 3 --nEpisode 2 --sched None --epochs 3 --eval
import sys
import torch
import pickle
import training

from pathlib import Path

from dataset import get_loaders
from utils import get_args_parser, get_optimizer
from model import DeiTForFewShot

def main(args):
    print(args.dataset)
    # Deal with output dir
    output_dir = Path(args.output_dir + "/" + args.experiment_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "log.txt").open("a") as f:
            f.write(" ".join(sys.argv) + "\n")

    # Set device
    device = torch.device(args.device)

    # Get data loaders
    print('Getting dataset')
    _, data_loader_val, global_labels_val = get_loaders(args)

    # Create model from pretrained backbone
    print('Uploading model')
    model = DeiTForFewShot(args)
    model = model.to(device)
    
    # Optimizers, LR and Loss function
    print('Defining optimizers')
    optimizer, lr_scheduler = get_optimizer(args, model)
    
    # Resume training from checkpoint
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')

        model.load_state_dict(checkpoint['model'])
        args.start_epoch = checkpoint['epoch']

        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.max_acc = checkpoint['max_acc']
            args.start_epoch += 1

        print(f'Resume from {args.resume} at epoch {args.start_epoch}.')

    # Define Trainer 
    trainer = training.Trainer(model=model, lr_scheduler=lr_scheduler, optimizer=optimizer, data_loader_train=None,
                               data_loader_val=data_loader_val, global_labels_val=global_labels_val, device=device, output_dir=output_dir,
                               experiment_name=args.experiment_name)

    # Extract Features
    features = trainer.extract_features(args)

    # Generate Pickle   
    torch.save(features, output_dir / 'features.pth')
    print(output_dir)

    # if args.output_dir:
    #     with (output_dir / 'features.pkl').open("wb") as f:
    #         pickle.dump(features, f)


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()

    main(args)