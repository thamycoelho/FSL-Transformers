import torch
import training
import json
import time
import datetime
import sys
import torch.multiprocessing as mp
import wandb

from timm.scheduler import create_scheduler
from pathlib import Path
from PIL import Image
Image.MAX_IMAGE_PIXELS = 933120000

from dataset import get_loaders
from utils import get_args_parser, get_optimizer
from model import DeiTForFewShot

def main(args):
    wandb.init(project='Sanity-Test', name=args.experiment_name, config=args)
    # Deal with output dir
    output_dir = Path(args.output_dir + "/" + args.dataset + "/" + args.experiment_name)
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
    trainer = training.Trainer(model=model, lr_scheduler=lr_scheduler, optimizer=optimizer, data_loader_train=data_loader_train,
                               data_loader_val=data_loader_val, global_labels_val=global_labels_val, device=device, output_dir=output_dir,
                               experiment_name=args.experiment_name)
    
    # Eval
    resume = True if args.resume else False
    evaluation_stats = trainer.evaluate(eval=args.eval, resume=resume)
    print(f"Accuracy on validation dataset: {evaluation_stats['acc']:.2f}% ± {evaluation_stats['confidence_interval']:.4f}%")

    if not args.eval:
        # Training 
        print(f"Start training for {args.epochs} epochs")
        start_time = time.time()
        
        _ = trainer.train(epochs=args.epochs, args=args)

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))

    wandb.finish()

if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()

    main(args)