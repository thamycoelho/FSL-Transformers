import torch
import training
import json
import time
import datetime
import sys
import torch.multiprocessing as mp

from timm.scheduler import create_scheduler
from pathlib import Path

from dataset import get_loaders
from utils import get_args_parser, generate_confusion_matrix, ddp_setup
from model import DeiTForFewShot

def main(rank, world_size, args):
    # Deal with output dir
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "log.txt").open("a") as f:
            f.write(" ".join(sys.argv) + "\n")

    # Set device
    ddp_setup(rank, world_size)  
  
    # Get data loaders
    print('Getting dataset')
    data_loader_train, data_loader_val, global_labels_val = get_loaders(args)
    
    # Create model from pretrained backbone
    print('Uploading model')
    model = DeiTForFewShot.from_pretrained("facebook/deit-base-distilled-patch16-224")
    model.config.update({"id2label": global_labels_val})
    
    # Optimizers, LR and Loss function
    print('Defining optimizers')
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    lr_scheduler , _ = create_scheduler(args, optimizer)

    # Define Trainer 
    trainer = training.Trainer(model=model, lr_scheduler=lr_scheduler, optimizer=optimizer, data_loader_train=data_loader_train,
                               data_loader_val=data_loader_val, global_labels_val=global_labels_val, gpu_id=rank, output_dir=output_dir)
    
    # Eval
    evaluation_stats = trainer.evaluate(eval=args.eval)
    print(f"Accuracy on validation dataset: {evaluation_stats['acc']:.2f}% ± {evaluation_stats['confidence_interval']:.4f}%")

    if not args.eval:
        # Training 
        print(f"Start training for {args.epochs} epochs")
        start_time = time.time()
        
        _ = trainer.train(epochs=args.epochs, args=args)

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))

if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    

    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, args), nprocs=world_size)
