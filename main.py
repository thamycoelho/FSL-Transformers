import torch
import training
import json
import time
import datetime
import torch.multiprocessing as mp

from timm.scheduler import create_scheduler


from dataset import get_loaders
from utils import get_args_parser, generate_confusion_matrix, ddp_setup
from model import DeiTForFewShot

def main(rank, world_size, args):
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
                               data_loader_val=data_loader_val, global_labels_val=global_labels_val, gpu_id=rank, output_dir=args.output_dir)
    
    # Eval
    evaluation_stats = trainer.evaluate(eval=args.eval)
    print(f"Accuracy on validation dataset: {evaluation_stats['acc']:.2f}% ± {evaluation_stats['confidence_interval']:.4f}%")

    if not args.eval:
        # Training 
        print(f"Start training for {args.epochs} epochs")
        start_time = time.time()
        
        _ = trainer.train(epochs=args.epochs)

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))

if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    

    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, args), nprocs=world_size)
