import torch
import training
import json
import time

from timm.utils import accuracy
from timm.scheduler import create_scheduler

from dataset import get_loaders
from utils import get_args_parser, generate_confusion_matrix, map_labels, to_device, get_device
from model import DeiTForFewShot
import utils.model as utils

def main(args):
    # Set device
    utils.init_distributed_mode(args)
    
    # Get data loaders
    print('Getting dataset')
    data_loader_train, data_loader_val, class_to_label = get_loaders(args, 1, 0)
    
    # Create model from pretrained backbone
    print('Uploading model')
    model = DeiTForFewShot.from_pretrained("facebook/deit-base-distilled-patch16-224")
    model.config.update({"id2label": class_to_label})
    model.to(device)
    
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu],
                                                          find_unused_parameters=args.unused_params)
        model_without_ddp = model.module
    
    # Optimizers, LR and Loss function
    print('Defining optimizers')
    optimizer = torch.optim.SGD(model_without_ddp.parameters(), lr=args.lr, momentum=args.momentum)

    lr_scheduler , _ = create_scheduler(args, optimizer)

    loss_function = torch.nn.CrossEntropyLoss()
    
    # Eval
    evaluation_stats = training.evaluate(data_loader_val, model, loss_function, device, class_to_label)
    print(f"Accuracy on validation dataset: {evaluation_stats['acc']:.2f}% ± {evaluation_stats['confidence_interval']:.4f}%")
    
    if args.eval:
        return 
    
    # Training 
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0
    

    for epoch in range(args.epochs):
        train_stats = training.train_one_epoch(data_loader_train, model, loss_function, lr_scheduler, optimizer, epoch, device)

        lr_scheduler.step(epoch)

        evaluation_stats = training.evaluate(data_loader_val, model, loss_function, device, class_to_label)

        print(f"Accuracy on validation dataset: {evaluation_stats['acc']:.2f}% ± {evaluation_stats['confidence_interval']:.4f}%")

        max_accuracy = max(max_accuracy, evaluation_stats['acc'])
        print(f'Max accuracy: {max_accuracy:.2f}%')


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    
    main(args)