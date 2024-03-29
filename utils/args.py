import argparse
import numpy as np


def get_args_parser():
   parser = argparse.ArgumentParser('Few-shot learning script', add_help=False)
   # General
   parser.add_argument('--batch-size', default=1, type=int)
   parser.add_argument('--epochs', default=100, type=int)
   parser.add_argument('--output_dir', default='outputs/tmp',
                     help='path where to save, empty for no saving')
   parser.add_argument('--seed', default=0, type=int)
   parser.add_argument('--deterministic', default=False, type=bool)
   parser.add_argument('--experiment_name', default="", help='name of the experiment running to create apropriate file.')
   
   # Extract features parameters
   parser.add_argument("--extract_features", action='store_true')
   parser.add_argument("--dataset_path", default="", type=str)

   # Classify parameters
   parser.add_argument("--classify", action="store_true")
   parser.add_argument("--support_file", type=str)
   parser.add_argument("--query_file", type=str)


   # Wandb parameters 
   parser.add_argument("--wandb", dest='wandb', action='store_true')
   parser.add_argument("--no-wandb", dest='wandb', action='store_false')
   parser.set_defaults(wandb=True)
   parser.add_argument("--project-name", default="FSL-Transformers", type=str)

   # Dataset parameters
   parser.add_argument("--dataset", choices=["places", "places_600", "test", "final_test", "csam", "litmus"],
                     default="places_600",
                     help="Which few-shot dataset.")

   # Few-shot parameters 
   parser.add_argument("--nClsEpisode", default=8, type=int,
                     help="Number of categories in each episode.")
   parser.add_argument("--nSupport", default=5, type=int,
                     help="Number of samples per category in the support set.")
   parser.add_argument("--nQuery", default=15, type=int,
                     help="Number of samples per category in the query set.")
   parser.add_argument("--nEpisode", default=2000, type=int,
                     help="Number of episodes for training / testing.")

   # Model params
   parser.add_argument('--pretrained_weights', default='', type=str, help="Path to pretrained weights to evaluate.")
   parser.add_argument('--backbone', default='deit_small',choices=['deit', 'resnet50', 'dino', 'resnet50_dino', 'deit_small', 'resnet18', 'vit_mini'])
   parser.add_argument('--aggregator', default='average', choices=['average', 'max', 'log_sum_exp', 'lp_pool', 'self_attn'])
   parser.add_argument('--temperature', default=0.1, type=float, help='temperature to be applyed to cosine similarities')
   
   # Deployment params
   parser.add_argument('--aug_prob', default=0.9, type=float, help='Probability of applying data augmentation during meta-testing')
   parser.add_argument('--aug_types', nargs="+", default=['color', 'translation'],
                     help='color, offset, offset_h, offset_v, translation, cutout')

   # Other model parameters
   parser.add_argument('--img-size', default=224, type=int, help='images input size')

   # Optimizer parameters
   parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                     help='SGD momentum (default: 0.9)')
   parser.add_argument('--optimizer', type=str, default='sgd', choices=['sgd', 'adam'])

   # Learning rate schedule parameters
   parser.add_argument('--sched', default='step', type=str, choices=["cosine", "step", "exponential", "None"], metavar='SCHEDULER',
                     help='LR scheduler (default: "step"')
   parser.add_argument('--lr', type=float, default=5e-5, metavar='LR',
                     help='learning rate (default: 5e-4)')
   parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                     help='learning rate noise on/off epoch percentages')
   parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                     help='learning rate noise limit percent (default: 0.67)')
   parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                     help='learning rate noise std-dev (default: 1.0)')
   parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                     help='warmup learning rate (default: 1e-6)')
   parser.add_argument('--min-lr', type=float, default=1e-6, metavar='LR',
                     help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

   parser.add_argument('--decay-epochs', type=float, default=10, metavar='N',
                     help='epoch interval to decay LR (step scheduler)')
   parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                     help='epochs to warmup LR, if scheduler supports')
   parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                     help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
   parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                     help='patience epochs for Plateau LR scheduler (default: 10')
   parser.add_argument('--decay-rate', '--dr', type=float, default=0.5, metavar='RATE',
                     help='LR decay rate (default: 0.1)')

   # Augmentation parameters
   parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                     help='Color jitter factor (default: 0.4)')
   parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                     help='Use AutoAugment policy. "v0" or "original". " + \
                           "(default: rand-m9-mstd0.5-inc1)'),
   parser.add_argument('--smoothing', type=float, default=0.0, help='Label smoothing (default: 0.1)')
   parser.add_argument('--train-interpolation', type=str, default='bicubic',
                     help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

   parser.add_argument('--repeated-aug', action='store_true')


   # Misc
   parser.add_argument('--resume', default='', help='resume from checkpoint')
   parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                     help='start epoch')
   parser.add_argument('--max_acc', default=None, type=tuple, help='Max accuracy obtained in training before')
   parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
   parser.add_argument('--num_workers', default=10, type=int)
   parser.add_argument('--pin-mem', action='store_true',
                     help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
   parser.set_defaults(pin_mem=True)
   
   # distributed training parameters
   parser.add_argument('--device', default='cuda',
                     help='cuda:gpu_id for single GPU training')
   return parser