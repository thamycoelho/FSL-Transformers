from dataset import get_loaders
from utils import get_args_parser

import numpy as np

def main(args):
    print('Getting dataset')
    data_loader_train, data_loader_val, global_labels_val = get_loaders(args)

    print("train")
    for epoch in range(args.epochs):
        print("epoch: ", epoch)
        for batch in data_loader_train:
            print("labels", batch[1])

    print("test")
    for batch in data_loader_val:
        print("labels", batch[1])

if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()

    main(args)