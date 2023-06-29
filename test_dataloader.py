from dataset import get_loaders
from utils import get_args_parser

import numpy as np
import matplotlib.pyplot as plt
import torch

def show_batch(image_batch, label_batch, save_file):
  plt.figure(figsize=(20,40))
  c = 0
  print(len(image_batch))
  for image in range(len(image_batch)):
    for n in range(len(image_batch[0])):
        print(len(image_batch),len(image_batch[0]))
        ax = plt.subplot(len(image_batch),len(image_batch[0]),c+1)
        plt.imshow(torch.permute(image_batch[image][n], (1, 2, 0)))
        plt.title(label_batch[image][n].item())
        plt.axis('off')
        c += 1
  plt.savefig(save_file)


def main(args):
    print('Getting dataset')
    data_loader_train, data_loader_val, global_labels_val = get_loaders(args)

    print("train")
    if not args.eval:
        support_images = []
        support_labels = []

        query_images = []
        query_labels = []
        for epoch in range(args.epochs):
            for batch in data_loader_train:
                support_images.append(batch[0][0])
                support_labels.append(batch[1][0])

                query_images.append(batch[2][0])
                query_labels.append(batch[3][0])
        show_batch(image_batch=support_images, label_batch=support_labels, save_file='train_support.png')
        show_batch(image_batch=query_images, label_batch=query_labels, save_file='train_query.png')

    print("test")
    support_images = []
    support_labels = []

    query_images = []
    query_labels = []
    for batch in data_loader_val:
        support_images.append(batch[0][0])
        support_labels.append(batch[1][0])

        query_images.append(batch[2][0])
        query_labels.append(batch[3][0])

    show_batch(image_batch=support_images, label_batch=support_labels, save_file='test_support.png')
    show_batch(image_batch=query_images, label_batch=query_labels, save_file='test_query.png')



if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()

    main(args)