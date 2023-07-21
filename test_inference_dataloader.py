from dataset import get_loaders
from utils import get_args_parser

def main(args):
    # Get data loaders
    print('Getting dataset')
    _, data_loader_val, global_labels_val = get_loaders(args)

    for batch in data_loader_val:
        img, label, labelToCls = batch
        print("img", img)
        print("label", label)
        print("label to class", labelToCls)
        break
    

if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()

    args.dataset = 'inference'
    args.dataset_path = '/home/thamiriscoelho/Documentos/landmark_images/test'
    args.batch_size  = 20

    main(args)