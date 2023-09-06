# FSL-Transformers

### Requirements 
- Python=3.9

```
pip install -r requirements.txt
```

### Extract Features

dataset_path can be:
* folder path
```
dataset   
│
└───folder
│   │
│   └───class_name1
│       │   image1.jpg
│       │   image2.jpg
│       │   ...
│   
└───folder2
    │
│   └───class_name2
│       │   image1.jpg
│       │   image2.jpg
│       │   ...
```

* csv
  
image_path, label

Run feature extraction:

```
python3 extract_features.py --output {output_file} --dataset final_test --backbone deit_small --project-name final_test --experiment_name {dataset_name} --batch-size 256 --extract_features --dataset_path {dataset_path}
```

### Classify Samples
```
python3 classify_samples.py --output {output_file} --dataset final_test --backbone deit_small --project-name final_test --experiment_name {dataset_name} --batch-size 256 --nEpisode 10000 --nClsEpisode 8 --classify --support_file {output_file}/{support_dataset_name}/features.pth --query_file {output_file}/{query_dataset_name}/features.pth
```
