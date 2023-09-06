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

### Experimentos 

1. Dataset de Cenas - Protocolo Few-Shot

```python3 main.py --output {output_file} --dataset csam --backbone deit_small --experiment_name FSL_CSAM_Cenas --project-name final_test --dataset_path {dataset_path} --nEpisode 10000 --eval --no-wandb```

2. Dataset de Cenas - Protolo Geral
   
       - Extrair Features
       - Classificação
           - Support: Places8
           - Queury: Cenas CSAM

3. Dataset de CSAM - Protocolo Few-Shot

   ```python3 main.py --output {output_file} --dataset csam --backbone deit_small --experiment_name FSL_CSAM --project-name final_test --dataset_path {dataset_path} --nEpisode 10000 --eval --no-wandb```
  
4. Dataset RCPD - Protocolo Few-Shot
   
   ```python3 main.py --output {output_file} --dataset csam --backbone deit_small --experiment_name FSL_RCPD --project-name final_test --dataset_path {dataset_path} --nEpisode 10000 --eval --no-wandb```
