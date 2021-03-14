# pl-image-classification
## Environment
```
bash make_env.sh
docker-compose up -d --build 
```

## Download CIFAR10 Dataset
```
cd src
python download_datasets.py --dataset CIFAR10
```

## Training CIFAR10
```
cd src
python train.py
```

## Inference CIFAR10
```
cd src
python inference.py
```
