# pl-image-classification
## Environment
### Container
- dev (Main Developing Environment)
- mlflow (Tracking ML)
- minio (S3 Compatible Storage)
- posgre (SQL)
### Develop
- Pytorch (ML)
- Pytorch Lightning (Useful Pytorch Wrapper)
- hydra (useful argument parser)

## Startup Docker Containers
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
