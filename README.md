# TrainRepo

## 시작 방법

1. build
```bash
docker build -t falconlee236/rl-image .
```

2. start
```bash
docker run -itd --name test --gpus all -p 3030:3030 --restart always falconlee236/rl-image
```