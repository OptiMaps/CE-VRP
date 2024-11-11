# TrainRepo

## 시작 방법

1. build
```bash
docker build -t falconlee236/rl-image:graph-mamba .
```

2. start
```bash
docker run -itd \ 
            --name rl-server \ 
            --gpus all \ 
            -p 3030:3030 \ 
            --restart always \ 
            -v /home/sangyleegcp1/gcs-bucket:/home/user/artifect \ 
            falconlee236/rl-image:graph-mamba
```

이때 sangyleegcp1는 각자 username을 적으면 된다.