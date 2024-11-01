TARGET=src/main.py
CONTAINER_NAME=rl-server
BASE_IMAGE=falconlee236/rl-image
TAG_NAME=graph-mamba
USER_NAME=user

all: container

train:
	@python3 $(TARGET)

container:
	@docker run -itd \
				--name $(CONTAINER_NAME) \
				--gpus all \
				-v /tmp/artifect:/home/$(USER_NAME)/artifect \
				-p 3030:3030 \
				--restart always \
				$(BASE_IMAGE):$(TAG_NAME)

build:
	@docker build -t $(BASE_IMAGE):$(TAG_NAME) .

pull:
	@docker pull $(BASE_IMAGE):$(TAG_NAME)

push:
	@docker push $(BASE_IMAGE):$(TAG_NAME)

clean:
	@docker rm $(CONTAINER_NAME) --force