TARGET=parco/train.py
CONTAINER_NAME=rl-server
BASE_IMAGE=falconlee236/rl-image
TAG_NAME=parco
USER_NAME=user
DOCKERFILE_SRC=dockerfiles/$(TAG_NAME)/Dockerfile

all: container

train:
	@python3 $(TARGET)

container:
	@docker run -itd \
				--name $(CONTAINER_NAME) \
				-v /tmp/artifect:/home/$(USER_NAME)/artifect \
				-p 3030:3030 \
				--restart always \
				$(BASE_IMAGE):$(TAG_NAME)

build:
	@docker build -t $(BASE_IMAGE):$(TAG_NAME) . -f $(DOCKERFILE_SRC)

pull:
	@docker pull $(BASE_IMAGE):$(TAG_NAME)

push:
	@docker push $(BASE_IMAGE):$(TAG_NAME)

clean:
	@docker rm $(CONTAINER_NAME) --force