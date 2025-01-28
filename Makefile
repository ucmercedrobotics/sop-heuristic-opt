CONTAINER_NAME := sop-opt
APP_NAME := sop-opt

build: 
	docker build . -t ${CONTAINER_NAME} --target local

bash:
	docker run -it --rm \
	--gpus all \
	-e NVIDIA_VISIBLE_DEVICES=all \
	-v $(shell pwd):/${APP_NAME} \
	${CONTAINER_NAME} \
	/bin/bash