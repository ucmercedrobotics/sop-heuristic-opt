CONTAINER_NAME := sop-opt
APP_NAME := sop-opt

repo-init:
	sudo apt install pre-commit && \
	pre-commit install

build:
	docker build . -t ${CONTAINER_NAME} --target local

bash:
	docker run -it --rm \
	--gpus all \
	-e NVIDIA_VISIBLE_DEVICES=all \
	-v $(shell pwd):/${APP_NAME} \
	${CONTAINER_NAME} \
	/bin/bash

run:
	python sop/scripts/improve_heuristic.py

dataset:
	python sop/scripts/generate_dataset.py
