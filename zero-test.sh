#!/usr/bin/env bash
## iris.dataの結果
docker run -v $PWD:/app -w /app -it els-python python ./zero.py -i iris.data
## iris.dataの結果(交差検証)
docker run -v $PWD:/app -w /app -it els-python python ./zero.py -i iris.data -c
