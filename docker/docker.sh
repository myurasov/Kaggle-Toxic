#!/bin/bash

UP1_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. >/dev/null && pwd)"
THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null && pwd)"
IMAGE_NAME=`cd "${UP1_DIR}" && echo ${PWD##*/} | tr '[:upper:]' '[:lower:]' | sed -e 's/[-_ ]//g'`
IMAGE_NAME="${IMAGE_NAME}-jupyter"
JUPYTER_PORT=8888
TENSORBORAD_PORT=6006
BUILD=1
KILL=1
CMD=""

# process named arguments
while [ $# -gt 0 ]; do
  case "$1" in
    --jupyter_port=*)
      JUPYTER_PORT="${1#*=}"
      ;;
    --tensorboard_port=*)
      TENSORBORAD_PORT="${1#*=}"
      ;;
    --no-kill)
      KILL=0
      ;;
    --no-build)
      BUILD=0
      ;;
    --help)
      echo "Usage: docker.sh [--jupyter_port=####|8888] [--tensorboard_port=####|6006] [--help] [command]"
      exit
      ;;
    *)
      CMD="${1}"
  esac
  shift
done

if [ $KILL -ge 1 ]
  then
    echo "Killing ${IMAGE_NAME}..."
    docker kill "${IMAGE_NAME}"
fi

if [ $BUILD -ge 1 ]
  then
    echo "Building ${IMAGE_NAME}..."
    docker build -f "${THIS_DIR}/Dockerfile" -t $IMAGE_NAME "${UP1_DIR}" || exit 1
fi

docker run --gpus=all --rm ${_OPTIONS_:-'-it'} -p $JUPYTER_PORT:8888  -p $TENSORBORAD_PORT:6006 \
	-v "${UP1_DIR}:/app" --name="${IMAGE_NAME}" \
	$IMAGE_NAME $CMD
