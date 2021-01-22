#!/bin/bash

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null && pwd)"
_OPTIONS_='-itd' "${THIS_DIR}"/docker.sh $@