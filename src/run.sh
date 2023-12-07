#!/bin/bash

output=$(python3 ./src/name_config.py)
python3 ./src/proGAN.py "$output"