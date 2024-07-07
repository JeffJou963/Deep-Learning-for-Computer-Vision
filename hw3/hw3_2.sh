#!/bin/bash

# TODO - run your inference Python3 code
python3 pb2_inference.py --data_dir $1 --out_dir $2 --decoder_weights $3
