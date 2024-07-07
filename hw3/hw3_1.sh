#!/bin/bash

# bash hw3_1.sh $1 $2 $3
# $1: path to the folder containing test images (images are named xxxx.png, where xxxx could be any string)
# $2: path to the id2label.json
# $3: path of the output csv file (e.g. output_p1/pred.csv)

# TODO - run your inference Python3 code
python3 pb1_inference.py --img_dir $1 --label_dir $2 --output_dir $3