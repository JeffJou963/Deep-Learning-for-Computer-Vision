#!/bin/bash

# TODO - run your inference Python3 code
#  python3 pb2_main.py --mode test
 python3 pb2_main.py --mode test --input_csv $1 --test_dir $2 --csv_dir $3

# TA:
#  bash hw1_2.sh 
#  /home/dlcv2023/grading/hw1/data/p2test_in.csv 
#  /home/dlcv2023/grading/hw1/data/p2test 
#  /home/dlcv2023/grading/hw1/hw1test/p2_output/test/output.csv