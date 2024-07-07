#!/bin/bash
    # echo "testing dir: $1";
    # echo "csv path: $2";
# TODO - run your inference Python3 code
    # python3 pb1_main.py --model_type modelB --mode test
    python3 pb1_main.py --model_type B --mode test --test_dir $1 --csv_dir $2