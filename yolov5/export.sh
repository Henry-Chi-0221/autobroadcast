#!/bin/bash
if [ "$1" = "" ]
then
    echo "Please enter the name of model."
    exit
fi

python export.py --weights ../models/$1 --include engine --opset 13 --imgsz 1280 1280 --device 0 --half