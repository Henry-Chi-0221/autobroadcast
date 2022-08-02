#!/bin/bash
if [ "$1" = "" ]
then
    echo "Please enter the class_num"
    exit
fi

python export.py --weights ../models/$1 --include engine --opset 13 --imgsz 320 1280 --device 0 --half