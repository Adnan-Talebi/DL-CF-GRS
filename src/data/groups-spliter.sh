#!/bin/bash

DIR=$(pwd)

for file in $(ls groups-*)
do
    file_name="${file%.*}"
    head -45001 $DIR/$file > $file_name"-train.csv"
    head -1 $DIR/$file > $file_name"-test.csv"
    tail -5000 $DIR/$file >> $file_name"-test.csv"
done