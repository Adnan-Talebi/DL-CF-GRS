#!/bin/zsh

DIR=$1
TRAIN=0.7
VAL=0.1

# round $1 to $2 decimal places
round() {
    printf "%.${2:-0}f" "$1"
}

for file in $(ls $DIR/groups-*[0-9].csv)
do
    size=$(cat $file | wc -l)
    size=$(($size-1))
    train=$(round $(($size*$TRAIN)) 0)
    val=$(round $(($size*$VAL)) 0)
    test=$(($size - $train - $val))
    echo "Total $size"
    echo "Train $train"
    echo "Val $val"
    echo "Test $test"
    
    file_name="${file%.*}"
    
    echo $file_name
    head -$(($train+1)) $file > $file_name"-train.csv"
    head -1 $file > $file_name"-test.csv"
    head -1 $file > $file_name"-val.csv"
    
    val_test=$(($val+$test))
    tail -$val_test $file | head -$val >> $file_name"-val.csv"
    tail -$test $file >> $file_name"-test.csv"
    
done