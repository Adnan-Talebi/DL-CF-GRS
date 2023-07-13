#!/bin/bash

echo "Run this script from project root"
echo ""
echo ""

DATA_PATH="data/grupos/"

DS="ft ml100k ml1m anime"

echo "\begin{table}[]"
echo "\begin{tabular}{|l|l|l|l|l|}"
echo "\begin{tabular}[c]{@{}l@{}}Group\\ size\end{tabular} & FilmTrust & MovieLens100K & MovieLens1M & MyAnimeList\\\\ \hline"
# línea ejemplo
# 2 & 10000/4000 &        \\ \hline
# <G> & train/test & ... 
echo "\hline"
for g in $(seq 2 10); do
    echo -n $g 
    for d in ${DS}; do
        train=$( wc -l < $DATA_PATH$d"/groups-$g-train.csv")
        test=$( wc -l < $DATA_PATH$d"/groups-$g-test.csv")
        echo -n " & "
        printf "%'d" $train
        echo -n "/"
        printf "%'d" $test
        echo -n "  "
    done
    echo " \\\\ \hline"
done

echo "\end{tabular}"
echo "\end{table}"