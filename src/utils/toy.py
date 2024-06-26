#!/bin/python3

from data_utils import init_random
init_random()

import argparse
import sys
import os

import pandas as pd

parser = argparse.ArgumentParser(description='Load data from a csv file')
parser.add_argument('--file_path', type=str, help='Path to the csv file')

args = parser.parse_args()

if args.file_path is None:
    print('Please specify the path to the file')
    sys.exit(1)

def load_data(file_path):
    return pd.read_csv(file_path)

data = load_data(args.file_path)

print(data)