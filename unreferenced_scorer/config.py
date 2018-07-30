# -*- coding: utf-8 -*-

import os

#Multi_Model
ADV_STATUS = False
DROP_OUT = [0.78, 0.83, 0.65, 0.6, 0.7, 0.5, 0.65, 0.5, 0.5]
CORPUS = 1

#Baseline
DROP_SINGLE = 0.5
LSTM_NET = True
STACK_STATUS = False
BI_DIRECTION = True

DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(DIR, 'models')

WORD_VEC_100 = os.path.join(MODEL_DIR, 'vec100.txt')


WORD_DICT = os.path.join(MODEL_DIR, 'train_words')
#This is used for generating multi_task train, dev, test input


TRAIN_FILE = ['../data/data_douban/train_mt.csv','../data/data_twitter/train_mt.csv']
DEV_FILE = ['../data/data_douban/dev_mt.csv','../data/data_twitter/dev_mt.csv']
TEST_FILE = ['../data/data_douban/test_mt.csv','../data/data_twitter/test_mt.csv']

DATA_FILE = ['../data/data_douban','../data/data_twitter']

TRAIN_FILE_RAW = ['../data/data_douban/train_mt.raw','../data/data_twitter/train_mt.raw']
DEV_FILE_RAW = ['../data/data_douban/dev_mt.raw','../data/data_twitter/dev_mt.raw']
TEST_FILE_RAW = ['../data/data_douban/test_mt.csv','../data/data_twitter/test_mt.csv']


MAX_LEN = 80

