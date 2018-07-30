# -*- coding: utf-8 -*-
import csv

from config import MAX_LEN, WORD_VEC_100, WORD_DICT
from config import TRAIN_FILE, DEV_FILE, TEST_FILE
from config import TRAIN_FILE_RAW, DEV_FILE_RAW, TEST_FILE_RAW
from config import CORPUS
from voc import Vocab
import argparse


class Data_index(object):
    def __init__(self, Vocabs):
        self.VOCABS = Vocabs

    def to_index(self, words):
        word_idx = []
        for word in words:
            if word in self.VOCABS.word2idx:
                word_idx.append(self.VOCABS.word2idx[word])
            else:
                word_idx.append(self.VOCABS.word2idx['UNK'])

        return ','.join(map(str, word_idx))

    def process_file(self, path, output, bigram=False):
        querys, replys = self.process_data(path)
        for query, reply in zip(querys, replys):
            query_idx = self.to_index(query)
            reply_idx = self.to_index(reply)
            length_query = len(query)
            length_reply = len(reply)
            output.writerow([query_idx, reply_idx, length_query, length_reply, max(length_query, length_reply)])

    def process_all_data(self):
        for i in range(CORPUS):
            f_train = open(TRAIN_FILE[i], 'w')
            f_dev = open(DEV_FILE[i], 'w')
            f_test = open(TEST_FILE[i], 'w')
            output_train = csv.writer(f_train)
            output_train.writerow(['query', 'reply', 'length_query', 'length_reply', 'length'])
            output_dev = csv.writer(f_dev)
            output_dev.writerow(['query', 'reply', 'length_query', 'length_reply', 'length'])
            output_test = csv.writer(f_test)
            output_test.writerow(['query', 'reply', 'length_query', 'length_reply', 'length'])

            f_train_raw = TRAIN_FILE_RAW[i]
            f_dev_raw = DEV_FILE_RAW[i]
            f_test_raw = TEST_FILE_RAW[i]

            self.process_file(f_train_raw, output_train)
            self.process_file(f_dev_raw, output_dev)
            self.process_file(f_test_raw, output_test)

    def process_data(self, path):
        query = []
        reply = []

        file_read = open(path, 'r')
        content = file_read.readlines()
        file_read.close()
        for line in content:
            line = unicode(line, "utf-8")
            line = line.split("######")
            line_query = line[0]
            line_reply = line[1]
            line_query = line_query.split()
            line_reply = line_reply.split()
            query.append(line_query)
            reply.append(line_reply)

        return query, reply



VOCABS = Vocab(WORD_VEC_100, None, single_task=False, bi_gram=False, frequency=5)
da_index = Data_index(VOCABS)
da_index.process_all_data()
