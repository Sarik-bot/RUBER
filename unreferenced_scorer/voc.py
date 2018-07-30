import numpy as np
from collections import defaultdict


class Vocab(object):
    def __init__(self, path_vec, train_word, single_task, bi_gram, frequency=15):
        self.path = path_vec
        self.table_path = train_word
        self.word2idx = defaultdict(int)
        self.word_vectors = None
        self.single = single_task
        self.bigram = bi_gram
        self.frequency = frequency
        self.table = set()
        self.load_data()

    def load_data(self):
        with open(self.path, 'r') as f:
            line = f.readline().strip().split(" ")
            N, dim = map(int, line)

            self.word_vectors = []
            idx = 0
            for k in range(N):
                line = unicode(f.readline(), 'utf-8').strip().split(" ")
                if line[0] in self.word2idx:
                    print line[0]
                    exit()
                self.word2idx[line[0]] = idx
                vector = np.asarray(map(float, line[1:]), dtype=np.float32)
                self.word_vectors.append(vector)
                idx += 1

            print 'Vocab size:', len(self.word_vectors)
            print 'word2idx:', len(self.word2idx)
            print 'index:', idx

            self.word_vectors = np.asarray(self.word_vectors, dtype=np.float32)

