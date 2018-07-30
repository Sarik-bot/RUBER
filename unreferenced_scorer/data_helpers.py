import pandas as pd
import numpy as np

class BucketedDataIterator():
    def __init__(self, df, num_buckets=10, is_test=False):
        if is_test:
            self.df = df
            self.pos = 0
            self.total = len(df)
            self.size = self.total
            
            return
        self.df = df
        self.total = len(df)
        df_sort = df.sort_values('length').reset_index(drop=True)
        self.size = self.total / num_buckets
        self.dfs = []
        for bucket in range(num_buckets - 1):
            self.dfs.append(df_sort.ix[bucket*self.size: (bucket + 1)*self.size - 1])
        self.dfs.append(df_sort.ix[(num_buckets-1)*self.size:])
        self.num_buckets = num_buckets

        # cursor[i] will be the cursor for the ith bucket
        self.cursor = np.array([0] * num_buckets)
        self.pos = 0
        self.shuffle()

        self.epochs = 0

    def shuffle(self):
        #sorts dataframe by sequence length, but keeps it random within the same length
        for i in range(self.num_buckets):
            self.dfs[i] = self.dfs[i].sample(frac=1).reset_index(drop=True)
            self.cursor[i] = 0

    def next_batch(self, batch_size, bigram=True, round = -1, classifier = False):
        
        if np.any(self.cursor + batch_size + 1 > self.size):
            self.epochs += 1
            self.shuffle()

        i = np.random.randint(0, self.num_buckets)

        res = self.dfs[i].ix[self.cursor[i]:self.cursor[i] + batch_size - 1]

        try:
            query = map(lambda x: map(int, x.split(",")), res['query'].tolist())
            reply = map(lambda x: map(int, x.split(",")), res['reply'].tolist())
        except Exception, e:
            print Exception ,e
            print res['query'].tolist()
            print res['reply'].tolist()
            exit()

        len_corpus = self.size-100
        sample = [np.random.randint(0, len_corpus) for ii in range(batch_size)]
        #print i
        #print sample
        res_neg = self.dfs[i].ix[sample]
        reply_neg = map(lambda x: map(int, x.split(",")), res_neg['reply'].tolist())
        reply_neg_len = []
        for ii in reply_neg:
            reply_neg_len.append(len(ii))


        self.cursor[i] += batch_size

        # Pad sequences with 0s so they are all the same length
        maxlen_query = max(res['length_query'])

        x = np.zeros([batch_size*2, maxlen_query], dtype=np.int32)
        for i, x_i in enumerate(x):
            x_i[:res['length_query'].values[i%batch_size]] = query[i%batch_size]


        maxlen_reply0 = max(res['length_reply'])
        maxlen_reply1 = max(reply_neg_len)
        maxlen_reply = max(maxlen_reply0, maxlen_reply1)
        y = np.zeros([batch_size*2, maxlen_reply], dtype=np.int32)
        for i, y_i in enumerate(y[:batch_size]):
            y_i[:res['length_reply'].values[i]] = reply[i]
        for i, y_i in enumerate(y[batch_size:]):
            y_i[:reply_neg_len[i]]=reply_neg[i]

        len_query_per = res['length_query'].values
        len_query = np.zeros([batch_size*2], dtype=np.int32)
        len_query[:batch_size] = len_query_per
        len_query[batch_size:] = len_query_per

        len_pos = res['length_reply'].values
        len_neg = reply_neg_len
        len_reply = np.zeros([batch_size*2], dtype=np.int32)
        len_reply[:batch_size] = len_pos
        len_reply[batch_size:] = len_neg
        
        """
        print "x: ", x
        print "y: ", y
        print "query: ", len_query
        print "reply: ", len_reply
        exit()
        """
        if classifier is False:
            return x, y, len_query, len_reply
        else:
            y_class = [round] * batch_size * 2
            return x, y, y_class, len_query, len_reply

    def next_pred_one(self, aa):
        res = self.df.ix[self.pos]
        query = map(int, res['query'].split(','))
        reply = map(int, res['reply'].split(','))
        length_query = res['length_query']
        length_reply = res['length_reply']
        self.pos += 1
        if self.pos == self.total:
            self.pos = 0
        return np.asarray([query],dtype=np.int32), np.asarray([reply],dtype=np.int32), np.asarray([length_query],dtype=np.int32), np.asarray([length_reply],dtype=np.int32)

    def next_all_batch(self, batch_size):
        if self.pos+batch_size>self.total:
            self.pos=0
            print "Wrong log by fzx at pos0"
        res = self.df.ix[self.pos : self.pos + batch_size -1]
        query = map(lambda x: map(int, x.split(",")), res['query'].tolist())
        reply = map(lambda x: map(int, x.split(",")), res['reply'].tolist())

        self.pos += batch_size

        maxlen_query = max(res['length_query'])
        x = np.zeros([batch_size, maxlen_query], dtype=np.int32)
        for i, x_i in enumerate(x):
            x_i[:res['length_query'].values[i]] = query[i]

        maxlen_reply = max(res['length_reply'])
        y = np.zeros([batch_size, maxlen_reply], dtype=np.int32)
        for i, y_i in enumerate(y):
            y_i[:res['length_reply'].values[i]] = reply[i]

        return x, y, res['length_query'].values, res['length_reply'].values

    def print_info(self):
        print 'dfs shape: ', [len(self.dfs[i]) for i in xrange(len(self.dfs))]
        print 'size: ', self.size


