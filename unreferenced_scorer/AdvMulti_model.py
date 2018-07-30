import numpy as np
from tensorflow.contrib.rnn.python.ops import core_rnn_cell_impl
import tensorflow as tf
import math

from voc import Vocab
from config import WORD_VEC_100
"""
    Build the multi_task_cws model.
    Args:
      num_corpus: int, The number of corpus used in multi_task.
      adv: boolean, If True, adversarial is added in the model.
      adv_weight: float, the weight of adversarial loss in the combined loss(cws_loss + hess_loss * weight)
"""
class MultiModel(object):
    def __init__(self, batch_size=100, vocab_size=5620,
                 word_dim=100, lstm_dim=100,
                 num_corpus = 8,
                 l2_reg_lambda=0.0,
                 adv_weight = 0.05,
                 lr=0.001,
                 clip=5,
                 adv=False,
                 margin=0.05):

        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.word_dim = word_dim
        self.lstm_dim = lstm_dim
        self.num_corpus = num_corpus
        self.l2_reg_lambda = l2_reg_lambda
        self.lr = lr
        self.clip = clip
        self.adv = adv
        self.margin = margin
        self.dense_dim = 1000


        # placeholders
        self.x = tf.placeholder(tf.int32, [None, None])
        self.y = tf.placeholder(tf.int32, [None, None])
        self.y_class = tf.placeholder(tf.int32, [None])#note by fzx
        self.seq_len_query = tf.placeholder(tf.int32, [None])
        self.seq_len_reply = tf.placeholder(tf.int32, [None])
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")


        with tf.variable_scope("embedding") as scope:
            self.embedding = tf.get_variable(
                shape=[vocab_size, word_dim],
                initializer=tf.truncated_normal_initializer(stddev=0.01),
                dtype=tf.float32,
                name="embedding",
                #regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg_lambda)
            )


        def _shared_layer(input_data_query, input_data_reply, seq_len_query, seq_len_reply):

            #input_data_query = tf.Print(input_data_query, [input_data_query], message="input_data_query")
            lstm_fw_cell = core_rnn_cell_impl.BasicLSTMCell(self.lstm_dim)
            lstm_bw_cell = core_rnn_cell_impl.BasicLSTMCell(self.lstm_dim)
            _, ((_, forward_output), (_, backward_output)) = tf.nn.bidirectional_dynamic_rnn(
                lstm_fw_cell,
                lstm_bw_cell,
                input_data_query,
                dtype=tf.float32,
                sequence_length=seq_len_query,
                scope = "query"
            )
            #to check
            forward_output = tf.reshape(forward_output, [size, self.lstm_dim])
            backward_output = tf.reshape(backward_output, [size, self.lstm_dim])
            output_query = tf.concat(axis=1, values=[forward_output, backward_output])
            output_query = tf.reshape(output_query, [size, self.lstm_dim*2])

            lstm_fw_cell = core_rnn_cell_impl.BasicLSTMCell(self.lstm_dim)
            lstm_bw_cell = core_rnn_cell_impl.BasicLSTMCell(self.lstm_dim)
            _, ((_, forward_output), (_, backward_output)) = tf.nn.bidirectional_dynamic_rnn(
                lstm_fw_cell,
                lstm_bw_cell,
                input_data_reply,
                dtype=tf.float32,
                sequence_length=seq_len_reply,
                scope = "reply"
            )
            forward_output = tf.reshape(forward_output, [size, self.lstm_dim])
            backward_output = tf.reshape(backward_output, [size, self.lstm_dim])
            output_reply = tf.concat(axis=1, values=[forward_output, backward_output])
            output_reply = tf.reshape(output_reply, [size, self.lstm_dim*2])


            Ms = tf.get_variable(
                shape=[self.lstm_dim*2, self.lstm_dim*2],
                initializer=tf.truncated_normal_initializer(stddev=0.01),
                dtype=tf.float32,
                name="Ms",
                #regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg_lambda)
            )
            #output_query = tf.Print(output_query, [output_query], message="output_query")
            #output_reply = tf.Print(output_reply, [output_reply], message="output_reply")
            #Ms = tf.Print(Ms, [Ms], message="Ms")
            M1 = tf.matmul(output_query,Ms)
            M2 = tf.multiply(M1, output_reply)
            #M2 = tf.Print(M2, [M2], message="M2:") 
            M3 = tf.reduce_sum(M2, axis=1)
            #M3 = tf.Print(M3, [M3], message="M3:")
            M4 = tf.nn.tanh(M3)

            return output_query, output_reply, M4

        def _private_layer(output_pub_query, output_pub_reply, input_data_query, input_data_reply, seq_len_query, seq_len_reply, shared_score):
            debug = False
            size_query = tf.shape(input_data_query)[0]
            size_reply = tf.shape(input_data_reply)[0]
            size = size_query
            
            if debug:
                input_data_query = tf.Print(input_data_query, [input_data_query], message="input_data_query:",summarize=100)
                input_data_reply = tf.Print(input_data_reply, [input_data_reply], message="input_data_reply:",summarize=100)

            lstm_fw_cell = core_rnn_cell_impl.BasicLSTMCell(self.lstm_dim)
            lstm_bw_cell = core_rnn_cell_impl.BasicLSTMCell(self.lstm_dim)
            _, ((_, forward_output_query), (_, backward_output_query))= tf.nn.bidirectional_dynamic_rnn(
                lstm_fw_cell,
                lstm_bw_cell,
                input_data_query,
                dtype=tf.float32,
                sequence_length=seq_len_query,
                scope = "query"
            )

            lstm_fw_cell = core_rnn_cell_impl.BasicLSTMCell(self.lstm_dim)
            lstm_bw_cell = core_rnn_cell_impl.BasicLSTMCell(self.lstm_dim)
            _, ((_, forward_output_reply), (_, backward_output_reply)) = tf.nn.bidirectional_dynamic_rnn(
                lstm_fw_cell,
                lstm_bw_cell,
                input_data_reply,
                dtype=tf.float32,
                sequence_length=seq_len_reply,
                scope = "reply"
            )
            output_query = tf.concat(axis=1, values=[forward_output_query, backward_output_query])
            output_reply = tf.concat(axis=1, values=[forward_output_reply, backward_output_reply])
            output_query = tf.reshape(output_query, [size, self.lstm_dim*2])
            output_reply = tf.reshape(output_reply, [size, self.lstm_dim*2])

            output_query = tf.concat(axis=1, values=[output_query, output_pub_query])
            output_reply = tf.concat(axis=1, values=[output_reply, output_pub_reply])
            output_query = tf.reshape(output_query, [size, self.lstm_dim * 4])
            output_reply = tf.reshape(output_reply, [size, self.lstm_dim * 4])
            output = tf.concat(axis=1, values=[output_query, output_reply])


            #do with output_query,output_reply,output            
            Mp = tf.get_variable(
                shape=[self.lstm_dim*4, self.lstm_dim*4], 
                initializer=tf.truncated_normal_initializer(stddev=0.01), 
                name="Mp", 
               # regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg_lambda)
            )
            M1 = tf.matmul(output_query,Mp)
            M2 = tf.multiply(M1, output_reply)
            M3 = tf.reduce_sum(M2, axis=1)
            M3 = tf.reshape(M3, [size,1])
            output_mlp = tf.concat(axis=1, values=[output, M3])
            if debug:
                output_mlp = tf.Print(output_mlp, [output_mlp], message="output_mlp:", summarize=100)

            W1 = tf.get_variable(
                shape=[self.lstm_dim * 8+1, 50],
                initializer=tf.truncated_normal_initializer(stddev=0.01),
                name="weights1",
                #regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg_lambda)
                )
            b1 = tf.Variable(0. , name="bias1")

            W2 = tf.get_variable(
                shape=[50 , 1],
                initializer=tf.truncated_normal_initializer(stddev=0.01),
                name="weights2",
                #regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg_lambda)
                )
            b2 = tf.Variable(0. , name="bias2")

            """
            W3 = tf.get_variable(
                shape=[self.dense_dim , self.dense_dim],
                initializer=tf.truncated_normal_initializer(stddev=0.01),
                name="weights3",
                #regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg_lambda)
                )
            b3 = tf.Variable(0. , name="bias3")

            W4 = tf.get_variable(
                shape=[self.dense_dim, 1],
                initializer=tf.truncated_normal_initializer(stddev=0.01),
                name="weights4",
                #regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg_lambda)
                )
            b4 = tf.Variable(0. , name="bias4")
            """

            matricized_unary_scores = tf.matmul(output_mlp, W1) + b1
            matricized_unary_scores = tf.nn.tanh(matricized_unary_scores)
            if debug:
                matricized_unary_scores = tf.Print(matricized_unary_scores, [matricized_unary_scores], message="matricized_unary_scores0:",summarize=100)
            matricized_unary_scores = tf.matmul(matricized_unary_scores, W2) + b2
            """
            matricized_unary_scores = tf.nn.tanh(matricized_unary_scores)
            matricized_unary_scores = tf.matmul(matricized_unary_scores, W3) + b3
            matricized_unary_scores = tf.nn.tanh(matricized_unary_scores)
            matricized_unary_scores = tf.matmul(matricized_unary_scores, W4) + b4
            """
            if debug:
                matricized_unary_scores = tf.Print(matricized_unary_scores, [matricized_unary_scores], message="matricized_unary_scores1:",summarize=100)
            matricized_unary_scores = tf.nn.sigmoid(matricized_unary_scores)



            unary_scores = tf.reshape( matricized_unary_scores, [size])
            if debug:
                unary_scores = tf.Print(unary_scores, [unary_scores], message="unary_scores:",summarize=100)

            scores_pos = tf.slice(unary_scores, [0], [size/2])
            if debug:
                scores_pos = tf.Print(scores_pos, [scores_pos], message="scores_pos:",summarize=100)
            scores_neg = tf.slice(unary_scores, [size/2], [size/2])
            if debug:
                scores_neg = tf.Print(scores_neg, [scores_neg], message="scores_neg:",summarize=100)
            costs = tf.maximum(0., self.margin - scores_pos + scores_neg)
            if debug:
                costs = tf.Print(costs, [costs], message="costs:")

            return unary_scores, costs, 0.

        #domain layer
        def _domain_layer(output_pub_query, output_pub_reply):  #output_pub batch_size * seq_len * (2 * lstm_dim)
            W_classifier = tf.get_variable(shape=[4 * lstm_dim, num_corpus],
                                           initializer=tf.truncated_normal_initializer(stddev=1.0 / math.sqrt(float(num_corpus))),
                                           name = 'W_classifier')
            bias = tf.Variable(
                tf.zeros([num_corpus],
                         name="class_bias"))
            #check axis
            output_pub = tf.concat(axis=1, values=[output_pub_query, output_pub_reply])
            output_avg = output_pub
            #output_avg = reduce_avg(output_pub, seq_len, 1)  #output_avg batch_size * (2 * lstm_dim)
            logits = tf.matmul(output_avg, W_classifier) + bias   #logits batch_size * num_corpus
            return logits

        def _Hloss(logits):
            log_soft = tf.nn.log_softmax(logits)  # batch_size * num_corpus
            soft = tf.nn.softmax(logits)
            H_mid = tf.reduce_mean(tf.multiply(soft, log_soft), axis=0)  # [num_corpus]
            H_loss = tf.reduce_sum(H_mid)
            return H_loss


        def _Dloss(logits, y_class):
            labels = tf.to_int64(y_class)
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits, labels=labels, name='xentropy')
            D_loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
            return D_loss

        def _loss(log_likelihood):

            loss = tf.reduce_mean(log_likelihood)

            return loss

        def _training(loss):

            optimizer = tf.train.AdamOptimizer(self.lr)
            global_step = tf.Variable(0, name="global_step", trainable=False)
            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), self.clip)
            train_op = optimizer.apply_gradients(zip(grads, tvars),
                                                      global_step = global_step)

            return train_op, global_step

        def _trainingPrivate(loss, taskid):
            optimizer = tf.train.AdamOptimizer(self.lr)
            global_step = tf.Variable(0, name="global_step", trainable=False)
            tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=taskid)
            grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), self.clip)
            train_op = optimizer.apply_gradients(zip(grads, tvars),
                                                 global_step=global_step)

            return train_op, global_step

        def _trainingDomain(loss):
            optimizer = tf.train.AdamOptimizer(self.lr)
            global_step = tf.Variable(0, name="global_step", trainable=False)
            tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='domain')
            grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), self.clip)
            train_op = optimizer.apply_gradients(zip(grads, tvars),
                                                 global_step=global_step)

            return train_op, global_step

        def _trainingShared(loss, taskid):
            optimizer = tf.train.AdamOptimizer(self.lr)
            global_step = tf.Variable(0, name="global_step", trainable=False)
            tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='shared') + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=taskid) + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='embedding')
            grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), self.clip)
            train_op = optimizer.apply_gradients(zip(grads, tvars),
                                      global_step=global_step)

            return train_op, global_step

        seq_len_query = tf.cast(self.seq_len_query, tf.int64)
        seq_len_reply = tf.cast(self.seq_len_reply, tf.int64)
        
        #xxx = tf.Print(self.x, [self.x], message="self.x:",summarize=100)
        #yyy = tf.Print(self.y, [self.y], message="self.y:",summarize=100)

        x = tf.nn.embedding_lookup(self.embedding, self.x)  # batch_size * (sequence*9) * word_dim
        y = tf.nn.embedding_lookup(self.embedding, self.y)
        #x = tf.nn.embedding_lookup(self.embedding, xxx)  # batch_size * (sequence*9) * word_dim
        #y = tf.nn.embedding_lookup(self.embedding, yyy)

        size = tf.shape(x)[0]
        # we use window_size 5 and bi_gram, which means for each position,
        # there will be 5+4=9 (character or word) features
        x = tf.reshape(x, [size, -1, word_dim])  # ba*se*(9*wd)
        y = tf.reshape(y, [size, -1, word_dim])  # ba*se*(9*wd)
        x = tf.nn.dropout(x, self.dropout_keep_prob)
        y = tf.nn.dropout(y, self.dropout_keep_prob)
        #task1:msr 2:as 3 pku 4 ctb 5 ckip 6 cityu 7 ncc 8 sxu 9 weibo
        with tf.variable_scope("shared"):
            output_pub_query, output_pub_reply, shared_score = _shared_layer(x, y, seq_len_query, seq_len_reply)

        #add adverisal op
        if self.adv:
            with tf.variable_scope("domain"):
                #note domain is on all the batch_size of only the right batch
                logits = _domain_layer(output_pub_query, output_pub_reply)
            self.H_loss = _Hloss(logits)
            self.D_loss = _Dloss(logits, self.y_class)

        self.scores = []
        #self.transition = []
        self.gate = []
        loglike = []
        #add task op
        for i in range(1, self.num_corpus+1):
            Taskid = 'task' + str(i)
            with tf.variable_scope(Taskid):
                condition = _private_layer(output_pub_query, output_pub_reply, x, y, seq_len_query, seq_len_reply, shared_score)
                self.scores.append(condition[0])
                loglike.append(condition[1])
                #self.transition.append(condition[2])
    

        #loss_com is combination loss(cws + hess), losses is basic loss(cws)
        self.losses = [_loss(o) for o in loglike]
        if self.adv:
            self.loss_com = [adv_weight * self.H_loss + o for o in self.losses]
            self.domain_op, self.global_step_domain = _trainingDomain(self.D_loss)
        #task_basic_op is for basic train
        self.task_basic_op = []
        self.global_basic_step = []
        for i in range(1, self.num_corpus+1):
            res = _training(self.losses[i-1])
            self.task_basic_op.append(res[0])
            self.global_basic_step.append(res[1])

        #task_op is for combination train(cws_loss + hess_loss * adv_weight)
        if self.adv:
            self.task_op = []
            self.global_step = []
            for i in range(1, self.num_corpus+1):
                Taskid = 'task' + str(i)
                res = _trainingShared(self.loss_com[i-1], taskid=Taskid)
                self.task_op.append(res[0])
                self.global_step.append(res[1])

        #task_op_ss is for private train
        self.task_op_ss = []
        self.global_pristep = []
        for i in range(1, self.num_corpus+1):
            Taskid = 'task' + str(i)
            res = _trainingPrivate(self.losses[i-1], Taskid)
            self.task_op_ss.append(res[0])
            self.global_pristep.append(res[1])

    #train all the basic model cwsloss, all parameters
    def train_step_basic(self, sess, x_batch, y_batch, seq_len_batch_query, seq_len_batch_reply, dropout_keep_prob, task_op, global_step, loss):

        feed_dict = {
            self.x: x_batch,
            self.y: y_batch,
            self.seq_len_query: seq_len_batch_query,
            self.seq_len_reply: seq_len_batch_reply,
            self.dropout_keep_prob: dropout_keep_prob
        }
        _, step, loss = sess.run(
            [task_op, global_step, loss],
            feed_dict)

        return step, loss

    # train all the cwsloss + hesloss VS advloss, main_line parameters Or cwsloss VS advloss(depends on taskop_type)
    def train_step_task(self, sess, x_batch, y_batch, seq_len_batch_query, seq_len_batch_reply, y_class_batch, dropout_keep_prob, task_op, global_step, loss,domain_op, global_step_domain, Dloss, Hloss):

        feed_dict = {
            self.x: x_batch,
            self.y: y_batch,
            self.y_class: y_class_batch,
            self.seq_len_query: seq_len_batch_query,
            self.seq_len_reply: seq_len_batch_reply,
            self.dropout_keep_prob: dropout_keep_prob
        }
        _, step_norm, loss_norm, _v, step_adv, loss_adv, loss_hess = sess.run(
            [task_op, global_step, loss, domain_op, global_step_domain, Dloss, Hloss],
            feed_dict)
        return step_norm, loss_norm, loss_adv, loss_hess


    # train only the private params, cwsloss
    def train_step_pritask(self, sess, x_batch, y_batch, seq_len_batch_query, seq_len_batch_reply, dropout_keep_prob, task_op, global_step, loss):

        
        feed_dict = {
            self.x: x_batch,
            self.y: y_batch,
            self.seq_len_query: seq_len_batch_query,
            self.seq_len_reply: seq_len_batch_reply,
            self.dropout_keep_prob: dropout_keep_prob
        }
        _, step, loss = sess.run(
            [task_op, global_step, loss],
            feed_dict)

        return step, loss


    #predict all for tasks
    def fast_all_predict(self, sess, N, batch_iterator, scores):
        num_batches = int((N - 1) / self.batch_size)
        res_scores = []
        for i in range(N):

            x_batch, y_batch, seq_len_batch_query, seq_len_batch_reply = batch_iterator.next_pred_one(self.batch_size)

            # infer predictions
            feed_dict = {
                self.x: x_batch,
                self.y: y_batch,
                self.seq_len_query: seq_len_batch_query,
                self.seq_len_reply: seq_len_batch_reply,
                self.dropout_keep_prob: 1.0
            }

            unary_scores = sess.run(
                    scores, feed_dict)
            for i in unary_scores:
                res_scores.append(i)

        return res_scores

    def fast_all_predict_cost(self, sess, N, batch_iterator, costs):
        num_batches = int((N - 1) / self.batch_size)
        res_scores = []
        for i in range(num_batches):

            x_batch, y_batch, seq_len_batch_query, seq_len_batch_reply = batch_iterator.next_batch(self.batch_size)

            # infer predictions
            feed_dict = {
                self.x: x_batch,
                self.y: y_batch,
                self.seq_len_query: seq_len_batch_query,
                self.seq_len_reply: seq_len_batch_reply,
                self.dropout_keep_prob: 1.0
            }

            unary_scores = sess.run(
                    costs, feed_dict)
            res_scores.append(unary_scores)

        return res_scores

    # predict one by one for tasks
    #useless
    def predict(self, sess, N, one_iterator, scores, transition_param):
        y_pred, y_true = [], []
        for i in xrange(N):
            x_one, y_one, len_one = one_iterator.next_pred_one()

            feed_dict = {
                self.x: x_one,
                self.y: y_one,
                self.seq_len: len_one,
                self.dropout_keep_prob: 1.0
            }

            unary_scores, transition_params = sess.run(
                [scores, transition_param], feed_dict)

            unary_scores_ = unary_scores[0]
            y_one_ = y_one[0]

            viterbi_sequence, _ = tf.contrib.crf.viterbi_decode(
                unary_scores_, transition_params)

            y_pred += viterbi_sequence
            y_true += y_one_[:len_one[0]].tolist()

        return y_true, y_pred

