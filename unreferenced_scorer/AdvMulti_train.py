import os
import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
import logging
import sys
import cPickle as pkl

from sklearn.metrics import accuracy_score

from voc import Vocab
from config import WORD_VEC_100, TRAIN_FILE, TEST_FILE, DEV_FILE, DATA_FILE, DROP_OUT, WORD_DICT, ADV_STATUS


from AdvMulti_model import MultiModel
import data_helpers

# ==================================================

init_embedding = Vocab(WORD_VEC_100, WORD_DICT, single_task=False, bi_gram=False).word_vectors
tf.flags.DEFINE_integer("vocab_size", init_embedding.shape[0], "vocab_size")

# Data parameters
tf.flags.DEFINE_integer("word_dim", 128, "word_dim")
tf.flags.DEFINE_integer("lstm_dim", 256, "lstm_dim")
tf.flags.DEFINE_integer("num_corpus", 1, "num_corpus")
tf.flags.DEFINE_boolean("real_status", True, "real_status")
tf.flags.DEFINE_boolean("train", False, "train_status")

# Model Hyperparameters[t]
tf.flags.DEFINE_float("lr", 0.001, "learning rate (default: 0.01)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.000, "L2 regularizaion lambda (default: 0.5)")
tf.flags.DEFINE_float("adv_weight", 0.06, "L2 regularizaion lambda (default: 0.5)")
tf.flags.DEFINE_float("clip", 5, "gradient clip")
tf.flags.DEFINE_float("margin", 0.05, "margin")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 128, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 150000, "Number of training epochs (default: 40)")
tf.flags.DEFINE_integer("num_epochs_private", 150000, "Number of training epochs (default: 40)")
tf.flags.DEFINE_integer("evaluate_every", 200, "Evaluate model on dev set after this many steps (default: 100)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()


print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items(),reverse=True):
    print("{}={} \n".format(attr.upper(), value))
print("")
#define log file
logger = logging.getLogger('record')
hdlr = logging.FileHandler('AdvMulti_train.log')
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
hdlr.setFormatter(formatter)
logger.addHandler(hdlr)
logger.setLevel(logging.INFO)

reuse_status = True
sep_status = True

stats = [ADV_STATUS]
posfix = map(lambda x: 'Y' if x else 'N', stats)
if ADV_STATUS:
    posfix.append(str(FLAGS.adv_weight))

#Load data
train_data_iterator = []
dev_data_iterator = []
test_data_iterator = []
dev_df = []
test_df = []
print("Loading data...")
for i in xrange(FLAGS.num_corpus):
    train_data_iterator.append(data_helpers.BucketedDataIterator(pd.read_csv(TRAIN_FILE[i])))
    dev_df.append(pd.read_csv(DEV_FILE[i]))
    dev_data_iterator.append(data_helpers.BucketedDataIterator(dev_df[i]))
    test_df.append(pd.read_csv(TEST_FILE[i]))
    test_data_iterator.append(data_helpers.BucketedDataIterator(test_df[i], is_test=True))

logger.info('-'*50)

# Training
# ==================================================
with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)

    session_conf.gpu_options.allow_growth = True

    sess = tf.Session(config=session_conf)
    with sess.as_default():

        # build model
        model = MultiModel(batch_size=FLAGS.batch_size,
                      vocab_size=FLAGS.vocab_size,
                      word_dim=FLAGS.word_dim,
                      lstm_dim=FLAGS.lstm_dim,
                      num_corpus=FLAGS.num_corpus,
                      lr=FLAGS.lr,
                      clip=FLAGS.clip,
                      l2_reg_lambda=FLAGS.l2_reg_lambda,
                      adv_weight = FLAGS.adv_weight,
                      adv=ADV_STATUS,
                      margin=FLAGS.margin)

        # Output directory for models
        model_name = 'multi_model'+ str(FLAGS.num_corpus)
        try:
            shutil.rmtree(os.path.join(os.path.curdir, "models", model_name))
        except:
            pass
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "models", model_name))
        print("Writing to {}\n".format(out_dir))

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        # modeli_embed_adv_gate_diff_dropout
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_all = []
        for i in xrange(1, FLAGS.num_corpus+1):
            filename = 'task' + str(i) + '_' + '_'.join(posfix)
            checkpoint_all.append(os.path.join(checkpoint_dir, filename))
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(),max_to_keep=20)

        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        # Initialize all the op
        # basictask is for basic cws loss, task is for combination loss, privatetask is for cws loss only on solo params
        # testTask is for prediction, taskID is for storage of loading from .csv
        basictask = []
        task = []
        privateTask = []
        testTask = []
        taskID = []
        for i in xrange(FLAGS.num_corpus):
            basictask.append([model.task_basic_op[i],model.global_basic_step[i],model.losses[i]])
            if model.adv:
                task.append([model.task_op[i], model.global_step[i], model.loss_com[i]])
            privateTask.append([model.task_op_ss[i], model.global_pristep[i], model.losses[i]])
            testTask.append([model.scores[i], model.losses[i]])
            taskID.append([train_data_iterator[i],dev_df[i],dev_data_iterator[i],test_df[i],test_data_iterator[i]])

        def train_step_basic(x_batch, y_batch, seq_len_batch_query, seq_len_batch_reply, id, ii):
            step, loss = model.train_step_basic(sess,
                x_batch, y_batch, seq_len_batch_query, seq_len_batch_reply, DROP_OUT[id-1], basictask[id-1][0], basictask[id-1][1], basictask[id-1][2])

            time_str = datetime.datetime.now().isoformat()
            if ii%40==0:
                print("Task_{}: {}: step {}, loss {:g}".format(id, time_str, step, loss))

            return step


        def train_step_all(x_batch, y_batch, y_class_batch, seq_len_batch_query, seq_len_batch_reply, id, ii):
            step, loss_cws, loss_adv, loss_hess = model.train_step_task(sess,
                   x_batch, y_batch, seq_len_batch_query, seq_len_batch_reply, y_class_batch, DROP_OUT[id-1], task[id-1][0], task[id-1][1], task[id-1][2], model.domain_op, model.global_step_domain, model.D_loss, model.H_loss)

            time_str = datetime.datetime.now().isoformat()
            if ii%40==0:
                print("Task_{}: {}: step {}, loss_cws {:g}, loss_adv {:g}, loss_hess {:g}".format(id, time_str, step, loss_cws, loss_adv, loss_hess))

            return step

        def train_step_private(x_batch, y_batch, seq_len_batch_query, seq_len_batch_reply, id, ii):
            step, loss = model.train_step_pritask(sess,
                x_batch, y_batch, seq_len_batch_query, seq_len_batch_reply, DROP_OUT[id-1], privateTask[id-1][0], privateTask[id-1][1], privateTask[id-1][2])

            time_str = datetime.datetime.now().isoformat()
            if ii%40==0:
                print("Task_{}: {}: step {}, loss {:g}".format(id, time_str, step, loss))

            return step


        def final_test_step(df, iterator, idx, test=False):
            N = df.shape[0]
            scores = model.fast_all_predict(sess, N, iterator, testTask[idx-1][0])
            if test:
                print "test:",
            else:
                print "dev:",
            return scores

        def final_test_step_cost(df, iterator, idx, test=False):
            N = df.shape[0]
            costs = model.fast_all_predict_cost(sess, N, iterator, testTask[idx-1][1])
            costs = np.array(costs)
            costs = np.sum(costs)
            if test:
                print "test:",
            else:
                print "dev:",
            return costs

        # train loop
        if FLAGS.train:
            best_accuary = [100.0] * FLAGS.num_corpus
            best_step_all = [0] * FLAGS.num_corpus
            best_val = [0.0] * FLAGS.num_corpus
            best_step_private = [0] * FLAGS.num_corpus
            flag = [False] * FLAGS.num_corpus
            logger.info('-------------Public train starts--------------')
            for i in range(FLAGS.num_epochs):
                for j in range(1, FLAGS.num_corpus + 1):
                    if model.adv:
                        x_batch, y_batch, y_class, seq_len_batch_query, seq_len_batch_reply = taskID[j - 1][0].next_batch(FLAGS.batch_size, round=j-1, classifier=True)
                        current_step = train_step_all(x_batch, y_batch, y_class, seq_len_batch_query, seq_len_batch_reply, j, i)
                    else:
                        x_batch, y_batch, seq_len_batch_query, seq_len_batch_reply = taskID[j - 1][0].next_batch(FLAGS.batch_size)
                        current_step = train_step_basic(x_batch, y_batch, seq_len_batch_query, seq_len_batch_reply, j, i)

                    if current_step % FLAGS.evaluate_every == 0:
                        tmp_f = final_test_step_cost(taskID[j-1][1], taskID[j-1][2], j)
                        print tmp_f/FLAGS.batch_size*2
                        if best_accuary[j - 1] > tmp_f:
                            best_accuary[j - 1] = tmp_f
                            best_step_all[j - 1] = current_step
                            if FLAGS.real_status:
                                path = saver.save(sess, checkpoint_all[j-1])
                                print("Saved model checkpoint to {}\n".format(path))
                            else:
                                print("This is only for trial and error\n")
                            #tmp_f = final_test_step_cost(taskID[j-1][3], taskID[j-1][4], j, test=True)
                            #best_val[j - 1] = tmp_f
            logger.info('-----------Public train ends-------------')
            for i in xrange(FLAGS.num_corpus):
                logger.info('Task{} best step is {} and {:.2f} '.format(i + 1, best_step_all[i],best_val[i]*100))

            for i in range(FLAGS.num_epochs_private):
                stop = True
                for j in range(FLAGS.num_corpus):
                    if flag[j] is False:
                        stop = False
                if stop is False:
                    for j in range(1, FLAGS.num_corpus + 1):
                        if flag[j - 1]:
                            continue
                        else:
                            x_batch, y_batch, seq_len_batch_query, seq_len_batch_reply = taskID[j - 1][0].next_batch(FLAGS.batch_size)
                            current_step = train_step_private(x_batch, y_batch, seq_len_batch_query, seq_len_batch_reply, j, i)
                            if current_step % FLAGS.evaluate_every == 0:
                                tmp_f = final_test_step_cost(taskID[j - 1][1], taskID[j - 1][2], j)
                                print tmp_f/FLAGS.batch_size*2
                                if best_accuary[j - 1] > tmp_f:
                                    best_accuary[j - 1] = tmp_f
                                    best_step_private[j - 1] = current_step
                                    if FLAGS.real_status:
                                        path = saver.save(sess, checkpoint_all[j - 1])
                                        print("Saved model checkpoint to {}\n".format(path))
                                    else:
                                        print("This is only for trial and error\n")
                                    #tmp_f = final_test_step_cost(taskID[j-1][3], taskID[j-1][4], j, test=True)
                                    #best_val[j-1] = tmp_f
                                #elif current_step - best_step_private[j - 1] > 2000:
                                #    print("Task_{} didn't get better results in more than 2000 steps".format(j))
                                #    flag[j - 1] = True
                else:
                    print 'Early stop triggered, all the tasks have been finished. Dropout:', DROP_OUT
                    break


        if FLAGS.real_status:
            logger.info('-------------Show the results------------')
            for i in xrange(FLAGS.num_corpus):
                filename = 'Model' + str(i+1) + '_' + '_'.join(posfix)
                saver.restore(sess, checkpoint_all[i])
                print 'Task:{}\n'.format(i+1)
                logger.info('Task:{}, filename:{}'.format(i+1, filename))
                yp = final_test_step(taskID[i][3], taskID[i][4], i+1, test=True)
                print yp
                pkl.dump(yp, open("result.pkl"+str(i), "wb"))
            exit()
