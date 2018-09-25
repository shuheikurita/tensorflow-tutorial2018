import tensorflow as tf
import numpy as np

from data_load import get_batch_data, load_vocab
from hyperparams import Hyperparams as hp
from modules import *
#from tqdm import tqdm

class Graph():
    def __init__(self, is_training=True):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.idx_q = tf.placeholder(tf.int32, shape=(None, hp.maxlen))
            self.idx_a = tf.placeholder(tf.int32, shape=(None, hp.maxlen))
            self.label = tf.placeholder(tf.int32, shape=(None))
            # Load vocabulary    
            de2idx, idx2de = load_vocab()
            vocab_size = len(de2idx) + 2
            
            # Encoder
            with tf.variable_scope("encoder"):
                ## Embedding
                self.enc_q = embedding(self.idx_q, 
                                      vocab_size=vocab_size, 
                                      num_units=hp.hidden_units, 
                                      scale=True,
                                      scope="enc_x_embed")

                self.enc_a = embedding(self.idx_a, 
                                      vocab_size=vocab_size, 
                                      num_units=hp.hidden_units, 
                                      scale=True,
                                      scope="enc_y_embed")
                ## Dropout
                self.enc_q = tf.layers.dropout(self.enc_q, 
                                            rate=hp.dropout_rate, 
                                            training=tf.convert_to_tensor(is_training))
                self.enc_a = tf.layers.dropout(self.enc_a, 
                                            rate=hp.dropout_rate, 
                                            training=tf.convert_to_tensor(is_training))

                hid_q = cudnn_cp_stack_bilstm(self.enc_q, hp.stack_num, hp.hidden_units, hp.maxlen, hp.dropout_rate, "q")
                hid_a = cudnn_cp_stack_bilstm(self.enc_a, hp.stack_num, hp.hidden_units, hp.maxlen, hp.dropout_rate, "a") 

                lstm_units = hp.hidden_units
                #rep_q = tf.concat([hid_q[:,-1,0:lstm_units+1],hid_q[:,0,lstm_units+1:]], axis=-1)
                #rep_a = tf.concat([hid_a[:,-1,0:lstm_units+1],hid_a[:,0,lstm_units+1:]], axis=-1)
                rep_q = tf.concat([hid_q[:,-1,:],hid_q[:,0,:]], axis=-1)
                rep_a = tf.concat([hid_a[:,-1,:],hid_a[:,0,:]], axis=-1)


                self.hl = tf.concat([rep_q,rep_a], -1)
                self.hl = tf.layers.dense(self.hl, hp.hidden_units, activation=tf.nn.relu)
                self.logits = tf.layers.dense(self.hl, 2)
                self.prob = tf.nn.softmax(self.logits)
            print()
            if is_training:  
                self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.label)
                # self.mean_loss = tf.reduce_sum(self.loss*self.istarget) / (tf.reduce_sum(self.istarget))
                self.mean_loss = tf.reduce_mean(self.loss)
                self.optimizer = tf.train.AdamOptimizer(learning_rate=hp.lr, beta1=0.9, beta2=0.98, epsilon=1e-8)
                self.train_op = self.optimizer.minimize(self.mean_loss)
                tf.summary.FileWriter(logdir='./graph/', graph=self.graph)

            #import ipdb; ipdb.set_trace()
            # saver
            self.saver = tf.train.Saver(max_to_keep=0)

            # session
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())

if __name__ == '__main__':                
    
    # Construct graph
    g = Graph("train"); print("Graph loaded")
    
    # Start session
    #sv = tf.train.Supervisor(graph=g.graph, 
    #                         logdir="log",
    #                         save_model_secs=0)
    #
    #self.x, self.y, self.label, self.num_batch = get_batch_data()
    #print("Now you are in Ipython in IPython notebook!")
    #import IPython; IPython.embed()
    
    datasets=[
            [np.array([ [1,4,5,8,3]+[0 for _ in range(11)], [1,1,2,6,3]+[0 for _ in range(11)] ]),
             np.array([ [1,2]+[0 for _ in range(14)],       [9,10]+[0 for _ in range(14)]])],
            [np.array([ [1,4,5,5,3]+[0 for _ in range(11)], [1,4,10,5,3]+[0 for _ in range(11)] ]),
             np.array([ [10]+[0 for _ in range(15)],        [10]+[0 for _ in range(15)]])],
            [np.array([ [4,4,5,6,3]+[0 for _ in range(11)], [9,4,2,6,3]+[0 for _ in range(11)] ]),
             np.array([ [1,2]+[0 for _ in range(14)],       [5,10,4,10]+[0 for _ in range(12)]])],
            [np.array([ [4,4,5,8,3]+[0 for _ in range(11)], [1,4,10,5,3]+[0 for _ in range(11)] ]),
             np.array([ [1,2]+[0 for _ in range(14)],        [1,2]+[0 for _ in range(14)]])],
    ]
    labels=[np.array([0,1]), np.array([1,1]), np.array([0,1]), np.array([0,0])]
#     print(datasets)
#     print(labels)
#     print(label)

    ## Train
    for epoch in range(1, hp.num_epochs):
        print("*** Epoch %d ***"%epoch)
        for mb,(data,lab) in enumerate(zip(datasets,labels)):
            feed_dict={
                    g.idx_q:data[0],
                    g.idx_a:data[1],
                    g.label:lab,
                    }
            log_loss,log_train_op = g.sess.run([g.mean_loss,g.train_op],feed_dict=feed_dict)
            print("Epoch: %d, Mini-batch: %d, loss: %f"%(epoch,mb,log_loss))
    print("Train Done.")
    
    ## Test
    for data,lab in zip(datasets,labels):
        feed_dict={
                g.idx_q:data[0],
                g.idx_a:data[1],
                g.label:lab,
                }
        log_prob = g.sess.run([g.prob],feed_dict=feed_dict)
        log_class = np.argmax(log_prob,axis=-1)
        print("Probability: "+str(log_prob))
        print("Argmax: "+str(log_class))
    print("Test Done.")    
