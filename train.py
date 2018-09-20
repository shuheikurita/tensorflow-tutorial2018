import tensorflow as tf
import numpy as np

from data_load import get_batch_data, load_vocab
from hyperparams import Hyperparams as hp
from modules import *
from tqdm import tqdm

class Graph():
    def __init__(self, is_training=True):
        self.graph = tf.Graph()
        with self.graph.as_default():
#            self.x, self.y, self.label, self.num_batch = get_batch_data()
            self.x = tf.placeholder(tf.int32, shape=(None, hp.maxlen))
            self.y = tf.placeholder(tf.int32, shape=(None, hp.maxlen))
            self.label = tf.placeholder(tf.int32, shape=(None))
            # Load vocabulary    
            de2idx, idx2de = load_vocab()
            vocab_size = len(de2idx) + 2
            
            # Encoder
            with tf.variable_scope("encoder"):
                ## Embedding
                self.enc_x = embedding(self.x, 
                                      vocab_size=vocab_size, 
                                      num_units=hp.hidden_units, 
                                      scale=True,
                                      scope="enc_x_embed")

                self.enc_y = embedding(self.y, 
                                      vocab_size=vocab_size, 
                                      num_units=hp.hidden_units, 
                                      scale=True,
                                      scope="enc_y_embed")
                ## Dropout
                self.enc_x = tf.layers.dropout(self.enc_x, 
                                            rate=hp.dropout_rate, 
                                            training=tf.convert_to_tensor(is_training))
                self.enc_y = tf.layers.dropout(self.enc_y, 
                                            rate=hp.dropout_rate, 
                                            training=tf.convert_to_tensor(is_training))
                ## Blocks transformerでは積んでた
                #with tf.variable_scope("num_blocks_{}".format(1)):
                with tf.variable_scope("enc_x"):
                    rnninput = tf.transpose(self.enc_x,perm=[1,0,2])
                    stack_num = 1
                    bilstmA = tf.contrib.cudnn_rnn.CudnnLSTM(stack_num, hp.hidden_units, dropout=hp.dropout_rate, name="forward_lstm")
                    bilstmB = tf.contrib.cudnn_rnn.CudnnLSTM(stack_num, hp.hidden_units, dropout=hp.dropout_rate, name="backward_lstm")
                    rnnoutputA,_ = bilstmA(rnninput)
                    rnnoutputB,_ = bilstmB(rnninput[::-1])
                    # 各word毎のhidden representationが欲しい時
                    # rnnoutput = tf.concat([rnnoutputA,rnnoutputB[::-1]],axis=2)
                    # rnnoutput = tf.transpose(rnnoutput,perm=[1,0,2])
                    # 最初と最後のhidden representationを取ってくる
                    rnnoutput_x = tf.concat([rnnoutputA,rnnoutputB],axis=2)[-1] # shape=[timestep, minibatch, emb]

                with tf.variable_scope("enc_y"):
                    rnninput = tf.transpose(self.enc_y,perm=[1,0,2])
                    stack_num = 1
                    bilstmA = tf.contrib.cudnn_rnn.CudnnLSTM(stack_num, hp.hidden_units, dropout=hp.dropout_rate, name="forward_lstm")
                    bilstmB = tf.contrib.cudnn_rnn.CudnnLSTM(stack_num, hp.hidden_units, dropout=hp.dropout_rate, name="backward_lstm")
                    rnnoutputA,_ = bilstmA(rnninput)
                    rnnoutputB,_ = bilstmB(rnninput[::-1])
                    # 各word毎のhidden representationが欲しい時
                    # rnnoutput = tf.concat([rnnoutputA,rnnoutputB[::-1]],axis=2)
                    # rnnoutput = tf.transpose(rnnoutput,perm=[1,0,2])
                    # 最初と最後のhidden representationを取ってくる
                    rnnoutput_y = tf.concat([rnnoutputA,rnnoutputB],axis=2)[-1] # shape=[timestep, minibatch, emb]
 

                rnnoutput_xy = tf.concat([rnnoutputA,rnnoutputB[::-1]],axis=2)

                #self.enc_x = feedforward(self.enc_x, num_units=[4*hp.hidden_units, hp.hidden_units])
                #self.enc_y = feedforward(self.enc_y, num_units=[4*hp.hidden_units, hp.hidden_units])

                #self.hl = tf.concat(self.enc_x, self.enc_y, -1)
                self.hl = tf.concat([rnnoutput_x,rnnoutput_y], -1)
                self.hl = tf.layers.dense(self.hl, hp.hidden_units, activation=tf.nn.relu)
                self.logits = tf.layers.dense(self.hl, 2)
            print()
            if is_training:  
                self.loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.label)
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
    for epoch in range(1, hp.num_epochs+1):
        print("*** Epoch %d ***"%epoch)
        datasets=[0,1]
        for data in datasets:
            feed_dict={
                    g.x:np.array([[1,2,3,4],[2,2,3,4]]),
                    g.y:np.array([[1,2,3,4],[1,2,0,0]]),
                    g.label:np.array([0,1]),
                    }
            loss,train_loss = sess.run([self.mean_loss,self.train_op],feed_dict=feed_dict)
            print("Epoch: %d, loss: %f"%(epoch,loss))

    with sv.managed_session() as sess:
        for epoch in range(1, hp.num_epochs+1):
            if sv.should_stop(): break
            for step in tqdm(range(g.num_batch), total=g.num_batch, ncols=70, leave=False, unit='b'):
                x = sess.run(g.logits)
                print(x)
#            sv.saver.save(sess, hp.logdir + '/model_epoch_%02d' % (epoch))
    print("Done")    
