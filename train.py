import tensorflow as tf

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
            en2idx, idx2en = load_vocab()
            
            # Encoder
            with tf.variable_scope("encoder"):
                ## Embedding
                self.enc_x = embedding(self.x, 
                                      vocab_size=len(de2idx), 
                                      num_units=hp.hidden_units, 
                                      scale=True,
                                      scope="enc_x_embed")

                self.enc_y = embedding(self.y, 
                                      vocab_size=len(de2idx), 
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
                import ipdb; ipdb.set_trace()
                with tf.variable_scope("num_blocks_{}".format(i)):
#                    self.enc_x = multihead_attention(queries=self.enc_x, 
#                                                    keys=self.enc_x, 
#                                                    num_units=hp.hidden_units, 
#                                                    num_heads=hp.num_heads, 
#                                                    dropout_rate=hp.dropout_rate,
#                                                    is_training=is_training,
#                                                    causality=False)
#                    self.enc_y = multihead_attention(queries=self.enc_y, 
#                                                    keys=self.enc_y, 
#                                                    num_units=hp.hidden_units, 
#                                                    num_heads=hp.num_heads, 
#                                                    dropout_rate=hp.dropout_rate,
#                                                    is_training=is_training,
#                                                    causality=False)
                    self.enc_x = feedforward(self.enc_x, num_units=[4*hp.hidden_units, hp.hidden_units])
                    self.enc_y = feedforward(self.enc_y, num_units=[4*hp.hidden_units, hp.hidden_units])
                    self.hl = tf.concat(self.enc_x, self.enc_y, -1)
                    self.hl = tf.layers.dense(self.hl, hp.hidden_units)
                    self.hl = tf.layers.relu(self.hl)
                    self.logits = tf.layers.dense(self.hl, 2)
            print()
            if is_training:  
                self.loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.hl, labels=self.label)
                self.mean_loss = tf.reduce_sum(self.loss*self.istarget) / (tf.reduce_sum(self.istarget))
#                self.optimizer = tf.train.AdamOptimizer(learning_rate=hp.lr, beta1=0.9, beta2=0.98, epsilon=1e-8)
#                self.train_op = self.optimizer.minimize(self.mean_loss)
                tf.summary.FileWriter(logdir='./graph/', graph=self.graph)

if __name__ == '__main__':                
    # Load vocabulary    
    word2idx, idx2word = load_vocab()
    
    # Construct graph
    g = Graph("train"); print("Graph loaded")
    
    # Start session
    sv = tf.train.Supervisor(graph=g.graph, 
                             logdir="log",
                             save_model_secs=0)
    with sv.managed_session() as sess:
        for epoch in range(1, hp.num_epochs+1): 
            if sv.should_stop(): break
            for step in tqdm(range(g.num_batch), total=g.num_batch, ncols=70, leave=False, unit='b'):
                x = sess.run(g.logits)
                print(x)
#            sv.saver.save(sess, hp.logdir + '/model_epoch_%02d' % (epoch))
    print("Done")    
