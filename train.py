import tensorflow as tf

from data_load import get_batch_data, load_vocab
from modules import *
from tqdm import tqdm

class Graph():
    def __init__(self, is_training=True):
        self.graph = tf.Graph()
        with self.graph.as_default():
            if is_training:
                self.x, self.y, self.label, self.num_batch = get_batch_data()
            else:
                self.x = tf.placeholder(tf.int32, shape=(hp.batchsize, hp.maxlen))
                self.y = tf.placeholder(tf.int32, shape=(hp.batchsize, hp.maxlen))
                self.label = tf.placeholder(tf.int32, shape=hp.batchsize)

            # Load vocabulary    
            de2idx, idx2de = load_de_vocab()
            en2idx, idx2en = load_en_vocab()
            
            # Encoder
            with tf.variable_scope("encoder"):
                ## Embedding
                self.enc_x = embedding(self.x, 
                                      vocab_size=len(de2idx), 
                                      num_units=hp.hidden_units, 
                                      scale=True,
                                      scope="enc_embed")

                self.enc_y = embedding(self.x, 
                                      vocab_size=len(de2idx), 
                                      num_units=hp.hidden_units, 
                                      scale=True,
                                      scope="enc_embed")
                ## Dropout
                self.enc_x = tf.layers.dropout(self.enc_x, 
                                            rate=hp.dropout_rate, 
                                            training=tf.convert_to_tensor(is_training))
                self.enc_y = tf.layers.dropout(self.enc_y, 
                                            rate=hp.dropout_rate, 
                                            training=tf.convert_to_tensor(is_training))
                ## Blocks
#                for i in range(hp.num_blocks):
                with tf.variable_scope("num_blocks_{}".format(i)):
                    self.enc_x = multihead_attention(queries=self.enc_x, 
                                                    keys=self.enc_x, 
                                                    num_units=hp.hidden_units, 
                                                    num_heads=hp.num_heads, 
                                                    dropout_rate=hp.dropout_rate,
                                                    is_training=is_training,
                                                    causality=False)
                    self.enc_y = multihead_attention(queries=self.enc_y, 
                                                    keys=self.enc_y, 
                                                    num_units=hp.hidden_units, 
                                                    num_heads=hp.num_heads, 
                                                    dropout_rate=hp.dropout_rate,
                                                    is_training=is_training,
                                                    causality=False)
                    self.enc_x = feedforward(self.enc_x, num_units=[4*hp.hidden_units, hp.hidden_units])
                    self.enc_y = feedforward(self.enc_y, num_units=[4*hp.hidden_units, hp.hidden_units])
                    self.enc = tf.concat(self.enc_x, self.enc_y)
                    

            train = optimizer.minimize(loss)
            if is_training:  
                self.loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y_smoothed)
                self.mean_loss = tf.reduce_sum(self.loss*self.istarget) / (tf.reduce_sum(self.istarget))
               
                # Training Scheme
                self.global_step = tf.Variable(0, name='global_step', trainable=False)
                self.optimizer = tf.train.AdamOptimizer(learning_rate=hp.lr, beta1=0.9, beta2=0.98, epsilon=1e-8)
                self.train_op = self.optimizer.minimize(self.mean_loss, global_step=self.global_step)
                   
                # Summary 
                tf.summary.scalar('mean_loss', self.mean_loss)
                self.merged = tf.summary.merge_all()
            

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
                sess.run(g.train_op)
#            gs = sess.run(g.global_step)   
            sv.saver.save(sess, hp.logdir + '/model_epoch_%02d_gs_%d' % (epoch, gs))
    print("Done")    
