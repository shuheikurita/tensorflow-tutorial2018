import tensorflow as tf

def embedding(inputs, 
              vocab_size, 
              num_units, 
              zero_pad=True, 
              scale=True,
              scope="embedding", 
              reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        lookup_table = tf.get_variable('lookup_table',
                                       dtype=tf.float32,
                                       shape=[vocab_size, num_units],
                                       initializer=tf.contrib.layers.xavier_initializer())
        if zero_pad:
            lookup_table = tf.concat((tf.zeros(shape=[1, num_units]),
                                      lookup_table[1:, :]), 0)
        outputs = tf.nn.embedding_lookup(lookup_table, inputs)
        if scale:
            outputs = outputs * (num_units ** 0.5) 
    return outputs

# input shape must be [mini_batch, time_sequence, embedding]
def cudnn_stack_bilstm(input, num_stacked, dim_hidden, timesteps, dropout_rate, name):
    # Define LSTMs
    bilstmA = tf.contrib.cudnn_rnn.CudnnLSTM(num_stacked, dim_hidden, dropout=dropout_rate, name="forward_lstm")
    bilstmB = tf.contrib.cudnn_rnn.CudnnLSTM(num_stacked, dim_hidden, dropout=dropout_rate, name="backward_lstm")
    # Define Computation Graph
    rnninput = tf.transpose(input, perm=[1,0,2])
    rnnoutputA,_ = bilstmA(rnninput)
    rnnoutputB,_ = bilstmB(rnninput[::-1])
    # 各word毎のhidden representationが欲しい時
    rnnoutput = tf.concat([rnnoutputA,rnnoutputB[::-1]],axis=2)
    rnnoutput = tf.transpose(rnnoutput,perm=[1,0,2])
    # 最初と最後のhidden representationを取ってくる
    # rnnoutput_y = tf.concat([rnnoutputA,rnnoutputB],axis=2)[-1] # shape=[timestep, minibatch, emb]
    return rnnoutput

# input shape must be [mini_batch, time_sequence, embedding]
def cudnn_cp_stack_bilstm(input, num_stacked, dim_hidden, timesteps, dropout, name):
    # Unstack to expand input to a list of 'timesteps' tensors of shape (mini_batch, embedding)
    input = tf.unstack(input, timesteps, 1)
    # Define lstm cells with tensorflow
    lstm_fw_cell = [tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(dim_hidden, reuse=tf.AUTO_REUSE) for i in range(num_stacked)]
    lstm_bw_cell = [tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(dim_hidden, reuse=tf.AUTO_REUSE) for i in range(num_stacked)]
    # Define computation graph of LSTMs
    outputs, _, _ = tf.contrib.rnn.stack_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, input, dtype=tf.float32)
    # Re-stack at axis of the timesteps.
    outputs=tf.stack(outputs,axis=1)
    return outputs

