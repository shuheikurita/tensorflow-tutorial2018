from hyperparams import Hyperparams as hp
import tensorflow as tf
import numpy as np

def load_vocab():
    idx = 2
    word2idx, idx2word = dict(), dict()
    for line in open("vocab.txt"):
        word2idx[line.strip()] = idx
        idx2word[idx] = line.strip()
        idx += 1
    word2idx["</S>"] = idx
    idx2word[idx] = "</S>"
    return word2idx, idx2word

def load_data(text_q_file, text_a_file):
    word2idx, idx2word = load_vocab()
    q_list, a_list = [], []
    for line in open(text_q_file):
        q_list.append(
                np.array([word2idx.get(item, 1) for item in (line.strip()+" </S>").split(" ")])
                )
    for line in open(text_a_file):
        a_list.append(
                np.array([word2idx.get(item, 1) for item in (line.strip()+" </S>").split(" ")])
                )
    q_array = np.zeros([len(q_list), hp.maxlen], np.int32)
    a_array = np.zeros([len(a_list), hp.maxlen], np.int32)
    for i, (x, y) in enumerate(zip(q_list, a_list)):
        q_array[i] = np.lib.pad(x, [0, hp.maxlen-len(x)], 'constant', constant_values=(0, 0))
        a_array[i] = np.lib.pad(y, [0, hp.maxlen-len(y)], 'constant', constant_values=(0, 0))
    return q_array, a_array

def load_label(text_label):
    with open(text_label,"rt") as f:
        return np.loadtxt(f)

def get_batch_data():
    q_list, a_list = load_data(hp.q_train_data_path, hp.a_train_data_path)
    l_list = load_label(hp.label_train_data_path)
    #print(q_list)
    q = tf.convert_to_tensor(q_list, tf.int32)
    a = tf.convert_to_tensor(a_list, tf.int32)
    l = tf.convert_to_tensor(l_list, tf.int32)
    batch_size = 1
    num_batch = len(q_list) // batch_size
    input_queues = tf.train.slice_input_producer([q, a, l])
    x, y, z = tf.train.shuffle_batch(input_queues,
                                num_threads=1,
                                batch_size=1, 
                                capacity=2,   
                                min_after_dequeue=1, 
                                allow_smaller_final_batch=False)
    return x, y, z, num_batch # (N, T), (N, T), ()

if __name__ == '__main__': 
    print(get_batch_data())
