
import tensorflow as tf
import numpy as np
from sklearn.metrics import mean_absolute_error

def rnn_predictor (X, T, hidden_dim):
      
        with tf.variable_scope("predictor", reuse = tf.AUTO_REUSE) as vs:
            
            d_cell = tf.nn.rnn_cell.GRUCell(num_units=hidden_dim, activation=tf.nn.tanh, name = 'd_cell')
                    
            d_outputs, d_last_states = tf.nn.dynamic_rnn(d_cell, X, dtype= tf.float32, sequence_length = T)
                
            Y_hat = tf.contrib.layers.fully_connected(d_outputs, 1, activation_fn=None) 
            
            Y_hat_Final = tf.nn.sigmoid(Y_hat)
            
            d_vars = [v for v in tf.all_variables() if v.name.startswith(vs.name)]
    
        return Y_hat_Final, d_vars

def train_synt_test_real (data_real, data_synt, params):

    tf.reset_default_graph()

    data_len = params['data_len']
    data_dim = params['data_dim']
    seq_len = params['seq_len']
    hidden_dim = params['hidden_dim']
    iterations = params['iterations']
    batch_size = params['batch_size']
    
    train_len = round(0.8*data_len)
    idx = np.random.permutation(data_len)
    train_idx = idx[:train_len]
    test_idx = idx[train_len:data_len]
    data_train_real = list(data_real[i] for i in train_idx)
    data_test_real = list(data_real[i] for i in test_idx)
    data_train_synt = list(data_synt[i] for i in train_idx)
    data_test_synt = list(data_synt[i] for i in test_idx)

    seq_len_list = []
    max_seq_len = seq_len

    for i in range(data_len):
        seq_len = len(data_real[i][:,0])
        seq_len_list.append(seq_len)
        if max_seq_len < seq_len:
            max_seq_len = seq_len

    
    X = tf.placeholder(tf.float32, [None, max_seq_len-1, data_dim-1])
    T = tf.placeholder(tf.int32, [None])    
    Y = tf.placeholder(tf.float32, [None, max_seq_len-1, 1])

    Y_pred, d_vars = rnn_predictor(X, T, hidden_dim)
    D_loss = tf.losses.absolute_difference(Y, Y_pred)
    D_opt = tf.train.AdamOptimizer().minimize(D_loss, var_list = d_vars)
    
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    # Training 
    for itt in range(iterations):

        idx = np.random.permutation(len(data_train_synt))
        train_idx = idx[:batch_size]     
            
        X_mb = list(data_train_synt[i][:-1,:(data_dim-1)] for i in train_idx)
        T_mb = list(seq_len-1 for i in train_idx)
        Y_mb = list(np.reshape(data_train_synt[i][1:,(data_dim-1)],[len(data_train_synt[i][1:,(data_dim-1)]),1]) for i in train_idx)        

        _, step_d_loss = sess.run([D_opt, D_loss], feed_dict={X: X_mb, T: T_mb, Y: Y_mb})            

    idx = np.random.permutation(len(data_test_real))
    train_idx = idx[:len(data_test_real)] 
    
    X_mb = list(data_test_real[i][:-1,:(data_dim-1)] for i in train_idx)
    T_mb = list(seq_len-1 for i in train_idx)
    Y_mb = list(np.reshape(data_test_real[i][1:,(data_dim-1)], [len(data_test_real[i][1:,(data_dim-1)]),1]) for i in train_idx)
    
    pred_Y_curr = sess.run(Y_pred, feed_dict={X: X_mb, T: T_mb})
    
    # Test MAE 
    MAE_Temp = 0
    for i in range(len(data_test_real)):
        MAE_Temp = MAE_Temp + mean_absolute_error(Y_mb[i], pred_Y_curr[i,:,:])
    
    MAE_tstr = MAE_Temp / len(data_test_real)
    
    return MAE_tstr
def train_real_test_real (data_real, params):
  
    tf.reset_default_graph()

    data_len = params['data_len']
    data_dim = params['data_dim']
    seq_len = params['seq_len']
    hidden_dim = params['hidden_dim']#max(int(data_dim/2),1)
    iterations = params['iterations']#5000
    batch_size = params['batch_size']#128
    
    train_len = round(0.8*data_len)
    idx = np.random.permutation(data_len)
    train_idx = idx[:train_len]
    test_idx = idx[train_len:data_len]
    data_train_real = list(data_real[i] for i in train_idx)
    data_test_real = list(data_real[i] for i in test_idx)

    seq_len_list = []
    max_seq_len = seq_len

    for i in range(data_len):
        seq_len = len(data_real[i][:,0])
        seq_len_list.append(seq_len)
        if max_seq_len < seq_len:
            max_seq_len = seq_len
 
    
    X = tf.placeholder(tf.float32, [None, max_seq_len-1, data_dim-1])
    T = tf.placeholder(tf.int32, [None])    
    Y = tf.placeholder(tf.float32, [None, max_seq_len-1, 1])

    Y_pred, d_vars = rnn_predictor(X, T, hidden_dim)
    D_loss = tf.losses.absolute_difference(Y, Y_pred)
    D_opt = tf.train.AdamOptimizer().minimize(D_loss, var_list = d_vars)
    
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    # Training 
    for itt in range(iterations):
          
        idx = np.random.permutation(len(data_train_real))
        train_idx = idx[:batch_size]     
            
        X_mb = list(data_train_real[i][:-1,:(data_dim-1)] for i in train_idx)
        T_mb = list(seq_len-1 for i in train_idx)
        Y_mb = list(np.reshape(data_train_real[i][1:,(data_dim-1)],[len(data_train_real[i][1:,(data_dim-1)]),1]) for i in train_idx)        

        _, step_d_loss = sess.run([D_opt, D_loss], feed_dict={X: X_mb, T: T_mb, Y: Y_mb})            

    idx = np.random.permutation(len(data_test_real))
    train_idx = idx[:len(data_test_real)] 
    
    X_mb = list(data_test_real[i][:-1,:(data_dim-1)] for i in train_idx)
    T_mb = list(seq_len-1 for i in train_idx)
    Y_mb = list(np.reshape(data_test_real[i][1:,(data_dim-1)], [len(data_test_real[i][1:,(data_dim-1)]),1]) for i in train_idx)

    pred_Y_curr = sess.run(Y_pred, feed_dict={X: X_mb, T: T_mb})
    
    # MAE
    MAE_Temp = 0
    for i in range(len(data_test_real)):
        MAE_Temp = MAE_Temp + mean_absolute_error(Y_mb[i], pred_Y_curr[i,:,:])
    
    MAE_trtr = MAE_Temp / len(data_test_real)
    
    return MAE_trtr
    