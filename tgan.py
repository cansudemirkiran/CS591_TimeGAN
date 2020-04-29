#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: cansu
Created on Thu Apr 16 15:52:50 2020

"""

import tensorflow as tf
import numpy as np
def rnn_cell(params):
    module_name = params['module_name']
  # GRU
    if (module_name == 'gru'):
        rnn_cell = tf.nn.rnn_cell.GRUCell(num_units=params['hidden_dim'], activation=tf.nn.tanh)
  # LSTM
    elif (module_name == 'lstm'):
        rnn_cell = tf.contrib.rnn.BasicLSTMCell(num_units=params['hidden_dim'], activation=tf.nn.tanh)
    return rnn_cell    
    
# Mapping Components
    
def embedder (X, T, params):      
      
    with tf.variable_scope("embedder", reuse = tf.AUTO_REUSE):
        
        e_cell = tf.nn.rnn_cell.MultiRNNCell([rnn_cell(params) for _ in range(params['num_layers'])])
            
        e_outputs, e_last_states = tf.nn.dynamic_rnn(e_cell, X, dtype=tf.float32, sequence_length = T)
        
        H = tf.contrib.layers.fully_connected(e_outputs, params['hidden_dim'], activation_fn=tf.nn.sigmoid)     

    return H
      
def recovery (H, T,params, data_dim):      
  
    with tf.variable_scope("recovery", reuse = tf.AUTO_REUSE):       
          
        r_cell = tf.nn.rnn_cell.MultiRNNCell([rnn_cell(params) for _ in range(params['num_layers'])])
            
        r_outputs, r_last_states = tf.nn.dynamic_rnn(r_cell, H, dtype=tf.float32, sequence_length = T)
        
        X_tilde = tf.contrib.layers.fully_connected(r_outputs, data_dim, activation_fn=tf.nn.sigmoid) 

    return X_tilde

# Adversarial Components
def generator (Z, T, params):      
      
    with tf.variable_scope("generator", reuse = tf.AUTO_REUSE):
        
        e_cell = tf.nn.rnn_cell.MultiRNNCell([rnn_cell(params) for _ in range(params['num_layers'])])
            
        e_outputs, e_last_states = tf.nn.dynamic_rnn(e_cell, Z, dtype=tf.float32, sequence_length = T)
        
        E = tf.contrib.layers.fully_connected(e_outputs, params['hidden_dim'], activation_fn=tf.nn.sigmoid)     

    return E
      
    
def discriminator (H, T, params):
  
    with tf.variable_scope("discriminator", reuse = tf.AUTO_REUSE):
        
        d_cell = tf.nn.rnn_cell.MultiRNNCell([rnn_cell(params) for _ in range(params['num_layers'])])
            
        d_outputs, d_last_states = tf.nn.dynamic_rnn(d_cell, H, dtype=tf.float32, sequence_length = T)
        
        Y_hat = tf.contrib.layers.fully_connected(d_outputs, 1, activation_fn=None) 

    return Y_hat   

# Supervisor Network
def supervisor (H, T, params):      
  
    with tf.variable_scope("supervisor", reuse = tf.AUTO_REUSE):
        
        e_cell = tf.nn.rnn_cell.MultiRNNCell([rnn_cell(params) for _ in range(params['num_layers']-1)])
            
        e_outputs, e_last_states = tf.nn.dynamic_rnn(e_cell, H, dtype=tf.float32, sequence_length = T)
        
        S = tf.contrib.layers.fully_connected(e_outputs, params['hidden_dim'], activation_fn=tf.nn.sigmoid)     

    return S

def random_generator (batch_size, z_dim, T_mb, Max_Seq_Len):
  
    Z_mb = list()
    
    for i in range(batch_size):
        
        Temp = np.zeros([Max_Seq_Len, z_dim])
        
        Temp_Z = np.random.uniform(0., 1, [T_mb[i], z_dim])
    
        Temp[:T_mb[i],:] = Temp_Z
        
        Z_mb.append(Temp_Z)
  
    return Z_mb
def get_losses(Y_real, Y_fake, Y_fake_sup, H, H_sup, X_tilde_fake_sup, X, X_tilde):
    
      #discriminator
      D_loss_real = tf.losses.sigmoid_cross_entropy(tf.ones_like(Y_real), Y_real) 
      D_loss_fake_sup= tf.losses.sigmoid_cross_entropy(tf.zeros_like(Y_fake_sup), Y_fake_sup)
      D_loss_fake = tf.losses.sigmoid_cross_entropy(tf.zeros_like(Y_fake), Y_fake)
      #discriminator overall
      D_loss = D_loss_real + D_loss_fake_sup + D_loss_fake

      #supervisor
      S_loss = tf.losses.mean_squared_error(H[:,1:,:], H_sup[:,1:,:])
      
      #generator adversarial
      G_loss_adv_sup = tf.losses.sigmoid_cross_entropy(tf.ones_like(Y_fake_sup), Y_fake_sup)
      G_loss_adv = tf.losses.sigmoid_cross_entropy(tf.ones_like(Y_fake), Y_fake)
      #generator mean and variance difference L1 loss
      G_loss_V1 = tf.reduce_mean(np.abs(tf.sqrt(tf.nn.moments(X_tilde_fake_sup,[0])[1] + 1e-6) - tf.sqrt(tf.nn.moments(X,[0])[1] + 1e-6)))
      G_loss_V2 = tf.reduce_mean(np.abs((tf.nn.moments(X_tilde_fake_sup,[0])[0]) - (tf.nn.moments(X,[0])[0])))
      G_loss_V = G_loss_V1 + G_loss_V2
      #generator overall 
      G_loss = G_loss_adv + G_loss_adv_sup + 100 * tf.sqrt(S_loss) + 100*G_loss_V 
      
      #Embedder-Recovery
      E_loss_only = 10*tf.sqrt(tf.losses.mean_squared_error(X, X_tilde)) #mean squared err 
      E_loss = E_loss_only  + 0.1*S_loss #combined with supervisor error
      
      return D_loss, G_loss, S_loss, E_loss, E_loss_only, G_loss_adv_sup, G_loss_V
      

    
def timegan(data, params):

    tf.reset_default_graph()
    data_len = len(data)
    data_dim = len(data[0][0,:])
    max_seq_len = len(data[0][:,0])
    seq_len_list = []

    for i in range(data_len):
        seq_len = len(data[i][:,0])
        seq_len_list.append(seq_len)
        if max_seq_len < seq_len:
            max_seq_len = seq_len
    
    # Get Parameters
    hidden_dim   = params['hidden_dim'] 
    num_layers   = params['num_layers']
    iterations   = params['iterations']
    batch_size   = params['batch_size']
    module_name  = params['module_name']  
    z_dim        = params['z_dim']
    gamma        = 1
    
    # input place holders
    
    X = tf.placeholder(tf.float32, [None, max_seq_len, data_dim], name = "input_x")
    Z = tf.placeholder(tf.float32, [None, max_seq_len, params['z_dim']], name = "input_z")
    T = tf.placeholder(tf.int32, [None], name = "input_t")

    
    
    H = embedder(X, T, params) # input mapped into the embedding space
    X_tilde = recovery(H, T, params, data_dim) # recovered input with recovery function
    
    # Generator
    H_fake = generator(Z, T, params) # Generated data
    H_fake_sup = supervisor(H_fake, T, params) # Supervising the fake data
    H_sup = supervisor(H, T, params) # Supervising the real data
    
    # Synthetic data
    X_tilde_fake_sup = recovery(H_fake_sup, T, params, data_dim) # Fake data recovered
    
    # Discriminator
    Y_fake_sup = discriminator(H_fake_sup, T, params)  #Supervised fake data classification outputs
    Y_real = discriminator(H, T, params) # Real data classification outputs
    Y_fake = discriminator(H_fake, T, params) #Fake data classification outputs
    
    # Variables        
    emb_vars = [v for v in tf.trainable_variables() if v.name.startswith('embedder')]
    rec_vars = [v for v in tf.trainable_variables() if v.name.startswith('recovery')]
    gen_vars = [v for v in tf.trainable_variables() if v.name.startswith('generator')]
    sup_vars = [v for v in tf.trainable_variables() if v.name.startswith('supervisor')]
    dis_vars = [v for v in tf.trainable_variables() if v.name.startswith('discriminator')]
    
    # LOSSES
   
    D_loss, G_loss, S_loss, E_loss, E_loss_only, G_loss_adv_sup, G_loss_V = get_losses(Y_real, Y_fake, Y_fake_sup, H, H_sup, X_tilde_fake_sup, X, X_tilde)
    
     
    # optimizer
    E0_opt = tf.train.AdamOptimizer().minimize(E_loss_only, var_list = emb_vars + rec_vars)
    E_opt = tf.train.AdamOptimizer().minimize(E_loss, var_list = emb_vars + rec_vars)
    D_opt = tf.train.AdamOptimizer().minimize(D_loss, var_list = dis_vars)
    G_opt = tf.train.AdamOptimizer().minimize(G_loss, var_list = gen_vars + sup_vars)      
    S_opt = tf.train.AdamOptimizer().minimize(S_loss, var_list = gen_vars + sup_vars)   
    
    
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    e_loss_list = []
    s_loss_list = []
    d_loss_list = []
    g_loss_adv_list = []
    g_loss_v_list = []
    e_loss_joint_list = []
    s_loss_joint_list = []        
    #Embedder training
    
    for i in range(params['iterations']):
        
        # Batch setting
        index = np.random.permutation(data_len)
        train_index = index[:params['batch_size']]   
            
        X_mb = list(data[j] for j in train_index)
        T_mb = list(seq_len_list[j] for j in train_index)
            
        # Train embedder        
        _, step_e_loss = session.run([E0_opt, E_loss_only], feed_dict={X: X_mb, T: T_mb})
        
        if i % 10 == 0:
            print('step: '+ str(i) + ', e_loss: ' + str(np.round(np.sqrt(step_e_loss),4)) ) 
            e_loss_list.append(np.round(np.sqrt(step_e_loss),4))
    
    # Supervisor training
    
    for i in range(params['iterations']):
        
        # Batch setting
        index = np.random.permutation(data_len)
        train_index = index[:params['batch_size']]   
            
        X_mb = list(data[j] for j in train_index)
        T_mb = list(seq_len_list[j] for j in train_index)      
        
        Z_mb = random_generator(params['batch_size'], params['z_dim'], T_mb, max_seq_len)
        
        # Train generator       
        _, step_s_loss = session.run([S_opt, S_loss], feed_dict={Z: Z_mb, X: X_mb, T: T_mb})

                           
        if i % 10 == 0:
            print('step: '+ str(i) + ', s_loss: ' + str(np.round(np.sqrt(step_s_loss),4)))
            s_loss_list.append(np.round(np.sqrt(step_s_loss),4))
    #Joint Training
            
    for i in range(params['iterations']):
      
        # Generator Training
        for kk in range(2): #FIXME: Why 2 ?
          
            # Batch setting
            index = np.random.permutation(data_len)
            train_index = index[:params['batch_size']] 
            
            X_mb = list(data[j] for j in train_index)
            T_mb = list(seq_len_list[j] for j in train_index)     
            
            # Random vector generation
            Z_mb = random_generator(params['batch_size'], params['z_dim'], T_mb, max_seq_len)
            
            # Train generator
            _, step_g_loss_adv, step_s_loss, step_g_loss_v = session.run([G_opt, G_loss_adv_sup, S_loss, G_loss_V], feed_dict={Z: Z_mb, X: X_mb, T: T_mb})
            
            # Train embedder        
            _, step_e_loss_only = session.run([E_opt, E_loss_only], feed_dict={Z: Z_mb, X: X_mb, T: T_mb})   
           
        # Discriminator Training
        
        # Batch setting
        index = np.random.permutation(data_len)
        train_index = index[:params['batch_size']]  
        
        X_mb = list(data[j] for j in train_index)
        T_mb = list(seq_len_list[j] for j in train_index)  
        
        # Random vector generation
        Z_mb = random_generator(params['batch_size'], params['z_dim'], T_mb, max_seq_len)
            
        
        d_loss = session.run(D_loss, feed_dict={X: X_mb, T: T_mb, Z: Z_mb})
        
        # Train discriminator
        
        if (d_loss > 0.15):        
            _, step_d_loss = session.run([D_opt, D_loss], feed_dict={X: X_mb, T: T_mb, Z: Z_mb})
        
        # Checkpoints
        if i % 10 == 0:
            print('step: '+ str(i) + 
                  ', d_loss: ' + str(np.round(step_d_loss,4)) + 
                  ', g_loss_adv: ' + str(np.round(step_g_loss_adv,4)) + 
                  ', s_loss: ' + str(np.round(np.sqrt(step_s_loss),4)) + 
                  ', g_loss_v: ' + str(np.round(step_g_loss_v,4)) + 
                  ', e_loss_t0: ' + str(np.round(np.sqrt(step_e_loss_only),4)))
            d_loss_list.append(np.round(step_d_loss,4))
            g_loss_adv_list.append(np.round(step_g_loss_adv,4))
            g_loss_v_list.append(np.round(step_g_loss_v,4))
            s_loss_joint_list.append(np.round(np.sqrt(step_s_loss),4))
            e_loss_joint_list.append(np.round(np.sqrt(step_e_loss_only),4))
            
        
    Z_mb = random_generator(data_len, params['z_dim'], seq_len_list, max_seq_len)
    out = session.run(X_tilde_fake_sup, feed_dict={Z: Z_mb, X: data, T: seq_len_list}) 
    loss_list = [e_loss_list, s_loss_list, d_loss_list, g_loss_adv_list, g_loss_v_list, e_loss_joint_list, s_loss_joint_list ]
    output = []
    for i in range(data_len):
        temp_out = out[i,:seq_len_list[i],:]
        output.append(temp_out)   
        
    return output, loss_list
    
   
    
    
   
    
                
            
    
     
      
