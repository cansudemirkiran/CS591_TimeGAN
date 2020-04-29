import numpy as np
import random
import pandas as pd 

def normalize(x):

    min_data = np.min(x,0)
    max_data = np.max(x,0)
    diff = max_data-min_data+1e-7
    normalized_data = (x-min_data)/diff
    return normalized_data

def load_google(seq_len):

	data = np.loadtxt('datasets/GOOGLE_BIG.csv', delimiter = ",", skiprows = 1)
	#flipping data, it goes backwards
	data = data[::-1] 
	data = normalize(data)
	length = len(data)
	sequences = []
	for i in range (0, length - seq_len):
		temp = data[i:i+seq_len]
		sequences.append(temp)
	random.shuffle(sequences)
	return sequences


def load_energy(seq_len):

    data = pd.read_csv('datasets/energydata_complete.csv', delimiter = ",", skiprows = 1, usecols = range(1,29))
    data = data.to_numpy()
    data = normalize(data)
    length = len(data)
    sequences = []
    for i in range (0, length - seq_len):
        temp = data[i:i+seq_len]
        sequences.append(temp)
    random.shuffle(sequences)
    return sequences

def generate_sine_data(num_of_seq,seq_len, num_of_subseq ):
    
    sequences = []
    k = np.arange(0,seq_len)
    for i in range(num_of_seq):    
        temp = []
        for j in range(num_of_subseq):
            freq = np.random.uniform(0,0.1)     
            phase = np.random.uniform(0,0.1)
            temp_sub = np.sin(freq*k+phase)
            temp.append(temp_sub)
        temp = np.transpose(np.asarray(temp))
        temp = normalize(temp)
        sequences.append(temp)
    return sequences
