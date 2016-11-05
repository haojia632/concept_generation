# This script loads the patent space, if the patent_space.npy exist, load the file.
# If the patent_space.npy does not exist, build the file from
# patent_basis_all_s.txt

import LoadFunctionSpace as lfs
from scipy.signal import fftconvolve
import os.path
import numpy as np
import util

origin_file = 'patent_basis_all_s.txt'
data_file = 'patent_space'

all_function_terms = []
all_function_vectors = []

def get_function_vector(function):
    global all_function_terms
    global all_function_vectors
    if len(all_function_terms) is 0:
        results = lfs.load_function_space()
        all_function_terms = results[0]
        all_function_vectors = results[1]
    function_index = all_function_terms.index(function)
    return all_function_vectors[function_index]

def create_patent_space():
    count = 0
    all_patent_vectors = []
    for line in open(origin_file):
        temp_patent_vector = []
        functions = line.split(' ')
        # delete the '\n'
        del(functions[len(functions)-1])
        if len(functions) is 0:
            continue
        else:
            temp_patent_vector = get_patent_vector(functions)
            all_patent_vectors.append(temp_patent_vector)
            count = count + 1
        
        if count % 500 is 0:
            print ('500 line processed' + 'Total is ' + str(count))

    return np.array(all_patent_vectors)

def load_patent_space():
    if os.path.exists(data_file + '.npy'):
        print('Patent_Space : start to load from file')
        all_patent_vectors = np.load(data_file + '.npy')
    else:
        print('Patent_Space : start to create from file')
        all_patent_vectors = create_patent_space()
        np.save(data_file, all_patent_vectors)
    return all_patent_vectors

def get_patent_vector(functions):
    if len(functions) is 0:
        temp_patent_vector = np.zeros(100)
    else:
        # If there are 1 or more functions, initial the vector
        temp_patent_vector = get_function_vector(functions[0])
        #  Iterate the whole lists
        for i in np.arange(1, len(functions)):
            temp_patent_vector = fftconvolve(temp_patent_vector, get_function_vector(functions[i]), mode='same')
    return temp_patent_vector

def get_patent_vector_a(functions_vectors):

    if len(functions_vectors) is 0:
        temp_patent_vector = np.zeros(100)
    else:
        # If there are 1 or more functions, initial the vector
        temp_patent_vector = functions_vectors[0]
        #  Iterate the whole lists
        for i in np.arange(1, len(functions_vectors)):
            temp_patent_vector = fftconvolve(temp_patent_vector, functions_vectors[i], mode='same')
    return temp_patent_vector


if __name__ == "__main__":
    ps = load_patent_space()
    
    print(ps.shape)
    
    print(np.max(ps))
    
    print(np.min(ps))
    
    print(len(np.sum(ps,axis=1)))
    
    print(np.sum(ps,axis=1))
    
    if 0 in np.sum(ps,axis=1):
        print("0")
    else:
        print("1")