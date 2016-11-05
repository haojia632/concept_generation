# This script loads the function space, if the function_space.bin exist, load the file.
# If the function_space.bin does not exist, build the file from
# vectorModelVectors_all_d100_w5_i1.txt 
import os.path
import numpy as np

data_file = 'function_space'
# orgin_file = 'vectorModelVectors_all_d100_w5_i1.txt' 
orgin_file = 'vectorModelVectors_all_d20_w2_i1.txt'

def create_function_space():
    all_function_terms = [] 
    all_function_vectors = []
    count = 0
    for line in open(orgin_file):
        function_with_vector = line.split(' ')
        all_function_terms.append(function_with_vector[0])
        all_function_vectors.append(np.array(function_with_vector[1:], dtype = np.float))
        count = count + 1
#         print(count)

    all_function_vectors = np.array(all_function_vectors)
    results = np.array([all_function_terms, all_function_vectors,[]])
    return results 


def load_function_space():
    print('load_function_space started')
    results = []
    if os.path.exists(data_file + '.npy'):
        print("FunctionSpace : Load data from ")
        results = np.load(data_file + '.npy')
    else:
        print('FunctionSpace : Create data from')
        results = create_function_space()
        np.save(data_file,results)
    return results

if __name__ == "__main__":
    # Load the function Space
    fs = load_function_space()
    ft = fs[0]
    fv = fs[1]
    
    print(np.min(fv))
    
    print(np.max(fv))
    
    term_index = [np.random.randint(len(ft)) for _ in range(10)]
    
    print([ft[index] for index in term_index])
    
    
    
    
    
    
    