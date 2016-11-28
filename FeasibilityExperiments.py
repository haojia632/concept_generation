# This script is to make the feasibility test
import LoadPatentSpace as lps
import LoadFunctionSpace as lfs

import scipy.spatial.distance as dis
import matplotlib.pyplot as plt

import numpy as np

# load the function space and patent space
function_space = lfs.load_function_space()
all_function_terms = function_space[0]
all_function_vectors = function_space[1]


# Compute the feasibility based a list of functions
def get_feasibility(functions):
    count = 0
    distance = 0
    for i in np.arange(0,len(functions)):
        for j in np.arange(i+1,len(functions)):
            v1_index = all_function_terms.index(functions[i])
            v1 = all_function_vectors[v1_index]
            v2_index = all_function_terms.index(functions[j])
            v2 = all_function_vectors[v2_index]
            distance = distance + abs(dis.cosine(v1,v2))
            count = count + 1

    return 1 - distance/count

# Compute the feasibility based a list of functions
def get_feasibility_a(functions_vectors):
    count = 0
    distance = 0
    for i in np.arange(0,len(functions_vectors)):
        for j in np.arange(i+1,len(functions_vectors)):
            v1 = functions_vectors[i]
            v2 = functions_vectors[j]
            distance = distance + abs(dis.cosine(v1,v2))
            count = count + 1

    return 1 - distance/count

def experiment5():

    feasibilities = []

    # variants = np.array([['hold','contain','filter'],
    #          ['hold', 'contain', 'carry'],
    #          ['hold', 'contain', 'seal'],
    #          ['hold', 'contain', 'time'],
    #          ['hold', 'contain', 'scale'],
    #          ['hold', 'contain', 'heat'],
    #          ['hold', 'contain', 'insulate'],
    #          ['hold', 'contain', 'stretch']])
    variants = np.array([['hold', 'trim', 'rotate'],
                         ['hold', 'trim', 'indicate'],
                         ['hold', 'trim', 'charge'],
                         ['hold', 'trim', 'waterproof'],
                         ['hold', 'trim', 'float'],
                         ['hold', 'trim', 'slip'],
                         ['hold', 'trim', 'clean'],
                         ['hold', 'trim', 'shake']])
    # variants = np.array([['blow','rotate','adjust','oscillate'],
    #          ['blow', 'rotate', 'adjust','time'],
    #          ['blow', 'rotate', 'adjust','mute'],
    #          ['blow', 'rotate', 'adjust','handle'],
    #          ['blow', 'rotate', 'adjust','charge'],
    #          ['blow', 'rotate', 'adjust','mist'],
    #          ['blow', 'rotate', 'adjust','purge'],
    #          ['blow', 'rotate', 'adjust','illuminate']])
    for v in variants:
        feasibilities.append(get_feasibility(v))

    return np.array(feasibilities)

def experiment4():
    existed_functions = ['blow', 'rotate']
    feasibilities = []
    for t in all_function_terms:
        feasibilities.append(get_feasibility(existed_functions + [t]))
        print(feasibilities)
    n, bins, patches = plt.hist(feasibilities, 40, normed=1, facecolor='green', alpha=0.75)
    plt.grid(True)
    plt.xlabel('feasibility')
    plt.ylabel('number of values')
    plt.show()


if __name__ == "__main__":
    experiment4()
#     result = 0
#     v = ['hold','contain','time']
#     print(get_feasibility(v))
#     
#     for _ in range(1000):
#         term_index = [np.random.randint(len(all_function_terms)) for _ in range(6)]
#         
#         patent = [all_function_terms[index] for index in term_index]
#         result = result + get_feasibility(patent)
#     
#     print(result)