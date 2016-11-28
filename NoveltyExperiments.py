# This script is to make the novelty test
import LoadPatentSpace as lps
import LoadFunctionSpace as lfs
import os.path
import scipy.spatial.distance as dis
import matplotlib.pyplot as plt

import numpy as np

# the number of test patents
n = 1000
# the total number patents
sn = 500000
# the interval for selection patent
interval = sn/n
# the number of closest points used
cn = 5
# The interations times for each experiment
iteration_number = 2
# the size of the base used to compute novelty
base_size = 5000
# the time scale
groups = 5
# the base space
base_space = []
# load the function space and patent space
function_space = lfs.load_function_space()
all_function_terms = function_space[0]
all_function_vectors = function_space[1]

patent_space = lps.load_patent_space()
# The total number of patent is changed, since some patent are zero (no function terms)
sn = len(patent_space)

base_patent_file = "base_patent_file"

# Select 2000 patent from the whole patent_space.
# Select 1 patent from every 250
def experiment1():
    multi_all_novelties = []

    # Iteration several times and record all results
    for t in np.arange(0,iteration_number):
        # Generate base space
        build_base_space()
        # Generate base space finished

    # Generate Test Patents
    test_indexs = np.random.randint(interval, size=(n,1))
    for i in np.arange(0,n):
        test_indexs[i] = test_indexs[i] + i*interval
    test_space = patent_space[test_indexs,:]
        # Generate Test Patents Finished!

    all_novelties = []
    count = 0

    for test_vector in test_space:
        #Compute the novelty
        all_novelties.append(get_novelty(test_vector))
        count = count + 1
        print( 'The ' + str(t) + ' iteration ' + str(count) + ' patent finished!')

    multi_all_novelties.append(all_novelties)

    # average all results
    mean_novelties = np.mean(np.array(multi_all_novelties),axis=0)

    group_novelties = mean_novelties.reshape((n/groups,groups),order='F')
    return np.mean(group_novelties,axis=0)


# Compute the novelty of real product
def experiment3():
    # Build the cup

    # variants = np.array([['hold','contain','filter'],
    #          ['hold', 'contain', 'carry'],
    #          ['hold', 'contain', 'seal'],
    #          ['hold', 'contain', 'time'],
    #          ['hold', 'contain', 'scale'],
    #          ['hold', 'contain', 'heat'],
    #          ['hold', 'contain', 'insulate'],
    #          ['hold', 'contain', 'stretch']])

    variants = np.array([['hold','trim','rotate'],
             ['hold', 'trim', 'indicate'],
             ['hold', 'trim', 'charge'],
             ['hold', 'trim', 'waterproof'],
             ['hold', 'trim', 'float'],
             ['hold', 'trim', 'slip'],
             ['hold', 'trim', 'clean'],
             ['hold', 'trim', 'lubricate'],
             ['hold', 'trim', 'shake'],
            ['hold', 'trim', 'automate']])
    # variants = np.array([['blow','rotate','adjust','oscillate'],
    #          ['blow', 'rotate', 'adjust','time'],
    #          ['blow', 'rotate', 'adjust','mute'],
    #          ['blow', 'rotate', 'adjust','handle'],
    #          ['blow', 'rotate', 'adjust','charge'],
    #          ['blow', 'rotate', 'adjust','mist'],
    #          ['blow', 'rotate', 'adjust','purge'],
    #          ['blow', 'rotate', 'adjust','illuminate'],
    #         ['blow', 'rotate', 'adjust','sterilize'],
    #         ['blow', 'rotate', 'adjust','oxygenate']])
    all_novelties = []
    for v in variants:
        novelty = get_novelty(lps.get_patent_vector(v))
        all_novelties.append(novelty)

    return all_novelties

def experiment4():
    existed_functions = ['blow', 'rotate']
    novelties = []
    for t in all_function_terms:
        novelties.append(get_novelty(lps.get_patent_vector(existed_functions + [t])))
        print(novelties)
    n, bins, patches = plt.hist(novelties, 40, normed=1, facecolor='green', alpha=0.75)
    plt.grid(True)
    plt.xlabel('novelty')
    plt.ylabel('number of values')
    plt.show()
    

#Compute the novelty base on base_space
def get_novelty(test_vector):
    top_dis = np.ones(cn)*2
    if not os.path.exists(base_patent_file + '.npy'):
        build_base_space()
    base_space = np.load(base_patent_file + '.npy')
    
    for v in base_space:
        # compute consine value
        dist = dis.cosine(test_vector,v)
        if dist is 0:
            continue
        if dist < np.max(top_dis):
            top_dis[np.argmax(top_dis)] = dist
    return np.mean(top_dis)


# Compute the novelty based on patent_space
def get_novelty_a(test_vector):
    top_dis = np.ones(cn) * 2
    for v in patent_space:
        # compute consine value
        dist = dis.cosine(test_vector, v)
        if dist is 0:
            continue
        if dist < np.max(top_dis):
            top_dis[np.argmax(top_dis)] = dist
    return np.mean(top_dis)

def build_base_space():
    # Generate base space
    base_indexs = np.random.randint(sn, size=(base_size, 1))
    base_space = patent_space[base_indexs, :]
    # Generate base space finished
    np.save(base_patent_file, base_space)

if __name__=="__main__":
    experiment4()