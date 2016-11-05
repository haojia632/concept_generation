import LoadPatentSpace as lps
import numpy as np
import scipy.spatial.distance as dis
patent_space = lps.load_patent_space()


def experiment6():
    group1 = np.array([['hold', 'contain', 'carry'],
                       ['hold', 'contain', 'handle'],
                       ['hold', 'contain', 'scale'],
                       ['hold', 'contain', 'measure'],
                       ['hold', 'contain', 'insulate'],
                       ['hold', 'contain', 'protect']])

    group2 = np.array([['hold', 'contain', 'carry'],
                       ['hold', 'contain', 'filter'],
                       ['hold', 'contain', 'scale'],
                       ['hold', 'contain', 'heat'],
                       ['hold', 'contain', 'stretch'],
                       ['hold', 'contain', 'insulate']])

    group3 = np.array([['hold', 'trim', 'indicate'],
                       ['hold', 'trim', 'show'],
                       ['hold', 'trim', 'shake'],
                       ['hold', 'trim', 'vibrate'],
                       ['hold', 'trim', 'waterproof'],
                       ['hold', 'trim', 'wash']])

    group4 = np.array([['hold', 'trim', 'clean'],
                       ['hold', 'trim', 'indicate'],
                       ['hold', 'trim', 'shake'],
                       ['hold', 'trim', 'charge'],
                       ['hold', 'trim', 'slip'],
                       ['hold', 'trim', 'float']])


    group5 = np.array([['blow', 'rotate', 'adjust','oscillate'],
                       ['blow', 'rotate', 'adjust','swing'],
                       ['blow', 'rotate', 'adjust','handle'],
                       ['blow', 'rotate', 'adjust','carry'],
                       ['blow', 'rotate', 'adjust','purge'],
                       ['hold', 'rotate', 'adjust','clean']])

    group6 = np.array([['blow', 'rotate', 'adjust', 'oscillate'],
                       ['blow', 'rotate', 'adjust', 'mute'],
                       ['blow', 'rotate', 'adjust', 'handle'],
                       ['blow', 'rotate', 'adjust', 'charge'],
                       ['blow', 'rotate', 'adjust', 'purge'],
                       ['hold', 'rotate', 'adjust', 'time']])

    group = group6
    count = 0
    distance = 0

    for i in np.arange(0,len(group)):
        for j in np.arange(i+1,len(group)):
            distance = distance  + dis.cosine(lps.get_patent_vector(group[i]),lps.get_patent_vector(group[j]))
            count = count + 1

    return distance/count


print(experiment6())

# np.savetxt('all_terms.txt',np.array(lps.all_function_terms),delimiter=' ',fmt='%s')