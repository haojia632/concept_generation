#!/usr/bin/env python
# Author Jia Hao from Beijing Institute of Technology
# encoding: utf-8
import sys
import copy
import pickle
import LoadPatentSpace as lps
import LoadFunctionSpace as lfs
import scipy.spatial.distance as dis
import scipy.stats as stats
import numpy as np
import random
import NoveltyExperiments as novelty
import FeasibilityExperiments as feasibility
import matplotlib.pyplot as plt
import os.path
from deap import algorithms,base,creator,tools
from astropy.units import fearthMass
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
# Dimention of individual
NDIM = 100
# Number of each population
NPOP = 40
# Generation Size
NGEN = 80
# Crossover rate
CXPB = 0.5
# Mutation rate
MUTPB = 0.5
# The bound of attribute
BOUND_LOW, BOUNT_UP = -1.0,1.0
# load the function space and patent space
function_space = lfs.load_function_space()
all_function_terms = function_space[0]
all_function_vectors = function_space[1]
existed_functions = ['blow', 'rotate', 'swing'] # Fan
# existed_functions = ['hold', 'contain', 'filter'] # Cup
# existed_functions = ['hold', 'trim', 'rotate'] # Shavor
def uniform(low,up,size=None):
    """
        Generate individual property
    """
    try:
        return [random.uniform(a,b) for a, b in zip(low,up)]
    except TypeError:
        return [random.uniform(a,b) for a,b in zip([low]*size,[up]*size)]

toolbox = base.Toolbox()
creator.create("FintnessMulti", base.Fitness,weights=(1.0,1.0))
creator.create("Individual",list,fitness=creator.FintnessMulti,term="default")
pareto_front = tools.ParetoFront()
# Define population
toolbox.register("attr_float",uniform, BOUND_LOW, BOUNT_UP,NDIM)
toolbox.register("individual",tools.initIterate,creator.Individual,toolbox.attr_float)
toolbox.register("population",tools.initRepeat,list,toolbox.individual)

existed_function_vector = [all_function_vectors[all_function_terms.index(term),:] for term in existed_functions ]



def termPrediction():
    all_term = all_function_terms
    for et in existed_functions:
        if et in all_term:
            all_term.remove(et)
    pop = toolbox.population(n=len(all_term))

    for i in range(len(all_term)):
        term = all_term[i]
        term_vector = all_function_vectors[all_function_terms.index(term)]
        pop[i].fitness.values = evaluation(term_vector)
        pop[i].term = term
        print(pop[i].term + " " + str(pop[i].fitness.values))

    feasibility = [ind.fitness.values[1] for ind in pop]
    novalty = [ind.fitness.values[0] for ind in pop]
    
    data_file = file("term_prediction2.pkl",'wb')
    pickle.dump(feasibility,data_file)
    pickle.dump(novalty,data_file)
    pickle.dump(all_term,data_file)
    pickle.dump(pop,data_file)

file_name = "term_prediction2.pkl"
file_name_r = "test_a.txt"
def run2():
    if not os.path.isfile(file_name):
        termPrediction()
    
    # Load data from stored file
    data_file = file(file_name,"rb")
    feasibility = pickle.load(data_file)
    novalty = pickle.load(data_file)
    all_term = pickle.load(data_file)
    pop = pickle.load(data_file)
    
    # paretofront sorting of population
    individuals = tools.sortNondominated(pop, len(pop))
    
    total = 30
    # get top 50 terms based on paretofront sorting
    top_pareto = []
    pareto_index = 0
    while(len(top_pareto)<total):
        if(total-len(top_pareto)>len(individuals[pareto_index])):
            top_pareto = top_pareto + individuals[pareto_index]
        else:
            top_pareto = top_pareto + individuals[pareto_index][0:(total-len(top_pareto))]
        pareto_index += 1
    
    # get top 50 terms based on novalty
    top_novalty = tools.selBest(pop, total)
    
    # get top 50 terms based on feasibility
    for ind in pop:
        ind.fitness.values = [ind.fitness.values[1],ind.fitness.values[0]]
    top_feasibility = tools.selBest(pop, total)
    for ind in pop:
        ind.fitness.values = [ind.fitness.values[1],ind.fitness.values[0]]
    
    # Randomly select "total" terms
    
    rd = random.sample(range(len(pop)),total)
    top_random = [pop[i] for i in rd]
#     pareto_test = []
#     novalty_test = []
#     feasibility_test =[]
#     random_test = []
#     
#     for i in range(6):
#         rd = random.sample(range(50), 10)
#         pareto_test.append([top_pareto[r] for r in rd])
#         
#         rd = random.sample(range(50), 10)
#         novalty_test.append([top_novalty[r] for r in rd])     
#         
#         rd = random.sample(range(50), 10)
#         feasibility_test.append([top_feasibility[r] for r in rd])
#         
#         rd = random.sample(range(len(pop)), 10) 
#         random_test.append([pop[r] for r in rd])
        
    f = open(file_name_r,'w')
    f.write( str(existed_functions) + "\n")
    f.write("###########################\n")
    f.write("Pareto Front Term\n")
    f.write("###########################\n")
    count = 0
    for ind in top_pareto:
        f.write(ind.term + "    ")
        count += 1
        if count%10 == 0:
            f.write("\n")
    
    f.write("###########################\n")
    f.write("Top Novalty Term\n")
    f.write("###########################\n")
    count = 0
    for ind in top_novalty:
        f.write(ind.term + "    ")
        count += 1
        if count%10 == 0:
            f.write("\n")
    f.write("###########################\n")
    f.write("Top Feasibility Term\n")
    f.write("###########################\n")
    count = 0
    for ind in top_feasibility:
        f.write(ind.term + "    ")
        count += 1
        if count%10 == 0:
            f.write("\n")
        
    f.write("###########################\n")
    f.write("Random Selected Term\n")
    f.write("###########################\n")
    for ind in top_random:
        f.write(ind.term + "    ")
        count += 1
        if count%10 == 0:
            f.write("\n")
    
def run():
    if not os.path.isfile("term_prediction2.pkl"):
        termPrediction()
        
    data_file = file("term_prediction2.pkl","rb")
    feasibility = pickle.load(data_file)
    novalty = pickle.load(data_file)
    all_term = pickle.load(data_file)
    pop = pickle.load(data_file)
    
    pareto_front.update(pop)
    print("Pareto Front is as follow"+ " "+ str(len(pareto_front)))
    print("#######################################################")
    print("#######################################################")
    print("#######################################################")
    for ind in pareto_front:
        print(ind.term + " " + str(ind.fitness.values))
    print("#######################################################")
    print("#######################################################")
    print("#######################################################")
    front = tools.selBest(pop, len(pop))
    f = open("result.txt",'w')
    count = 1
    for ind in front:
        f.write(str(count) + "  : " +ind.term + " " + str(ind.fitness.values) + "\n")
        count += 1
    print("Pareto Front")
    print("#######################################################")
    print("#######################################################")
    print("#######################################################")      
    individuals = tools.sortNondominated(pop, len(pop))
    print(len(individuals[0]+individuals[1]+individuals[3]))
    for ind in individuals[0]+individuals[1]+individuals[3]:
        print(ind.term + " " + str(ind.fitness.values))
    
     
    plt.subplot(1,3,1)
    plt.hist(feasibility,100)
    
    plt.subplot(1,3,2)
    plt.hist(novalty,100)
    
    plt.subplot(1,3,3)
    plt.ylim([0,0.01])
    plt.scatter(feasibility, novalty)
   
    kde = stats.gaussian_kde(feasibility)
    xmin, xmax = np.min(feasibility),np.max(feasibility)
    x = np.linspace(xmin, xmax, 1000) 

    plt.plot(x,kde(x),'r')
    
    print(kde.covariance_factor())
    print(np.mean(feasibility))
    plt.show()
    threshold_up = np.mean(feasibility) + 0.7*kde.covariance_factor()
    threshold_down = np.mean(feasibility) + 0.5*kde.covariance_factor()

    suggest_term = [all_term[feasibility.index(fv)] for fv in feasibility if fv > threshold_down]
    print(len(suggest_term))
    for term in suggest_term:
        print(term)
        
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    kde = stats.gaussian_kde([novalty,feasibility])
    xmin, xmax = np.min(novalty),np.max(novalty)    
    ymin, ymax = np.min(feasibility),np.max(feasibility)  
    x = np.linspace(xmin, xmax, 1000) 
    y = np.linspace(ymin, ymax, 1000) 
    x, y = np.meshgrid(x, y)
    print(kde.evaluate([x,y]))
    surf = ax.scatter(x, y, kde.evaluate([x,y]))
    plt.show()
        
def evaluation(individual):
    """
        This funciton is used to evaluate each individual
    """
    # Get the closest term through individual vector
    individual_term = individual
    # make circular convolution the generation virtual patent
    vectors4novelty = np.concatenate((existed_function_vector,np.array([individual_term])))
    patent_vector = lps.get_patent_vector_a(vectors4novelty)
    # Calculate Novelty Value based on patent space
    # We use the base space to test the algorithm temperorely
    novelty_value = novelty.get_novelty(patent_vector)
    # Calcualte Fesibility value based on function space
    feasibility_value = feasibility.get_feasibility_a(vectors4novelty)
    return [novelty_value,feasibility_value]


if __name__ == "__main__":

    run2()

