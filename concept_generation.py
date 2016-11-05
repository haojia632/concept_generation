#!/usr/bin/env python
# Author Jia Hao from Beijing Institute of Technology
# encoding: utf-8
import sys
import copy
import pickle
import LoadPatentSpace as lps
import LoadFunctionSpace as lfs
import scipy.spatial.distance as dis
import numpy as np
import random
import NoveltyExperiments as novelty
import FeasibilityExperiments as feasibility
import matplotlib.pyplot as plt
import os.path
from deap import algorithms,base,creator,tools
# Dimention of individual
NDIM = 20
# Number of each population
NPOP = 60
# Generation Size
NGEN = 60
# Crossover rate
CXPB = 0.3
# Mutation rate
MUTPB = 0.7

ETA = 1/NDIM
# load the function space and patent space
function_space = lfs.load_function_space()
all_function_terms = function_space[0]
all_function_vectors = function_space[1]
existed_functions = ['blow', 'rotate']
existed_functions_vector = np.zeros([len(existed_functions),NDIM])

# The bound of attribute
BOUND_LOW, BOUNT_UP = np.min(all_function_vectors),np.max(all_function_vectors)

def termPrediction():
    """
        This is  the algorithm used to predict the next function term
        based on NSGA-II
        :param functions: A list of existing functions, based on which
                          to predict the next function term
        :return result: A group of generated function terms
    """

    for i in range(len(existed_functions)):
        existed_functions_vector[i] = all_function_vectors[all_function_terms.index(existed_functions[i])]

    # Generate a group of function terms by nsga2
    result,logbook = nsga2()
    # Mapping the nsga2 result to real function terms
    result_vector = mapping2terms(result)

    if os.path.isfile("term_prediction.pkl"):
        os.remove("term_prediction.pkl")
    data_file = file("term_prediction.pkl",'wb')
    pickle.dump(result,data_file,True)
    pickle.dump(result_vector,data_file,True)
    pickle.dump(logbook,data_file,True)
    data_file.close()

    analysis_result()


def uniform(low,up,size=None):
    """
        Generate individual property
    """
    try:
        return [random.uniform(a,b) for a, b in zip(low,up)]
    except TypeError:
        return [random.uniform(a,b) for a,b in zip([low]*size,[up]*size)]

def evaluation(individual):
    """
        This funciton is used to evaluate each individual
        Other problem is that How to evaluate the population
    """
    # Get the closest term through individual vector
    #individual_term = mapping2terms([individual])
    individual_term = individual
    #print(existed_functions + individual_term);
    # make circular convolution the generation virtual patent
    # patent_vector = all_function_vectors[all_function_terms.index(individual_term[0])]
    vectors4novelty = np.concatenate((existed_functions_vector,np.array([individual_term])))
    patent_vector = lps.get_patent_vector_a(vectors4novelty)
    # Calculate Novelty Value based on patent space
    # We use the base space to test the algorithm temperorely
    novelty_value = novelty.get_novelty(patent_vector)
    # Calcualte Fesibility value based on function space
    feasibility_value = feasibility.get_feasibility_a(vectors4novelty)
    #print(str(existed_functions+individual_term) + " Novelty:"+str(novelty_value)+" Feasibility:"+str(feasibility_value))
    return [novelty_value,feasibility_value]

toolbox = base.Toolbox()
#creator.create("FintnessMulti", base.Fitness,weights=(1.0,1.0))
creator.create("FintnessMulti", base.Fitness,weights=(1.0,0.8))
creator.create("Individual",list,fitness=creator.FintnessMulti)
# Define population
toolbox.register("attr_float",uniform, BOUND_LOW, BOUNT_UP,NDIM)
toolbox.register("individual",tools.initIterate,creator.Individual,toolbox.attr_float)
toolbox.register("population",tools.initRepeat,list,toolbox.individual)
# Define operators
toolbox.register("evaluate",evaluation)
toolbox.register("mate",tools.cxSimulatedBinaryBounded,low=BOUND_LOW,up=BOUNT_UP,eta=ETA)
toolbox.register("mutate",tools.mutPolynomialBounded,low=BOUND_LOW,up=BOUNT_UP,eta=ETA,indpb=MUTPB)
toolbox.register("select",tools.selNSGA2)
# Define Statistics object
stats = tools.Statistics(lambda ind:ind.fitness.values)
stats.register("avg",np.mean, axis=0)
stats.register("std",np.std, axis=0)
stats.register("min",np.min, axis=0)
stats.register("max",np.max, axis=0)
# Define Log object
logbook = tools.Logbook()
pareto_front = tools.ParetoFront()


def nsga2():
    """
        NSGA-II method
    """
    logbook.header = "gen", "evals","std","min","avg","max"
    # Initialize the population
    pop = toolbox.population(n=NPOP)
    # Evaluate the individuals with an invalid fitness
    print("Start to evaluate the initial population")
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # This is just to assign the crowding distance to the individuals
    # no actual selection is done
    pop = toolbox.select(pop, len(pop))
    # calculate the Statistics
    record = stats.compile(pop)
    logbook.record(gen=0, evals=len(invalid_ind), **record)
    print(logbook.stream)

    pareto_front.update(pop)
    print(list(set(mapping2terms(pareto_front))))

    #print("Start the evolution process")
    # Begin the generational process
    for gen in range(1, NGEN):
        #print("The " + str(gen) + " Generation started")
        # Vary the population
        offspring = tools.selTournamentDCD(pop, len(pop))
        offspring = [toolbox.clone(ind) for ind in offspring]

        for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
            if random.random() <= CXPB:
                toolbox.mate(ind1, ind2)

            toolbox.mutate(ind1)
            toolbox.mutate(ind2)
            del ind1.fitness.values, ind2.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Select the next generation population
        pop = toolbox.select(pop + offspring, NPOP)
        record = stats.compile(pop)
        logbook.record(gen=gen, evals=len(invalid_ind), **record)
        print(logbook.stream)

        pareto_front.update(pop)
        print(list(set(mapping2terms(pareto_front))))
    return pop, logbook


def mapping2terms(functions):
    """
        maping the virtual function vector to the closest function terms
    """
    # Initialize the result and position
    # The result and position will keep the top 3 cloest terms
    result = 65536*np.ones([len(functions),20])
    position = np.zeros([len(functions),20],dtype=np.int)
    # Start to calculate the cloest term
    for i in range(len(functions)):
        for j in range(len(all_function_vectors)):
            temp_dis = dis.cosine(functions[i],all_function_vectors[j])
            index_max = np.argmax(result[i,:])
            if temp_dis < result[i,index_max]:
                result[i,index_max] = temp_dis
                position[i,index_max] = j
    term_result = []
    for p in range(len(position)):
        flag = True
        while(flag):
            index_min = np.argmin(result[p,:])
            temp_term = all_function_terms[position[p,index_min]]
            if temp_term in existed_functions:
                result[p,index_min] = 65536
            else:
                flag = False
                term_result = term_result + [temp_term]
    return term_result

def analysis_result():
    """
        This is used to analysis the data
    """
    data_file = file("term_prediction.pkl","rb")
    result = pickle.load(data_file)
    result_vector = pickle.load(data_file)
    logbook = pickle.load(data_file)

    pareto_front = tools.ParetoFront()
    pareto_front.update(result)
    
    individuals = tools.sortNondominated(result, len(result))
    
    
    print(list(set(mapping2terms(pareto_front))))

    front = np.array([ind.fitness.values for ind in result])
    plt.subplot(1,3,1)
    plt.scatter(front[:,0], front[:,1], c="b")
    plt.grid(True)

    gen = logbook.select("gen")
    avg = np.array(logbook.select("max"))

    plt.subplot(1,3,2)
    plt.plot(gen,avg[:,0])
    plt.grid(True)

    plt.subplot(1,3,3)
    plt.plot(gen,avg[:,1])
    plt.grid(True)

    plt.show()
if __name__ == "__main__":

    termPrediction()
#     analysis_result()

