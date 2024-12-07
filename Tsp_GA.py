import random
import numpy as np
from geopy.geocoders import Nominatim
from math import sqrt
import matplotlib.pyplot as plt

# Function to read cities from file and get coordinates
def readCities(PNames):
    P = []
    geolocator = Nominatim(user_agent="gplApp")
    with open("India_cities.txt", "r") as file:
        for j, line in enumerate(file):
            city = line.rstrip('\n')
            if city == "":
                continue
            theLocation = city + ", India"
            pt = geolocator.geocode(theLocation, timeout=10)
            if pt:
                y = round(pt.latitude, 2)
                x = round(pt.longitude, 2)
                print(f"City[{j}] = {city} ({x}, {y})")
                P.append([x, y])
                PNames.append(city)
    return P

# Fitness class definition
class Fitness:
    def __init__(self, route):
        self.route = route
        self.distance = 0
        self.fitness = 0.0

    def routeDistance(self):
        if self.distance == 0:
            pathDistance = 0
            for i in range(len(self.route)):
                fromCity = self.route[i]
                toCity = None
                if i + 1 < len(self.route):
                    toCity = self.route[i + 1]
                else:
                    toCity = self.route[0]
                pathDistance += sqrt((fromCity[0] - toCity[0])**2 + (fromCity[1] - toCity[1])**2)
            self.distance = pathDistance
        return self.distance

    def routeFitness(self):
        if self.fitness == 0:
            self.fitness = 1 / float(self.routeDistance())
        return self.fitness

# Generate population
def createRoute(cityList):
    return random.sample(cityList, len(cityList))

def initialPopulation(popSize, cityList):
    population = []
    for _ in range(popSize):
        population.append(createRoute(cityList))
    return population

# Rank routes by fitness
def rankRoutes(population):
    fitnessResults = {}
    for i in range(len(population)):
        fitnessResults[i] = Fitness(population[i]).routeFitness()
    return sorted(fitnessResults.items(), key=lambda x: x[1], reverse=True)

# Selection of parents (top 8.5%)
def selection(popRanked, eliteSize):
    selectionResults = []
    df = sum([individual[1] for individual in popRanked])
    for i in range(eliteSize):
        selectionResults.append(popRanked[i][0])
    for _ in range(len(popRanked) - eliteSize):
        pick = random.uniform(0, df)
        current = 0
        for i in range(len(popRanked)):
            current += popRanked[i][1]
            if current > pick:
                selectionResults.append(popRanked[i][0])
                break
    return selectionResults

# Mating function
def breed(parent1, parent2):
    child = []
    geneA = int(random.random() * len(parent1))
    geneB = int(random.random() * len(parent1))
    startGene = min(geneA, geneB)
    endGene = max(geneA, geneB)
    for i in range(startGene, endGene):
        child.append(parent1[i])
    child += [gene for gene in parent2 if gene not in child]
    return child

def breedPopulation(matingPool, eliteSize):
    children = []
    length = len(matingPool) - eliteSize
    pool = random.sample(matingPool, len(matingPool))
    for i in range(0, eliteSize):
        children.append(matingPool[i])
    for i in range(0, length):
        child = breed(pool[i], pool[len(matingPool) - i - 1])
        children.append(child)
    return children

# Mutation function
def mutate(individual, mutationRate):
    for swapped in range(len(individual)):
        if random.random() < mutationRate:
            swapWith = int(random.random() * len(individual))
            city1 = individual[swapped]
            city2 = individual[swapWith]
            individual[swapped] = city2
            individual[swapWith] = city1
    return individual

def mutatePopulation(population, mutationRate):
    mutatedPop = []
    for ind in range(0, len(population)):
        mutatedInd = mutate(population[ind], mutationRate)
        mutatedPop.append(mutatedInd)
    return mutatedPop

# Main GA loop with fitness tracking
def nextGeneration(currentGen, eliteSize, mutationRate):
    popRanked = rankRoutes(currentGen)
    selectionResults = selection(popRanked, eliteSize)
    matingPool = [currentGen[i] for i in selectionResults]
    children = breedPopulation(matingPool, eliteSize)
    nextGen = mutatePopulation(children, mutationRate)
    return nextGen

# average fitness
def geneticAlgorithm(cityList, popSize, eliteSize, mutationRate, generations):
    pop = initialPopulation(popSize, cityList)
    progress = []  # To track average fitness
    
    # Calculate average fitness
    avgFitness = np.mean([Fitness(route).routeFitness() for route in pop])
    progress.append(avgFitness)
    
    print("Initial distance: " + str(1 / rankRoutes(pop)[0][1]))
    
   
    for i in range(0, generations):
        pop = nextGeneration(pop, eliteSize, mutationRate)
        avgFitness = np.mean([Fitness(route).routeFitness() for route in pop])
        progress.append(avgFitness)
    
    print("Final distance: " + str(1 / rankRoutes(pop)[0][1]))
    bestRouteIndex = rankRoutes(pop)[0][0]
    bestRoute = pop[bestRouteIndex]
    return bestRoute, progress

# plots
def plotRoute(cityList, bestRoute):
    bestRoute = np.array(bestRoute)
    plt.figure(figsize=(10, 6))
    plt.scatter(bestRoute[:, 0], bestRoute[:, 1], c='red', marker='o')  # Plot cities
    for i, txt in enumerate(PNames):
        plt.annotate(txt, (bestRoute[i][0], bestRoute[i][1]), fontsize=10)
    plt.plot(np.append(bestRoute[:, 0], bestRoute[0, 0]), np.append(bestRoute[:, 1], bestRoute[0, 1]), 'b-')  # Plot path
    plt.xlabel("Latitude")
    plt.ylabel("Longitude")
    plt.title("Best Route Found by Genetic Algorithm")
    plt.savefig('Best_Route_by_Genetic_Algorithm')
    plt.show()

# Plotting the progress (average fitness)
def plotFitness(progress):
    plt.plot(progress)
    plt.ylabel('Average Fitness')
    plt.xlabel('Generation')
    plt.title('Average Fitness Over Generations')
    plt.show()

# call function
PNames = []
cities = readCities(PNames)
bestRoute, progress = geneticAlgorithm(cities, popSize=100, eliteSize=8, mutationRate=0.01, generations=500)
print("Best route: ", bestRoute)

# Plot the best route
plotRoute(cities, bestRoute)

# Plot the increase in average fitness
plotFitness(progress)
