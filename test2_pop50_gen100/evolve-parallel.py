import numpy as np
import pickle
import os
import neat
import sys

xvec = []
svec = []
num_eval = 10

def eval_genome(genome, config):
    # create the neural network from the genotype
    net = neat.nn.RecurrentNetwork.create(genome, config)
    # test cases stored as global variables
    global xvec
    global svec
    # number of test cases per individual per generation
    global num_eval

    error = 0.0
    for i in range(num_eval):
        # draws a random data point for evaluation
        i = np.random.randint(0,len(xvec))
        x = xvec[i]
        s = svec[i]
        output = net.activate(x)

        # error is SSE between target and output
        error += np.sum(np.power(np.subtract(output,s),2))

    error = error/num_eval
    # print ("error: ", error)
    # print ("-- error per entry: ", np.sqrt(error/len(svec[0])))

    # return fitness = inverse error,
    # because neat module seems to be stuck on finess maximization
    return 1/error

def run(num_generations):
    # Determine path to configuration file.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward')
    print ("Loading config...")
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    # save the config settings used for the run
    config.save('run_config.txt')

    # create population using config
    print("Creating population...")
    pop = neat.Population(config)

    # initialize stat tracking
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.StdOutReporter(False))

    # run!
    print("Running for %i generations..." % num_generations)
    pe = neat.ParallelEvaluator(4,eval_genome)
    winner = pop.run(pe.evaluate, num_generations)

    # Log statistics.
    stats.save()

    # pickle the best individual
    winner_net = neat.nn.RecurrentNetwork.create(winner, config)
    pickle.dump(winner_net,open("winner_net.p","wb"))

def main():
    if len(sys.argv) != 3:
        print ("Wrong number of arguments. Usage: evolve-parallel.py <NUM_GENERATIONS> <NUM_EVAL>")
        return
    num_generations = int(sys.argv[1])
    global num_eval
    num_eval = int(sys.argv[2])

    global xvec
    xvec = pickle.load(open("xvec.p", "rb"), encoding='latin1')
    global svec
    svec = pickle.load(open("svec.p", "rb"), encoding='latin1')
    run(num_generations)

if __name__ == '__main__':
	main()
