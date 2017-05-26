import numpy as np
import pickle
import os
import neat

xvec = []
svec = []

def eval_genome(genome, config):
    net = neat.nn.RecurrentNetwork.create(genome, config)
    global xvec
    global svec
    num_eval=10
    error = 0.0
    for i in range(num_eval):
        # draws a random data point for evaluation
        i = np.random.randint(0,len(xvec))
        x = xvec[i]
        s = svec[i]
        output = net.activate(x)

        # print("s: ",max(s)," | ",min(s))
        # print("sp: ",max(output)," | ",min(output))
        error += np.sum(np.power(np.subtract(output,s),2))

    error = error/num_eval
    print ("error: ", error)
    print ("-- error per entry: ", np.sqrt(error/257))

    # return fitness = inverse error,
    # because neat module seems to be stuck on finess maximization
    return 1/error


def eval_genomes(genomes, config):
    fitnesses = []
    for genome_id, genome in genomes:
        genome.fitness = eval_genome(genome, config)
        fitnesses.append(genome.fitness)
    f = open("results.txt","a")
    f.write("avg, " + str(np.mean(fitnesses)) + ", best, " + str(max(fitnesses)) + "\n")
    f.close()


def run():
    # Determine path to configuration file.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward')
    print ("Loading config...")
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    # Demonstration of saving a configuration back to a text file.
    config.save('test_save_config.txt')

    print("Creating population...")
    pop = neat.Population(config)
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.StdOutReporter(True))

    num_generations=20
    print("Running for %i generations..." % num_generations)
    winner = pop.run(eval_genomes, num_generations)

    # Log statistics.
    stats.save()

    #print('\nBest genome:\n{!s}'.format(winner))
    winner_net = neat.nn.RecurrentNetwork.create(winner, config)
    pickle.dump(winner_net,open("winner_net.p","wb"))

def main():
	global xvec
	xvec = pickle.load(open("xvec.p", "rb"), encoding='latin1')
	global svec
	svec = pickle.load(open("svec.p", "rb"), encoding='latin1')
	run()

if __name__ == '__main__':
	main()
