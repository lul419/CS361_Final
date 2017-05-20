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
    error = 0.0
    for i in range(len(xvec)):
    	x = xvec[i]
    	s = svec[i]
    	output = net.activate(x)

    	error += np.sum(np.power(np.subtract(output,s),2))

    print ("error: ", error)

    return error/len(xvec)


def eval_genomes(genomes, config):
	print ("--------- evaluating ---------")
	for genome_id, genome in genomes:
		genome.fitness = eval_genome(genome, config)
	print()


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

    print("Running for 10 generations...")
    pe = neat.ParallelEvaluator(4, eval_genome)
    winner = pop.run(pe.evaluate, 10)
    

    # Log statistics.
    stats.save()

    # Show output of the most fit genome against a random input.
    print('\nBest genome:\n{!s}'.format(winner))
    # print('\nOutput:')
    # winner_net = neat.nn.RecurrentNetwork.create(winner, config)
    # num_correct = 0
    # for n in range(num_tests):
    #     print('\nRun {0} output:'.format(n))
    #     seq = [random.choice((0.0, 1.0)) for _ in range(N)]
    #     winner_net.reset()
    #     for s in seq:
    #         inputs = [s, 0.0]
    #         winner_net.activate(inputs)
    #         print('\tseq {0}'.format(inputs))

    #     correct = True
    #     for s in seq:
    #         output = winner_net.activate([0, 1])
    #         print("\texpected {0:1.5f} got {1:1.5f}".format(s, output[0]))
    #         correct = correct and round(output[0]) == s
    #     print("OK" if correct else "FAIL")
    #     num_correct += 1 if correct else 0

    # print("{0} of {1} correct {2:.2f}%".format(num_correct, num_tests, num_correct/num_tests))

    # node_names = {-1: 'input', -2: 'gate', 0: 'output'}
    # visualize.draw_net(config, winner, True, node_names=node_names)
    # visualize.plot_stats(stats, ylog=False, view=True)

def main():
	global xvec 
	xvec = pickle.load(open("xvec.p", "rb"))
	global svec
	svec = pickle.load(open("svec.p", "rb"))
	run()

if __name__ == '__main__':
	main()
	

