
def relu(weights, inputs):
	'''
	weights: {k:(w,b)}
	inputs: (x,k)
	'''
	x = inputs[0]
	k = inputs[1]
	w = weights[k][0]
	w = weights[k][1]
	
	return max(0,w*x+b)