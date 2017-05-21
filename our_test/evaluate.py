import numpy as np
import pickle
import os
import neat
from scipy.io import wavfile as wf
from scipy import signal as sig

def main():
    nn = pickle.load(open("winner_net.p","rb"), encoding='latin1')
    xvec = pickle.load(open("xvec.p", "rb"), encoding='latin1')
    argfg = pickle.load(open("argfg.p", "rb"), encoding='latin1')

    # evalute the ENN on the sound file
    output = []
    for x in xvec:
        sp = nn.activate(x)
        #print("sp: ", sp)
        output.append(sp)
    output = np.array(output)

    # reconstruct Zxx
    # we need (513,254) vector row = complex freq, col = data
    C = 1
    ratefg=44100
    L=512
    Zxx = []
    for i in range(C,len(argfg)-C):
        mag = output[i-C]
        arg = argfg[i]
        c = np.multiply(mag,np.exp(np.multiply(arg,1j)))
        Zxx.append(c)
    Zxx = np.array(Zxx).T
    print(Zxx)
    time,recovered = sig.istft(Zxx, fs=ratefg, nperseg=L)
    pickle.dump(recovered, open("pickled_recovered.p", "wb"))
    recovered = np.multiply(recovered, 0.005)
    wf.write("winner_output.wav",ratefg,recovered)

main()
