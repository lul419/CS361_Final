# CS361 Evol. Comp. (Spring 2017)
# Final Project

from scipy.io import wavfile as wf
from scipy import signal as sig
import numpy as np
import pickle
import sys
import random

def main():
    # arguments:
    # C, L, file_main, file_bg
    if len(sys.argv) != 5:
        print ("Wrong number of arguments. Usage: preproc.py <C> <L> <file1> <file2>.")
        return
    str1 = sys.argv[1]
    str2 = sys.argv[2]
    f1 = sys.argv[3]
    f2 = sys.argv[4]

    if not (str1.isdigit() and int(str1) > 0):
        print ("C needs to be a positive integer.")
        return
    if not (str2.isdigit() and int(str2) > 0):
        print ("L needs to be a positive integer.")
        return
    C = int(str1)
    L = int(str2)

    # open .wav files, apply fft to get list of frames
    ratefg,fg = wf.read(f1)
    ratebg,bg = wf.read(f2)

    if not ratefg == ratebg:
        print("Error: file sample rates do not match.")
        return

    # extract single audio channel only
    fg = (fg[:,0]).T
    bg = (bg[:,0]).T

    # perform STFT, get frequency frames
    freq,time,Zxxfg = sig.stft(fg, fs=ratefg, nperseg=L)
    freq,time,Zxxbg = sig.stft(bg, fs=ratebg, nperseg=L)
    argfg = np.angle(np.array(Zxxfg).T)
    argbg = np.angle(np.array(Zxxbg).T)
    fftfg = np.abs(np.array(Zxxfg).T)
    fftbg = np.abs(np.array(Zxxbg).T)
    numframes = min(len(fftfg),len(fftbg))

    # stack 2C+1 frames to get f and v
    fvec = []
    vvec = []
    for i in range(C,numframes-C):
        f = fftfg[i-C:i+C+1].flatten()
        v = fftbg[i-C:i+C+1].flatten()
        fvec.append(f)
        vvec.append(v)
    fvec = np.array(fvec)
    vvec = np.array(vvec)

    # construct x and s by mixing f and v
    xvec = []
    svec = []
    coeff = []
    for i in range(len(fvec)):
        f = fvec[i]
        v = vvec[i]

        # mix using random amplitude weights
        alpha0 = np.random.uniform(low=0.01,high=1.0)
        alpha1 = np.random.uniform(low=0.01,high=1.0)
        t = np.add(np.multiply(f,alpha0), np.multiply(v,alpha1))
        gamma = np.max(t)

        # construct x and s
        x = np.divide(t,gamma)
        # s = np.multiply(f, alpha0/gamma)
        s = np.multiply(f, alpha0/gamma)
        l = len(fftfg[0])
        s = s[C*l:(C+1)*l]

        xvec.append(x)
        svec.append(s)
        coeff.append((gamma,alpha0,alpha1))

    xvec = np.array(xvec)
    svec = np.array(svec)
    coeff = np.array(coeff)

    # pickle xvec, svec, coeff, argfg and argbg
    pickle.dump(xvec, open("xvec.p", "wb"))
    pickle.dump(svec, open("svec.p", "wb"))
    pickle.dump(coeff, open("coeff.p", "wb"))
    pickle.dump(argfg, open("argfg.p", "wb"))

    # Test inverse transform
    # reconstruct Zxx
    # we need (513,254) vector row = complex freq, col = data
    # Zxx = []
    # for i in range(C,len(argfg)-C):
    #     mag = svec[i-C]
    #     arg = argfg[i]
    #     c = np.multiply(mag,np.exp(np.multiply(arg,1j)))
    #     Zxx.append(c)
    # Zxx = np.array(Zxx).T
    # time,recovered = sig.istft(Zxx, fs=ratefg, nperseg=L)

    # recovered = np.multiply(recovered,0.00005)
    # wf.write("test.wav",ratefg,recovered)


main()
