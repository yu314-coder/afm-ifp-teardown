"""ANE de-swizzle for AFM-ifp weights.  Tile = 8x128 (=1024, one scale block),
grid row-major, tile column-major.  Validated: real data -> R>>1, random -> R~1."""
import numpy as np
from crack_lut import smax, load, pairs_by_freq, build

MAGS=np.array([0.13,0.39,0.66,0.94,1.26,1.62,2.07,2.73],dtype=np.float32)
TH,TW=8,128   # ANE tile

def get_lut():
    nib,_=load(); return build(pairs_by_freq(nib),MAGS), nib

def deswizzle(v, Cout, Cin, th=TH, tw=TW):
    """v: flat values in ANE storage order -> logical [Cout,Cin]. grid-RM, tile-CM."""
    gh,gw=Cout//th, Cin//tw
    return v.reshape(gh,gw,tw,th).transpose(0,3,1,2).reshape(Cout,Cin)

def Rclean(W, seed=1):
    W=W.astype(np.float32); s1=smax(W)
    f=W.flatten().copy(); np.random.default_rng(seed).shuffle(f); s2=smax(f.reshape(W.shape))
    return (s1/s2)**2

def fits(Cout,Cin,th=TH,tw=TW): return Cout%th==0 and Cin%tw==0
