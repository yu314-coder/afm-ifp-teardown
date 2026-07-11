"""Fast LUT (codebook) recovery for the AFM-ifp palettized weights.
Metric: structure ratio R = sigma_max(W)^2 / sigma_max(shuffle(W))^2  (shuffle preserves ||W||_F).
Real weights: R ~ 9 ; noise: R ~ 1.  Uses power iteration for sigma_max (fast)."""
import numpy as np

def smax(W, iters=30, seed=0):
    rng=np.random.default_rng(seed)
    v=rng.standard_normal(W.shape[1]).astype(np.float32); v/=np.linalg.norm(v)
    for _ in range(iters):
        u=W@v; v=W.T@u; n=np.linalg.norm(v);
        if n==0: return 0.0
        v/=n
    return float(np.linalg.norm(W@v))

def R(W, seed=1):
    W=W.astype(np.float32)
    s1=smax(W)
    f=W.flatten().copy(); np.random.default_rng(seed).shuffle(f)
    s2=smax(f.reshape(W.shape))
    return (s1/s2)**2 if s2>0 else 0.0

def load():
    b=np.frombuffer(open("idx_exact.bin","rb").read(),dtype=np.uint8)
    scales=np.frombuffer(open("fp16_region.bin","rb").read(),dtype=np.float16).astype(np.float32)
    lo=(b&0xf); hi=(b>>4); nib=np.empty(b.size*2,dtype=np.uint8); nib[0::2]=lo; nib[1::2]=hi
    return nib, scales

def pairs_by_freq(nib):
    h=np.bincount(nib,minlength=16); p=[(i,15-i) for i in range(8)]
    p.sort(key=lambda q:-(h[q[0]]+h[q[1]])); return p

def build(pairs, mags):
    lut=np.zeros(16,dtype=np.float32)
    for (a,c),m in zip(pairs,mags): lut[a]=m; lut[c]=-m
    return lut

# canonical tensor layout for the leading (sparse) region: qkv[3072,1536], out[1536,2048]
LAYOUT=[("qkv",3072,1536),("out",1536,2048)]

def decode_tensor(nib, scales, lut, pos, spos, o, i):
    N=o*i
    W=(lut[nib[pos:pos+N]]*np.repeat(scales[spos:spos+N//1024],1024)[:N]).reshape(o,i)
    return W, pos+N, spos+N//1024

if __name__=="__main__":
    nib,scales=load(); pairs=pairs_by_freq(nib)
    # score = R on tensor0 (qkv) + tensor2 (qkv) averaged
    def score(mags):
        lut=build(pairs,mags); pos=spos=0; rs=[]
        for k,(nm,o,i) in enumerate(LAYOUT*3):
            W,pos,spos=decode_tensor(nib,scales,lut,pos,spos,o,i)
            if nm=="qkv": rs.append(R(W))
        return np.mean(rs)
    mags=np.array([0.13,0.39,0.66,0.94,1.26,1.62,2.07,2.73],dtype=np.float32)
    cur=score(mags); print("init R=%.2f"%cur)
    for it in range(25):
        improved=False
        for k in range(8):
            for d in (0.85,1.18,0.95,1.05,0.99,1.01):
                t=mags.copy(); t[k]*=d
                if np.all(np.diff(t)>0):
                    s=score(t)
                    if s>cur+1e-3: mags,cur=t,s; improved=True
        if not improved: break
    lut=build(pairs,mags)
    np.save("afm_lut.npy",lut)
    print("optimized mean R (qkv tensors) = %.2f"%cur)
    print("LUT =",np.round(lut,4).tolist())
    # full verification with correct shapes
    print("\nper-tensor verification:")
    pos=spos=0
    for k,(nm,o,i) in enumerate((LAYOUT*3)):
        W,pos,spos=decode_tensor(nib,scales,lut,pos,spos,o,i)
        print("  t%d %-4s [%d,%d] R=%.2f std=%.4f"%(k,nm,o,i,R(W),W.std()))
    print("(real ~9, noise ~1)")
