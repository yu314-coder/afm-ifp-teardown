import json,numpy as np,torch,torch.nn.functional as F,sys,importlib.util
d=np.memmap(sys.argv[1],dtype=np.uint8,mode='r');EOF=d.size;D=1536;VOCAB=152064
cb=np.load("afm_codebook_deswz.npy").ravel().astype(np.float32)
SCALE0=0x60;INDEX0=0x1078000;BACK=0xc0c8000;NSC=(INDEX0-SCALE0)//2
SC=np.frombuffer(d[SCALE0:SCALE0+NSC*2],dtype=np.float16).astype(np.float32)
NQ=16;NKV=8;HD=128;THETA=500000.0;EXP=256;K=10;Ss=16384;Sd=3072;EMBSC=0.02
h=json.load(open("const_hierarchy.json"))
seq=[('S' if ('IFP' in ' '.join(h[str(N)]['hier']) or 'KVReuse' in ' '.join(h[str(N)]['hier'])) else 'D') if str(N) in h else 'S' for N in range(396)]
coff=[];off=BACK
for N in range(396):coff.append(off);off+=(Ss if seq[N]=='S' else Sd)*D//2
def dec(off,Co,Ci=D):
    n=Co*Ci;raw=np.array(d[off:off+n//2],dtype=np.uint8);idx=np.empty(n,dtype=np.int64);idx[0::2]=raw&0xf;idx[1::2]=raw>>4
    blk=np.clip(((off-INDEX0)*2+np.arange(n))//1024,0,NSC-1)
    return torch.from_numpy((cb[idx]*SC[blk]).reshape(Co//8,Ci//128,128,8).transpose(0,3,1,2).reshape(Co,Ci).astype(np.float32))
def rms(x,e=1e-6):return x/x.pow(2).mean(-1,keepdim=True).add(e).sqrt()
def rope(x):
    T=x.shape[-2];inv=1/(THETA**(torch.arange(0,HD,2).float()/HD));a=torch.outer(torch.arange(T).float(),inv)
    c=torch.cat([a.cos()]*2,-1);s=torch.cat([a.sin()]*2,-1);x1,x2=x[...,:HD//2],x[...,HD//2:];return x*c+torch.cat([-x2,x1],-1)*s
def attn(hh,Wq,Wo):
    T=hh.shape[0];pr=rms(hh)@Wq.T;q=pr[:,:NQ*HD].view(T,NQ,HD).transpose(0,1);k=pr[:,NQ*HD:NQ*HD+NKV*HD].view(T,NKV,HD).transpose(0,1);v=pr[:,NQ*HD+NKV*HD:NQ*HD+2*NKV*HD].view(T,NKV,HD).transpose(0,1)
    q=rope(rms(q));k=rope(rms(k));k=k.repeat_interleave(2,0);v=v.repeat_interleave(2,0)
    return hh+F.scaled_dot_product_attention(q,k,v,is_causal=True).transpose(0,1).reshape(T,NQ*HD)@Wo.T
def ffn(hh,L):
    hn=rms(hh);out=torch.zeros_like(hh)
    for mod in range(3):
        b=9*L+3*mod
        if b+2>=396:break
        I=Ss if seq[b]=='S' else Sd
        try:g=hn@dec(coff[b],I).T;u=hn@dec(coff[b+1],I).T;dn=dec(coff[b+2],I)
        except:continue
        NE=I//EXP;score=g[:,:NE*EXP].reshape(g.shape[0],NE,EXP).abs().mean(-1);tk=score.topk(min(K,NE),-1).indices
        m=torch.zeros_like(g)
        for tt in range(g.shape[0]):
            for e in tk[tt]:m[tt,e*EXP:(e+1)*EXP]=1
        out=out+((F.silu(g)*u)*m)@dn
    return hh+out
# DE-SWIZZLED embedding: rows in 8-row tiles
EMB=EOF-VOCAB*D//2
def emb_deswz(ids):
    r=[]
    for tid in ids:
        tile=tid//8;pos=tid%8;o=EMB+tile*8*D//2
        raw=np.array(d[o:o+8*D//2],dtype=np.uint8);idx=np.empty(8*D,dtype=np.float32);idx[0::2]=raw&0xf;idx[1::2]=raw>>4;idx-=7.5
        W8=idx.reshape(1,D//128,128,8).transpose(0,3,1,2).reshape(8,D);r.append(W8[pos]*EMBSC)
    return torch.tensor(np.array(r),dtype=torch.float32)
spec=importlib.util.spec_from_file_location("t","afm_tokenizer.py");t=importlib.util.module_from_spec(spec);spec.loader.exec_module(t)
tk=t.AFMTokenizer("tok_vocab.json")
sd=torch.load("/Volumes/D/fix/afmplus_v11_ifp_FULL_v2.pt",map_location='cpu',weights_only=False)["state_dict"]
import re
NL=len(sorted({int(re.search(r'layers\.(\d+)\.',k).group(1)) for k in sd if k.startswith("model.layers.")}))
ids=tk.encode("The capital of France is",add_bos=True)
hh=emb_deswz(ids)
for L in range(NL):
    p=f"model.layers.{L}.attn."
    if (p+"qkv.weight") in sd:hh=attn(hh,sd[p+"qkv.weight"].float(),sd[p+"o.weight"].float())
    hh=ffn(hh,L)
hf=rms(hh[-1]);logits=np.zeros(VOCAB,dtype=np.float32)
# de-swizzled unembed: decode in 8-row tiles
for tile in range(0,VOCAB//8):
    o=EMB+tile*8*D//2;raw=np.array(d[o:o+8*D//2],dtype=np.uint8);idx=np.empty(8*D,dtype=np.float32);idx[0::2]=raw&0xf;idx[1::2]=raw>>4;idx-=7.5
    W8=torch.from_numpy(idx.reshape(1,D//128,128,8).transpose(0,3,1,2).reshape(8,D))
    logits[tile*8:tile*8+8]=(W8@hf).numpy()
print(f"DE-SWIZZLED embedding, {NL}-layer prediction for 'The capital of France is':",flush=True)
for i in np.argsort(logits)[::-1][:12]:print(f"   '{tk.vocab[i] if i<len(tk.vocab) else '?'}' ({logits[i]:.0f})")
