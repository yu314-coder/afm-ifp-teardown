"""odix_fb.py -- minimal FlatBuffer reader for Apple's `odix` container (magic odix\\0\\0\\5\\0).

Used to decompile main-h16g.odix. Recovers: root table, the 38 export-config functions
(field[9]: prompt_opt/extend x dense_only/sparse_only x context x rank), the 114-entry value
table (field[2], period-3 gate/up/down type cadence), and the NDArray.alloc_const op format
[flags, offset, size, dtype]. The remaining gap to per-constant FFN shapes is the type-index ->
type-table -> interned-dimension-symbol chain (see ODIX_DECOMPILER.md).
"""
import struct,numpy as np
class FB:
    def __init__(s,path,limit=None):
        f=open(path,'rb');f.seek(0);h=f.read(24)
        s.ir_start,s.ir_size=struct.unpack("<QQ",h[8:24])
        f.seek(0);s.d=f.read(limit) if limit else f.read()
        s.N=len(s.d)
    def u8(s,o):return s.d[o]
    def u16(s,o):return struct.unpack("<H",s.d[o:o+2])[0]
    def u32(s,o):return struct.unpack("<I",s.d[o:o+4])[0]
    def i32(s,o):return struct.unpack("<i",s.d[o:o+4])[0]
    def root(s):return s.ir_start+s.u32(s.ir_start)
    def vt(s,tpos):     # returns dict field_index->absolute field pos (only present fields)
        vtp=tpos-s.i32(tpos)
        if not(s.ir_start<=vtp<s.N-4):return None
        vs=s.u16(vtp)
        if vs<4 or vs>4096 or vs%2: return None
        nf=(vs-4)//2;out={}
        for i in range(nf):
            fo=s.u16(vtp+4+2*i)
            if fo: out[i]=tpos+fo
        return out
    def follow(s,p):    # uoffset at p -> target
        return p+s.u32(p)
    def vec(s,p):       # p points to uoffset to vector; returns (elems_start,count)
        t=s.follow(p);return t+4,s.u32(t)
    def string(s,p):
        t=s.follow(p);L=s.u32(t)
        if L>2000:return None
        try:return s.d[t+4:t+4+L].decode('latin1')
        except:return None
if __name__=="__main__":
    import sys
    fb=FB(sys.argv[1])
    r=fb.root();print("root@0x%x"%r)
    f=fb.vt(r)
    print("root fields:",{k:hex(v) for k,v in f.items()})
    for k,p in f.items():
        raw=fb.u32(p)
        # try vector
        try:
            es,cnt=fb.vec(p)
            if 0<cnt<10_000_000: print("  field[%d]: vec count=%d @0x%x"%(k,cnt,es));continue
        except:pass
        print("  field[%d]: scalar=0x%x(%d)"%(k,raw,raw))
