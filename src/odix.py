"""
odix.py — parser for Apple's on-device-inference 'odix' container (magic 'odix\\0\\0\\5\\0').
Project: reconstruct the IFP MoE weight layout for AFM 'afmplus-v11.0-ifp'.

Confirmed so far:
  * Header: u64[0]=ir_start(0x40), u64[1]=ir_size. IR is a set of length-prefixed sections.
  * Back-references are SELF-RELATIVE signed 32-bit offsets: target = pos + int32(word),
    high byte 0xff => it's a ref. They point into an interned string/symbol pool.
  * One IR section is a LOCATION/DEBUG tree: keys 'location_type','named_child_loc','name',
    'op_id','line','filename','column' — gives op NAMES (e.g. ANE_RoPETransform_2277) and the
    tensor names (_wrapped_model_..._palettized_indices_raw), but NOT shapes/offsets.
  * Shapes/dtypes/data-offsets live in the early sections (the 2.36 MB block at 0x40). [WIP]
"""
import struct, re

class Odix:
    def __init__(self, path):
        self.d = open(path, "rb").read()
        assert self.d[:4] == b"odix", "bad magic"
        self.ir_start, self.ir_size = struct.unpack("<QQ", self.d[8:24])
        self.ir_end = self.ir_start + self.ir_size

    def u32(self, o): return struct.unpack("<I", self.d[o:o+4])[0]
    def s32(self, o):
        v = self.u32(o); return v - (1 << 32) if v >> 31 else v
    def is_ref(self, o): return (self.u32(o) >> 24) == 0xff
    def deref(self, o): return o + self.s32(o)          # self-relative target

    def cstr(self, o, maxlen=128):
        m = re.match(rb"[ -~]{1,%d}" % maxlen, self.d[o:o+maxlen])
        return m.group().decode() if m else ""

    def sections(self):
        """Length-prefixed top-level sections: [u32 tag][u32 len][len bytes], 4-aligned."""
        out, o = [], self.ir_start
        while o + 8 <= self.ir_end:
            tag, ln = self.u32(o), self.u32(o + 4)
            if ln == 0 or o + 8 + ln > len(self.d):
                break
            out.append((o, tag, ln))
            o += (8 + ln + 3) & ~3
        return out

    def find_all(self, needle):
        out, i = [], self.d.find(needle)
        while i != -1:
            out.append(i); i = self.d.find(needle, i + 1)
        return out

    # --- tensor-name inventory in file order (physical dedup happens later) ---
    def constant_names(self):
        pat = re.compile(rb"(?:p_|b_)?_wrapped_model[A-Za-z0-9_]+")
        seen, out = set(), []
        for m in pat.finditer(self.d):
            n = m.group().decode().lstrip("pb_")
            if n not in seen:
                seen.add(n); out.append((m.start(), n))
        return out


if __name__ == "__main__":
    import sys
    o = Odix(sys.argv[1])
    print("ir 0x%x..0x%x (%.1f MB)  file %.1f MB"
          % (o.ir_start, o.ir_end, o.ir_size/1e6, len(o.d)/1e6))
    for off, tag, ln in o.sections()[:8]:
        print("  section @0x%07x tag=0x%08x len=%d (%.2f MB)" % (off, tag, ln, ln/1e6))
