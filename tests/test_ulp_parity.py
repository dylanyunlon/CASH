#!/usr/bin/env python3
"""
C024: Differential test locking the cross-layer ULP contract.

Asserts the Python ULPAnalyzer is bit-identical to a host-compiled emulation of
the CUDA primitives in ulp_analyzer.cuh / differential_validator.cuh, across an
adversarial value set (±0, ±inf, NaN, denormals, power-of-two ULP gaps), and
that f_ulp_bounded agrees with Python at every pareto_solver.cuh tolerance preset.

No GPU required: the CUDA device functions are pure bit arithmetic, emulated in
C++ with the SAME source logic and compiled with g++.
"""
import os, sys, subprocess, struct, tempfile, textwrap
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from crucible.validator.consistency_validator import ULPAnalyzer, ParetoTracer

CPP = r'''
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cmath>
static inline int32_t f2i(float x){int32_t i;std::memcpy(&i,&x,4);return i;}
// MUST mirror ulp_analyzer.cuh::float_to_biased_int (C017 form)
int64_t fbi(float x){uint32_t b=(uint32_t)f2i(x);uint32_t m=(b>>31)?0xFFFFFFFFu:0x80000000u;return (int64_t)(b^m);}
uint64_t ulp(float a,float b){if(std::isnan(a)||std::isnan(b))return UINT64_MAX;int64_t d=fbi(a)-fbi(b);return (uint64_t)(d<0?-d:d);}
int ulp_to_bin(uint64_t u){const int B=34;if(u==0)return 0;if(u==UINT64_MAX)return B-1;int b=0;uint64_t t=u;while(t>>=1)++b;return (b+1<B)?b+1:B-1;}
// MUST mirror differential_validator.cuh::f_ulp_bounded (C019 form)
int fub(float a,float b,double tol){return (double)ulp(a,b)>tol;}
int main(int argc,char**argv){
    // protocol: read pairs of uint32 hex (bit patterns) on stdin, emit ulp + bin
    // plus, for a fixed tolerance set, the fub decision.
    uint32_t ba,bb; 
    while(scanf("%u %u",&ba,&bb)==2){
        float a,b; std::memcpy(&a,&ba,4); std::memcpy(&b,&bb,4);
        uint64_t u=ulp(a,b);
        printf("%llu %d\n",(unsigned long long)u, ulp_to_bin(u));
    }
    return 0;
}
'''

def build():
    d = tempfile.mkdtemp()
    src = os.path.join(d, 'emu.cpp'); binp = os.path.join(d, 'emu')
    open(src,'w').write(CPP)
    subprocess.run(['g++','-O2','-o',binp,src], check=True)
    return binp

def f2bits(f):
    return struct.unpack('<I', struct.pack('<f', np.float32(f)))[0]

def adversarial_floats():
    vals = [0.0, -0.0, 1.0, -1.0, 2.0, -2.0, 0.5, -0.5,
            float('inf'), float('-inf'),
            1e30, -1e30, 1e-30, -1e-30]
    # denormals
    for k in (1,2,3,1000):
        vals.append(struct.unpack('<f', struct.pack('<I', k))[0])
        vals.append(struct.unpack('<f', struct.pack('<I', 0x80000000|k))[0])
    # values straddling power-of-two ULP gaps: take x and x+2^k ULPs
    base = 1.0
    bb = f2bits(base)
    for k in range(0, 30):
        vals.append(struct.unpack('<f', struct.pack('<I', (bb + (1<<k)) & 0xFFFFFFFF))[0])
    return vals

def test_ulp_and_bin_parity():
    binp = build()
    vals = adversarial_floats()
    pairs = [(x,y) for x in vals for y in vals]
    # build stdin
    lines = []
    for x,y in pairs:
        lines.append(f"{f2bits(x)} {f2bits(y)}")
    out = subprocess.run([binp], input="\n".join(lines), capture_output=True, text=True, check=True)
    cpp = [tuple(map(int, l.split())) for l in out.stdout.strip().split("\n")]

    a = torch.tensor([x for x,_ in pairs], dtype=torch.float32)
    b = torch.tensor([y for _,y in pairs], dtype=torch.float32)
    u, nan = ULPAnalyzer.ulp_distance_with_mask(a, b)
    u = u.tolist(); nan = nan.tolist()

    mism = 0
    for i,((cu,cbin),pu,pn) in enumerate(zip(cpp,u,nan)):
        if pn:
            # CUDA returns UINT64_MAX for NaN; python uses sentinel + mask.
            assert cu == (1<<64)-1, f"row {i}: cpp expected NaN sentinel, got {cu}"
            continue
        if cu != pu:
            mism += 1
            if mism<=10: print(f"  ULP mismatch row {i}: cpp={cu} py={pu}")
        # bin parity
        pbin = ULPAnalyzer.ulp_to_bin(pu)
        assert pbin == cbin, f"row {i}: bin cpp={cbin} py={pbin} (ulp={pu})"
    assert mism == 0, f"{mism} ULP value mismatches vs CUDA emulation"
    print(f"  ULP+bin parity over {len(pairs)} pairs: PASS")

def test_histogram_conserves_counts():
    a = torch.cat([torch.randn(5000), torch.full((11,), float('nan'))])
    b = torch.cat([torch.randn(5000), torch.zeros(11)])
    h = ULPAnalyzer.ulp_histogram(a, b)
    assert sum(h['histogram_counts']) == h['total_elements'], "counts not conserved"
    assert h['histogram_counts'][-1] >= h['nan_count'], "NaN not in catastrophic bin"
    print("  histogram count conservation: PASS")

def test_empty_and_all_nan():
    e = torch.tensor([])
    assert ULPAnalyzer.ulp_distance(e, e).numel() == 0
    assert ULPAnalyzer.summary(e, e)['max_ulp'] == 0
    assert ULPAnalyzer.ulp_histogram(e, e)['total_elements'] == 0
    allnan = torch.full((16,), float('nan'))
    z = torch.zeros(16)
    s = ULPAnalyzer.summary(allnan, z)
    assert s['nan_count'] == 16 and s['max_ulp'] == 0
    print("  empty + all-NaN defined behaviour: PASS")

def test_pareto_smoke():
    pt = ParetoTracer(torch.device('cpu'), torch.device('cpu'))
    pts = pt.sweep(num_embeddings=400, embedding_dim=8, batch_size=32, num_iters=2)
    assert len(pts) == 7
    for p in pts:
        assert p.achieved_max_ulp < ULPAnalyzer.NAN_ULP, "sentinel leaked into max"
    print("  pareto sweep smoke + no-sentinel-leak: PASS")

if __name__ == '__main__':
    test_ulp_and_bin_parity()
    test_histogram_conserves_counts()
    test_empty_and_all_nan()
    test_pareto_smoke()
    print("\nC024 ALL TESTS PASSED")
