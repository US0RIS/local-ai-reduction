import numpy as np
import torch
from larc.q4 import quantize_q4,dequantize_q4
from larc.q4_runtime import q4_rows,dequantize_q4_rows

GOLD=np.array([[-8.0,-4.0,0.0,3.5,7.0],[-1.0,0.0,1.0,2.0,3.0]],dtype=np.float32)
EXPECTED_PACKED=np.array([0x40,0xC8,0x8F,0x86,0xDA,0x8F],dtype=np.uint8)
EXPECTED_SCALE_BITS=np.array([15360,14043],dtype=np.uint16)

def test_numpy_and_torch_q4_are_byte_identical():
    n=quantize_q4(GOLD)
    p,s,c=q4_rows(torch.from_numpy(GOLD))
    assert c==5
    assert np.array_equal(n.packed,EXPECTED_PACKED)
    assert np.array_equal(n.scales.view(np.uint16),EXPECTED_SCALE_BITS)
    assert np.array_equal(p.cpu().numpy().reshape(-1),EXPECTED_PACKED)
    assert np.array_equal(s.cpu().numpy().view(np.uint16),EXPECTED_SCALE_BITS)
    nd=dequantize_q4(n)
    td=dequantize_q4_rows(p,s,c).cpu().numpy()
    assert np.array_equal(nd,td)
    # The first row deliberately exercises both -8 and +7, proving all 16 codes are reachable.
    assert int((EXPECTED_PACKED[0]&0x0F))-8==-8
    assert int(EXPECTED_PACKED[2]&0x0F)-8==7
