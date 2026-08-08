import tempfile
from pathlib import Path
from larc.paged_container import LARCv2File,PageSpec,write_larc_v2,CODEC_Q4_ROW,CODEC_HRVQ64,FLAG_REQUIRED,FLAG_SHARED,FLAG_REFINEMENT,FLAG_STREAMABLE

def test_v2_pages_are_aligned_checksummed_and_random_access():
    with tempfile.TemporaryDirectory() as td:
        p=Path(td)/'x.larc';pages=[PageSpec(1,CODEC_Q4_ROW,FLAG_REQUIRED|FLAG_SHARED,b'abc'*100,600,1),PageSpec(2,CODEC_HRVQ64,FLAG_REFINEMENT|FLAG_STREAMABLE,b'xyz'*37,222,1)]
        write_larc_v2(p,pages,{'model':{'name':'test'}},4096)
        with LARCv2File(p) as f:
            assert f.manifest['model']['name']=='test'
            assert bytes(f.page_view(1,True))==b'abc'*100
            assert f.pages[1].offset%4096==0 and f.pages[2].offset%4096==0
            assert f.resident_payload_bytes([1,1,2])==411
