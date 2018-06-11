import matplotlib.pyplot as plt
from io import StringIO
import pathlib
import tarfile
import numpy as np


def process_srfs(fname):
    tar = tarfile.open(fname, "r:gz")
    bands = {}
    for member in tar.getmembers():
        f = tar.extractfile(member)
        if f:
            content = f.read()
            s = StringIO(content.decode("utf-8"))
            x=np.genfromtxt(s, skip_header=4)
            chname = member.name
            chname = chname.replace("rtcoef_", "")
            chname = chname.replace("_srf", "")
            chname = chname.replace(".txt", "")
            x[:, 0] = 10000000/x[:, 0]
            bands[chname] = x
    print(f"Done {fname.name:s}. {len(bands):d} bands")
    return bands

path = pathlib.Path("/data/netapp_3/ucfajlg/python/gp_emulator/gp_emulator/SRFs/")
library = {}
for fname in path.glob("*.tar.gz"):
    bands = process_srfs(fname)
    library.update(bands)
np.savez("band_pass_library.npz", **library)
f=np.load("band_pass_library.npz")
ch=[f for f in f.keys() if f.find("landsat_8_oli") >= 0]
for c in ch:
    plt.plot(f[c][:, 0], f[c][:, 1], '-')
