# https://docs.deepmodeling.com/projects/dpdata/en/master/index.html
import dpdata
from pathlib import Path

base = Path("QDpiDataset-main/data")

# load all subsets
data = dpdata.MultiSystems()
#data.from_deepmd_hdf5(data/"neutral/spice.hdf5")
data.from_deepmd_hdf5(base/"neutral/ani.hdf5")
data.from_deepmd_hdf5(base/"neutral/geom.hdf5")
print(data)
for key, values in data.systems.items():
    print(key)
    print(len(values))
    sys = values#[0]
    #print(sys["atom_names"])
    #print(sys["atom_numbs"])
    print(sys["atom_types"])
    print(sys["coords"])
    print(sys["energies"])
    print(sys["forces"])
    print(sys["cells"])
    print(sys["virials"])
    break
exit(0)

data.from_deepmd_hdf5(base/"neutral/freesolvmd.hdf5")
data.from_deepmd_hdf5(base/"neutral/re.hdf5")
data.from_deepmd_hdf5(base/"neutral/remd.hdf5")
data.from_deepmd_hdf5(base/"neutral/comp6.hdf5")

# dump combined data
data.to_deepmd_hdf5("qdpi-1.0.hdf5")

# print the summary of data
print(data)

# get subsystems
subsystems = list(data.systems.values())

# get the data from one of the subsystem
print(subsystems[0].data.keys())
print(subsystems[0].data)
