import pickle, sys
from pathlib import Path
from topfed.analysis import e3
cells = e3.load_cells(Path("results"))
pickle.dump(cells, open(sys.argv[1], "wb"))
print(len(cells), sum(len(v) for v in cells.values()))
ds = sorted({(k.dataset, k.attack, k.defense) for k in cells if k.attack in ("adaptive","direction_aligned") and k.model=="cnn"})
from collections import defaultdict
d = defaultdict(set)
for a,b,c in ds: d[(a,b)].add(c)
for k,v in d.items(): print(k, sorted(v))
