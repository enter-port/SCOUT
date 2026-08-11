"""Materialize an SOE-style core_N subset (demos with ind % interval == 0) into
a self-contained hdf5. LPB RobomimicReplayImageDataset loads ALL demos, so we
materialize the subset instead of using a mask/filter_key."""
import sys, h5py
src, dst, interval = sys.argv[1], sys.argv[2], int(sys.argv[3])
with h5py.File(src, 'r') as fin:
    demos = sorted(fin['data'].keys(), key=lambda k: int(k.split('_')[-1]))
    keep = [d for i, d in enumerate(demos) if i % interval == 0]
    total = int(sum(fin[f'data/{d}/actions'].shape[0] for d in keep))
    with h5py.File(dst, 'w') as fout:
        for top in fin.keys():
            if top != 'data':
                fin.copy(top, fout)
        g = fout.create_group('data')
        for k, v in fin['data'].attrs.items():
            g.attrs[k] = v
        for ni, d in enumerate(keep):
            fin.copy(f'data/{d}', g, name=f'demo_{ni}')
        g.attrs['total'] = total
print(f"OK wrote {dst}: {len(keep)} demos, total={total} steps")
