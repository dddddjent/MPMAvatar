import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse

# Template command (mpmavatar environment, MPMAvatar directory):
# python merge_meshes.py --seq s170_t1 --output_dir output/phys --data_dir data

parser = argparse.ArgumentParser()
parser.add_argument("--seq", type=str, required=True)
parser.add_argument("--output_dir", type=str, default="./output/phys")
parser.add_argument("--data_dir", type=str, default="./data")
args = parser.parse_args()

output_dir = Path(args.output_dir)
data_dir = Path(args.data_dir)
split_idx_upper = np.load(data_dir / args.seq / "split_idx_upper.npz")

upper_dir = output_dir / f"{args.seq}_upper" / "seed0/uvmesh"
lower_dir = output_dir / f"{args.seq}_lower" / "seed0/uvmesh"
mesh_upper_files = {path.name: path for path in upper_dir.glob("*.obj")}
mesh_lower_files = {path.name: path for path in lower_dir.glob("*.obj")}
assert mesh_upper_files, f"No upper simulation meshes: {upper_dir}"
assert mesh_upper_files.keys() == mesh_lower_files.keys(), "Upper and lower simulation frames differ"

merged_dir = output_dir / args.seq / "seed0/uvmesh"
merged_dir.mkdir(parents=True, exist_ok=True)

for filename in tqdm(sorted(mesh_upper_files), desc="Merging meshes..."):
    upper_file, lower_file = mesh_upper_files[filename], mesh_lower_files[filename]
    upper_v, lower_v, lines = [], [], []
    
    with open(upper_file, 'r') as f:
        for line in f:
            if line.startswith('v '):
                parts = line.strip().split()
                upper_v.append([float(parts[1]), float(parts[2]), float(parts[3])])
            else:
                lines.append(line)

    with open(lower_file, 'r') as f:
        for line in f:
            if line.startswith('v '):
                parts = line.strip().split()
                lower_v.append([float(parts[1]), float(parts[2]), float(parts[3])])
    
    upper_v = np.array(upper_v, dtype=np.float32)
    lower_v = np.array(lower_v, dtype=np.float32)
    assert upper_v.shape == lower_v.shape, f"Vertex counts differ: {filename}"
    cloth_v_idx_upper = split_idx_upper["reordered_cloth_v_idx"]
    lower_v[cloth_v_idx_upper] = upper_v[cloth_v_idx_upper]
    
    with (merged_dir / filename).open('w') as f:
        f.writelines(['v %f %f %f\n' % (v[0], v[1], v[2]) for v in lower_v])
        f.writelines(lines)
