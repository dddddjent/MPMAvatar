import os
import bpy
from glob import glob
import sys
import argparse

# Template command (synthetic_avatar environment, allocated GPU):
# python MPMAvatar/blender/bake.py -- --output_path data/MPMAvatar/4DDress/examples/s190_t2/output/phys/s190_t2/seed0 --ao_res 256 --resume

class ArgumentParserForBlender(argparse.ArgumentParser):
    def _get_argv_after_doubledash(self) -> list[str]:
        try:
            idx = sys.argv.index("--")
            return sys.argv[idx+1:]
        except ValueError as e:
            return []
    def parse_args(self) -> argparse.Namespace:
        return super().parse_args(args=self._get_argv_after_doubledash())

parser = ArgumentParserForBlender()
parser.add_argument("--output_path", type=str, default="./output")
parser.add_argument("--ao_res", type=int, default=256)
parser.add_argument("--resume", action="store_true", help="Keep completed AO maps and save new maps atomically")
args = parser.parse_args()

bpy.data.scenes[0].render.engine = "CYCLES"

# Set the device_type
bpy.context.preferences.addons[
    "cycles"
].preferences.compute_device_type = "CUDA" # or "OPENCL"

# Set the device and feature set
bpy.context.scene.cycles.device = "GPU"

# get_devices() to let Blender detects GPU device
bpy.context.preferences.addons["cycles"].preferences.get_devices()
for d in bpy.context.preferences.addons["cycles"].preferences.devices:
    d["use"] = 1 # Using all devices, include GPU and CPU

bpy.context.scene.render.bake.margin = args.ao_res // 256

meshdir = os.path.join(args.output_path, "uvmesh")
aomapdir = os.path.join(args.output_path, "aomap")
os.makedirs(aomapdir, exist_ok=True)

meshfiles = sorted(glob(os.path.join(meshdir, "*.obj")))

for idx, meshfile in enumerate(meshfiles):
    image_path = os.path.join(aomapdir, os.path.basename(meshfile).replace("obj", "png"))
    if args.resume and os.path.isfile(image_path):
        completed = bpy.data.images.load(image_path, check_existing=False)
        assert tuple(completed.size) == (args.ao_res, args.ao_res), image_path
        bpy.data.images.remove(completed)
        print(f"Keeping completed AO map: {image_path}", flush=True)
        continue
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.object.select_all()
    bpy.ops.object.delete()

    bpy.ops.wm.obj_import(filepath=meshfile)

    current_mesh = bpy.context.scene.objects[0]
    bpy.context.view_layer.objects.active = current_mesh

    mat = bpy.data.materials.new(name="Material")
    current_mesh.data.materials.append(mat)

    mat = current_mesh.active_material
    mat.use_nodes = True
    matnodes = mat.node_tree.nodes

    bpy.ops.image.new(name="AO", width=args.ao_res, height=args.ao_res)
    image = bpy.data.images['AO']

    tex = matnodes.new('ShaderNodeTexImage')
    img = image
    tex.image = img

    disp = matnodes['Material Output'].inputs[2]
    mat.node_tree.links.new(disp, tex.outputs[0])

    # Bake the lightmap
    bpy.ops.object.bake(type='AO')

    # Save the baked image
    image.filepath_raw = image_path + ".tmp.png" if args.resume else image_path
    image.file_format = "PNG"
    image.save()
    if args.resume:
        os.replace(image.filepath_raw, image_path)

    bpy.data.images.remove(image)
