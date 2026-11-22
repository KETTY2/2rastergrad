
import os, shutil, sys
import pycolmap

src = sys.argv[1]
inp = os.path.join(src, "distorted", "sparse", "0")
img = os.path.join(src, "input")
assert os.path.isdir(inp), f": {inp}"
assert os.path.isdir(img), f": {img}"


pycolmap.undistort_images(src, inp, img, output_type="COLMAP")


sparse = os.path.join(src, "sparse")
os.makedirs(os.path.join(sparse, "0"), exist_ok=True)
for f in os.listdir(sparse):
    if f != "0":
        shutil.move(os.path.join(sparse, f), os.path.join(sparse, "0", f))

rec = pycolmap.Reconstruction(os.path.join(sparse, "0"))
for cam in rec.cameras.values():
    print("camera model:", cam.model)
print("images:", len(rec.images))
print("Done.")