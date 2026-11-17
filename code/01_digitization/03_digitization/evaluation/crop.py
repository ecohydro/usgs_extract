"""Save a zoomed crop of a page image, for reading small print.
Run: python crop.py IMAGE OUT.png X0 Y0 X1 Y1 [ROTATE_DEG]   (X, Y as fractions 0-1 of the page,
     taken after rotation; output is upscaled so the long side is about 2000 px)."""
import sys
from PIL import Image

img, out, x0, y0, x1, y1 = sys.argv[1], sys.argv[2], *map(float, sys.argv[3:7])
im = Image.open(img).convert("L")
if len(sys.argv) > 7:
    im = im.rotate(float(sys.argv[7]), expand=True)
w, h = im.size
c = im.crop((int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h)))
s = 2000 / max(c.size)
if s > 1:
    c = c.resize((int(c.width * s), int(c.height * s)), Image.LANCZOS)
c.save(out)
print(out, c.size)
