import fontforge
import string
F = fontforge.open("masallera.ttf")
for name in F:
    if name in string.ascii_lowercase:
      filename = name + ".png"
      F[name].export(filename, 500)
