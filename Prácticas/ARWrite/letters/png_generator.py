import fontforge
import string
F = fontforge.open("masallera.ttf")
for name in F:
    filename = None
    if name in string.ascii_lowercase and name != "s":
      filename = name + ".png"
      F[name].export(filename, 500)
    elif name == 'S':
      filename = name.lower() + ".png"
    if filename:
      F[name].export(filename, 500)
