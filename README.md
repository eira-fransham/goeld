# Göld - Löve but for Goldsrc

This is designed to be a game engine for hacking together 3D games using old tech. It's based on the
simple mental model of PyGame or Löve, but for Goldsrc/Quake-era tech. Right now it can load Quake 2
maps and display them, including the skybox textures, but Goldsrc
models and display 

![Screenshot 1](screenshots/01.png) 

![Screenshot 2](screenshots/02.png) 

![Screenshot 3](screenshots/03.png)

![Screenshot 4](screenshots/04.png)

I've also got a parser for the monstrosity that is the Goldsrc .mdl format, so my next task is to
get Half-Life 1 models loading, displaying and animating in this engine. It's very, very likely that
I'll switch to updating the already-existing bindings to `assimp` and using those instead, since I'm
certain that my existing parser is buggy (since the format is a mess) and integrating with `assimp`
allows me to support every model format that it supports, instead of only .mdl.

## How to use

Extract Quake 2's .pak files into a folder called `data` in the working directory that you'll be
executing the program in (so probably the project root, if you're executing with `cargo run`). You
can do this with a tool like [pakextract](https://github.com/yquake2/pakextract). I don't parse
`.wal` textures, so you'll have to bulk-convert any textures stored as `.wal` to `.tga`, `.png`,
`.gif` or suchlike. I'm just using the `image` library for my image parsing, since Goldsrc uses
`.png` and `.tga` textures anyway and the game that I have in mind as an "end goal" for this project
primarily reuses Goldsrc assets. To convert `.wal` to `.tga`, I would like to recommend Wally since
everyone else online does, but I've found that it consistently crashes when batch processing images
under Wine, and so the best way I've found to do the conversion is by a simple Python script using
the Pillow library:

```python
from PIL import WalImageFile

for vals in os.walk("."):
    root, dirs, files = vals
    for file in files:
        pre, ext = os.path.splitext(file)
        try:
            WalImageFile.open(root + "/" + file).save(root + "/" + pre + ".png")
        except:
            pass
```

If you don't do this, you'll see something which I'm sure will be familiar to many modders and
tinkerers of this era of games:

![Missing texture](screenshots/missing.png)
