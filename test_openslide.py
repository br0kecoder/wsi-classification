from openslide import open_slide
import openslide
from openslide.deepzoom import DeepZoomGenerator
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
slide=open_slide('TCGA-A6-2686-01Z-00-DX1.0540a027-2a0c-46c7-9af0-7b8672631de7.svs')
#slide=open_slide('DHMC_0041.png')
slide_props=slide.properties
print(slide_props)
print(f"Vendor is {slide_props['openslide.vendor']}")
print(f"Pixel size of X in um is {slide_props['openslide.mpp-x']}")
print(f"Pixel size of Y in um is {slide_props['openslide.mpp-y']}")

#objective=float(slide.properties[openslide.PROPERTY_NAME_OBJECTIVE_POWER])
objective=float(slide.properties['openslide.objective-power'])
print(f"The objective power is {objective}")

slide_dims=slide.dimensions
print(f"Dimensions: {slide_dims}")

slide_thumb=slide.get_thumbnail(size=(1000,1000))
slide_thumb.show()

tiles=DeepZoomGenerator(slide,tile_size=256,overlap=0,limit_bounds=False)

print(f"{tiles.level_count} {tiles.tile_count} {tiles.level_dimensions}")

cols, rows=tiles.level_tiles[tiles.level_count-1]
import os
tile_dir="images/saved_tiles/svs_original_tiles"
if not os.path.exists(tile_dir):
    os.makedirs(tile_dir)
for row in range(rows):
    for col in range(cols):
        tile_name=os.path.join(tile_dir,'%d_%d' %(col,row))
        temp_tile=tiles.get_tile(tiles.level_count-1, (col,row))
        temp_tile_RGB=temp_tile.convert('RGB')
        temp_tile_np=np.array(temp_tile_RGB)
        plt.imsave(tile_name+".png",temp_tile_np)