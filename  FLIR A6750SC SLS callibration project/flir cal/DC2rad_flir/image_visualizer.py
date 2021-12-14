import matplotlib.pyplot as plt
import rasterio
import glob

target_dir = 'drone_collect_images'
output = 'figs'

test_paths = glob.glob(target_dir + '/*.img')
headers = glob.glob(target_dir + '/*.hdr')
multiple_length = len(test_paths)

g = 0
def process(filename: str=None) -> None:
    image = rasterio.open(filename)
    image = image.read(1)
    
    fig, ax = plt.subplots()
    im = ax.imshow(image, cmap=plt.get_cmap('hsv'))#,vmin = 0, vmax = 20
    fig.colorbar(im)
    plt.title(g)
    plt.show()
    #im.savefig(output + '/graph' + g + '.png')


for file in test_paths[1:20]:
    g += 1
    process(file)

