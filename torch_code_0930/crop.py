import os
from PIL import Image
from glob import glob
import sys


def crop2imgs(image, shape):
    img = Image.open(image)
    (orig_x, orig_y) = img.size
    (box_x, box_y) = (orig_x//shape[0], orig_y//shape[1])
    for i in range(shape[1]):
        for j in range(shape[0]):
            box = (j*box_x, i*box_y, (j+1)*box_x, (i+1)*box_y)
            yield img.crop(box)
            

if __name__ == '__main__':
    os.makedirs('croped',exist_ok=True)
    os.makedirs(sys.argv[1]+'croped/'+sys.argv[2],exist_ok=True)
    shape = [int(x) for x in sys.argv[2].split(',')]
    for img in glob(sys.argv[1]+'/*.jpg'):
        for i,v in enumerate(crop2imgs(img,shape)):
            # print(img[img.rfind('/')+1:img.rfind('.')])
            v.save(sys.argv[1]+'croped/'+sys.argv[2]+'/'+img[img.rfind('/')+1:img.rfind('.')]+'_%d.png'%i)
        


