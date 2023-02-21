import os
from argparse import ArgumentParser
import json
import sys
import getopt

from gluoncv import model_zoo, data

class Indexer:

    def __init__(self, path):
        self.net = model_zoo.get_model('yolo3_darknet53_voc', pretrained=True)
        self.path = path



    def _extract_objects(self, im_fname):

        x, img = data.transforms.presets.yolo.load_test(im_fname, short=512)
        print('Shape of pre-processed image:', x.shape)

        class_IDs, scores, bounding_boxs = self.net(x)

        # [net.classes[int(i)] if i!=-1 else -1 for i in class_IDs[0].asnumpy().flat]
        labels = [self.net.classes[int(i)] for i in class_IDs[0].asnumpy().flat if i != -1]

        return labels



    def index_from_scratch(self):
        index = []
        for i, fname in enumerate(os.listdir(self.path)):
            if not (fname.endswith('.jpg') or fname.endswith('.png') or fname.endswith('.jpeg')):
                continue

            im_path = os.path.join(self.path, fname)
            try:
                labels = self._extract_objects(im_path)
                index.append((i, fname, labels))
            except Exception as e:
                print(e)
                continue

        json.dump(index,open(os.path.join('index.json'), 'w'))


    def index_update(self):
        index = []
        for i, fname in enumerate(os.listdir(self.path)):
            if not (fname.endswith('.jpg') or fname.endswith('.png') or fname.endswith('.jpeg')):
                continue

            im_path = os.path.join(self.path, fname)
            try:
                labels = self._extract_objects(im_path)
                index.append((i, fname, labels))
            except Exception as e:
                print(e)
                continue

        json.dump(index,open(os.path.join('index.json'), 'w'))



def main(args):
    images_dir = "collected_images/"
    from_scratch = True

    try:
        opts, args = getopt.getopt(args, "i:s:", ["images_dir=","from_scratch="])
    except:
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-i", "--images_dir"):
            images_dir = arg
        elif opt in ("-s", "--from_scratch"):
            from_scratch = arg == 'y'


    indexer = Indexer(images_dir)
    if from_scratch:
        indexer.index_from_scratch()
    else:
        indexer.index_update()



if __name__ == '__main__':
    main(sys.argv[1:])