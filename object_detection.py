from gluoncv import model_zoo, data, utils
from matplotlib import pyplot as plt

net = model_zoo.get_model('yolo3_darknet53_voc', pretrained=True)

im_fname = utils.download('https://raw.githubusercontent.com/zhreshold/' +
                          'mxnet-ssd/master/data/demo/dog.jpg',
                          path='input_images/dog.jpg')
x, img = data.transforms.presets.yolo.load_test(im_fname, short=512)
print('Shape of pre-processed image:', x.shape)

class_IDs, scores, bounding_boxs = net(x)

# [net.classes[int(i)] if i!=-1 else -1 for i in class_IDs[0].asnumpy().flat]
labels = [net.classes[int(i)] for i in class_IDs[0].asnumpy().flat if i!=-1 ]

print(labels)

# ax = utils.viz.plot_bbox(img, bounding_boxs[0], scores[0],
#                          class_IDs[0], class_names=net.classes)
# plt.show()