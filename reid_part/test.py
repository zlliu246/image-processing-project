import os
from run_market import load_model, check_images
import market1501_dataset

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

model = load_model('reid_model_market.h5')

test_ids = market1501_dataset.get_id('datasets/Market-1501', 'bounding_box_test')
pair = market1501_dataset.get_pair('datasets/Market-1501', 'bounding_box_test', test_ids, True)
print(check_images(model, pair[0], pair[1]))