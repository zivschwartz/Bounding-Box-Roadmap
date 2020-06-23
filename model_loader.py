"""
You need to implement all four functions in this file and also put your team info as a variable
Then you should submit the python file with your model class, the state_dict, and this file
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

# import your model class
from models import * 
from utils.datasets import *
from utils.utils import *

# Put your transform function here, we will use it for our dataloader
# For bounding boxes task
def get_transform_task1(): 

    transform = torchvision.transforms.Compose([
                    torchvision.transforms.Resize([400,267]),
                    torchvision.transforms.ToTensor()
                    ])
    return transform

# For road map task
def get_transform_task2(): 

    return torchvision.transforms.ToTensor()

class ModelLoader():
    # Fill the information for your team
    team_name = 'Alfs Angels'
    team_member = ['Atul Gandhi', 'Elliot Silva', 'Ziv Schwartz']
    contact_email = 'ag6776/egs345/zs1349@nyu.edu'

    def __init__(self, model_file='put_your_model_file(or files)_name_here'):
        # You should 
        #       1. create the model object
        #       2. load your state_dict
        #       3. call cuda()
        # self.model = ...
        # 
        self.img_size = 320
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        cfg = 'cfg/yolov3-spp.cfg'
        weights = 'best.pt'

        self.model = Darknet(cfg, self.img_size).to(self.device).eval()

        self.model.load_state_dict(torch.load(weights, map_location=self.device)['model'])

        img = torch.zeros((1, 3, self.img_size, self.img_size), device=self.device)  # init img
        _ = self.model(img.float()) if self.device.type != 'cpu' else None  # run once


    def get_bounding_boxes(self, samples):
        # samples is a cuda tensor with size [batch_size, 6, 3, 256, 306]
        # You need to return a tuple with size 'batch_size' and each element is a cuda tensor [N, 2, 4]
        # where N is the number of object

        batch_tensor=samples
        batch_tensor[:,3:] = torch.flip(batch_tensor[:,3:],[3])
        row1 = torch.cat((batch_tensor[:,0],batch_tensor[:,1],batch_tensor[:,2]), dim=3, out=None)
        row2 = torch.cat((batch_tensor[:,3],batch_tensor[:,4],batch_tensor[:,5]), dim=3, out=None)
        stitched = torch.cat((row1,row2),dim=2, out = None).transpose(2,3).to(self.device)

        all_pred = []
        
        for img in stitched:
            
            img = img.cpu().numpy()[:, :, ::-1].transpose(1, 2, 0)
            img = letterbox(img, new_shape=self.img_size)[0]

            # # Convert
            img = img[:, :, ::-1].transpose(2, 0, 1)  
            img = np.ascontiguousarray(img)
            img = torch.from_numpy(img).to(self.device)

            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            pred = self.model(img)[0]

            pred = non_max_suppression(pred)

            boxes = []

            for i, det in enumerate(pred):
                if det is not None and len(det):

                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], (800,800)).round()
                    for box in det:
                        coords = box[:4]
                        print(coords)
                        coords /= 10
                        coords -= 40

                        x1, y1, x3, y3  = coords

                        x2 = x1
                        x4 = x3

                        y2 = y3
                        y4 = y1

                        new_box = [[x1.item(),x2.item(),x3.item(),x4.item()],[y1.item(),y2.item(),y3.item(),y4.item()]]
                        boxes.append(new_box)
                else:
                    boxes.append([[0,0,0,0],[0,0,0,0]])

            all_pred.append(torch.tensor(boxes))


        return all_pred

    def get_binary_road_map(self, samples):
        # samples is a cuda tensor with size [batch_size, 6, 3, 256, 306]
        # You need to return a cuda tensor with size [batch_size, 800, 800]

        all_roads_ = torch.load('all_roads')
        binary = [i*1.0 for i in all_roads_]
        binary_mean = torch.mean(torch.stack(binary), axis=0) > 0.5

        return binary_mean
