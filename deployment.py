import numpy as np
import sys
import os
import torch
from flask import Flask, request, jsonify
import json
import argparse
from model import UNetWrapper
from minio import Minio
from myutil.logconf import logging
import torch
import torch.nn as nn
import torch.optim

import torch
import torch.cuda
import torch.nn.functional as F
from torch.utils.data import Dataset

from torch.optim import SGD, Adam
from torch.utils.data import DataLoader
import zipfile
import io
import SimpleITK as sitk
import glob
from myutil.util import enumerateWithEstimate
from collections import namedtuple
import uuid
import datetime
from PIL import Image
import shutil
import random
PredictTuple = namedtuple('PredictTuple', 'index, image')

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
# log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)

app = Flask(__name__)

class PredictionDataset(Dataset):
    def __init__(self, data=None):
        self.data= data
    def __len__(self):
        return len(self.data)

    def __getitem__(self, ndx): 
        return self.data[ndx]    


class Server:
    def __init__(self, sys_argv=None):
        # call the parent class constructor
        if sys_argv is None:
            sys_argv = sys.argv[1:]

        parser = argparse.ArgumentParser()
        parser.add_argument('--batch-size',
            help='Batch size to use for training',
            default=16,
            type=int,
        )
        parser.add_argument('--num-workers',
            help='Number of worker processes for background data loading',
            default=8,
            type=int,
        )
        parser.add_argument('--model',
            help='Model Path for Prediction',
            default='seg_2024-02-09_11.00.01_reduceFN.best.state'
        )
        parser.add_argument('--url',
            help='Url for the minio server',
            default='localhost:9005'
        )
        parser.add_argument('--access-key',
            help='access key for the minio server',
            default='minioadmin'
        )
        parser.add_argument('--secret-key',
            help="secret-key for the minio",
            default='minioadmin',
        )
        parser.add_argument('--backet-name',
            help="backet name for the minio",
            default='cts',
        )
        parser.add_argument('--backet-name-out',
            help="backet name for the minio store the output image",
            default='ctsout',
        )
        self.cli_args = parser.parse_args(sys_argv)
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")


        # load the model state from the file
        self.segmentation_model = self.initModel()
       


        # create a Minio object with your credentials
        self.minio_client = self.initMinio()

    
    def getitem_fullSlice(self, hu, slice_ndx):
        #ct_t = torch.zeros((self.contextSlices_count * 2 + 1, 512, 512))

        ct_t = torch.zeros((3 * 2 + 1, 512, 512))

        start_ndx = slice_ndx - 3
        end_ndx = slice_ndx + 3 + 1
        for i, context_ndx in enumerate(range(start_ndx, end_ndx)):
            context_ndx = max(context_ndx, 0)
            context_ndx = min(context_ndx, hu.shape[0] - 1)
            ct_t[i] = torch.from_numpy(hu[context_ndx].astype(np.float32))

        # CTs are natively expressed in https://en.wikipedia.org/wiki/Hounsfield_scale
        # HU are scaled oddly, with 0 g/cc (air, approximately) being -1000 and 1 g/cc (water) being 0.
        # The lower bound gets rid of negative density stuff used to indicate out-of-FOV
        # The upper bound nukes any weird hotspots and clamps bone down
        ct_t.clamp_(-1000, 1000)

        #pos_t = torch.from_numpy(ct.positive_mask[slice_ndx]).unsqueeze(0)

        return ct_t    
    def initMinio(self):
        return Minio(self.cli_args.url, access_key=self.cli_args.access_key, secret_key=self.cli_args.secret_key, secure=False)
    
    def initModel(self):
        segmentation_model = UNetWrapper(
            in_channels=7,
            n_classes=1,
            depth=3,
            wf=4,
            padding=True,
            batch_norm=True,
            up_mode='upconv',
        )

       
        if self.cli_args.model:
            d = torch.load(self.cli_args.model, map_location='cpu')
            segmentation_model.load_state_dict(d['model_state'])
            

        if self.use_cuda:
            log.info("Using CUDA; {} devices.".format(torch.cuda.device_count()))
            if torch.cuda.device_count() > 1:
                segmentation_model = nn.DataParallel(segmentation_model)
                augmentation_model = nn.DataParallel(augmentation_model)
            segmentation_model = segmentation_model.to(self.device)
            augmentation_model = augmentation_model.to(self.device)
            segmentation_model.eval()
        return segmentation_model
    
    def getObject(self, name):
        print("getting object")
        response = self.minio_client.get_object(self.cli_args.backet_name, name)
        # read the data from the response object
        
        data = response.read()
        print("done getting the object")
        # close the response object
        response.close()
        # create a BytesIO object from the data
        data_stream = io.BytesIO(data)

        # create a ZipFile object from the data stream
        zip_file = zipfile.ZipFile(data_stream)
        random_number = random.randint(10**9, 10**10 - 1)
        fileName = str (uuid.uuid4 ()) + str(random_number) 
        dateString = datetime.date.today ().strftime("%Y-%m-%d-%H-%M-%S")
        self.objectName = dateString + fileName
        # extract the .mhd and .raw files to a temporary folder
        zip_file.extractall(self.objectName)

    def initDataLoader(self, data):
        batch_size = self.cli_args.batch_size
        if self.use_cuda:
            batch_size *= torch.cuda.device_count()
        dataloader = DataLoader(
            data,
            batch_size=batch_size,
            num_workers=self.cli_args.num_workers,
            pin_memory=self.use_cuda,
        )

        return dataloader   

    


    def getImageNameGroup(self, name):
        fileName = str (uuid.uuid4 ()) + name 
        dateString = datetime.date.today ().strftime ("%Y-%m-%d")
        objectName = dateString + "/" + fileName
        return objectName
    
    def rescale(self, arr):
        arr_min = arr.min()
        arr_max = arr.max()
        return (arr - arr_min) / (arr_max - arr_min)
    
    def uplaodImage(self, name, obj):
        image = self.rescale(obj) * 255.0
        img = Image.fromarray(image.astype(np.uint8), 'RGB')
        out_img = io.BytesIO()
        img.save(out_img, format='png')
        out_img.seek(0)
        length = out_img.getbuffer().nbytes
        self.minio_client.put_object(self.cli_args.backet_name_out, name, out_img, length,"image/png" )
        # os.remove("temp_file")
        return name    

    def handle_post_predict(self):
        data = request.get_json()
        fileName = data.get('fileName')
        self.getObject(fileName)
        mhd_path = glob.glob(
            '{}/**.mhd'.format(self.objectName),  recursive=True
        )[0]

        ct_mhd = sitk.ReadImage(mhd_path)
        self.hu_a = np.array(sitk.GetArrayFromImage(ct_mhd), dtype=np.float32)
        images_all = np.empty((self.hu_a.shape[0], 512, 512, 3), dtype=np.float32)
        #delete the temporary file 
        shutil.rmtree(self.objectName)
        data_all = []
       
        for slice_ndx in range(self.hu_a.shape[0]):
                #ct_ndx = slice_ndx * (hu_a.shape[0] - 1) // 5
                ct_t = self.getitem_fullSlice(self.hu_a, slice_ndx)

                input_g = ct_t.to(self.device)
                data_all.append(
                    PredictTuple(slice_ndx, input_g)
                )

        self.dataloader = self.initDataLoader(data_all) 

        batch_iter = enumerateWithEstimate(
            self.dataloader,
            "Predicting",
            start_ndx=self.dataloader.num_workers,
        )       

        universalName= self.getImageNameGroup("output")
        indices= []
        output = {}
        output["name"] = universalName
        output["size"] = self.hu_a.shape[0]
        for batch_ndx, batch_tup in batch_iter: 
                slice_ndx_list, input_g = batch_tup
                prediction_g = self.segmentation_model(input_g)[0]
                prediction_a = prediction_g.to('cpu').detach().numpy()[0] > 0.5
                for i, slice_ndx in enumerate(slice_ndx_list):
                    if np.any(prediction_a): # Check if any element in the i-th prediction array is True
                        indices.append(slice_ndx.item()) # Append the corresponding slice index to the list
                    ct_t = self.getitem_fullSlice(self.hu_a, slice_ndx)
                    ct_t[:-1,:,:] /= 2000
                    ct_t[:-1,:,:] += 0.5
                    ctSlice_a = ct_t[3].numpy()
                    image_a = np.zeros((512, 512, 3), dtype=np.float32)
                    image_a[:,:,:] = ctSlice_a.reshape((512,512,1))
                    image_a[:,:,1] += prediction_a 
                    image_a *= 0.5
                    image_a.clip(0, 1, image_a)
                    images_all[slice_ndx]= image_a
                    name= universalName + '_' + str(slice_ndx.item()) + '.png'
                    self.uplaodImage(name, image_a) 
            
        
        output["indices"] = indices       
        
        
        
        return jsonify(output)
    
    def main(self):
        # create an instance of the class with the file name
        # register the handle_post method as the route for the Flask app
        app.route("/predict", methods=["POST"])(self.handle_post_predict)
        # run the Flask app
        app.run(host='0.0.0.0', port=8999)

if __name__ == '__main__':
    Server().main()


