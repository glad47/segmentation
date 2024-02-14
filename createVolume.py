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
import pywavefront
import uuid
import datetime
import pyvista as pv
import numpy as np
import sys
import os
import io
import torch
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from minio import Minio
from PIL import Image
import imagecodecs
from matplotlib import cm

PredictTuple = namedtuple('PredictTuple', 'index, image')


class Test:
    
    def __init__(self):
        mhd_path = glob.glob(
            'temp/**.mhd',  recursive=True
        )[0]

        self.ct_mhd = sitk.ReadImage(mhd_path)
        self.hu_a = np.array(sitk.GetArrayFromImage(self.ct_mhd), dtype=np.float32)
        self.minio_client = self.initMinio()

    def initMinio(self):
        return Minio('localhost:9005', 'minioadmin', 'minioadmin', secure=False)
    
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
        self.minio_client.put_object('ctsout', name, out_img, length,"image/png" )
        # os.remove("temp_file")
        return name
    
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
    
    def initDataLoader(self, data):
        dataloader = DataLoader(
            data,
            batch_size=8,
            num_workers=4
        )

        return dataloader


    def main(self):
        
        
        images_all = np.empty((self.hu_a.shape[0], 512, 512, 3), dtype=np.float32)
        data_all = []
        for slice_ndx in range(self.hu_a.shape[0]):
                #ct_ndx = slice_ndx * (hu_a.shape[0] - 1) // 5
                ct_t = self.getitem_fullSlice(self.hu_a, slice_ndx)

                input_g = ct_t
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
        for batch_ndx, batch_tup in batch_iter: 
                slice_ndx_list, input_g = batch_tup
                prediction_g = torch.rand(len(batch_tup))
                prediction_a = prediction_g.to('cpu').detach().numpy()[0] > 0.5
                for slice_ndx in slice_ndx_list:
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

        # image_3d = np.array(images_all)          
        # flattened_volume = image_3d.flatten(order='C').astype('short')
        # # Assuming 'stacked_volume' is your 4D array (z, x, y, channels)
        # itk_image = sitk.GetImageFromArray(image_3d)

        # # Set metadata (spacing, origin, etc.) if needed
        # itk_image.SetSpacing(self.ct_mhd.GetSpacing())
        # itk_image.SetOrigin(self.ct_mhd.GetOrigin())

        # # Save the ITK image to an .mhd file
        # sitk.WriteImage(itk_image, "OutputFile.mhd")

        # # Save the flattened array to a .raw file
        # flattened_volume.tofile("OutputFile.raw")

        # # Reads the image using SimpleITK
        # itkimage = sitk.ReadImage("1.3.6.1.4.1.14519.5.2.1.6279.6001.105756658031515062000744821260.mhd")

        # # Convert the image to a numpy array first and then shuffle the dimensions to get axis in the order z,y,x
        # ct_scan = sitk.GetArrayFromImage(itkimage)

        # # Get the first image and plot it
        # # im1 = ct_scan[0,:,:]
        # plt.imshow(ct_scan, cmap=plt.cm.gray_r)
        # plt.show()


if __name__ == '__main__':
    Test().main()        