from dataclasses import field
from turtle import width
import streamlit as st
from PIL import Image

import os
from glob import glob
from typing import Optional

import cv2
import numpy as np
import torch
import yaml
from fire import Fire
from tqdm import tqdm

from aug import get_normalize
from models.networks import get_generator


class Predictor:
    def __init__(self, weights_path: str, model_name: str = ''):
        with open('config/config.yaml') as cfg:
            config = yaml.load(cfg, Loader=yaml.FullLoader)
        model = get_generator(model_name or config['model'])
        model.load_state_dict(torch.load(weights_path)['model'])
        self.model = model.cuda()
        self.model.train(True)
        # GAN inference should be in train mode to use actual stats in norm layers,
        # it's not a bug
        self.normalize_fn = get_normalize()

    @staticmethod
    def _array_to_batch(x):
        x = np.transpose(x, (2, 0, 1))
        x = np.expand_dims(x, 0)
        return torch.from_numpy(x)

    def _preprocess(self, x: np.ndarray, mask: Optional[np.ndarray]):
        x, _ = self.normalize_fn(x, x)
        if mask is None:
            mask = np.ones_like(x, dtype=np.float32)
        else:
            mask = np.round(mask.astype('float32') / 255)

        h, w, _ = x.shape
        block_size = 32
        min_height = (h // block_size + 1) * block_size
        min_width = (w // block_size + 1) * block_size

        pad_params = {'mode': 'constant',
                      'constant_values': 0,
                      'pad_width': ((0, min_height - h), (0, min_width - w), (0, 0))
                      }
        x = np.pad(x, **pad_params)
        mask = np.pad(mask, **pad_params)

        return map(self._array_to_batch, (x, mask)), h, w

    @staticmethod
    def _postprocess(x: torch.Tensor) -> np.ndarray:
        x, = x
        x = x.detach().cpu().float().numpy()
        x = (np.transpose(x, (1, 2, 0)) + 1) / 2.0 * 255.0
        return x.astype('uint8')

    def __call__(self, img: np.ndarray, mask: Optional[np.ndarray], ignore_mask=True) -> np.ndarray:
        (img, mask), h, w = self._preprocess(img, mask)
        with torch.no_grad():
            inputs = [img.cuda()]
            if not ignore_mask:
                inputs += [mask]
            pred = self.model(*inputs)
        return self._postprocess(pred)[:h, :w, :]

# predict class end

def process_video(pairs, predictor, output_dir):
    for video_filepath, mask in tqdm(pairs):
        video_filename = os.path.basename(video_filepath)
        output_filepath = os.path.join(output_dir, os.path.splitext(video_filename)[0]+'_deblur.mp4')
        video_in = cv2.VideoCapture(video_filepath)
        fps = video_in.get(cv2.CAP_PROP_FPS)
        width = int(video_in.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video_in.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frame_num = int(video_in.get(cv2.CAP_PROP_FRAME_COUNT))
        video_out = cv2.VideoWriter(output_filepath, cv2.VideoWriter_fourcc(*'MP4V'), fps, (width, height))
        tqdm.write(f'process {video_filepath} to {output_filepath}, {fps}fps, resolution: {width}x{height}')
        for frame_num in tqdm(range(total_frame_num), desc=video_filename):
            res, img = video_in.read()
            if not res:
                break
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pred = predictor(img, mask)
            pred = cv2.cvtColor(pred, cv2.COLOR_RGB2BGR)
            video_out.write(pred)


# def brighten_image(image, amount):
#     img_bright = cv2.convertScaleAbs(image, beta=amount)
#     return img_bright


# def blur_image(image, amount):
#     blur_img = cv2.GaussianBlur(image, (11, 11), amount)
#     return blur_img


# def enhance_details(img):
#     hdr = cv2.detailEnhance(img, sigma_s=12, sigma_r=0.15)
#     return hdr



def main():

  weights_path='fpn_inception.h5'
  predictor = Predictor(weights_path=weights_path)

  image = Image.open("./logo.png") #Brand logo image (optional)
  image = np.array(image)

  #Create two columns with different width
  col1, col2 = st.columns( [0.8, 0.2])
  with col1:               # To display the header text using css style
      st.markdown(""" <style> .font {
      font-size:35px ; font-family: 'Cooper Black'; color: #3EC70B;} 
      </style> """, unsafe_allow_html=True)
      st.markdown('<p class="font">Upload your photo here...</p>', unsafe_allow_html=True)
      
  with col2:               # To display brand logo
      st.image(image,  width=150)


      #Add a header and expander in side bar
  st.sidebar.markdown('<p class="font">Naver Team</p>', unsafe_allow_html=True)
  with st.sidebar.expander("About the App"):
      st.write("""
        Licence Deblurring app \n  \nThis app was created by Naver Team special for Capstone Project
      """)


  #Add file uploader to allow users to upload photos
  uploaded_file = st.file_uploader("", type=['jpg','png','jpeg'])

  # Add 'before' and 'after' columns

  if uploaded_file is not None:
      uploaded_image = Image.open(uploaded_file)
      original_image = np.array(uploaded_image)

      col1, col2 = st.columns( [0.5, 0.5])
      with col1:
          st.markdown('<p style="text-align: center;">Before</p>',unsafe_allow_html=True)
          st.image(original_image,width=300)  
   
      result = st.button("Run on image")
      
      with col2:
          st.markdown('<p style="text-align: center;">After</p>',unsafe_allow_html=True)

          if result:
             
             img = cv2.cvtColor(original_image,cv2.COLOR_BGR2RGB)

             pred = predictor(img, original_image)
             pred = cv2.cvtColor(pred, cv2.COLOR_RGB2BGR)
             st.image(pred, width=300)

             st.write("Done!!!")
             
     

if __name__ == "__main__":
  main()
