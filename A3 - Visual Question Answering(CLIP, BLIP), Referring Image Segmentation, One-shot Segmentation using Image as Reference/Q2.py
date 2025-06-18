#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:





# In[2]:


get_ipython().system('pip install ftfy packaging regex tqdm torch torchvision huggingface_hub safetensors timm open_clip_torch fairscale==0.4.4 pycocoevalcap # requests filelock mmcv-full==1.3.12 mmsegmentation pycocotools==2.0.2 opencv-python==4.5.3.56')


# In[3]:


import numpy as np
import matplotlib.pyplot as plt
from transformers import CLIPProcessor, CLIPModel, BlipProcessor, BlipForQuestionAnswering, BlipForConditionalGeneration
import torch
from PIL import Image
import requests
import torch.nn.functional as F


# In[4]:


image_path = "/kaggle/input/sample_image.jpg"
image = Image.open(image_path)


# In[ ]:





# # 2. Visual QA

# In[5]:


device = "cuda" if torch.cuda.is_available() else "cpu"


# In[6]:


blip_model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base").to(device)
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")


# In[7]:


image_2 = image.convert("RGB")
question = "Where is the dog present in the image?"


# In[8]:


inputs_3 = blip_processor(image_2, question, return_tensors="pt").to(device)

with torch.no_grad():
    output_3 = blip_model.generate(**inputs_3)

answer_3 = blip_processor.decode(output_3[0], skip_special_tokens=True)
print("Question:", question)
print("Reply:", answer_3)


# In[9]:


question_2 = "Where is the man present in the image?"
inputs_3_2 = blip_processor(image_2, question_2, return_tensors="pt").to(device)

with torch.no_grad():
    output_3_2 = blip_model.generate(**inputs_3_2)

answer_3_2 = blip_processor.decode(output_3_2[0], skip_special_tokens=True)
print("Question:", question_2)
print("Reply:", answer_3_2)


# In[ ]:




