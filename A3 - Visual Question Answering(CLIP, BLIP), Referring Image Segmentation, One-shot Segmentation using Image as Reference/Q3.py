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


# In[2]:


get_ipython().system('pip install ftfy packaging regex tqdm torch torchvision huggingface_hub safetensors timm open_clip_torch fairscale==0.4.4 pycocoevalcap #requests filelock mmcv-full==1.3.12 mmsegmentation pycocotools==2.0.2 opencv-python==4.5.3.56')


# In[3]:


get_ipython().system('pip install open_clip_torch')


# In[4]:


import numpy as np
import matplotlib.pyplot as plt
from transformers import CLIPProcessor, CLIPModel, BlipProcessor, BlipForQuestionAnswering, BlipForConditionalGeneration
import torch
from PIL import Image
import requests
from open_clip import create_model_from_pretrained, get_tokenizer
import torch.nn.functional as F


# In[5]:


clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


# In[6]:


clips_model, clips_processor = create_model_from_pretrained('hf-hub:UCSC-VLAA/ViT-L-14-CLIPS-224-Recap-DataComp-1B')
clips_tokenizer = get_tokenizer('hf-hub:UCSC-VLAA/ViT-L-14-CLIPS-Recap-DataComp-1B')


# In[ ]:





# # 3. BLIP vs CLIP

# In[7]:


image_dir = "/kaggle/input/samples-20250417T083449Z-001/samples/"
device = "cuda" if torch.cuda.is_available() else "cpu"


# In[8]:


blipc_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
blipc_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")


# In[9]:


all_captions = {}

for filename in os.listdir(image_dir):
    image_path = os.path.join(image_dir, filename)
    cur_image = Image.open(image_path).convert("RGB")

    cur_inputs = blipc_processor(images=cur_image, return_tensors="pt").to(device)
    with torch.no_grad():
        cur_output = blipc_model.generate(**cur_inputs)

    cur_caption = blipc_processor.decode(cur_output[0], skip_special_tokens=True)
    all_captions[image_path] = cur_caption


# In[10]:


for image_path in all_captions.keys():
    print("Caption:", all_captions[image_path])
    plt.imshow(Image.open(image_path))
    plt.axis("off")
    plt.title(image_path.split("/")[-1])
    plt.show()

    print()


# In[11]:


clip_eval = {}

for image_path in all_captions.keys():
    cur_image = Image.open(image_path)
    cur_input = clip_processor(text=all_captions[image_path] , images=cur_image, return_tensors="pt", padding=True)

    outputs = clip_model(**cur_input)
    logits_per_image = outputs.logits_per_image
    
    clip_eval[image_path] = logits_per_image[0][0].item()


# In[12]:


for image_path in all_captions.keys():
    print("Caption:", all_captions[image_path])
    print("CLIP Eval Score:", clip_eval[image_path])
    plt.imshow(Image.open(image_path))
    plt.axis("off")
    plt.title(image_path.split("/")[-1])
    plt.show()

    print()


# In[13]:


clips_eval = {}

for image_path in all_captions.keys():
    cur_image = Image.open(image_path)
    cur_input = clips_processor(cur_image).unsqueeze(0)
    text_tokens = clips_tokenizer([all_captions[image_path]], context_length=clips_model.context_length)

    with torch.no_grad():
        image_features = clips_model.encode_image(cur_input)
        text_features = clips_model.encode_text(text_tokens)

    image_features = F.normalize(image_features, dim=-1)
    text_features = F.normalize(text_features, dim=-1)

    text_probs = (100 * image_features @ text_features.T)

    clips_eval[image_path] = text_probs[0][0].item()


# In[14]:


for image_path in all_captions.keys():
    print("Caption:", all_captions[image_path])
    print("CLIPS Eval Score:", clips_eval[image_path])
    plt.imshow(Image.open(image_path))
    plt.axis("off")
    plt.title(image_path.split("/")[-1])
    plt.show()

    print()


# In[ ]:




