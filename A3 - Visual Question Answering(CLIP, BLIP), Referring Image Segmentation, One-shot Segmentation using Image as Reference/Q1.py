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


# # 1. CLIP vs CLIPS (Contrastive Language-Image Pretraining)

# In[2]:


get_ipython().system('pip install ftfy packaging regex tqdm torch torchvision huggingface_hub safetensors timm open_clip_torch # fairscale==0.4.4 pycocoevalcap requests filelock mmcv-full==1.3.12 mmsegmentation pycocotools==2.0.2 opencv-python==4.5.3.56')


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
import os


# In[5]:


clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


# In[6]:


image_path = "/kaggle/input/sample_image.jpg"
image = Image.open(image_path)

texts = [
    "a person walking with a dog",
    "a man and his pet",
    "a human and canine companion",
    "a person holding a dog lika a baby",
    "a man playing with his dog",
    "a person and dog resting",
    "a human holding a pet",
    "a man and his loyal companion",
    "a person with their furry friend",
    "a human and dog enjoying home time"
]

inputs = clip_processor(text=texts, images=image, return_tensors="pt", padding=True)


# In[7]:


plt.imshow(Image.open(image_path))
plt.axis("off")
plt.title("Given Image")
plt.show()


# In[8]:


outputs = clip_model(**inputs)
logits_per_image = outputs.logits_per_image
# probs = logits_per_image.softmax(dim=1)
probs = logits_per_image


# In[9]:


print("Similarity scores for each text description:")
for text, prob in zip(texts, probs[0]):
    print(f"{text}: {prob.item():.4f}")


# In[10]:


clips_model, clips_processor = create_model_from_pretrained('hf-hub:UCSC-VLAA/ViT-L-14-CLIPS-224-Recap-DataComp-1B')
clips_tokenizer = get_tokenizer('hf-hub:UCSC-VLAA/ViT-L-14-CLIPS-Recap-DataComp-1B')


# In[11]:


inputs_2 = clips_processor(image).unsqueeze(0)
text_tokens = clips_tokenizer(texts, context_length=clips_model.context_length)


# In[12]:


with torch.no_grad():
    image_features = clips_model.encode_image(inputs_2)
    text_features = clips_model.encode_text(text_tokens)

image_features = F.normalize(image_features, dim=-1)
text_features = F.normalize(text_features, dim=-1)


# In[13]:


# text_probs = (100 * image_features @ text_features.T).softmax(dim=-1)
text_probs = (100 * image_features @ text_features.T)

print("CLIPS Similarity scores for each text:")
for text, prob in zip(texts, text_probs[0]):
    print(f"{text}: {prob.item():.4f}")


# In[ ]:




