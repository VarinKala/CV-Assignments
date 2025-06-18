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


get_ipython().system('pip install ftfy packaging regex tqdm torch torchvision huggingface_hub safetensors timm open_clip_torch fairscale==0.4.4 pycocoevalcap requests filelock mmcv-full==1.3.12 mmsegmentation pycocotools==2.0.2 opencv-python==4.5.3.56')


# In[3]:


get_ipython().system('pip install h5py tokenizers==0.8.1rc1')
# !pip install tokenizers


# In[5]:


import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image
import requests
import torch.nn.functional as F


# In[6]:


device = "cuda" if torch.cuda.is_available() else "cpu"


# In[ ]:





# # 4. Referring Image Segmentation (RIS)

# In[7]:


references = {}

with open("/kaggle/input/reference.txt", "r") as f:
    references_text = f.readlines()

for line in references_text:
    image, desc = line.split(":")
    image, desc = image.strip(), desc.strip().strip('"')

    references[image] = desc


# In[8]:


print(references)


# In[9]:


get_ipython().system('git clone https://github.com/yz93/LAVT-RIS.git')


# In[10]:


get_ipython().run_line_magic('cd', 'LAVT-RIS')


# In[11]:


get_ipython().system('pip install mmcv-full==1.3.12')


# In[12]:


get_ipython().system('pip install mmsegmentation==0.17.0')


# In[13]:


get_ipython().system('pip install pycocotools==2.0.2')


# In[ ]:


# ## REFERENCE: LAVT-RIS Github
# ## demo_inference.py

# image_path = './demo/demo.jpg'
# sentence = 'the most handsome guy'
# weights = './checkpoints/refcoco.pth'
# device = 'cuda:0'

# # pre-process the input image
# from PIL import Image
# import torchvision.transforms as T
# import numpy as np
# img = Image.open(image_path).convert("RGB")
# img_ndarray = np.array(img)  # (orig_h, orig_w, 3); for visualization
# original_w, original_h = img.size  # PIL .size returns width first and height second

# image_transforms = T.Compose(
#     [
#      T.Resize(480),
#      T.ToTensor(),
#      T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ]
# )

# img = image_transforms(img).unsqueeze(0)  # (1, 3, 480, 480)
# img = img.to(device)  # for inference (input)

# # pre-process the raw sentence
# from bert.tokenization_bert import BertTokenizer
# import torch
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# sentence_tokenized = tokenizer.encode(text=sentence, add_special_tokens=True)
# sentence_tokenized = sentence_tokenized[:20]  # if the sentence is longer than 20, then this truncates it to 20 words
# # pad the tokenized sentence
# padded_sent_toks = [0] * 20
# padded_sent_toks[:len(sentence_tokenized)] = sentence_tokenized
# # create a sentence token mask: 1 for real words; 0 for padded tokens
# attention_mask = [0] * 20
# attention_mask[:len(sentence_tokenized)] = [1]*len(sentence_tokenized)
# # convert lists to tensors
# padded_sent_toks = torch.tensor(padded_sent_toks).unsqueeze(0)  # (1, 20)
# attention_mask = torch.tensor(attention_mask).unsqueeze(0)  # (1, 20)
# padded_sent_toks = padded_sent_toks.to(device)  # for inference (input)
# attention_mask = attention_mask.to(device)  # for inference (input)

# # initialize model and load weights
# from bert.modeling_bert import BertModel
# from lib import segmentation

# # construct a mini args class; like from a config file


# class args:
#     swin_type = 'base'
#     window12 = True
#     mha = ''
#     fusion_drop = 0.0


# single_model = segmentation.__dict__['lavt'](pretrained='', args=args)
# single_model.to(device)
# model_class = BertModel
# single_bert_model = model_class.from_pretrained('bert-base-uncased')
# single_bert_model.pooler = None

# checkpoint = torch.load(weights, map_location='cpu')
# single_bert_model.load_state_dict(checkpoint['bert_model'])
# single_model.load_state_dict(checkpoint['model'])
# model = single_model.to(device)
# bert_model = single_bert_model.to(device)


# # inference
# import torch.nn.functional as F
# last_hidden_states = bert_model(padded_sent_toks, attention_mask=attention_mask)[0]
# embedding = last_hidden_states.permute(0, 2, 1)
# output = model(img, embedding, l_mask=attention_mask.unsqueeze(-1))
# output = output.argmax(1, keepdim=True)  # (1, 1, 480, 480)
# output = F.interpolate(output.float(), (original_h, original_w))  # 'nearest'; resize to the original image size
# output = output.squeeze()  # (orig_h, orig_w)
# output = output.cpu().data.numpy()  # (orig_h, orig_w)


# # show/save results
# def overlay_davis(image, mask, colors=[[0, 0, 0], [255, 0, 0]], cscale=1, alpha=0.4):
#     from scipy.ndimage.morphology import binary_dilation

#     colors = np.reshape(colors, (-1, 3))
#     colors = np.atleast_2d(colors) * cscale

#     im_overlay = image.copy()
#     object_ids = np.unique(mask)

#     for object_id in object_ids[1:]:
#         # Overlay color on  binary mask
#         foreground = image*alpha + np.ones(image.shape)*(1-alpha) * np.array(colors[object_id])
#         binary_mask = mask == object_id

#         # Compose image
#         im_overlay[binary_mask] = foreground[binary_mask]

#         # countours = skimage.morphology.binary.binary_dilation(binary_mask) - binary_mask
#         countours = binary_dilation(binary_mask) ^ binary_mask
#         # countours = cv2.dilate(binary_mask, cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))) - binary_mask
#         im_overlay[countours, :] = 0

#     return im_overlay.astype(image.dtype)


# output = output.astype(np.uint8)  # (orig_h, orig_w), np.uint8
# # Overlay the mask on the image
# visualization = overlay_davis(img_ndarray, output)  # red
# visualization = Image.fromarray(visualization)
# # show the visualization
# #visualization.show()
# # Save the visualization
# visualization.save('./demo/demo_result.jpg')


# In[14]:


from PIL import Image
import torchvision.transforms as T
import numpy as np
from bert.tokenization_bert import BertTokenizer
import torch
# from scipy.ndimage.morphology import binary_dilation
from scipy.ndimage import binary_dilation

samples_dir = "/kaggle/input/samples-20250417T083449Z-001/samples/"


# In[72]:


get_ipython().system('find . -name "refcoco.pth"')


# In[15]:


from bert.modeling_bert import BertModel
from lib import segmentation

weights = '/kaggle/input/refcoco.pth'

class args:
    swin_type = 'base'
    window12 = True
    mha = ''
    fusion_drop = 0.0


single_model = segmentation.__dict__['lavt'](pretrained='', args=args)
single_model.to(device)
model_class = BertModel
single_bert_model = model_class.from_pretrained('bert-base-uncased')
single_bert_model.pooler = None

checkpoint = torch.load(weights, map_location='cpu')
single_bert_model.load_state_dict(checkpoint['bert_model'])
single_model.load_state_dict(checkpoint['model'])
model = single_model.to(device)
bert_model = single_bert_model.to(device)


# In[ ]:





# In[16]:


segmented_samples = {}
y1_features = {}

for filename in os.listdir(samples_dir):
    image_path = os.path.join(samples_dir, filename)
    sentence = references[filename]


    img = Image.open(image_path).convert("RGB")
    img_ndarray = np.array(img)
    original_w, original_h = img.size
    
    image_transforms = T.Compose(
        [
         T.Resize(480),
         T.ToTensor(),
         T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    )
    
    img = image_transforms(img).unsqueeze(0)
    img = img.to(device)
    
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    sentence_tokenized = tokenizer.encode(text=sentence, add_special_tokens=True)
    sentence_tokenized = sentence_tokenized[:20]

    padded_sent_toks = [0] * 20
    padded_sent_toks[:len(sentence_tokenized)] = sentence_tokenized

    attention_mask = [0] * 20
    attention_mask[:len(sentence_tokenized)] = [1]*len(sentence_tokenized)

    padded_sent_toks = torch.tensor(padded_sent_toks).unsqueeze(0)
    attention_mask = torch.tensor(attention_mask).unsqueeze(0)
    padded_sent_toks = padded_sent_toks.to(device)
    attention_mask = attention_mask.to(device)

    import torch.nn.functional as F
    last_hidden_states = bert_model(padded_sent_toks, attention_mask=attention_mask)[0]
    embedding = last_hidden_states.permute(0, 2, 1)
    output = model(img, embedding, l_mask=attention_mask.unsqueeze(-1))
    y1_features[filename] = output
    output = output.argmax(1, keepdim=True)
    output = F.interpolate(output.float(), (original_h, original_w))
    output = output.squeeze()
    output = output.cpu().data.numpy()


    def overlay_davis(image, mask, colors=[[0, 0, 0], [255, 0, 0]], cscale=1, alpha=0.4):
        colors = np.reshape(colors, (-1, 3))
        colors = np.atleast_2d(colors) * cscale
    
        im_overlay = image.copy()
        object_ids = np.unique(mask)
    
        for object_id in object_ids[1:]:
            foreground = image*alpha + np.ones(image.shape)*(1-alpha) * np.array(colors[object_id])
            binary_mask = mask == object_id
    
            im_overlay[binary_mask] = foreground[binary_mask]
    
            countours = binary_dilation(binary_mask) ^ binary_mask
            im_overlay[countours, :] = 0
    
        return im_overlay.astype(image.dtype)
    
    
    output = output.astype(np.uint8)

    visualization = overlay_davis(img_ndarray, output)
    visualization = Image.fromarray(visualization)
    visualization.save('../' + filename)

    segmented_samples[filename] = visualization
    print(filename, "saved")


# In[ ]:





# In[17]:


for filename in os.listdir(samples_dir):
    print(filename, ":", references[filename])
    plt.imshow(segmented_samples[filename])
    plt.axis("off")
    plt.title(filename)
    plt.show()


# In[18]:


for filename in os.listdir(samples_dir):
    cur_y1_fm = y1_features[filename].squeeze(0).mean(dim=0)

    plt.imshow(cur_y1_fm.cpu().detach().numpy(), cmap="viridis")
    plt.title("Y1 Feature Map: " + filename)
    plt.axis("off")
    plt.show()


# In[ ]:





# In[19]:


fail_references = {
    'ILSVRC2012_test_00000004.jpg' : "the black eyes of the dog",
    'ILSVRC2012_test_00000022.jpg' : "the brown marble floor in the picture",
    'ILSVRC2012_test_00000023.jpg' : "the blue dustbin in the picture",
    'ILSVRC2012_test_00000026.jpg' : "the brown hair in the painting",
    'ILSVRC2012_test_00000018.jpg' : "the red icecream in the children's hands",
    'ILSVRC2012_test_00000003.jpg' : "the golden logo in the picture",
    'ILSVRC2012_test_00000019.jpg' : "the green plant in the picture",
    'ILSVRC2012_test_00000030.jpg' : "the yellow beak of the duck",
    'ILSVRC2012_test_00000034.jpg' : "the black handle of the coffee machine",
    'ILSVRC2012_test_00000025.jpg' : "the green leaves just below the butterfly"
}


# In[20]:


fail_segmented_samples = {}

for filename in os.listdir(samples_dir):
    image_path = os.path.join(samples_dir, filename)
    sentence = fail_references[filename]


    img = Image.open(image_path).convert("RGB")
    img_ndarray = np.array(img)
    original_w, original_h = img.size
    
    image_transforms = T.Compose(
        [
         T.Resize(480),
         T.ToTensor(),
         T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    )
    
    img = image_transforms(img).unsqueeze(0)
    img = img.to(device)
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    sentence_tokenized = tokenizer.encode(text=sentence, add_special_tokens=True)
    sentence_tokenized = sentence_tokenized[:20]
    
    padded_sent_toks = [0] * 20
    padded_sent_toks[:len(sentence_tokenized)] = sentence_tokenized

    attention_mask = [0] * 20
    attention_mask[:len(sentence_tokenized)] = [1]*len(sentence_tokenized)

    padded_sent_toks = torch.tensor(padded_sent_toks).unsqueeze(0)
    attention_mask = torch.tensor(attention_mask).unsqueeze(0)
    padded_sent_toks = padded_sent_toks.to(device)
    attention_mask = attention_mask.to(device)

    import torch.nn.functional as F
    last_hidden_states = bert_model(padded_sent_toks, attention_mask=attention_mask)[0]
    embedding = last_hidden_states.permute(0, 2, 1)
    output = model(img, embedding, l_mask=attention_mask.unsqueeze(-1))
    output = output.argmax(1, keepdim=True)
    output = F.interpolate(output.float(), (original_h, original_w))
    output = output.squeeze()
    output = output.cpu().data.numpy()

    def overlay_davis(image, mask, colors=[[0, 0, 0], [255, 0, 0]], cscale=1, alpha=0.4):
        colors = np.reshape(colors, (-1, 3))
        colors = np.atleast_2d(colors) * cscale
    
        im_overlay = image.copy()
        object_ids = np.unique(mask)
    
        for object_id in object_ids[1:]:
            foreground = image*alpha + np.ones(image.shape)*(1-alpha) * np.array(colors[object_id])
            binary_mask = mask == object_id
    
            im_overlay[binary_mask] = foreground[binary_mask]
    
            countours = binary_dilation(binary_mask) ^ binary_mask
            im_overlay[countours, :] = 0
    
        return im_overlay.astype(image.dtype)
    
    
    output = output.astype(np.uint8)
    
    visualization = overlay_davis(img_ndarray, output)
    visualization = Image.fromarray(visualization)
    visualization.save('../f_' + filename)

    fail_segmented_samples[filename] = visualization
    print(filename, "saved")


# In[ ]:





# In[21]:


for filename in os.listdir(samples_dir):
    print(filename, ":", fail_references[filename])
    plt.imshow(fail_segmented_samples[filename])
    plt.axis("off")
    plt.title(filename)
    plt.show()


# In[ ]:




