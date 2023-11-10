import torch
from PIL import Image
import open_clip

import os
import csv
import numpy as np

from tqdm import tqdm, trange

import random

import torch
from PIL import Image
import requests

import json

import base64
from io import BytesIO

import time
from timeit import default_timer as timer

import seaborn as sns
import pandas as pd  
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"
print("\nUsing %s\n" % device)

def getflanT5():

    from transformers import AutoTokenizer, T5EncoderModel

    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xxl")
    model = T5EncoderModel.from_pretrained("google/flan-t5-xxl")

    model.to(device)
    print("\nModel on %s\n" % next(model.parameters()).device)

    return model, tokenizer 

def getOpenCLIP():

    import open_clip

    #clip_vision_encoder_pretrained = "E:/Source/open_clip/logs/2023_08_08-22_48_12-model_ViT-L-14-lr_0.0001-b_64-j_8-p_amp/checkpoints/epoch_latest.pt"
    clip_vision_encoder_pretrained = "laion2b_s32b_b82k"

    model, _, image_processor = open_clip.create_model_and_transforms( "ViT-L-14", pretrained=clip_vision_encoder_pretrained)
    tokenizer = open_clip.get_tokenizer( "ViT-L-14")

    #model.visual.output_tokens = True

    model.to(device)
    print("\nModel on %s\n" % next(model.parameters()).device)

    return model, image_processor, tokenizer 

def getOpenFlamingo():
    import open_flamingo
    from huggingface_hub import hf_hub_download

    #clip_vision_encoder_pretrained = "E:/Source/open_clip/logs/2023_08_08-22_48_12-model_ViT-L-14-lr_0.0001-b_64-j_8-p_amp/checkpoints/epoch_latest.pt"
    clip_vision_encoder_pretrained = "laion2b_s32b_b82k"

    model, image_processor, tokenizer = open_flamingo.create_model_and_transforms(
        clip_vision_encoder_path = "ViT-L-14",
        clip_vision_encoder_pretrained = clip_vision_encoder_pretrained,
        lang_encoder_path = "mosaicml/mpt-1b-redpajama-200b-dolly",
        tokenizer_path = "mosaicml/mpt-1b-redpajama-200b-dolly",
        cross_attn_every_n_layers=1
    )
    model.to(device)
    print("\nModel on %s\n" % next(model.parameters()).device)

    tokenizer.padding_side = "left" # For generation padding tokens should be on the left
    checkpoint_path = hf_hub_download("openflamingo/OpenFlamingo-3B-vitl-mpt1b-langinstruct", "checkpoint.pt")
    model.load_state_dict(torch.load(checkpoint_path), strict=False)

    return model, image_processor, tokenizer 

path = "E:/Source/GRIT/grit-20m/images/"

def create_features_t5():

    # Load the data

    rows = []
    with open(os.path.join(path, 'images_0000_to_0009_interleaved.csv'), newline='', encoding="utf8") as csvfile: 
        reader = csv.reader(csvfile)
        next(reader)
        for row in reader:
            rows.append(row)

    rows = rows[0:10000]

    print("num samples %d" % len(rows))
    num_prompts = len(rows[0])-1

    # Load the model
    model, tokenizer = getflanT5()
    batchSize = 256

    text_features = np.zeros((num_prompts, len(rows), 768))

    for i in tqdm(range(0, len(rows), batchSize)):

        prompts_batch = []
        for k in range(num_prompts):
            prompts_batch.append([])
        
        num = min(batchSize, len(rows)-i)

        for j in range(num):
            row = rows[i+j]
        
            for k in range(num_prompts):
                prompt = row[1+k]
                prompts_batch[k].append(prompt)

        # Calculate features
        for k in range(num_prompts):
            with torch.no_grad():
                texts = tokenizer(prompts_batch[k])
                text_feature = model(texts).last_hidden_state

            text_features[k,i:i+num,:] = text_feature.cpu().numpy()

    np.save(os.path.join(path, 'text_features_t5.npy'), text_features)

def create_features(useOF):

    # Load the data

    rows = []
    with open(os.path.join(path, 'images_0000_to_0009_interleaved.csv'), newline='', encoding="utf8") as csvfile: 
        reader = csv.reader(csvfile)
        next(reader)
        for row in reader:
            rows.append(row)

    rows = rows[0:10000]

    print("num samples %d" % len(rows))
    num_prompts = len(rows[0])-1

    # Load the model

    if useOF:
        model, preprocess, tokenizer = getOpenFlamingo()
        batchSize = 16
    else:
        model, preprocess, tokenizer = getOpenCLIP()
        batchSize = 256

    image_features = np.zeros((len(rows), 768))
    text_features = np.zeros((num_prompts, len(rows), 768))

    for i in tqdm(range(0, len(rows), batchSize)):

        images_batch = []
        prompts_batch = []
        for k in range(num_prompts):
            prompts_batch.append([])
        
        num = min(batchSize, len(rows)-i)

        for j in range(num):
            row = rows[i+j]
            img_file = os.path.join(path, row[0])

            image = preprocess(Image.open(img_file)).unsqueeze(0).to(device)
            images_batch.append(image)
        
            for k in range(num_prompts):
                prompt = row[1+k]
                prompts_batch[k].append(prompt)

        images_batch = torch.cat(images_batch, dim=0)

        # Calculate features
        with torch.no_grad():
            if useOF:
                image_feature = model.vision_encoder(images_batch)[0]
            else:
                image_feature = model.encode_image(images_batch)

        image_features[i:i+num,:] = image_feature.cpu().numpy()

        # Calculate features
        for k in range(num_prompts):
            with torch.no_grad():
                texts = tokenizer(prompts_batch[k])

            if useOF:
                text_feature  = model.lang_encoder(images_batch)
            else:
                text_feature = model.encode_text(texts)

            text_features[k,i:i+num,:] = text_feature.cpu().numpy()

    if useOF:
        np.save(os.path.join(path, 'image_features_of.npy'), image_features)
        np.save(os.path.join(path, 'text_features_of.npy'), text_features)
    else:
        np.save(os.path.join(path, 'image_features_clip.npy'), image_features)
        np.save(os.path.join(path, 'text_features_clip.npy'), text_features)

def compute_tsne(useOF):
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA

    if useOF:
        image_features = np.load(os.path.join(path, 'image_features_of.npy'))
        text_features = np.load(os.path.join(path, 'text_features_of.npy'))
    else:
        image_features = np.load(os.path.join(path, 'image_features_clip.npy'))
        text_features = np.load(os.path.join(path, 'text_features_clip.npy'))

    features = np.concatenate((image_features, text_features.reshape(text_features.shape[0]*text_features.shape[1], text_features.shape[2])))

    print("pca")

    start = timer()    
    pca = PCA(n_components=50)
    features_pca = pca.fit_transform(features) 
    end = timer()

    print("pca time %.02f mins" % ((end - start)/60))

    print("tsne")

    start = timer() 
    tsne = TSNE(n_components=2)
    features_tsne = tsne.fit_transform(features_pca)
    end = timer()

    print("tsne time %.02f mins" % ((end - start)/60))

    image_features_tsne = features_tsne[0:len(image_features),:]
    text_features_tsne = features_tsne[len(image_features):,:]
    text_features_tsne = text_features_tsne.reshape(text_features.shape[0], text_features.shape[1], text_features_tsne.shape[1])

    if useOF:
        np.save(os.path.join(path, 'image_features_of_tsne.npy'), image_features_tsne)
        np.save(os.path.join(path, 'text_features_of_tsne.npy'), text_features_tsne)
    else:
        np.save(os.path.join(path, 'image_features_clip_tsne.npy'), image_features_tsne)
        np.save(os.path.join(path, 'text_features_clip_tsne.npy'), text_features_tsne)

def display(useOF):

    rows = []
    with open(os.path.join(path, 'images_0000_to_0009_interleaved.csv'), newline='', encoding="utf8") as csvfile: 
        reader = csv.reader(csvfile)
        next(reader)
        for row in reader:
            rows.append(row)

    rows = rows[0:10000]

    if useOF:
        image_features = np.load(os.path.join(path, 'image_features_of.npy'))
        text_features = np.load(os.path.join(path, 'text_features_of.npy'))
    else:
        image_features = np.load(os.path.join(path, 'image_features_clip.npy'))
        text_features = np.load(os.path.join(path, 'text_features_clip.npy'))

    df = pd.DataFrame()

    xs = []
    ys = []
    colors = []
    ss = []

    bs = 2

    cmap = plt.colormaps['rainbow']

    for i in range(0, image_features.shape[0], bs):

        plt.clf()

        for j in range(bs):
            x = image_features[i+j,0].tolist()
            y = image_features[i+j,1].tolist()

            rgb = cmap(random.randint(0,cmap.N))
            r = rgb[0]
            g = rgb[1]
            b = rgb[2]

            color = [r, g, b]
            label = "i%d" % (i+j)
            print("%s: %s" % (label, rows[i+j][0]))

            plt.scatter(x, y, c=color, s=150)
            plt.annotate(label, # this is the text
                 (x,y), # these are the coordinates to position the label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center') 

            # plt.xlim(-.5, .5)
            # plt.ylim(-2.5, 1)

            for k in range(text_features.shape[0]):
                x = text_features[k,i+j,0].tolist()
                y = text_features[k,i+j,1].tolist()

                a = 1 - 0.5*k/text_features.shape[0]
                s = (1 - 0.5*k/text_features.shape[0])*100

                color = [r*a, g*a, b*a]
                label = "t%d_%d" % (i+j, k)
                print("%s: %s" % (label, rows[i+j][k+1]))

                plt.scatter(x, y, c=color, s=s)
                plt.annotate(label, # this is the text
                (x,y), # these are the coordinates to position the label
                textcoords="offset points", # how to position the text
                xytext=(0,10), # distance from text to points (x,y)
                ha='center') 

        plt.show()

if __name__ == "__main__":
    #create_features(False)
    #create_features(True)
    #compute_tsne(False)
    #compute_tsne(True)

    display(False)

    #create_features_t5()