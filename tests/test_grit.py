from random import random
import pandas as pd

from collections import Counter

from tqdm import tqdm

import string
import csv

import multiprocessing

import os
from datasets import load_dataset

import re

import json  
import os  
import requests  
from urllib.parse import urlparse  
from requests.exceptions import HTTPError  

import sys
from pathlib import Path
import textwrap

import ast
import os
import numpy as np

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
pylab.rcParams['figure.figsize'] = 20, 12

import cv2
import base64
import io

import glob

from concurrent.futures import ThreadPoolExecutor

import openai

openai.api_key = 'ca5aa4348fc14d83a193a7048e728a94'
openai.api_base = 'https://openai-models-east.openai.azure.com/'
openai.api_type = 'azure'
openai.api_version = '2023-05-15' # this may change in the future
deployment_name='text-davinci-003'

#import torch
#import transformers
#from transformers import AutoTokenizer, AutoModelForCausalLM

#MIN_TRANSFORMERS_VERSION = '4.25.1'

## check transformers version
#assert transformers.__version__ >= MIN_TRANSFORMERS_VERSION, f'Please upgrade transformers to version {MIN_TRANSFORMERS_VERSION} or higher.'

## init
#tokenizer = AutoTokenizer.from_pretrained("togethercomputer/RedPajama-INCITE-Chat-3B-v1")
#model = AutoModelForCausalLM.from_pretrained("togethercomputer/RedPajama-INCITE-Chat-3B-v1", torch_dtype=torch.float16)
#model = model.to('cuda:0')

def download_images_from_parquet(p):
    file = 'E:\\Source\\GRIT\\grit-20m\\meta\\coyo_%d_snappy.parquet' % p
    output_folder = 'E:\\Source\\GRIT\\grit-20m\\tmp'

    print("opening %s" % file)

    df = pd.read_parquet(file, engine='pyarrow', columns=['key', 'url', 'caption', 'noun_chunks','ref_exps','width', 'height'])

    for index, json_obj in tqdm(df.iterrows(), total=df.shape[0]):

        #if ("Disabled man" in json_obj["caption"]):
        augment_caption(json_obj, output_folder)  
        vis_image(json_obj, output_folder)  

def folder_to_csv(start, stop):
    pRange = range(start, stop+1)
    out_path_csv = ('E:\\Source\\GRIT\\grit-20m\\images\\images_%04d_to_%04d_interleaved.csv' % (start, stop))
    output_folder = 'E:\\Source\\GRIT\\grit-20m\\images\\'

    with open(out_path_csv, 'w', newline='', encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile, delimiter=',')

        for p in pRange:
            path = 'E:\\Source\\GRIT\\grit-20m\\images\\%05d' % p
            print("opening %s" % path)

            for file in tqdm(glob.glob(path + '**/*.json', recursive=True)):
                # print the path name of selected files
                # print(file)

                with open(file) as json_file:
                    json_obj = json.load(json_file)

                    if ("Laura" in json_obj['caption']):
                        g = 0

                    #augmented_caption_gpt, caption, augmented_caption = augment_caption(json_obj, output_folder, False)  
                    augmented_captions = augment_caption_interleaved(json_obj)  

                    #aug_caption = "A photo and a diagram are shown. In the photo, a paper towel is dipped into a bowl full of a red liquid sitting on a countertop. The red liquid is traveling up the lower part of the paper towel, and this section of the photo has a square drawn around it. A right-facing arrow leads from this square to the image. The image is square and has a background of two types of molecules, mixed together. The first type of molecule is composed of two bonded black spheres, one of which is single bonded to three white spheres and one of which is single bonded to two white spheres and a red sphere that is itself bonded to a white sphere. The other type of molecule is composed of six black spheres bonded together in a row and bonded to other red and white spheres. Six upward-facing arrows are drawn on top of this background. They have positive signs on their lower ends and negative signs on their heads. Four upward-facing arrows are drawn with their signs reversed. The red liquid is in the bottom left of the image, and a paper towel is on the left of the image, and Six upward-facing arrows are in the top right of the image."
                    #x = tokenizer.encode(aug_caption)
                    #aug_caption_dec = tokenizer.decode(x)
                    #print(aug_caption + "\n\n")
                    #print(aug_caption_dec)

                    image_file = file.replace(output_folder, "").replace("json","jpg")

                    row = [image_file] + augmented_captions
                    writer.writerow(row)

def parquet_to_csv(p):
    file = 'E:\\Source\\GRIT\\grit-20m\\meta\\coyo_%d_snappy.parquet' % p
    output_folder = 'E:\\Source\\GRIT\\grit-20m\\tmp'

    #print("opening %s" % file)

    df = pd.read_parquet(file, engine='pyarrow', columns=['key', 'url', 'caption', 'noun_chunks','ref_exps','width', 'height'])

    out_path_csv = 'E:\\Source\\GRIT\\grit-20m\\data\\coyo_%d_snappy.csv' % p

    with open(out_path_csv, 'w', newline='', encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile, delimiter=',')

        for index, json_obj in tqdm(df.iterrows(), total=df.shape[0]):

            aug_caption = augment_caption(json_obj, output_folder, False)  

            row = [index, json_obj["url"], aug_caption]
            writer.writerow(row)

def download_from_csv(p):
    file = 'E:\\Source\\GRIT\\grit-20m\\data\\coyo_%d_snappy.csv' % p 
    out_path_csv = 'E:\\Source\\GRIT\\grit-20m\\data\\coyo_%d_snappy_dl.csv' % p

    global output_base_folder
    output_base_folder = 'E:\\Source\\GRIT\\grit-20m\\data\\coyo_%d_snappy' % p

    print("opening %s" % file)

    images = []
    with open(file, newline='', encoding="utf-8") as csvfile: 
        reader = csv.reader(csvfile)
        for row in tqdm(reader):
            images.append(row)

            #if (len(images)>10):
            #    break

    with ThreadPoolExecutor(max_workers = 20) as executor:
        results = list(tqdm(executor.map(download_image, images), total=len(images)))

    #for image in tqdm(images):
    #    download_image(image, output_base_folder)

    with open(out_path_csv, 'w', newline='', encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile, delimiter=',')

        for i, image in tqdm(enumerate(images)):
            if (results[i]):
                rel_path = results[i].replace("E:\\Source\\GRIT\\grit-20m\\data\\", "")
                row = [rel_path, image[2]]
                writer.writerow(row)

def download_image(image):  

    index, url, _  = image

    try:  
        response = requests.get(url)  
        response.raise_for_status()  
    except Exception as e:  
        #print(f"Error while downloading {url}: {e}")  
        return None  
  
    output_folder = "%08d" % (int(float(index) / 10000))
    output_folder = os.path.join(output_base_folder, output_folder)

    file_name = "%08d" % int(index)
    ext = os.path.splitext(os.path.basename(urlparse(url).path) )[1] 
  
    ext = ext if (ext) else ".jpg" 

    output_path = os.path.join(output_folder, file_name + ext)  

    if (not os.path.isdir(output_folder)):
        os.makedirs(output_folder) 

    with open(output_path, 'wb') as file:  
        file.write(response.content) 

    return output_path
        
def imshow(img, file_name = "tmp.jpg", caption='test'):
    # Create figure and axis objects
    fig, ax = plt.subplots()
    # Show image on axis
    ax.imshow(img[:, :, [2, 1, 0]])
    ax.set_axis_off()
    # Set caption text
    # Add caption below image
    ax.text(0.5, -0.2, '\n'.join(textwrap.wrap(caption, 120)), ha='center', transform=ax.transAxes, fontsize=18)
    plt.savefig(file_name, bbox_inches='tight')
    plt.close()
    
def get_phrase(phrase):
    if (phrase[-1] == "s"):
        return phrase + " are "
    else:
        return phrase + " is "

def get_relation(relation, on=False):
    if (on):
        return "on the " + relation + " of the image"
    else:
        return "in the " + relation + " of the image"

def get_relation_phrase(grid_x, grid_y):
    if (grid_x == 0 and grid_y == 0):
        return get_relation("top left")
    elif (grid_x == 1 and grid_y == 0):
        return  get_relation("top center")
    elif (grid_x == 2 and grid_y == 0):
        return get_relation("top right")
    elif (grid_x == 0 and grid_y == 1):
        return  get_relation("left", True)
    elif (grid_x == 1 and grid_y == 1):
        return  get_relation("center")
    elif (grid_x == 2 and grid_y == 1):
        return  get_relation("right", True)
    elif (grid_x == 0 and grid_y == 2):
        return  get_relation("bottom left")
    elif (grid_x == 1 and grid_y == 2):
        return get_relation("bottom center")
    else: #if (grid_x == 2 and grid_y == 2):
        return  get_relation("bottom right")

def augment_caption_interleaved(json_obj, variations=8): 
    caption = json_obj['caption']

    #grounding_list = json_obj['ref_exps']
    grounding_list = json_obj['noun_chunks']
    image_w = json_obj['width']
    image_h = json_obj['height']

    grounding_list = sorted(grounding_list, key=lambda x: x[0])

    augmented_captions = []

    candidates_all = []
    for i, (phrase_s, phrase_e, x1_norm, y1_norm, x2_norm, y2_norm, score) in enumerate(grounding_list): 

        center_x = x1_norm+(x2_norm-x1_norm)/2
        center_y = y1_norm+(y2_norm-y1_norm)/2

        grid_x = int(center_x/.33)
        grid_y = int(center_y/.33)

        candidates = []
        for x in range(0,3):
            for y in range(0,3):
                if (grid_x != x or grid_y != y):
                    candidates.append((x, y))

        np.random.shuffle(candidates)
        candidates_all.append(candidates)

    for var in range(variations):

        augmented_caption = ""

        phrase_l = 0
        for i, (phrase_s, phrase_e, x1_norm, y1_norm, x2_norm, y2_norm, score) in enumerate(grounding_list): 

            phrase_prefix = caption[phrase_l:int(phrase_s)]
            phrase = caption[int(phrase_s):int(phrase_e)]

            center_x = x1_norm+(x2_norm-x1_norm)/2
            center_y = y1_norm+(y2_norm-y1_norm)/2

            grid_x = int(center_x/.33)
            grid_y = int(center_y/.33)

            if (var != 0):
                (grid_x_rand, grid_y_rand) = candidates_all[i][var-1]
   
                augmented_caption += phrase_prefix + phrase + " (" + get_relation_phrase(grid_x_rand, grid_y_rand) + ") "
            else:
                augmented_caption += phrase_prefix + phrase + " (" + get_relation_phrase(grid_x, grid_y) + ") "

            phrase_l = int(phrase_e)

        augmented_caption += caption[int(phrase_e):]

        augmented_captions.append(augmented_caption.replace("  ", " "))

    return augmented_captions

def augment_caption(json_obj, output_folder, display=False): 
    caption = json_obj['caption']

    #grounding_list = json_obj['ref_exps']
    grounding_list = json_obj['noun_chunks']
    image_w = json_obj['width']
    image_h = json_obj['height']

    augmented_caption = ""

    for i, (phrase_s, phrase_e, x1_norm, y1_norm, x2_norm, y2_norm, score) in enumerate(grounding_list):  
        phrase = caption[int(phrase_s):int(phrase_e)]

        center_x = x1_norm+(x2_norm-x1_norm)/2
        center_y = y1_norm+(y2_norm-y1_norm)/2

        grid_x = int(center_x/.33)
        grid_y = int(center_y/.33)

        augmented_caption += get_phrase(phrase) + get_relation_phrase(grid_x, grid_y)

        if (i == len(grounding_list)-1):
            augmented_caption += "."
        else:
            augmented_caption += ", and "

        first_word = augmented_caption.split(" ")[0].lower()
    
        if (not (first_word == "the" or first_word == "a" or first_word == "an")):
            augmented_caption = "The " + augmented_caption

        try:
            start_phrase = 'Please combine these two statements to create a new statement.  Statement 1 describes the image and Statement 2 includes the spatial relationships.  For the new statement, keep the style and details of Statement 1 and incorporate the spatial relationships described in Statement 2. Statement 1: %s and Statement 2: %s' % (caption, augmented_caption)
            response = openai.Completion.create(engine=deployment_name, prompt=start_phrase, max_tokens=100)
            augmented_caption_gpt = response['choices'][0]['text'].replace('\n', '').replace(' .', '.').strip()
        except:
            print(start_phrase)

            if (caption[-1] == "." or caption[-1] == "!" or caption[-1] == "?"):
                augmented_caption_gpt = caption + " " + augmented_caption        
            else:
                augmented_caption_gpt = caption + ". " + augmented_caption
            pass

    #print(caption)
    #print(augmented_caption)

    #inputs = tokenizer(start_phrase, return_tensors='pt').to(model.device)
    #input_length = inputs.input_ids.shape[1]
    #outputs = model.generate(
    #    **inputs, max_new_tokens=128, do_sample=True, temperature=0.7, top_p=0.7, top_k=50, return_dict_in_generate=True
    #)
    #token = outputs.sequences[0, input_length:]
    #augmented_caption_gpt = tokenizer.decode(token)

    #if (caption[-1] == "." or caption[-1] == "!" or caption[-1] == "?"):
    #    augmented_caption = caption + " " + augmented_caption        
    #else:
    #    augmented_caption = caption + ". " + augmented_caption   

    #if (display):
    #    url = json_obj['url']  
    #    try:  
    #        response = requests.get(url)  
    #        response.raise_for_status()  
        
    #        file_name = os.path.basename(urlparse(url).path)  
    #        # output_path = os.path.join(output_folder, file_name) 
    #        file_key_name = json_obj['key'] + os.path.splitext(file_name)[1]
    #        output_path = os.path.join(output_folder, file_key_name) 
        
    #    except Exception as e:  
    #        print(f"Error while downloading {url}: {e}")  
    #        return    

    #    with open(output_path, 'wb') as file:  
    #        file.write(response.content) 
    
    #    try:
    #        pil_img = Image.open(output_path).convert("RGB")
    #    except:
    #        return 
    #    image = np.array(pil_img)[:, :, [2, 1, 0]]

    #    #file_key_name = json_obj['key'] + '_exp' + os.path.splitext(file_name)[1]
    #    #output_path = os.path.join(output_folder, file_key_name)         
    #    #imshow(image, file_name= output_path, caption=caption)

    #    file_key_name = json_obj['key'] + '_exp_aug' + os.path.splitext(file_name)[1]
    #    output_path = os.path.join(output_folder, file_key_name)         
    #    imshow(image, file_name= output_path, caption = augmented_caption)

    return augmented_caption_gpt, caption, augmented_caption

def vis_image(json_obj, output_folder): 
    url = json_obj['url']  
    try:  
        response = requests.get(url)  
        response.raise_for_status()  
        
        file_name = os.path.basename(urlparse(url).path)  
        # output_path = os.path.join(output_folder, file_name) 
        file_key_name = json_obj['key'] + os.path.splitext(file_name)[1]
        output_path = os.path.join(output_folder, file_key_name) 
        
    except Exception as e:  
        print(f"Error while downloading {url}: {e}")  
        return    

    with open(output_path, 'wb') as file:  
        file.write(response.content) 
    
    try:
        pil_img = Image.open(output_path).convert("RGB")
    except:
        return 
    image = np.array(pil_img)[:, :, [2, 1, 0]]
    image_h = pil_img.height
    image_w = pil_img.width
    caption = json_obj['caption']
    
    def is_overlapping(rect1, rect2):  
        x1, y1, x2, y2 = rect1  
        x3, y3, x4, y4 = rect2  
        return not (x2 < x3 or x1 > x4 or y2 < y3 or y1 > y4) 
    
    #grounding_list = json_obj['ref_exps']
    grounding_list = json_obj['noun_chunks']
    
    new_image = image.copy()
    previous_locations = []
    previous_bboxes = []
    text_offset = 10
    text_offset_original = 4
    text_size = max(0.07 * min(image_h, image_w) / 100, 0.5)
    text_line = int(max(1 * min(image_h, image_w) / 512, 1))
    box_line = int(max(2 * min(image_h, image_w) / 512, 2))
    text_height = text_offset # init
    # pdb.set_trace()
    for (phrase_s, phrase_e, x1_norm, y1_norm, x2_norm, y2_norm, score) in grounding_list:  
        phrase = caption[int(phrase_s):int(phrase_e)]
        x1, y1, x2, y2 = int(x1_norm * image_w), int(y1_norm * image_h), int(x2_norm * image_w), int(y2_norm * image_h)
        print(f"Decode results: {phrase} - {[x1, y1, x2, y2]}")
        # draw bbox
        # random color
        color = tuple(np.random.randint(0, 255, size=3).tolist())
        new_image = cv2.rectangle(new_image, (x1, y1), (x2, y2), color, box_line)
        
        # add phrase name  
        # decide the text location first  
        for x_prev, y_prev in previous_locations:  
            if abs(x1 - x_prev) < abs(text_offset) and abs(y1 - y_prev) < abs(text_offset):  
                y1 += text_height  

        if y1 < 2 * text_offset:  
            y1 += text_offset + text_offset_original  

        # add text background
        (text_width, text_height), _ = cv2.getTextSize(phrase, cv2.FONT_HERSHEY_SIMPLEX, text_size, text_line)  
        text_bg_x1, text_bg_y1, text_bg_x2, text_bg_y2 = x1, y1 - text_height - text_offset_original, x1 + text_width, y1  
        
        for prev_bbox in previous_bboxes:  
            while is_overlapping((text_bg_x1, text_bg_y1, text_bg_x2, text_bg_y2), prev_bbox):  
                text_bg_y1 += text_offset  
                text_bg_y2 += text_offset  
                y1 += text_offset 
                
                if text_bg_y2 >= image_h:  
                    text_bg_y1 = max(0, image_h - text_height - text_offset_original)  
                    text_bg_y2 = image_h  
                    y1 = max(0, image_h - text_height - text_offset_original + text_offset)  
                    break 
        
        alpha = 0.5  
        for i in range(text_bg_y1, text_bg_y2):  
            for j in range(text_bg_x1, text_bg_x2):  
                if i < image_h and j < image_w: 
                    new_image[i, j] = (alpha * new_image[i, j] + (1 - alpha) * np.array(color)).astype(np.uint8) 
        
        cv2.putText(  
            new_image, phrase, (x1, y1 - text_offset_original), cv2.FONT_HERSHEY_SIMPLEX, text_size, (0, 0, 0), text_line, cv2.LINE_AA  
        )  
        previous_locations.append((x1, y1))  
        previous_bboxes.append((text_bg_x1, text_bg_y1, text_bg_x2, text_bg_y2))
    
    try:
        file_key_name = json_obj['key'] + '_exp' + os.path.splitext(file_name)[1]
        output_path = os.path.join(output_folder, file_key_name) 
        
        imshow(new_image, file_name= output_path, caption=caption)
    except:
        # Out of (supported formats: eps, jpeg, jpg, pdf, pgf, png, ps, raw, rgba, svg, svgz, tif, tiff, webp)
        return 
    
if __name__ == '__main__':

    #data = download_images_from_parquet(0)
    #data = parquet_to_csv(0)
    #data = download_from_csv(0)
    folder_to_csv(0,9)
    #folder_to_csv(10,99)