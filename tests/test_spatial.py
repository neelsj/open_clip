import torch
from PIL import Image
import open_clip

import open_flamingo
from huggingface_hub import hf_hub_download

import os
import csv
import numpy as np

from tqdm import tqdm

import random

device = "cuda" if torch.cuda.is_available() else "cpu"

useOF = False

def getOpenCLIP(arch, pretrained):
    
    model, _, preprocess = open_clip.create_model_and_transforms(arch, pretrained=pretrained)
    tokenizer = open_clip.get_tokenizer(arch)
    model.to(device)

    return model, preprocess, tokenizer

def getOpenFlamingo(arch, pretrained):

    model, image_processor, tokenizer = open_flamingo.create_model_and_transforms(
        clip_vision_encoder_path = arch,
        clip_vision_encoder_pretrained = pretrained,
        lang_encoder_path = "mosaicml/mpt-1b-redpajama-200b-dolly",
        tokenizer_path = "mosaicml/mpt-1b-redpajama-200b-dolly",
        cross_attn_every_n_layers=1
    )
    model.to(device)
    print("\nModel on %s\n" % next(model.parameters()).device)

    # set image / mean metadata from pretrained_cfg if available, or use default
    # model.vision_encoder.image_mean = (0.5, 0.5, 0.5)
    # model.vision_encoder.image_std = (0.5, 0.5, 0.5)

    checkpoint_path = hf_hub_download("openflamingo/OpenFlamingo-3B-vitl-mpt1b-langinstruct", "checkpoint.pt")
    model.load_state_dict(torch.load(checkpoint_path), strict=False)

    return model, image_processor, tokenizer

def get_article(a):
    if (a[0] in ("a", "e", "i", "o", "u")):
        return "an " + a 
    else:
        return "a " + a

def get_relation_single(relation):   
    return " on the " + relation + " of "

def get_relation(relation):
    if (relation in ("left", "right")):
        return " to the " + relation + " of "
    else:
        return " " + relation + " "

def mirror_relation(relation):
    if (relation == "left"):
        return "right"
    elif (relation == "right"):
        return "left"
    elif (relation == "above"):
        return "below"
    else:
        return "above"

def create_prompts(a, b, relation, background=None, allFour=True):

    if (allFour):       

        relations = ["above", "below", "left", "right"]

        prompts = [get_article(a) + get_relation(relation) + get_article(b)]

        relations_extra = set(relations).difference([relation])

        for relation in relations_extra:
            prompts.append(get_article(a) + get_relation(relation) + get_article(b))
    else:
        prompta = get_article(a) + get_relation(relation) + get_article(b)
        promptb = get_article(a) + get_relation(mirror_relation(relation)) + get_article(b)

        prompts = [prompta, promptb]

    if (background):
        background = background.replace("-", " ").replace("_", " ")

        for i in range(len(prompts)):
            prompts[i] +=  " in a " + background        

    return prompts

def create_prompt(a, b, relation, background=None, allFour=True):

    prompt = get_article(a) + get_relation(relation) + get_article(b)
       
    if (background):
        background = background.replace("-", " ").replace("_", " ")
        
        prompt +=  " in a " + background        

    return prompt

def create_prompts_single(a, relation):

    relations = ["top", "bottom", "left", "right"]

    prompts = [get_article(a) + get_relation_single(relation) + "the image"]

    relations_extra = set(relations).difference([relation])

    for relation in relations_extra:
        prompts.append(get_article(a) + get_relation_single(relation) + "the image")    

    return prompts

def test_spatial(arch, pretrained):
    # Load the model
    device = "cuda" if torch.cuda.is_available() else "cpu"

    batchSize = 512

    model, preprocess, tokenizer = getOpenCLIP()

    path = "e:/Source/EffortlessCVSystem/Data/coco_spatial_pair_backgrounds_finetune"

    rows = []
    with open(os.path.join(path, 'val.csv'), newline='') as csvfile: 
        reader = csv.reader(csvfile)
        for row in reader:
            rows.append(row)

    #rows = random.sample(rows[1:], 2048)

    print("num samples %d" % len(rows))

    image_features = np.zeros((len(rows), 768))

    for i in tqdm(range(0, len(rows), batchSize)):

        images = []
    
        num = min(batchSize, len(rows)-i)

        for j in range(num):
            row = rows[i+j]
            image = os.path.join(path, row[0])

            image = preprocess(Image.open(image)).unsqueeze(0).to(device)
            images.append(image)
        
        images = torch.cat(images, dim=0)

        # Calculate features
        with torch.no_grad():
            image_feature = model.encode_image(images)

        image_features[i:i+num,:] = image_feature.cpu().numpy()

    ##np.save(os.path.join(path, 'image_features_val.npy'), image_features)
    ##image_features = np.load(os.path.join(path, 'image_features_val.npy'))

    probs = []
    probs_spatial = {}
    probs_spatial["above"] = []
    probs_spatial["below"] = []
    probs_spatial["left"] = []
    probs_spatial["right"] = []

    counts_spatial = {}
    counts_spatial["above"] = 0
    counts_spatial["below"] = 0
    counts_spatial["left"] = 0
    counts_spatial["right"] = 0

    for i, row in enumerate(tqdm(rows)):
        image = row[0]
        prompt = row[1]
        parts = image.split("\\")
        a, b = parts[1].split("_")
        relation = parts[2]
        background = parts[3]

        prompts = create_prompts(a, b, relation, background)
        text = tokenizer(prompts).to(device)

        with torch.no_grad():
            text_features = model.encode_text(text).cpu().numpy()

            text_probs = (100.0 * torch.tensor(image_features[i]) @ text_features.T).softmax(dim=-1)

            prob = text_probs.cpu().numpy()[0]

            probs.append(prob)

            probs_spatial[relation].append(prob)
            
            counts_spatial[relation] += 1

    print("Accuracy %f" % np.mean(probs))

    for relation in ["above", "below", "left", "right"]:
        print("Accuracy %s %f" % (relation, np.mean(probs_spatial[relation])))

    print(counts_spatial)
 
def test_spatial_single(arch, pretrained):
    # Load the model
    device = "cuda" if torch.cuda.is_available() else "cpu"

    batchSize = 1 if useOF else 512

    model, preprocess, tokenizer = getOpenCLIP(arch, pretrained)
    #model, preprocess, _ = getOpenFlamingo(arch, pretrained)
    
    path = "e:/Source/EffortlessCVSystem/Data/coco_spatial_single_backgrounds_finetune"

    rows = []
    with open(os.path.join(path, 'val.csv'), newline='') as csvfile: 
        reader = csv.reader(csvfile)
        for row in reader:
            rows.append(row)

    #rows = random.sample(rows[1:], 2048)

    print("Spatial testing num samples %d" % len(rows))
    
    image_features = np.zeros((len(rows), model.visual.output_dim))

    for i in tqdm(range(0, len(rows), batchSize)):

        images = []
    
        num = min(batchSize, len(rows)-i)

        for j in range(num):
            row = rows[i+j]
            image = os.path.join(path, row[0]).replace("\\", "/")

            image = preprocess(Image.open(image)).unsqueeze(0).to(device)
            images.append(image)
        
        images = torch.cat(images, dim=0)

        # Calculate features
        with torch.no_grad():
            if (useOF):
                image_feature = model.vision_encoder(images)[0]                
            else:
                image_feature = model.encode_image(images)

        image_features[i:i+num,:] = image_feature.cpu().numpy()

    ##np.save(os.path.join(path, 'image_features_val.npy'), image_features)
    ##image_features = np.load(os.path.join(path, 'image_features_val.npy'))

    probs = []
    probs_spatial = {}
    probs_spatial["top"] = []
    probs_spatial["bottom"] = []
    probs_spatial["left"] = []
    probs_spatial["right"] = []

    counts_spatial = {}
    counts_spatial["top"] = 0
    counts_spatial["bottom"] = 0
    counts_spatial["left"] = 0
    counts_spatial["right"] = 0

    for i, row in enumerate(tqdm(rows)):
        image = row[0]
        prompt = row[1]
        parts = image.split("\\")
        a = parts[1]
        relation = parts[2]
        background = parts[3]

        prompts = create_prompts_single(a, relation)
        text = tokenizer(prompts).to(device)

        with torch.no_grad():           
            text_features = model.encode_text(text).cpu().numpy()

            text_probs = (100.0 * torch.tensor(image_features[i]) @ text_features.T).softmax(dim=-1)

            prob = text_probs.cpu().numpy()[0]

            probs.append(prob)

            probs_spatial[relation].append(prob)
            
            counts_spatial[relation] += 1


    print("Spatial testing accuracy %f" % np.mean(probs))

    for relation in probs_spatial.keys():
        print("Spatial testing accuracy %s %f" % (relation, np.mean(probs_spatial[relation])))

    #for relation in counts_spatial.keys():
    #    logging.info("Spatial testing counts %s %f" % (relation,counts_spatial[relation]))   

def create_all_prompts():
    path = "e:/Source/EffortlessCVSystem/Data/coco_spatial_backgrounds"

    rows = []
    with open(os.path.join(path, 'pairs.csv'), newline='') as csvfile: 
        reader = csv.reader(csvfile)
        for row in reader:
            rows.append(row)

    prompts = []
    for i, row in enumerate(tqdm(rows)):
        a = row[0]
        b = row[1]
        relation = row[2]
        background = row[3]
        image = row[7]
        prompt = create_prompt(a, b, relation, background)

        prompts.append([image, prompt])

    with open(os.path.join(path, 'val.csv'), 'w', newline='', encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        
        for i, row in enumerate(tqdm(prompts)):
            writer.writerow(row)

if __name__ == "__main__":

    arch = "ViT-L-14"
    pretrained = "E:/Source/open_clip/logs/best_spatial_checkpoint/epoch_1.pt"
    #pretrained = 'laion2b_s32b_b82k'

    test_spatial_single(arch, pretrained)

    #create_all_prompts()