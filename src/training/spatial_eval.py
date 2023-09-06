import torch
from PIL import Image
import open_clip

import os
import csv
import numpy as np

from tqdm import tqdm

import random

import logging

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

def create_prompts(a, b, relation):

    relations = ["above", "below", "left", "right"]

    prompts = [get_article(a) + get_relation(relation) + get_article(b)]

    relations_extra = set(relations).difference([relation])

    for relation in relations_extra:
        prompts.append(get_article(a) + get_relation(relation) + get_article(b))    

    return prompts

def create_prompts_single(a, relation):

    relations = ["top", "bottom", "left", "right"]

    prompts = [get_article(a) + get_relation_single(relation) + "the image"]

    relations_extra = set(relations).difference([relation])

    for relation in relations_extra:
        prompts.append(get_article(a) + get_relation_single(relation) + "the image")    

    return prompts

def test_spatial(model, preprocess, epoch, args):
    # Load the model

    if (args.test_spatial is None):
         return {}       
    if args.zeroshot_frequency == 0:
        return {}
    if (epoch % args.zeroshot_frequency) != 0 and epoch != args.epochs:
        return {}

    tokenizer = open_clip.get_tokenizer(args.model)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    batchSize = args.batch_size

    path = args.test_spatial

    rows = []
    with open(os.path.join(path, 'val.csv'), newline='') as csvfile: 
        reader = csv.reader(csvfile)
        for row in reader:
            rows.append(row)

    #rows = random.sample(rows[1:], 2048)

    logging.info("Spatial testing num samples %d" % len(rows))

    image_features = np.zeros((len(rows), 768))

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

        prompts = create_prompts(a, b, relation)
        text = tokenizer(prompts).to(device)

        with torch.no_grad():
            text_features = model.encode_text(text).cpu().numpy()

            text_probs = (100.0 * torch.tensor(image_features[i]) @ text_features.T).softmax(dim=-1)

            prob = text_probs.cpu().numpy()[0]

            probs.append(prob)

            probs_spatial[relation].append(prob)
            
            counts_spatial[relation] += 1

    results = {}

    logging.info("Spatial testing accuracy %f" % np.mean(probs))
    results['spatial-testing-accuracy'] = np.mean(probs)

    for relation in probs_spatial.keys():
        logging.info("Spatial testing accuracy %s %f" % (relation, np.mean(probs_spatial[relation])))
        results['spatial-testing-accuracy-%s' % relation] = np.mean(probs_spatial[relation])

    logging.info('Finished spatial testing.')

    return results

def test_spatial_single(model, preprocess, epoch, args):
    # Load the model

    if (args.test_spatial_single is None):
         return {}       
    if args.zeroshot_frequency == 0:
        return {}
    if (epoch % args.zeroshot_frequency) != 0 and epoch != args.epochs:
        return {}

    tokenizer = open_clip.get_tokenizer(args.model)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    batchSize = args.batch_size

    path = args.test_spatial_single

    rows = []
    with open(os.path.join(path, 'val.csv'), newline='') as csvfile: 
        reader = csv.reader(csvfile)
        for row in reader:
            rows.append(row)

    #rows = random.sample(rows[1:], 2048)

    logging.info("Spatial testing num samples %d" % len(rows))

    image_features = np.zeros((len(rows), 768))

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

    results = {}

    logging.info("Spatial testing accuracy %f" % np.mean(probs))
    results['spatial-testing-accuracy'] = np.mean(probs)

    for relation in probs_spatial.keys():
        logging.info("Spatial testing accuracy %s %f" % (relation, np.mean(probs_spatial[relation])))
        results['spatial-testing-accuracy-%s' % relation] = np.mean(probs_spatial[relation])

    logging.info('Finished spatial testing.')

    return results