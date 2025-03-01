from sympy import Q
import torch
from torch._dynamo.utils import ifdyn
import torchvision
import torchvision.transforms as T
from PIL import Image
import open_clip

import os
import csv
import numpy as np

from tqdm import tqdm, trange

import random

from PIL import Image
import requests

import cv2

import json

import base64
from io import BytesIO

import time
from timeit import default_timer as timer

import matplotlib.pyplot as plt 

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from datetime import datetime

import re

device = "cuda" if torch.cuda.is_available() else "cpu"

#region Prompts

def get_article(a):
    if (a[0] in ("a", "e", "i", "o", "u")):
        return "an " + a 
    else:
        return "a " + a

def get_relation(relation):
    if (relation in ("left", "right")):
        return relation + " of"
    else:
        return relation

def get_relation_single_clip(relation):   
    return " on the " + relation + " of "

def get_relation_clip(relation):
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

def create_prompts(a, relation, relations, b=None, options=False):

    relationsA = relations      
    random.shuffle(relationsA)

    verb = "Are" if a[-1] == "s" else "Is"

    if (b):
        relationsA = ", ".join(relationsA[:-1]) + ", or " + get_relation(relationsA[-1])        
        prompt = verb + " the " + a + " " + relationsA + " the " + b  + " in the image?"
    else:
        relationsA = ", ".join(relationsA[:-1]) + ", or " + relationsA[-1]
        
        prompt = verb + " the " + a + " on the " + relationsA + " of the image?"

    if (options):
        relationsChoices = relations
        random.shuffle(relationsChoices)
        relationsChoices = ", ".join(relationsChoices[:-1]) + ", or " + relationsChoices[-1]
        prompt += ("\nAnswer with one of (%s) only." % relationsChoices)
         
    print(prompt)

    return prompt

def create_prompts_clip(a, b, relation, relations, background=None, allFour=True):

    prompts = [get_article(a) + get_relation_clip(relation) + get_article(b)]

    relations_extra = set(relations).difference([relation])

    for relation in relations_extra:
        prompts.append(get_article(a) + get_relation_clip(relation) + get_article(b))
     
    return prompts

def create_prompts_single_clip(a, relation, relations):

    prompts = [get_article(a) + get_relation_single_clip(relation) + "the image"]

    relations_extra = set(relations).difference([relation])

    for relation in relations_extra:
        prompts.append(get_article(a) + get_relation_single_clip(relation) + "the image")    

    return prompts

def check_answer(correct_relation, relations, answer_text):

    answer_text = answer_text.lower()
    answer_text = answer_text.replace("sofa", "couch")
    answer_text = answer_text.replace("sneakers", "shoes")

    total_count = 0
    relations_count = {}
    for relation in relations:
        x = answer_text.count(relation)
        relations_count[relation] = x
        
    total_count = sum(relations_count.values())
    
    pred = "correct" if (total_count == 1 and relations_count[correct_relation] == 1) else "incorrect" if (total_count == 1 and relations_count[correct_relation] == 0) else "none"

    return pred

#endregion

#region CreateModels

def getGPT4():
    from azure.identity import DefaultAzureCredential 
    RESOURCE_NAME = "gpt-visual-api" 
    DEPLOYMENT_NAME = "gpt-visual" 

    # Get the AAD token for current logged-in user 
    credential = DefaultAzureCredential() 

    # Get the token. The token will expire after 1 hour, so if your script is running for a long time call GetToken again before that to refesh. 
    token = credential.get_token("https://cognitiveservices.azure.com/.default") 

    base_url = f"https://{RESOURCE_NAME}.openai.azure.com/openai/deployments/{DEPLOYMENT_NAME}" 

    headers = {   
        "Content-Type": "application/json",   
        "Authorization": f"Bearer {token.token}" 
    } 

    # Prepare endpoint, headers, and request body 
    #endpoint = f"{base_url}/rainbow?api-version=2023-03-15-preview" 
    #endpoint = f"{base_url}/rainbow?api-version=2023-09-15-preview" 
    #endpoint = f"{base_url}/chat/completions?api-version=2023-09-15-preview"
    endpoint = f"{base_url}/chat/completions?api-version=2023-08-01-preview"
        
    return headers, endpoint 

def getOpenFlamingo():
    import open_flamingo
    from huggingface_hub import hf_hub_download

    clip_vision_encoder_pretrained = "E:/Source/open_clip/logs/best_spatial_checkpoint/epoch_2.pt"
    #clip_vision_encoder_pretrained = "openai"
    #clip_vision_encoder_pretrained = "laion2b_s32b_b82k"
    
    model, image_processor, tokenizer = open_flamingo.create_model_and_transforms(
        clip_vision_encoder_path = "ViT-L-14",
        clip_vision_encoder_pretrained = clip_vision_encoder_pretrained,
        lang_encoder_path = "anas-awadalla/mpt-1b-redpajama-200b-dolly",
        tokenizer_path = "anas-awadalla/mpt-1b-redpajama-200b-dolly",
        cross_attn_every_n_layers=1
    )
    model.to(device)
    model.eval()
    print("\nModel on %s\n" % next(model.parameters()).device)

    # set image / mean metadata from pretrained_cfg if available, or use default
    # model.vision_encoder.image_mean = (0.5, 0.5, 0.5)
    # model.vision_encoder.image_std = (0.5, 0.5, 0.5)

    checkpoint_path = hf_hub_download("openflamingo/OpenFlamingo-3B-vitl-mpt1b", "checkpoint.pt")
    model.load_state_dict(torch.load(checkpoint_path), strict=False)

    return model, image_processor, tokenizer 

def getLLaVA():
    from llava.mm_utils import get_model_name_from_path
    from llava.model.builder import load_pretrained_model
    from llava.utils import disable_torch_init

    disable_torch_init()
    
    model_path = "/mnt/e/Source/LLaVA/checkpoints/liuhaotian/llava-v1.5-7b/"
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(model_path, None, model_name, load_4bit=True)
    model.eval()
    
    return model, image_processor, tokenizer 

def getOpenCLIP(arch = "ViT-L-14", pretrained = 'openai'):
    
    #arch = "ViT-L-14"
    #arch = 'ViT-L-14-336'
    #pretrained = 'openai'
    #pretrained = "E:/Source/open_clip/logs/best_spatial_checkpoint/epoch_1.pt"
    
    model, _, preprocess = open_clip.create_model_and_transforms(arch, pretrained=pretrained)
    tokenizer = open_clip.get_tokenizer(arch)
    model.to(device)

    return model, preprocess, tokenizer

def getFasterRCNN():
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
    model.eval()
    
    return model
#endregion

#region Generatetext

def generate_text_gpt(headers, endpoint, query_image, query_text):

    buffered = BytesIO()
    query_image.save(buffered, format="JPEG")
    base64_bytes = base64.b64encode(buffered.getvalue())
    base64_string = base64_bytes.decode('utf-8')

    data = {
        "messages": [
            # { "role": "system", 
            #  "content": "You are a helpful assistant." }, # Content can be a string, OR
            { "role": "user", 
             "content": [                                   # It can be an array containing strings and images.
                query_text,
                { "image": base64_string }                          # Images are represented like this.q
            ] }
        ],  
        "max_tokens": 200,  
        "temperature": 0.7,  
        "n": 1
    }  

    # Make the API call   

    response = requests.post(endpoint, headers=headers, data=json.dumps(data))    
        
    #print(response.json())

    if (response.status_code == 200):
        answer_text = response.json()["choices"][0]["message"]["content"].strip()

        return answer_text, False

    else:
        answer_text = response.json()["error"]["message"]
        print("\n"+answer_text)
        #answer_text = [int(s) for s in msg.split() if s.isdigit()][0]

        return answer_text, False if "filtered" in answer_text else True

def generate_text_phi2v(query_image, query_text):

    query_text = query_text.replace("image", "<|image_1|>") + " Answer:"
    #query_text = query_text + " <|image_1|>"
    
    print(query_text)

    buffered = BytesIO()
    query_image.save(buffered, format="JPEG")
    base64_bytes = base64.b64encode(buffered.getvalue())
    base64_string = base64_bytes.decode('utf-8')

    header = {'Authorization': 'V5VIE6rOq3zdZL4FXE03zLnDiZF4iyNj'}
    data = {"model": "phi-2-vision-v1", 
            "temperature": 0.3, 
            "prompt": query_text, 
            "max_tokens": 200, 
            "stop": ["<|endoftext|>"], 
            "images":[base64_string]
            }
    url = "https://blog.yintat.com/dqjoBCx0P2k7/api/models/unified-completion"

    # Make the API call   
    response = requests.post(url, headers=header, json=data, timeout=600)
            
    if (response.status_code == 200):
        answer_text = response.json()["choices"][0]["text"].strip()
    else:
        answer_text = response.json()["error"]["message"]
        print("\n"+answer_text)
        #answer_text = [int(s) for s in msg.split() if s.isdigit()][0]

    return answer_text       

def generate_text_of(model, image_processor, tokenizer, query_image, query_text):
    
    query_text = "<image>Question: " + query_text + " Answer:"

    """
    Step 2: Preprocessing images
    Details: For OpenFlamingo, we expect the image to be a torch tensor of shape 
        batch_size x num_media x num_frames x channels x height x width. 
        In this case batch_size = 1, num_media = 3, num_frames = 1,
        channels = 3, height = 224, width = 224.
    """
    vision_x = image_processor(query_image).unsqueeze(0).unsqueeze(0).unsqueeze(0).to(device)
    
    """
    Step 3: Preprocessing text
    Details: In the text we expect an <image> special token to indicate where an image is.
        We also expect an <|endofchunk|> special token to indicate the end of the text 
        portion associated with an image.
    """
    
    tokenizer.padding_side = "left" # For generation padding tokens should be on the left
    lang_x = tokenizer(
        [query_text],
        return_tensors="pt",
    )

    """
    Step 4: Generate text
    """
    generated_text = model.generate(
        vision_x=vision_x,
        lang_x=lang_x["input_ids"].to(device),
        attention_mask=lang_x["attention_mask"].to(device),
        max_new_tokens=20,
        num_beams=3
    )

    gen_text = tokenizer.decode(generated_text[0])
    answer_text = gen_text.replace(query_text, "")

    return answer_text

def generate_text_llava(model, image_processor, tokenizer, query_image, query_text):
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
    from llava.conversation import conv_templates, SeparatorStyle
    from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
    from transformers import TextStreamer

    temperature = 0.2    
    top_p = None
    num_beams = 1
    
    if model.config.mm_use_im_start_end:
        query_text = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + query_text
    else:
        query_text = DEFAULT_IMAGE_TOKEN + '\n' + query_text

    conv_mode = "llava_v1"
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], query_text)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)

    image_tensor = image_processor.preprocess(query_image, return_tensors='pt')['pixel_values'][0]
        
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor.unsqueeze(0).half().cuda(),
            do_sample=True if temperature > 0 else False,
            temperature=temperature,
            top_p=top_p,
            num_beams=num_beams,
            # no_repeat_ngram_size=3,
            max_new_tokens=1024,
            use_cache=True)

    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[:-len(stop_str)]
    outputs = outputs.strip()

    return outputs

def generate_zeroshot_clip(model, image_processor, tokenizer, query_image, a, b, relation, relations):
        
    if (b):
        prompts = create_prompts_clip(a, b, relation, relations)
    else:
        prompts = create_prompts_single_clip(a, relation, relations)
            
    with torch.no_grad():
        image = image_processor(query_image).unsqueeze(0).to(device)
        image_feature = model.encode_image(image).cpu().numpy()
        
        text = tokenizer(prompts).to(device)
        text_features = model.encode_text(text).cpu().numpy()
        
        text_probs = (100.0 * torch.tensor(image_feature) @ text_features.T).softmax(dim=-1)
        res, ind = text_probs.topk(1)
        
    answer_text = prompts[ind]

    return answer_text

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

def get_prediction_fasterrcn(model, img, confidence=0.5):
    """
    get_prediction
    parameters:
        - img_path - path of the input image
        - confidence - threshold value for prediction score
    method:
        - Image is obtained from the image path
        - the image is converted to image tensor using PyTorch's Transforms
        - image is passed through the model to get the predictions
        - class, box coordinates are obtained, but only prediction score > threshold
        are chosen.
    
    """
    transform = T.Compose([T.ToTensor()])
    img = transform(img)
    pred = model([img])
    pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].numpy())]
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().numpy())]
    pred_score = list(pred[0]['scores'].detach().numpy())
    pred_t = [pred_score.index(x) for x in pred_score if x>confidence]
    pred_boxes = [pred_boxes[i] for i in pred_t]
    pred_class = [pred_class[i] for i in pred_t]
    pred_score = [pred_score[i] for i in pred_t]
    return pred_boxes, pred_class, pred_score

def detect_object_fasterrcn(model, img, confidence=0.5, rect_th=2, text_size=2, text_th=2):
    boxes, pred_cls = get_prediction_fasterrcn(model, img, confidence)

    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # print(len(boxes))
    for i, box in enumerate(boxes):
        pt1 = (int(np.round(box[0][0])), int(np.round(box[0][1])))
        pt2 = (int(np.round(box[1][0])), int(np.round(box[1][1])))

        cv2.rectangle(img, pt1, pt2, color=(0, 255, 0), thickness=rect_th)
        cv2.putText(img,pred_cls[i], pt1, cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,255,0),thickness=text_th)
        
    plt.figure(figsize=(20,30))
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    plt.show()

    return boxes, pred_cls

def get_location(query_image, box):

    w = query_image.width
    h = query_image.height
    
    locations = {}
    locations["left"] = (w*.25, h*.5)
    locations["right"] = (w*.75, h*.5)
    locations["top"] = (w*.5, h*.25)
    locations["bottom"] = (w*.5, h*.75)
    
    dists = {}
    for loc in locations:
        coords = locations[loc]

        cx = (box[0][0]+box[1][0])/2
        cy = (box[0][1]+box[1][1])/2
        
        d = np.sqrt((coords[0] - cx)**2 + (coords[1] - cy)**2)
        dists[loc] = d
        
    ind = np.argmin(list(dists.values()))

    return list(locations.keys())[ind]

def generate_fasterrcn(model, query_image, a, b, relation, relations, task):
    boxes, pred_cls, pred_score = get_prediction_fasterrcn(model, query_image, confidence=0.5)

    if (task == "SPATIAL_REASONING"):

        locations_a = []
        for i, cl in enumerate(pred_cls):
            if (a == cl):
                loc_a = get_location(query_image, boxes[i])
                locations_a.append(loc_a)           

        if (b):
            locations_b = []
            for i, cl in enumerate(pred_cls):
                if (b == cl):
                    loc_b = get_location(query_image, boxes[i])
                    locations_b.append(loc_b)     

            for i, loc_a in enumerate(locations_a):
                for j, loc_b in enumerate(locations_b):
      
                    if (loc_a == "left" and loc_b == "right" and relation == "left"):
                        return relation
                    elif (loc_a == "right" and loc_b == "left" and relation == "right"):
                        return relation
                    elif (loc_a == "top" and loc_b == "bottom" and relation == "above"):
                        return relation  
                    elif (loc_a == "bottom" and loc_b == "top" and relation == "below"):
                        return relation
               
            return (locations_a[0] if len(locations_a)>0 else "none") + "_" + (locations_b[0] if len(locations_b)>0 else "none")
        else:
            for i, loc_a in enumerate(locations_a):
                if (loc_a == relation):
                    return relation
           
            return locations_a[0] if len(locations_a)>0 else "none"
        
    elif (task == "RECOGNITION" or task == "VISUAL_PROMPTING"):

        answer_text = " ".join(pred_cls)
        
    elif (task == "OBJECT_DETECTION"):

        answer_text = ""

        w, h = query_image.size

        for i, box in enumerate(boxes):
            answer_text += "[%f, %f, %f, %f] - %s - %f\n" % (box[0][0]/w, box[0][1]/h, box[1][0]/w, box[1][1]/h, pred_cls[i], pred_score[i])

        return answer_text
#endregion

def llm_eval(model_type, model, image_processor, headers, endpoint, tokenizer, image, query_image, query_text, a, b, relation, relations, task):
    if (model_type == "GPT"):
        answer_text, error = generate_text_gpt(headers, endpoint, query_image, query_text)    

        # retry
        while (error):
            print("Waiting 1 seconds...")
            time.sleep(1)
            answer_text, error = generate_text_gpt(headers, endpoint, query_image, query_text)
                    
    elif (model_type == "OF"):                
        answer_text = generate_text_of(model, image_processor, tokenizer, query_image, query_text)     
            
    elif (model_type == "LLAVA"):                
        answer_text = generate_text_llava(model, image_processor, tokenizer, query_image, query_text)  
                    
    elif (model_type == "PHI2V"):                                
        answer_text = generate_text_phi2v(query_image, query_text)  

    elif (model_type == "CLIP" or model_type == "CLIP_336" or model_type == "CLIP_SFT"): 
        answer_text = generate_zeroshot_clip(model, image_processor, tokenizer, query_image, a, b, relation, relations)  
        
    elif (model_type == "FASTERRCNN"):
        answer_text = generate_fasterrcn(model, query_image, a, b, relation, relations, task)
        
    return answer_text

def check_dectection_answer(answer_text, a, b=None):

    answer_text = answer_text.lower()
    answer_text = answer_text.replace("sofa", "couch")
    answer_text = answer_text.replace("sneakers", "shoes")
    answer_text = answer_text.replace("sneaker", "shoes")
    answer_text = answer_text.replace("shoe", "shoes")
    answer_text = answer_text.replace("weight", "dumbbell")
    
    if (a): 
        a = a.lower()
    if (b):
        b = b.lower()

    a_parts = a.split(" ")                
    ps = []

    for p in a_parts:
        correct = True if p in answer_text.lower() else False
        ps.append(correct)
                     
    pred_a_bool = any(ps)                    

    if (usePairs):
        b_parts = b.split(" ")                
        ps = []

        for p in b_parts:
            correct = True if p in answer_text.lower() else False
            ps.append(correct)
                    
        pred_b_bool = any(ps)

        pred = "correct" if (pred_a_bool and pred_b_bool) else "incorrect"

        relation = a + " and " + b
    else:
        pred = "correct" if pred_a_bool else "incorrect"
                
        relation = a

    return relation, pred

def test_spatial(path, model_type, usePairs, task, detectionPrompt=None):    

    rows = []
            
    with open(os.path.join(path, 'val_prompts.csv'), newline='') as csvfile: 
        reader = csv.reader(csvfile)
        for row in reader:
            rows.append(row)
    
    out_path = os.path.join(path, 'val_results_%s_%s.csv' % (task.lower(), model_type.lower()))

    questions_answers = {}
    preds = {}
    
    if (os.path.exists(out_path)):
        with open(out_path, newline='') as csvfile: 
            reader = csv.reader(csvfile)
            for row in reader:
                image = row[0]
                caption = row[1] 
                query_text = detectionPrompt if (detectionPrompt) else row[2] 
                relation = row[3] 
                answer_text = row[4]
                pred = row[5]
                               
                if (pred != "error"):                    
                    questions_answers[image] = [caption, query_text, relation, answer_text, pred]
                    preds[image] = pred
                    
    print("num samples %d" % len(rows))

    headers = None
    endpoint = None
    model = None
    image_processor = None
    tokenizer = None
    
    if (model_type == "GPT"):
        headers, endpoint = getGPT4()
        start = timer()
    elif (model_type == "OF"):
        model, image_processor, tokenizer = getOpenFlamingo()
    elif (model_type == "LLAVA"):
        model, image_processor, tokenizer = getLLaVA()
    elif (model_type == "CLIP"):
        model, image_processor, tokenizer = getOpenCLIP(arch = "ViT-L-14", pretrained = 'openai')
    elif (model_type == "CLIP_SFT"):
        model, image_processor, tokenizer = getOpenCLIP(arch = "ViT-L-14", pretrained = 'E:/Source/open_clip/logs/best_spatial_checkpoint/epoch_2.pt')
    elif (model_type == "CLIP_336"):
        model, image_processor, tokenizer = getOpenCLIP(arch = "ViT-L-14-336", pretrained = 'openai')        
    elif (model_type == "FASTERRCNN"):
        model = getFasterRCNN()        
        
    if (usePairs):
        relations = ["left", "right", "above", "below"]  
    else:
        relations = ["left", "right", "top", "bottom"]  

    for i, row in tqdm(enumerate(rows)): 

        image = row[0]            
        caption = row[1]  
        query_text = detectionPrompt if (detectionPrompt) else row[2]
        relation = row[3]
        
        if (image in questions_answers):
            continue
        try:

            image_path = os.path.join(path, image).replace("\\","/")
            query_image = Image.open(image_path)

            parts = image.split("\\")

            if (usePairs):
                a, b = parts[1].split("_")
            else:
                a = parts[1]
                b = None

            answer_text = llm_eval(model_type, model, image_processor, headers, endpoint, tokenizer, image, query_image, query_text, a, b, relation, relations, task)
                
            if (detectionPrompt):
                correct_text, pred = check_dectection_answer(answer_text, a, b)
            else:
                pred = check_answer(relation, relations, answer_text) 
                correct_text = relation
            
            # answer_text2 = llm_eval(model_type, model, image_processor, headers, endpoint, tokenizer, image, query_image, query_text, a, b, relation, relations, "RECOGNITION")
            # correct_text2, pred2 = check_dectection_answer(answer_text2, a, b)

            # if (pred == "correct" and pred2 != "correct"):
            #     print("%d: %s\n" % (i, image_path))

            print("Query text: %s\nAnswer text: %s\nCorrect: %s\n" % (query_text, answer_text, correct_text))

            preds[image] = pred

            #if (i % 10  == 0):
                    
            correct = sum(1 for pred in preds.values() if pred == "correct")
            incorrect = sum(1 for pred in preds.values() if pred == "incorrect")
            indetermined = sum(1 for pred in preds.values() if pred == "none")        
            n = sum(1 for pred in preds.values() if pred != "error")
            
            print("Mean correct %f incorrect %f indetermined %f" % (correct/n, incorrect/n, indetermined/n))

            #query_image.show()
   
        except Exception as e:
            print(e)
            answer_text = str(e.args)
            pred = "error"
            pass          
        
        except BaseException as e:
            print(e)
            answer_text = str(e.args)
            pred = "error"            
            pass        

        questions_answers[image] = [caption, query_text, relation, answer_text.replace("\n","\\n"), pred]

        with open(out_path, 'w', newline='', encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile, delimiter=',')

            for row in questions_answers.keys():
                writer.writerow([row] + questions_answers[row])
                
        if (model_type == "GPT"):

            time.sleep(2)

            end = timer()
            if (end - start > 600): #get new token every 30 min
                print("Getting new GPT token after %.02f mins" % ((end - start)/60))
                headers, endpoint = getGPT4()
                start = timer()
            else:
                print("GPT token is %.02f mins old" % ((end - start)/60))

    correct = sum(1 for pred in preds.values() if pred == "correct")
    incorrect = sum(1 for pred in preds.values() if pred == "incorrect")
    indetermined = sum(1 for pred in preds.values() if pred == "none")        
    n = sum(1 for pred in preds.values() if pred != "error")

    print("Mean correct %f incorrect %f indetermined %f" % (correct/n, incorrect/n, indetermined/n))

    with open(out_path, 'w', newline='', encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile, delimiter=',')

        for row in questions_answers.keys():
            writer.writerow([row] + questions_answers[row])

def score_prompting_and_rec(path, model_type, usePairs, recognition, detectionPrompt=None):    

    if (task == "SPATIAL_REASONING"):
        out_path = os.path.join(path, 'val_results_%s.csv' % (model_type.lower()))
    elif (task == "VISUAL_PROMPTING"):
        out_path = os.path.join(path, 'val_results_detection_prompt_%s.csv' % (model_type.lower()))        
    else:
        out_path = os.path.join(path, 'val_results_%s_%s.csv' % (task.lower(), model_type.lower()))
    
    rows = []
            
    with open(out_path, newline='', encoding="utf-8") as csvfile: 
        reader = csv.reader(csvfile)
        for row in reader:
            rows.append(row)
    
    print("num samples %d" % len(rows))     
        
    if (usePairs):
        relations = ["left", "right", "above", "below"]  
    else:
        relations = ["left", "right", "top", "bottom"]  

    preds = []

    for i, row in enumerate(rows): 

        if (len(row) == 6):
            image = row[0]
            caption = row[1] 
            query_text = row[2] 
            relation = row[3] if (row[3]) else image.split("\\")[2]
            answer_text = row[4]
            pred = row[5]
        else:
            image = row[0]
            query_text = row[1] 
            relation = image.split("\\")[2]
            answer_text = row[2]
            pred = row[3]
        
        parts = image.split("\\")

        if (usePairs):
            a, b = parts[1].split("_")
        else:
            a = parts[1]
            b = ""

        if ("filtered" in answer_text):
            pred = "none"
        else:
            if (detectionPrompt):
                correct_text, pred = check_dectection_answer(answer_text, a, b)
            else:
                pred = check_answer(relation, relations, answer_text) 
                correct_text = relation

        if (pred == "incorrect" or pred == "error"):
            print("Query text: %s\nAnswer text: %s\nCorrect: %s\n" % (query_text, answer_text, correct_text))

        preds.append(pred)

    correct = sum(1 for pred in preds if pred == "correct")
    incorrect = sum(1 for pred in preds if pred == "incorrect")
    indetermined = sum(1 for pred in preds if pred == "none")        
    n = sum(1 for pred in preds if pred != "error")

    print("Mean correct %f incorrect %f indetermined %f" % (correct/n, incorrect/n, indetermined/n))

def score_detection(path, model_type):    
    
    json_path = os.path.join(path, 'coco_instances.json')

    coco = COCO(json_path)
    coco_img_ids = coco.getImgIds()
    coco_imgs = coco.loadImgs(coco_img_ids)
    coco_file_name_to_id = {}

    for c in coco_imgs:
        coco_file_name_to_id[c["file_name"]] = c["id"]

    coco_cat_ids = coco.getCatIds()
    coco_cats = coco.loadCats(coco_cat_ids)
    coco_cat_name_to_id = {}
    
    for c in coco_cats:
        coco_cat_name_to_id[c["name"]] = c["id"]

    out_path = os.path.join(path, 'val_results_object_detection_%s.csv' % (model_type.lower()))
    
    rows = []
            
    with open(out_path, newline='', encoding="utf-8") as csvfile: 
        reader = csv.reader(csvfile)
        for row in reader:
            rows.append(row)
    
    print("num samples %d" % len(rows))     
    
    img_infos = []
    
    tp = 0
    fp = 0
    tn = 0
    fn = 0    

    annotations = []

    for image_id, row in enumerate(rows): 
        image = row[0]
        answer_text = row[4]

        if ("[" in answer_text):
            
            img = Image.open(os.path.join(path, image))
            w, h = img.size

            # img = np.array(img)

            dets = answer_text.split("\\n")

            for det in dets:
                try:
                    parts = det.split("-")
                    if (len(parts)==3):
                        box_string, label, confidence = parts
                        confidence = float(confidence.strip())
                    elif (len(parts)==2):
                        box_string, label = parts
                        confidence = .5
                    else:
                        continue
            
                    box_string = box_string.replace("x0","")
                    box_string = box_string.replace("y0","")
                    box_string = box_string.replace("x1","")
                    box_string = box_string.replace("y1","")
                    
                    box = re.findall(r"[-+]?\d*\.\d+|\d+", box_string)

                    box = [float(b) for b in box]
                    xmin, ymin, xmax, ymax = box
                    box = [xmin*w, ymin*h, (xmax-xmin)*w, (ymax-ymin)*h]
                        
                    # pt1 = (int(np.round(box[0])), int(np.round(box[1])))
                    # pt2 = (int(np.round(box[2])), int(np.round(box[3])))

                    # rect_th=2
                    # text_size=2
                    # text_th=2
                        
                    # cv2.rectangle(img, pt1, pt2, color=(0, 255, 0), thickness=rect_th)
                    # cv2.putText(img,label, pt1, cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,255,0),thickness=text_th)

                    label = label.strip()

                    if (label in coco_cat_name_to_id.keys()):
                        annotation = {
                            "image_id": coco_file_name_to_id[image],
                            'category_id':  coco_cat_name_to_id[label],
                            'bbox': box,
                            'score': confidence,
                        }
                        annotations.append(annotation)
                        
                except Exception as e:
                    print(e)
                    pass          
        
                except BaseException as e:
                    print(e)
                    pass   

    json_out_path = os.path.join(path, 'val_results_object_detection_%s.json' % (model_type.lower()))

    with open(json_out_path, "w") as f:
        json.dump(annotations, f, indent=4)

    cocovalPrediction = coco.loadRes(json_out_path)

    cocoEval = COCOeval(coco, cocovalPrediction, "bbox")    
    cocoEval.params.useCats = 0
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

def generate_prompts_spatial(path, usePairs):    
    
    rows = []
            
    with open(os.path.join(path, 'val.csv'), newline='') as csvfile: 
        reader = csv.reader(csvfile)
        for row in reader:
            rows.append(row)
            if (usePairs):
                next(reader)
            
    query_texts = []

    if (usePairs):
        relations = ["left", "right", "above", "below"]  
    else:
        relations = ["left", "right", "top", "bottom"]  

    for i, row in tqdm(enumerate(rows)):     
        
        image = row[0]

        parts = image.split("\\")

        if (usePairs):
            a, b = parts[1].split("_")
            a = a.lower()
            b = b.lower()            
        else:
            a = parts[1]
            a = a.lower()
            b = None            
        
        relation = parts[2]

        query_text = create_prompts(a, relation, relations, b=b, options=True)
        query_texts.append(query_text)
            
        rows[i] += [query_text, relation]

    out_path = os.path.join(path, 'val_prompts.csv')   

    with open(out_path, 'w', newline='', encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        for row in rows:
            writer.writerow(row)

if __name__ == "__main__":
    #test_spatial()
    
    if (os.name == "nt"):
        drive = "E:/"    
    else:
        drive = "/mnt/e"
    
    usePairs = False
    
    task = "RECOGNITION"
    #task = "VISUAL_PROMPTING"
    #task = "SPATIAL_REASONING"
    #task = "OBJECT_DETECTION"
    
    model_type = "GPT"
    #model_type = "OF"
    #model_type = "LLAVA"
    #model_type = "PHI2V"
    #model_type = "CLIP"
    #model_type = "CLIP_SFT"
    #model_type = "CLIP_336"
    #model_type = "FASTERRCNN"

    if (task == "RECOGNITION"):
        if (usePairs):
            path = drive + "/Source/EffortlessCVSystem/Data/nococo_spatial_pairs_backgrounds"
        else:
            path = drive + "/Source/EffortlessCVSystem/Data/nococo_spatial_single_backgrounds"

        detectionPrompt = "What objects are in this image?"
        
    elif (task == "VISUAL_PROMPTING"):
        if (usePairs):
            path = drive + "/Source/EffortlessCVSystem/Data/coco_spatial_pairs_boxes"
            
            detectionPrompt = "What objects are in the red and yellow box in this image?"
        else:
            path = drive + "/Source/EffortlessCVSystem/Data/coco_spatial_single_boxes"
            
            detectionPrompt = "What object is in the red box in this image?"
            
    elif (task == "SPATIAL_REASONING"):

        if (usePairs):
            path = drive + "/Source/EffortlessCVSystem/Data/nococo_spatial_pairs_backgrounds"
        else:
            path = drive + "/Source/EffortlessCVSystem/Data/nococo_spatial_single_backgrounds"

        detectionPrompt = None

    elif (task == "OBJECT_DETECTION"):
        
        if (usePairs):
            path = drive + "/Source/EffortlessCVSystem/Data/coco_spatial_pairs_detection"
        else:
            path = drive + "/Source/EffortlessCVSystem/Data/coco_spatial_single_detection"

        detectionPrompt = "Return the label and the bounding box normalized coordinates of the objects in the following image in the following format: [x0, y0, x1, y1] - class - confidence"
        
    #generate_prompts_spatial(path, usePairs)

    test_spatial(path, model_type, usePairs, task, detectionPrompt)

    #score_prompting_and_rec(path, model_type, usePairs, task, detectionPrompt)

    #score_detection(path, model_type)