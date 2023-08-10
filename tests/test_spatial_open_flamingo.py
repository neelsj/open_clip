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

useGPT4 = False

usePairs = False

if (usePairs):
    relations = ["left", "right", "above", "below"]  
else:
    relations = ["left", "right", "top", "bottom"]  

device = "cuda" if torch.cuda.is_available() else "cpu"
print("\nUsing %s\n" % device)

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
    endpoint = f"{base_url}/rainbow?api-version=2023-03-15-preview" 

    return headers, endpoint

def getOpenFlamingo():
    import open_flamingo
    from huggingface_hub import hf_hub_download

    clip_vision_encoder_pretrained = "E:/Source/open_clip/logs/2023_08_08-22_48_12-model_ViT-L-14-lr_0.0001-b_64-j_8-p_amp/checkpoints/epoch_latest.pt"
    #clip_vision_encoder_pretrained = "datacomp_xl_s13b_b90k"

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

def getLLaVA():
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
    from llava.conversation import conv_templates, SeparatorStyle
    from llava.model.builder import load_pretrained_model
    from llava.utils import disable_torch_init
    from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

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

def mirror_relation(relation):
    if (relation == "left"):
        return "right"
    elif (relation == "right"):
        return "left"
    elif (relation == "above"):
        return "below"
    else:
        return "above"

def create_prompts(a, b, relation, options = False):

    relationsA = relations      
    random.shuffle(relationsA)

    if (usePairs):
        relationsA = ", ".join(relationsA[:-1]) + ", or " + get_relation(relationsA[-1])        
        prompt = "Is the " + a + " " + relationsA + " the " + b  + "?"
    else:
        relationsA = ", ".join(relationsA[:-1]) + ", or " + relationsA[-1]
        
        prompt = "Is the " + a + " on the " + relationsA + " of the image?"

    if (options):
        relationsB = relations
        random.shuffle(relationsB)
        relationsB = ", ".join(relationsB[:-1]) + ", or " + relationsB[-1]
        prompt += (" Answer with one of (%s) only." % relationsB)

    if (not useGPT4):
        prompt = "<image>Question: " + prompt + " Answer:"

    return prompt

def check_answer(correct_relation, answer_text):

    answer_text = answer_text.lower()

    total_count = 0
    relations_count = {}
    for relation in relations:
        x = answer_text.count(relation)
        relations_count[relation] = x
        total_count += x

    pred = "none" if total_count == 0 else "correct" if ((relations_count[correct_relation] >= 1) and ((total_count-relations_count[correct_relation])==0)) else "incorrect"

    return pred

def generate_text_gpt(headers, endpoint, query_image, query_text):

    buffered = BytesIO()
    query_image.save(buffered, format="JPEG")
    base64_bytes = base64.b64encode(buffered.getvalue())
    base64_string = base64_bytes.decode('utf-8')

    data = {   
        "transcript": [ 
            { "type": "text", "data": query_text }, 
            { "type": "image", "data": base64_string } 

        ],   

        "max_tokens": 50,   
        "temperature": 0.7,   
        "n": 1 
    }   

    # Make the API call   

    response = requests.post(endpoint, headers=headers, data=json.dumps(data))    
        
    print(response.json())

    if (response.status_code == 200):
        answer_text = response.json()["choices"][0]["text"].strip()
    else:
        msg = response.json()["error"]["message"]
        print("\n"+msg)
        answer_text = [int(s) for s in msg.split() if s.isdigit()][0]

    return answer_text

def generate_text_of(model, image_processor, tokenizer, query_image, query_text):
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

def test_spatial():
    # Load the model

    if (usePairs):
        path = "e:/Source/EffortlessCVSystem/Data/coco_spatial_pair_backgrounds_finetune"
    else:
        path = "e:/Source/EffortlessCVSystem/Data/coco_spatial_single_backgrounds_finetune"

    images = []
    with open(os.path.join(path, 'val.csv'), newline='') as csvfile: 
        reader = csv.reader(csvfile)
        for row in reader:
            images.append(row[0])

    images = list(set(images))

    #out_path = os.path.join(path, 'val_results_gpt.csv' if useGPT4 else 'val_results_of.csv')

    #if (os.path.exists(out_path)):
    #    preds = []
    #    questions_answers = []

    #    with open(out_path, newline='') as csvfile: 
    #        reader = csv.reader(csvfile)
    #        for row in reader:
    #            image = row[0]
    #            query_text = row[1]
    #            answer_text = row[2]
    #            pred = row[3]

    #            preds.append(pred)
    #            questions_answers.append([image, query_text, answer_text, pred])
    #else:

    preds = []
    questions_answers = []

    start_row = int(len(preds))

    print("num samples %d" % len(images))

    if (useGPT4):
        headers, endpoint = getGPT4()
        start = timer()
    else:
        model, image_processor, tokenizer = getOpenFlamingo()

    try:
        for i, image in tqdm(enumerate(images), initial=start_row): 
            parts = image.split("\\")

            if (usePairs):
                a, b = parts[1].split("_")
            else:
                a = parts[1]
                b = None

            relation = parts[2]

            image_path = os.path.join(path, image).replace("\\","/")
            query_image = Image.open(image_path)
            query_image.show()

            if (useGPT4):
                query_text = create_prompts(a, b, relation, options=True)
                answer_text = generate_text_gpt(headers, endpoint, query_image, query_text)    

                while (not isinstance(answer_text, str)):
                    print("Waiting %d seconds..." % answer_text)
                    time.sleep(float(answer_text) + .1)
                    answer_text = generate_text_gpt(headers, endpoint, query_image, query_text)

            else:
                query_text = create_prompts(a, b, relation)
                answer_text = generate_text_of(model, image_processor, tokenizer, query_image, query_text)     
                   
            print("Query text: %s\nAnswer text: %s" % (query_text, answer_text))

            pred = check_answer(relation, answer_text) 
            preds.append(pred)

            #if (i % 10  == 0):
                    
            correct = sum(1 for pred in preds if pred == "correct")
            incorrect = sum(1 for pred in preds if pred == "incorrect")
            indetermined = sum(1 for pred in preds if pred == "none")        

            questions_answers.append([image, query_text, answer_text, pred])

            print("Mean correct %f incorrect %f indetermined %f" % (correct/len(preds), incorrect/len(preds), indetermined/len(preds)))

            if (useGPT4):
                time.sleep(60)

                end = timer()
                if (end - start > 600): #get new token every 30 min
                    print("Getting new GPT token after %.02f mins" % ((end - start)/60))
                    headers, endpoint = getGPT4()
                    start = timer()
                else:
                    print("GPT token is %.02f mins old" % ((end - start)/60))
                #if (i > 3):
                #    break

    except Exception as e:
        print(e)
        pass          
    except BaseException as e:
        print(e)
        pass  

    correct = sum(1 for pred in preds if pred == "correct")
    incorrect = sum(1 for pred in preds if pred == "incorrect")
    indetermined = sum(1 for pred in preds if pred == "none")

    print("Mean correct %f incorrect %f indetermined %f" % (correct/len(preds), incorrect/len(preds), indetermined/len(preds)))

    out_path = os.path.join(path, 'val_results_gpt2.csv' if useGPT4 else 'val_results_of2.csv')

    with open(out_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')

        for row in questions_answers:
            writer.writerow(row)

if __name__ == "__main__":
    test_spatial()