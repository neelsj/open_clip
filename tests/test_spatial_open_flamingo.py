import torch
from PIL import Image
#import open_clip

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

model_type = "GPT"
#model_type = "OF"
#model_type = "LLAVA"
#model_type = "PSI2V"

usePairs = False

if (usePairs):
    relations = ["left", "right", "above", "below"]  
else:
    relations = ["left", "right", "top", "bottom"]  

if (model_type == "LLAVA" or model_type == "OF"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("\nUsing %s\n" % device)

def getGPT4():
    from azure.identity import DefaultAzureCredential 
    RESOURCE_NAME = "gpt-visual-wus" 
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
    endpoint = f"{base_url}/chat/completions?api-version=2023-09-15-preview"
    
    return headers, endpoint

def getOpenFlamingo():
    import open_flamingo
    from huggingface_hub import hf_hub_download

    #clip_vision_encoder_pretrained = "E:/Source/open_clip/logs/best_spatial_checkpoint/epoch_1.pt"
    clip_vision_encoder_pretrained = "openai"
    #clip_vision_encoder_pretrained = "laion2b_s32b_b82k"
    
    model, image_processor, tokenizer = open_flamingo.create_model_and_transforms(
        clip_vision_encoder_path = "ViT-L-14",
        clip_vision_encoder_pretrained = clip_vision_encoder_pretrained,
        lang_encoder_path = "anas-awadalla/mpt-1b-redpajama-200b-dolly",
        tokenizer_path = "anas-awadalla/mpt-1b-redpajama-200b-dolly",
        cross_attn_every_n_layers=1
    )
    model.to(device)
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

    return model, image_processor, tokenizer 

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

    if (model_type == "OF"):
        prompt = "<image>Question: " + prompt + " Answer:"
        
    elif (model_type == "PSI2V"):
         #prompt = prompt + "\n<|image_1|>"
         prompt = prompt.replace("image", "<|image_1|>")
         
    print(prompt)

    return prompt

def check_answer(correct_relation, answer_text):

    answer_text = answer_text.lower()

    total_count = 0
    relations_count = {}
    for relation in relations:
        x = answer_text.count(relation)
        relations_count[relation] = x
        
    total_count = sum(relations_count.values())
    
    pred = "correct" if (total_count == 1 and relations_count[correct_relation] == 1) else "incorrect" if (total_count == 1 and relations_count[correct_relation] == 0) else "none"

    return pred

def generate_text_gpt(headers, endpoint, query_image, query_text):

    buffered = BytesIO()
    query_image.save(buffered, format="JPEG")
    base64_bytes = base64.b64encode(buffered.getvalue())
    base64_string = base64_bytes.decode('utf-8')

    # data = {   
    #     "transcript": [ 
    #         { "type": "text", "data": query_text }, 
    #         { "type": "image", "data": base64_string } 

    #     ],   

    #     "max_tokens": 50,   
    #     "temperature": 0.7,   
    #     "n": 1 
    # }   

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
        "max_tokens": 50,  
        "temperature": 0.7,  
        "n": 1
    }  

    # Make the API call   

    response = requests.post(endpoint, headers=headers, data=json.dumps(data))    
        
    #print(response.json())

    if (response.status_code == 200):
        answer_text = response.json()["choices"][0]["message"]["content"].strip()
    else:
        msg = response.json()["error"]["message"]
        print("\n"+msg)
        answer_text = [int(s) for s in msg.split() if s.isdigit()][0]

    return answer_text

def generate_text_psi2v(query_image, query_text):

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

def test_spatial():    
    if (os.name == "nt"):
        drive = "E:/"    
    else:
        drive = "/mnt/e"
    
    if (usePairs):
        path = drive + "/Source/EffortlessCVSystem/Data/coco_spatial_pairs_backgrounds"
    else:
        path = drive + "/Source/EffortlessCVSystem/Data/coco_spatial_single_backgrounds"
    
    images = []
    with open(os.path.join(path, 'val.csv'), newline='') as csvfile: 
        reader = csv.reader(csvfile)
        for row in reader:
            images.append(row[0])

    # images = list(set(images))

    out_path = os.path.join(path, 'val_results_gpt.csv' if model_type == "GPT" else 'val_results_of.csv' if model_type == "OF" else 'val_results_llava.csv' if model_type == "LLAVA" else 'val_results_psi2v.csv')    
    
    if (os.path.exists(out_path)):
       preds = []
       questions_answers = []

       with open(out_path, newline='') as csvfile: 
           reader = csv.reader(csvfile)
           for row in reader:
               pred = row[3]

               preds.append(pred)
               questions_answers.append(row)
    else:
        preds = []
        questions_answers = []

    start_row = int(len(preds))

    print("num samples %d start at %d" % (len(images), start_row))

    if (model_type == "GPT"):
        headers, endpoint = getGPT4()
        start = timer()
    elif (model_type == "OF"):
        model, image_processor, tokenizer = getOpenFlamingo()
    elif (model_type == "LLAVA"):
        model, image_processor, tokenizer = getLLaVA()
        
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
            
            if (model_type == "GPT"):
                query_text = create_prompts(a, b, relation, options=True)
                answer_text = generate_text_gpt(headers, endpoint, query_image, query_text)    

                while (not isinstance(answer_text, str)):
                    print("Waiting %d seconds..." % answer_text)
                    time.sleep(float(answer_text) + .1)
                    answer_text = generate_text_gpt(headers, endpoint, query_image, query_text)

            elif (model_type == "OF"):
                query_text = create_prompts(a, b, relation, options=True)
                answer_text = generate_text_of(model, image_processor, tokenizer, query_image, query_text)     
            elif (model_type == "LLAVA"):
                query_text = create_prompts(a, b, relation, options=True)
                answer_text = generate_text_llava(model, image_processor, tokenizer, query_image, query_text)  
            elif (model_type == "PSI2V"):                
                query_text = create_prompts(a, b, relation, options=True)
                answer_text = generate_text_psi2v(query_image, query_text)  
                
            print("Query text: %s\nAnswer text: %s\nCorrect: %s\n" % (query_text, answer_text, relation))

            pred = check_answer(relation, answer_text) 
            preds.append(pred)

            #if (i % 10  == 0):
                    
            correct = sum(1 for pred in preds if pred == "correct")
            incorrect = sum(1 for pred in preds if pred == "incorrect")
            indetermined = sum(1 for pred in preds if pred == "none")        

            questions_answers.append([image, query_text, answer_text.replace("\n","\\n"), pred])

            print("Mean correct %f incorrect %f indetermined %f" % (correct/len(preds), incorrect/len(preds), indetermined/len(preds)))

            #query_image.show()

            if (model_type == "GPT"):

                with open(out_path, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile, delimiter=',')

                    for row in questions_answers:
                        writer.writerow(row)

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
    
    with open(out_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')

        for row in questions_answers:
            writer.writerow(row)

if __name__ == "__main__":
    test_spatial()