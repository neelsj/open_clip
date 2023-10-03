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
    endpoint = f"{base_url}/rainbow?api-version=2023-07-01-preview" 
    
    return headers, endpoint

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

def test_spatial(path):
    # Load the model

    data = []
    with open(os.path.join(path, 'val.csv'), newline='') as csvfile: 
        reader = csv.reader(csvfile)
        for row in reader:
            data.append(row)

    out_path = os.path.join(path, 'val_results_gpt.csv')

    questions_answers = []

    if (os.path.exists(out_path)):
       preds = []
       questions_answers = []

       with open(out_path, newline='') as csvfile: 
           reader = csv.reader(csvfile)
           for row in reader:
               query_image = row[0]
               query_text = row[1]
               answer_text = row[2]

               questions_answers.append([query_image, query_text, answer_text])
                  

    start_row = int(len(questions_answers))

    print("num samples %d" % len(data))

    headers, endpoint = getGPT4()
    start = timer()

    for i, row in tqdm(enumerate(data), initial=start_row): 
            
        try:
            query_image = row[0]
            query_text = row[1]
            
            answer_text = generate_text_gpt(headers, endpoint, query_image, query_text)    

            while (not isinstance(answer_text, str)):
                print("Waiting %d seconds..." % answer_text)
                time.sleep(float(answer_text) + .1)
                answer_text = generate_text_gpt(headers, endpoint, query_image, query_text)
                   
            print("Query text: %s\nAnswer text: %s" % (query_text, answer_text))    

            questions_answers.append([query_image, query_text, answer_text])

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

    with open(out_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')

        for row in questions_answers:
            writer.writerow(row)

if __name__ == "__main__":
    test_spatial()