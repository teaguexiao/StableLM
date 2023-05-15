from flask import Flask, request, send_file, render_template
from diffusers import (
    StableDiffusionPipeline,
    EulerDiscreteScheduler,
    StableDiffusionImg2ImgPipeline,
)
import torch
from PIL import Image
from io import BytesIO
import requests
from flask_cors import CORS, cross_origin
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
import json
from datetime import datetime
from uuid import uuid4
import os
import glob
from urllib.parse import urljoin

import logging
logging.basicConfig(filename='/var/log/flask.log', level=logging.DEBUG)

import configparser
# Initialize OPEN AI Key
config = configparser.ConfigParser()
config.read('config.ini')
os.environ["OPENAI_API_KEY"] = config.get('OPENAI-API', 'key')

#for OPENAI
from langchain import OpenAI, ConversationChain
from langchain import PromptTemplate

app = Flask(__name__, template_folder="frontend", static_folder="frontend")
CORS(app, support_credentials=True)

@app.route("/api")
def index():
    return "This is the API backend by Flask", 200

@app.get("/api/health")
def health_check():
    return "Healthy", 200

@app.post("/api/create/list_all_img")
def list_all_img():
    data = request.json
    pageSize = int(data["pageSize"])
    pageNum = int(data['pageNum'])
    #{pageSize: 999, pageNum: 1}
    
    directory = "/etc/nginx/html/aigc-static/generated_img"
    images = []
    images = glob.glob(os.path.join(directory, '*'))
    images.sort(reverse=True)

    base_url = "http://54.249.140.115:8890"
    images = [s.replace("/etc/nginx/html/aigc-static/generated_img", base_url+"/aigc-static/generated_img") for s in images]
    
    #Paging
    total_pages = (len(images) + pageSize - 1) // pageSize  # Calculate the total number of pages

    if pageNum < 1 or pageNum > total_pages:
        # Invalid page number, return an empty list
        result = {
            "code" : 200,
            "data" : [],# Retrieve the page using list slicing
            "total" : 0,
            "msg" : "无效的pageNum参数"
        }

    start_index = (pageNum - 1) * pageSize
    end_index = start_index + pageSize

    result = {
        "code" : 200,
        "data" : images[start_index:end_index],# Retrieve the page using list slicing
        "total" : len(images),
        "msg" : "图片列表展示成功！"
    }
    
    return json.dumps(result), 200


# @app.post("/api/create/create_img")
def create_img():
    data = request.json
    #payload example
    '''
    {
        "koc_id": 1,
        "koc_name": "JOJO",
        "koc_type": "代言人",
        "koc_url": "/aigcimgs/koc/1.svg",
        "product_id": 1,
        "product_name": "OPPO Find N2 Flip",
        "product_type": "手机",
        "product_url": "/aigcimgs/product/1.png",
        "scen_id": 1,
        "scen_name": "海边",
        "style_id": 1,
        "style_name": "拟真"
    }
    '''
    #model_id = "andite/anything-v4.0" # 默认的，高品质、高细节的动漫风格
    #model_id = 'Envvi/Inkpunk-Diffusion' # 温克朋克风格，提示词 nvinkpunk
    #model_id = 'nousr/robo-diffusion-2-base' # 看起来很酷的机器人，提示词 nousr robot
    #model_id = 'prompthero/openjourney' # openjorney 风格,提示词 mdjrny-v4 style
    #model_id = 'dreamlike-art/dreamlike-photoreal-2.0' #写实，真实风格，提示词 photo
    #model_id = "stabilityai/stable-diffusion-2"
    model_id = "ductridev/chilloutmix_NiPrunedFp32Fix"
    #local pipeline
    #model_id = './models/chilloutmix_NiPrunedFp32Fix.safetensors'
    #model_id = './models/dreamlike-photoreal-2.0.safetensors'
    #model_id = './models/v1-5-pruned-emaonly.safetensors'
    
    filename = datetime.now().strftime('%Y%m%d-%H%M%S-') + str(uuid4()) + ".png"
    output = "/etc/nginx/html/aigc-static/generated_img/" + filename
    print(output)

    scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, scheduler=scheduler, torch_dtype=torch.float16, safety_checker=None
    )
    pipe = pipe.to("cuda")
    #prompt = "(RAW photo, best quality), (realistic, photo-realistic:1.2),1girl holding a Oppo phone, on the street, outdoor, smile, (high detailed skin:1.4), soft lighting, high quality,fair skin, looking at viewer, straight-on,full body, colorful startrails, gothic architecture, photorealistic,oil painting (medium), solo,long_hair, grey_hair_grey_eyes, eye contact, standing"
    #negative_prompt = "ng_deepnegative_v1_75t, paintings, sketches, (worst quality:2), (low quality:2), (normal quality:2), lowres, ((monochrome)), (grayscale:1.2), skin spots, acnes, skin blemishes, age spot, glans,extra fingers,fewer fingers,(watermark:1.2),(letters:1.2),(nsfw:1.2),teeth,mole,multiple breasts, (mutated hands and fingers:1.5 ), (long body :1.3), (mutation, poorly drawn :1.2) ,blurred,one hand with more than 5 fingers,one hand with less than 5 fingers,one hand with more than 5 digit,one hand with less than 5 digit,more than two shoes,more than 2 nipples,different nipples,more than 1 left hand,more than 1 right hand,more than 2 thighs,more than 2 legs,worst quality,low quality,normal quality"
    #欧美范
    prompt = "olgaabrom (sharp focus:1.2), photo, (beautiful face:1.1), detailed eyes, luscious lips, (cat eye makeup:0.85), wearing (bikini:1.2) on a (beach:1.2), depth of field, bokeh, 4K, HDR. by (James C. Christensen:1.2|Jeremy Lipking:1.1)."
    negative_prompt = "shiny skin, smooth skin, out of frame, (bad fingers:1.2), (bad hands:1.2), (bad limbs:1.1), (bad proportions:1.2), (bad quality:1.1), (poorly drawn face:1.3), ugly face, (poorly drawn:1.2), asymmetrical body, (bad anatomy:1.3), (flesh merge:1.2), (large breasts:1.0), (fleshpile:1.2), tiling, (generic:1.1), glitchy, (gross proportions:1.2), (malformed:1.2), extra arms, extra fingers, extra hands, extra legs, extra limbs, extra body parts, extra breasts, mangled fingers"
    
    image = pipe(prompt, negative_prompt=negative_prompt, num_images_per_prompt=2, guidance_scale=7.5, num_inference_steps=40, height=512, width=384).images[0]
    image.save(output)  
    return send_file(output), 200

# @app.post("/api/create/create_img_img2img")
@app.post("/api/create/create_img")
def img2img():
    data = request.json
    scen_id = data["scen_id"]
    
    model_id = "ductridev/chilloutmix_NiPrunedFp32Fix"
    filename = datetime.now().strftime('%Y%m-%d%H-%M%S-') + str(uuid4()) + ".png"
    output = "/etc/nginx/html/aigc-static/generated_img/" + filename
    print(output)

    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        model_id, torch_dtype=torch.float16
    )
    pipe = pipe.to("cuda")
    if scen_id == "1":
        url = "http://54.249.140.115:8890/aigc-static/img2img_sample/0.jpg"
    else:
        url = "http://54.249.140.115:8890/aigc-static/img2img_sample/0.jpg"

    response = requests.get(url)
    init_image = Image.open(BytesIO(response.content)).convert("RGB")
    
    prompt = "(RAW photo, best quality), (realistic, photo-realistic:1.2),1girl holding a Oppo phone in the hand, on the street, outdoor, smile, (high detailed skin:1.4), soft lighting, high quality,fair skin, looking at viewer, straight-on,full body, colorful startrails, gothic architecture, photorealistic,oil painting (medium), solo,long_hair, grey_hair_grey_eyes, eye contact, standing"
    negative_prompt = "ng_deepnegative_v1_75t, paintings, sketches, (worst quality:2), (low quality:2), (normal quality:2), lowres, ((monochrome)), (grayscale:1.2), skin spots, acnes, skin blemishes, age spot, glans,extra fingers,fewer fingers,(watermark:1.2),(letters:1.2),(nsfw:1.2),teeth,mole,multiple breasts, (mutated hands and fingers:1.5 ), (long body :1.3), (mutation, poorly drawn :1.2) ,blurred,one hand with more than 5 fingers,one hand with less than 5 fingers,one hand with more than 5 digit,one hand with less than 5 digit,more than two shoes,more than 2 nipples,different nipples,more than 1 left hand,more than 1 right hand,more than 2 thighs,more than 2 legs,worst quality,low quality,normal quality"

    image = pipe(
        prompt, negative_prompt=negative_prompt, image=init_image, strength=0.75, guidance_scale=7.5
    ).images[0]

    image.save(output)  
    return send_file(output), 200

@app.post("/api/create/create_text")
def create_text():
    #data = request.json
    #JSON Example
    '''
    {
        "scen_id": 1,
        "scen_name": "海边",
        "style_id": 1,
        "style_name": "拟真",
        "product_id": 1,
        "media_id": 1
    }
    '''
    llm = OpenAI(model_name="gpt-3.5-turbo")
    
    multiple_input_prompt = PromptTemplate(
        input_variables=["scen_name", "style_name", "product_name"], 
        template="从现在开始，你就是一名资深的旅游博主，你喜欢全世界环游，你热爱时尚和数码产品，你热爱生活，你很有激情，你喜欢分享各种内容到社交媒体。现在你要给{product_name}制作一篇使用体验文章，目的是吸引别人种草，你的配图场景是{scen_name}，画面风格是{style_name}。文章需要突出的产品功能是强劲性能，高刷新屏幕，5G通信，充电快，有手写笔等特点，但是不能很生硬地说出这些特点，你也不需要把以上的特点都描述出来，可以只体现其中几个特点就可以。\n还要求：\n1，尽可能体现真实使用过程\n2，多使用带有可爱的emoji\n3，多用短句，每句话不超过20个字\n4，多用空行和分段\n5，带有一些感叹或者是偏感受的描述字眼\n6，最后加一些相关的tag\n"
    )
    content = multiple_input_prompt.format(scen_name="海边", style_name="梦幻的", product_name="Oppo Pad 2 平板电脑")
    
    conversation = ConversationChain(llm=llm, verbose=True)
    output = conversation.predict(input=content)
    
    print(output)
    result = {
        "code" : 200,
        "data" : output,
        "msg" : "文本生成成功！"
    }
    
    return json.dumps(result), 200

app.run(host='0.0.0.0', port=5000, debug=True)
