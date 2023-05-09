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

#for OPENAI
from langchain import OpenAI, ConversationChain
from langchain import PromptTemplate

app = Flask(__name__, template_folder="frontend", static_folder="frontend")
CORS(app, support_credentials=True)

@app.route("/")
def index():
    return "Healthy", 200

@app.get("/health")
def health_check():
    return "Healthy", 200

@app.post("/create/create_img")
def create_img():
    data = request.json
    #model_id = "andite/anything-v4.0" # 默认的，高品质、高细节的动漫风格
    #model_id = 'Envvi/Inkpunk-Diffusion' # 温克朋克风格，提示词 nvinkpunk
    #model_id = 'nousr/robo-diffusion-2-base' # 看起来很酷的机器人，提示词 nousr robot
    #model_id = 'prompthero/openjourney' # openjorney 风格,提示词 mdjrny-v4 style
    #model_id = 'dreamlike-art/dreamlike-photoreal-2.0' #写实，真实风格，提示词 photo
    model_id = "stabilityai/stable-diffusion-2"
    output = "output_txt2img.png"

    scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, scheduler=scheduler, torch_dtype=torch.float16
    )
    pipe = pipe.to("cuda")
    prompt = "(RAW photo, best quality), (realistic, photo-realistic:1.2),1girl, on the street, mini skirt, shirt, crowded, outdoor, smile, (high detailed skin:1.4), soft lighting, high quality,fair skin, busty breasts, looking at viewer, straight-on,full body,depth of field,colorful startrails,gothic architecture,photorealistic,oil painting (medium) ,solo,long_hair,grey_hair_grey_eyes,eye contact,standing,tying hai"
    #neg_prompts = "ng_deepnegative_v1_75t, paintings, sketches, (worst quality:2), (low quality:2), (normal quality:2), lowres, ((monochrome)), (grayscale:1.2), skin spots, acnes, skin blemishes, age spot, glans,extra fingers,fewer fingers,(watermark:1.2),(letters:1.2),(nsfw:1.2),teeth,mole,multiple breasts, (mutated hands and fingers:1.5 ), (long body :1.3), (mutation, poorly drawn :1.2) ,blurred,one hand with more than 5 fingers,one hand with less than 5 fingers,one hand with more than 5 digit,one hand with less than 5 digit,more than two shoes,more than 2 nipples,different nipples,more than 1 left hand,more than 1 right hand,more than 2 thighs,more than 2 legs,worst quality,low quality,normal quality"

    image = pipe(prompt, guidance_scale=7.5, num_inference_steps=20,height=384, width=512).images[0]
    image.save(output)  
    return send_file(output), 200
#
@app.post("/create/create_text")
def create_text():
    #data = request.json
    llm = OpenAI(model_name="gpt-3.5-turbo")
    
    multiple_input_prompt = PromptTemplate(
        input_variables=["scen_name", "style_name", "product_name"], 
        template="从现在开始，你就是一名资深的旅游博主，你喜欢全世界环游，你热爱时尚和数码产品，你热爱生活，你很有激情，你喜欢分享各种内容到社交媒体。现在你要给{product_name}制作一篇使用体验文章，目的是吸引别人种草，你的配图场景是{scen_name}，画面风格是{style_name}。文章需要突出的产品功能是强劲性能，高刷新屏幕，5G通信，充电快，有手写笔等特点，但是不能很生硬地说出这些特点，你也不需要把以上的特点都描述出来，可以只体现其中几个特点就可以。\n还要求：\n1，尽可能体现真实使用过程\n2，多使用带有可爱的emoji\n3，多用短句，每句话不超过20个字\n4，多用空行和分段\n5，带有一些感叹或者是偏感受的描述字眼\n6，最后加一些相关的tag\n"
    )
    content = multiple_input_prompt.format(scen_name="海边", style_name="梦幻的", product_name="Oppo Pad 2 平板电脑")
    
    conversation = ConversationChain(llm=llm, verbose=True)
    output = conversation.predict(input=content)

    print(output)
    return output, 200

app.run(host='0.0.0.0', port=80)
