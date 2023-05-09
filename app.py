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
from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate, LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

app = Flask(__name__, template_folder="frontend", static_folder="frontend")
CORS(app, support_credentials=True)

@app.route("/")
def index():
    return render_template("index.html")

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
    image = pipe(data["prompt"], guidance_scale=7.5, num_inference_steps=20,height=data["height"], width=data["width"]).images[0]

    image.save(output)
    return send_file(output), 200

@app.post("/create/create_text")
def create_text():
    chat = ChatOpenAI(temperature=0)
    #data = request.json
    result = chat([HumanMessage(content="从现在开始，你就是化身为叫作JOJO的女旅游博主，你喜欢全世界环游，你热爱时尚和数码产品，你热爱生活，你很有激情，你也很可爱。现在你要给OPPO Pad 2平板电脑制作一篇在户外旅行期间的使用体验文章，目的是吸引别人种草，你的配图场景是海边，画面风格是梦幻的 ，文章需要突出的产品功能是强劲性能，高刷新屏幕，5G通信，充电快，有手写笔等特点，但是不能很生硬地说出这些特点，你也不需要把以上的特点都描述出来，可以只体现其中几个特点就可以。\n还要求：\n1，尽可能体现真实使用过程\n2，多使用带有可爱的emoji\n3，多用短句，每句 话不超过20个字\n4，多用空行和分段\n5，带有一些感叹或者是偏感受的描述字眼\n6，最后加一些相关的tag\n7，你不必强调你是JOJO.")])
    return result, 200

app.run(host='0.0.0.0', port=5000)
