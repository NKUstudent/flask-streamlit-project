import io
from PIL import Image
import json
import PIL
import requests

import sys
sys.path.append('C:\\Users\\墨池洗砚\\Desktop\\学习\\暑期实训\\WorkSpace\\langchain-demo')
from model_choice import modelchoice

modelchoice.setenv()

from langchain_community.chat_models import ChatZhipuAI
from langchain.prompts import PromptTemplate

_template = """
以下提示用于指导Al绘画模型创建图像。它们包括人物外观、背景、颜色和光影效果，以及图像的主题和风格等各种细节。这些提示的格式通常包括带权重的数字括号，用于指定某些细节的重要性或强调。例如，"(masterpiece:1.4)"表示作品的质量非常重要。以下是一些示例：
1. (8k, RAW photo, best quality, masterpiece:1.2),(realistic, photo-realistic:1.37), ultra-detailed, 1girl, cute, solo, beautiful detailed sky, detailed cafe, night, sitting, dating, (nose blush), (smile:1.1),(closed mouth), medium breasts, beautiful detailed eyes, (collared shirt:1.1), bowtie, pleated skirt, (short hair:1.2), floating hair, ((masterpiece)), ((best quality)),
2. (masterpiece, finely detailed beautiful eyes: 1.2), ultra-detailed, illustration, 1 girl, blue hair black hair, japanese clothes, cherry blossoms, tori, street full of cherry blossoms, detailed background, realistic, volumetric light, sunbeam, light rays, sky, cloud,
3. highres, highest quallity, illustration, cinematic light, ultra detailed, detailed face, (detailed eyes, best quality, hyper detailed, masterpiece, (detailed face), blue hairlwhite hair, purple eyes, highest details, luminous eyes, medium breats, black halo, white clothes, backlighting, (midriff:1.4), light rays, (high contrast), (colorful)

仿照之前的提示，写一段描写如下要素的提示：
{input}

你应该仅以 JSON 格式响应，如下所述:
返回格式如下:
{{
  "question":"$YOUR_QUESTION_HERE",
  "answer": "$YOUR_ANSWER_HERE"
}}
确保响应可以被 Python json.loads 解析。
"""

class RefinePrompt:
  
    #chat_llm = ChatSparkLLM(temperature = 0)
    chat_llm = ChatZhipuAI(model="glm-4")
    
    prompt = PromptTemplate(
        input_variables=["input"],
        template = _template,
    )

    '''
    chat_prompt = ChatPromptTemplate.from_messages([
        ('system' , _template),
        ('user', '{input}'),
    ])
    '''
    '''
    chain = LLMChain(
        prompt = chat_prompt,
        llm = chat_llm,
    )
    '''
    chain = prompt | chat_llm

    #chain = LLMChain(prompt=prompt,llm=llm)
    def run(self,text):
        ret = self.chain.invoke(text)
        res = ret.content
        # 解析json
        result = json.loads(res)
        return result["answer"]

#ENDPOINT = "http://localhost:8080"
ENDPOINT = "https://modelslab.com/api/v3/text2img"

def do_webui_request(url, **kwargs):
    reqbody = {
        "key": "erlaYr2YdmmpiGwlenww3CB9lY3dWuzRq4xeMZCjeesDSUWq6nGMyisdCbsi",
        "prompt": "best quality, extremely detailed",
        "negative_prompt": "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality",
        "seed": -1,
        "subseed": -1,
        "subseed_strength": 0,
        "batch_size": 1,
        "n_iter": 1,
        "steps": 15,
        "cfg_scale": 7,
        "width": 512,
        "height": 768,
        "restore_faces": True,
        "eta": 0,
        "sampler_index": "Euler a",
        "controlnet_input_images": [],
        "controlnet_module": 'canny',
        "controlnet_model": 'control_canny-fp16 [e3fe7712]',
        "controlnet_guidance": 1.0,
        
    }
    reqbody.update(kwargs)
    #print("reqbody:",reqbody)
    r = requests.post(url, json=reqbody)
    return r.json()

class T2I:
    def __init__(self):
        self.text_refine = RefinePrompt()
        
    def inference(self, text):
        #image_filename = os.path.join('output/image', str(uuid.uuid4())[0:8] + ".png")
        refined_text = self.text_refine.run(text)
        #print(f'{text} refined to {refined_text}')
        resp = do_webui_request(
            url=ENDPOINT,
            #url = ENDPOINT + "/api/v3/txt2img",
            prompt = refined_text,
        )
        #print(type(resp))
        #print(resp)
        #image = Image.open(io.BytesIO(base64.b64decode(resp["output"][0])))
        image_url = resp['output'][0]
        #image.save(image_filename)
        print(f"Processed T2I.run, text: {text}, image_url: {image_url}")
        return image_url

def draw(input):
    t2i = T2I()
    #input = input('Input: ')
    image_url = t2i.inference(input)
    #print("fileurl:",image_url)
    response = requests.get(image_url)
    #display(image_url)
    #webbrowser.open(image_url[0])
    #print(response.content)
    #image = Image.open(io.BytesIO(response.content))
    #image.show()
    return image_url
'''
    if response.status_code == 200:
        # 检查内容类型是否为图像
        if 'image' in response.headers.get('Content-Type', ''):
            try:
                # 使用PIL打开图片
                image = Image.open(io.BytesIO(response.content))
                # 显示图片（可选）
                image.show()
            except PIL.UnidentifiedImageError:
                print("无法识别的图像文件。请检查URL是否指向有效的图像文件。")
        else:
            print("URL不指向图像文件。内容类型为：", response.headers.get('Content-Type'))
    else:
        print("无法获取图像。HTTP状态码：", response.status_code)

print(response.headers.get('Content-Type', ''))
img = Image.open(io.BytesIO(response.content))
img.show()
draw('Please draw a yellow kitten')

'''
