import json,base64
import logging
from json.decoder import JSONDecodeError
logger = logging.getLogger("API")
'''
请求的报文：
{
	"img":"/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAkGBwgH....AH/9k=",
	"url":"",
	"prob":false,
	"charInfo":false,
	"rotate":false,
	"table":false
}

处理调用的参数，还原图片
'''
def process_request(request):
    # base64_data = request.form.get('img','')
    str_data = request.get_data()
    logger.debug("Got data:%d bytes",len(str_data))
    # import requests
    # requests.get(url).json()

    data = str_data.decode('utf-8')

    try:
        data = data.replace('\r\n', '')
        data = data.replace('\n', '')
        data = json.loads(data)
    except JSONDecodeError as e:
        logger.error(data)
        logger.error("JSon数据格式错误")
        raise  Exception("JSon数据格式错误:"+str(e))

    base64_data = data['img']
    logger.debug("Got image ,size:%d",len(base64_data))
    # 去掉可能传过来的“data:image/jpeg;base64,”HTML tag头部信息

    index = base64_data.find(",")
    if index!=-1: base64_data = base64_data[index+1:]
    # print(base64_data)
    # 降base64转化成byte数组
    buffer = base64.b64decode(base64_data)
    logger.debug("Convert image to bytes by base64, lenght:%d",len(buffer))

    return buffer


'''
入参 result
{
    'name':     'xxx.png',
    'box':      [[1, 1, 1, 1],[2, 2, 2, 2],....]
    'image':    <draw image numpy array>,
    'f1':       0.78,
    text:       ['xxx','yyy',...]
}
    
返回：
{
    "sid":"6c8f999fe943bd5ad325483fb860a3f0645e9b214bdffb4b6a4d9ae96ea9debb6b861c69",
    "prism_version":"1.0.9",
    "prism_wnum":182,
    "prism_wordsInfo":[
    {
        "word":"供审核使用",
        "pos":[{"x":340,"y":46},{"x":582,"y":46},{"x":582,"y":72},{"x":340,"y":72}]
    },
    {
        "word":"核实图片",
        "pos":[{"x":602,"y":44},{"x":836,"y":40},{"x":837,"y":68},{"x":602,"y":71}]
    }],
    "height":1200,
    "width":1600,
    "orgHeight":1200,
    "orgWidth":1600,
    "content":"仅供审核使用 核实图片 2峂 201311300746 3 ...."
}
'''
def post_process(result,width,height):
    prism_wordsInfo = []
    text = ""
    for i,b in enumerate(result['boxes']):
        one = {}
        one['pos'] = b.tolist()
        one['word'] = result['text'][i]
        prism_wordsInfo.append(one)

        text+= result['text'][i] + " "

    return \
    {
        "sid": "NS-OCR-RECOGNITION",
        "prism_version": "1.0",
        "prism_wnum": len(result),
        "prism_wordsInfo": prism_wordsInfo,
        "height": height,
        "width": width,
        "orgHeight": height,
        "orgWidth": width,
        "content": text
    }