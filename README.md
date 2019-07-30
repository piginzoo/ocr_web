This is a glue project, which combine the project [ctpn](https://github.com/piginzoo/text-detection-ctpn) and [crnn](https://github.com/piginzoo/CRNN_Tensorflow), actually, it is a pipe , the ctpn first detects the text zone for an image,then it will extract the small fragments, which contains all texts, then these fragments will send to crnn network, it will populate all texts.

The project use flask as its web server, and run with gunicorn gateway, here some issues when startup. There will be a exception to tell you tensorflow flags arguments error, the error is caused by gunicorn's argument will be parsed by tensorflow, too. The solutions is add them in tensorflow tf.app.flags.

The ocr can support you to send your image to us by 2 ways: base64 or http post, the result will be a picture drawn by all detected zones, and also texts recognized by crrn.

# crnn json format
request json:
[
    {"img":"GHJKFGHJKFGHJKGHJKGHJK"},
    {"img":"GHJKFGHJKFGHJKGHJKGHJK"},
    {"img":"GHJKFGHJKFGHJKGHJKGHJK"},
    {"img":"GHJKFGHJKFGHJKGHJKGHJK"},
    {"img":"GHJKFGHJKFGHJKGHJKGHJK"},
]

response json:
{
    "sid":"6c8f999fe94",
    "prism_wordsInfo":[
        {"word":"供审核使用1"},
        {"word":"核实图片1"},
        {"word":"供审核使用1"},
        {"word":"核实图片1"},
        {"word":"供审核使用1"}        
    ]
}