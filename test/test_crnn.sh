PORT=8080

# 测试异常case
if [ "$1" == "error" ];
then
    echo "异常case测试..."
    #我靠，搞了半天，终于搞定了，必须用管道格式，否则，报错
    curl -v -d @- -H "Content-type: application/json" http://localhost:$PORT/ocr <<CURL_DATA
{
    "img": "GHJKFGHJKFGHJKGHJKGHJK",
	"prob":false,
	"charInfo":false,
	"rotate":false,
	"table":false;
}
CURL_DATA
    exit -1
fi

# 测试异常case
if [ "$1" != "" ]; then
    echo "自定义端口：$1"
    PORT=$1
fi

image1="$(cat test1.png | base64)"
image2="$(cat test2.png | base64)"
image3="$(cat test3.png | base64)"
image4="$(cat test4.png | base64)"

data=$(cat<<CURL_DATA
[
    {"img": "$image1"},
    {"img": "$image2"},
    {"img": "$image3"},
    {"img": "$image4"}
]
CURL_DATA
)

# 简单的测试
echo "测试功能"
echo $data|curl -v -d @- -H "Content-type: application/json" http://127.0.0.1:$PORT/crnn
