echo "OCR Json调用测试脚本..."
port=8080
server=http://101.200.57.40
image="$(cat test.png | base64)"
curl -v -d @- -H "Content-type: application/json" $server:$port/ocr <<CURL_DATA
{
    "img": "$image",
	"prob":false,
	"charInfo":false,
	"rotate":false,
	"table":false
}
CURL_DATA

