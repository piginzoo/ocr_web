#{
#	"img":"/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAkGBwgH....AH/9k=",
#	"url":"",
#	"prob":false,
#	"charInfo":false,
#	"rotate":false,
#	"table":false
#}
echo "Test OCR ..."
port=8080

if [ "$1" == "error" ];
then
    echo "异常case测试..."
    #我靠，搞了半天，终于搞定了，必须用管道格式，否则，报错
    curl -v -d @- -H "Content-type: application/json" http://localhost:$port/ocr <<CURL_DATA
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

if [ ! -z $@ ]
then
  echo "set port to $1"
  port=$1
fi

image="$(cat test.png | base64)"

#echo $image
echo "Port : \t$port"
echo "URL : \thttp://localhost:$port/ocr"

#我靠，搞了半天，终于搞定了，必须用管道格式，否则，报错
curl -v -d @- -H "Content-type: application/json" http://localhost:$port/ocr <<CURL_DATA
{
    "img": "$image",
	"prob":false,
	"charInfo":false,
	"rotate":false,
	"table":false
}
CURL_DATA



