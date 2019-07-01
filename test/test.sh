PORT=8080
CON=1
NUM=1

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


ARGS=`getopt -o p:c:n: --long port:,con:,num: -n 'help.bash' -- "$@"`
if [ $? != 0 ]; then
    help
    exit 1
fi
eval set -- "${ARGS}"
while true ;
do
        case "$1" in
                -p|--port)
                    echo "端口号：$2"
                    PORT=$2
                    shift 2
                    ;;
                -c|--con)
                    echo "并发数：$2"
                    CON=$2
                    shift 2
                    ;;
                -n|--num)
                    echo "压测数 #$2"
                    NUM=$2
                    shift 2
                    ;;
                --) shift ; break ;;
                *) help; exit 1 ;;
        esac
done

image="$(cat test.png | base64)"

data=$(cat<<CURL_DATA
{
    "img": "$image",
	"prob":false,
	"charInfo":false,
	"rotate":false,
	"table":false
}
CURL_DATA
)

# 压力测试
if [ "$1" == "stress" ];
then
    echo "压力测试..."
    echo $data > test.data
    echo "ab -s 300 -c $CON -n $NUM -p test.data -T 'application/json'  http://127.0.0.1:$PORT/ocr"
    ab -s 300 -c $CON -n $NUM -p test.data -T 'application/json'  http://127.0.0.1:$PORT/ocr
    echo "压力测试完毕！"
    rm test.data
    exit;
fi

# 简单的测试
echo "测试功能"
echo $data|curl -v -d @- -H "Content-type: application/json" http://127.0.0.1:$PORT/ocr
