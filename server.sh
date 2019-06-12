Date=$(date +%Y%m%d%H%M)

function help(){
    echo "命令格式："
    echo "  server.sh start --port|-p [默认8080] --worker|-w [默认9]"
    echo "  server.sh stop"
    exit
}


if [ -z "$*" ]; then
    help
    exit
fi

if [ "$1" = "stop" ]; then
    echo "停止 OCR Web 服务"
    ps aux|grep gunicorn|grep -v grep|awk '{print $2}'|xargs kill -9
    exit
fi

# 默认
PORT=8080
WORKER=9

ARGS=`getopt -o p:w: --long port:,worker: -n 'example.bash' -- "$@"`
if [ $? != 0 ]; then
    help
    exit 1
fi

eval set -- "${ARGS}"

while true ;
do
        case "$1" in
                -p|--port)
                    echo "自定义端口号：$2"
                    PORT=$2
                    shift 2
                    ;;
                -w|--worker)
                    echo "自定义进程数：$2"
                    WORKER=$2
                    shift 2
                    ;;
                --) shift ; break ;;
                *) help; exit 1 ;;
        esac
done

if [ $? != 0 ]; then
    help
    exit 1
fi

echo "服务器即将启动, 端口:$PORT 工作进程:$WORKER"

CUDA_VISIBLE_DEVICES=0 gunicorn \
    --workers=$WORKER \
    --worker-class=gevent \
    --bind=0.0.0.0:$PORT \
    --timeout=300 \
    server:app \
    \>> ./logs/ocr_server_$Date.log 2>&1 &

