#!/usr/bin/env bash
Date=$(date +%Y%m%d%H%M)

function help(){
    echo " 启动基于tf-serving的OCR Web服务："
    echo "  server.sh start --port|-p [默认8080] --worker|-w [默认9] --gpu [0|1]"
    echo "  server.sh stop"
    exit
}

if [ -z "$*" ]; then
    help
    exit
fi

if [ "$1" = "debug" ]; then
    echo "基于Tf-serving的OCR Web服务  ：  调试模式"
    gunicorn --workers=1 --bind=0.0.0.0:8080 --timeout=300 server.server_tfs:app
    exit
fi

if [ "$1" = "stop" ]; then
    echo "停止 基于Tf-serving的OCR Web服务"
    ps aux|grep server.server_tfs|grep -v grep|awk '{print $2}'|xargs kill -9
    exit
fi

if [ ! "$1" = "start" ]; then
    help
    exit
fi

# 默认
PORT=8081
CONNECTION=10
GPU=1
WORKER=1

ARGS=`getopt -o p:g:w: --long port:,gpu:,worker: -n 'help.bash' -- "$@"`
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
                    echo "自定义Worker数：$2"
                    WORKER=$2
                    shift 2
                    ;;
                -g|--gpu)
                    echo "自定义#GPU：  #$2"
                    GPU=$2
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

echo "基于Tf-serving的OCR Web服务器启动... 端口:$PORT 工作进程:$CONNECTION"
# 参考：https://medium.com/building-the-system/gunicorn-3-means-of-concurrency-efbb547674b7
# worker=4是根据GPU的显存数调整出来的，ration=0.2，大概一个进程占满为2.5G,4x2.5=10G显存
_CMD="CUDA_VISIBLE_DEVICES=$GPU nohup gunicorn \
    --workers=$WORKER \
    --bind=0.0.0.0:$PORT \
    --timeout=300 \
    server.server_tfs:app \
    \>> ./logs/ocr_server_$Date.log 2>&1 &"
echo "启动服务命令："
echo "$_CMD"
eval $_CMD

