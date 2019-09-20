#!/usr/bin/env bash
Date=$(date +%Y%m%d%H%M)

function help(){
    echo "命令格式："
    echo "  server.sh start --port|-p [默认8080] --worker|-w [默认3] --gpu [0|1] --mode|-m [tfserving|single]"
    echo "  server.sh stop"
    exit
}

if [ -z "$*" ]; then
    help
    exit
fi

if [ "$1" = "debug" ]; then
    echo "OCR Web 服务调试模式"
    gunicorn --workers=1 --name=ocr_web_server --bind=0.0.0.0:8080 --timeout=300 server:app
    exit
fi

if [ "$1" = "stop" ]; then
    echo "停止 OCR Web 服务"
    ps aux|grep ocr_web_server|grep -v grep|awk '{print $2}'|xargs kill -9
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
WORKER=3
MODE=tfserving #tf-serving方式，single是单独加载模型

ARGS=`getopt -o p:c:g:w:m: --long port:,connection:,gpu:,worker:,mode: -n 'help.bash' -- "$@"`
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
                -c|--connection)
                    echo "自定义并发数：$2"
                    CONNECTION=$2
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
                -m|--mode)
                    echo "自定义模式：  #$2"
                    MODE=$2
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

if [ "$MODE" == "tfserving" ]; then
    echo "基于Tf-Serving的OCR Web服务器启动... 端口:$PORT 工作进程:$WORKER"
    # 参考：https://medium.com/building-the-system/gunicorn-3-means-of-concurrency-efbb547674b7
    # worker=4是根据GPU的显存数调整出来的，ration=0.2，大概一个进程占满为2.5G,4x2.5=10G显存
    _CMD="gunicorn \
    --workers=$WORKER \
    --bind=0.0.0.0:$PORT \
    --timeout=300 \
    server.server:app\
    --env mode=$MODE"
#    >> ./logs/ocr_server_$Date.log 2>&1 &"
    echo "启动服务："
    echo "$_CMD"
    eval $_CMD
    exit 0
fi


if [ "$MODE" == "single" ]; then
    echo "非Docker服务器启动... 端口:$PORT 工作进程:$CONNECTION"
    # 参考：https://medium.com/building-the-system/gunicorn-3-means-of-concurrency-efbb547674b7
    # worker=4是根据GPU的显存数调整出来的，ration=0.2，大概一个进程占满为2.5G,4x2.5=10G显存
    _CMD="CUDA_VISIBLE_DEVICES=$GPU gunicorn \
        --name=ocr_web_server \
        --workers=$WORKER \
        --worker-class=gevent \
        --worker-connections=$CONNECTION \
        --bind=0.0.0.0:$PORT \
        --timeout=300 \
        server.server:app\
        --env mode=$MODE"
#        \>> ./logs/ocr_server_$Date.log 2>&1 &"
    echo "启动服务："
    echo "$_CMD"
    eval $_CMD
    exit 0
fi

echo "无法识别的服务器类型：MODE=$MODE"