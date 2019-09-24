if [ "$1" == "" ];then
    echo "命令：ocr_docker.sh start|stop"
    exit
fi


if [ "$1" == "stop" ];then
    docker ps|grep tensorflow/serving|awk '{print $1}'|xargs -I {} docker stop {}
    echo "容器已经停止！"
    exit
fi

version=`date +%Y%m%d%H%M`

if [ "$1" == "build" ] && [ "$2" == "base" ] && [ "$3" == "proxy" ];then
    echo "构建基础容器：python3.7 以及各个基础python packages(代理Proxy模式)"
    docker build \
        -f config/Dockerfile.base \
        --build-arg http_proxy=http://172.17.0.1:8123 \
        --build-arg https_proxy=http://172.17.0.1:8123\ \
        -t ocr.base .
    exit
fi


if [ "$1" == "build" ] && [ "$2" == "base" ];then
    echo "构建基础容器：python3.7 以及各个基础python packages"
    docker build -f config/Dockerfile.base -t ocr.base .
    exit
fi


if [ "$1" == "build" ];then
    echo "构建容器：部署代码"
    docker build -f config/Dockerfile.base -t ocr.$version
    exit
fi

if [ "$1" == "start" ];then
    echo "启动Nvidia-Docker...."

    TF_VERSION=1.12.3
    if [ "$2" != "" ];then
        TF_VERSION=$2
        echo "自定义容器  ：$2"
    fi
    BASE_DIR=$(pwd)
    CRNN_MODEL=$BASE_DIR/model/crnn
    CTPN_MODEL=$BASE_DIR/model/ctpn
    CONFIG=$BASE_DIR/config/model.cfg

    echo "Docker镜像  ：tensorflow/serving:$TF_VERSION-gpu"
    echo "CTPN模型目录：$CTPN_MODEL"
    echo "CRNN模型目录：$CTPN_MODEL"
    echo "Config配置  ：$CONFIG"


    # "--runtime=nvidia":启动nvidia-docker，是一种特殊的docker，支持GPU资源管理
    docker run \
     --runtime=nvidia  \
     -e NVIDIA_VISIBLE_DEVICES=1 \
     -t --rm  \
     -p 8500:8500 \
     --cpus=10 \
     --mount type=bind,source=$CRNN_MODEL,target=/model/crnn \
     --mount type=bind,source=$CTPN_MODEL,target=/model/ctpn \
     --mount type=bind,source=$CONFIG,target=/model/model.cfg \
     tensorflow/serving:$TF_VERSION-gpu \
     --model_config_file=/model/model.cfg
fi