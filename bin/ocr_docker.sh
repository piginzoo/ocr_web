if [ "$1" == "stop" ];then
    echo "命令：ocr_docker.sh start|stop"
    exit
fi


if [ "$1" == "stop" ];then
    docker ps|grep tensorflow/serving|awk '{print $1}'|xargs -I {} docker stop {}
    echo "容器已经停止！"
    exit
fi


if [ "$1" == "start" ];then
    echo "启动Nvidia-Docker...."

    TF-VERSION=1.12.3
    BASE_DIR=$(pwd)
    CRNN_MODEL=$BASE_DIR/model/crnn
    CTPN_MODEL=$BASE_DIR/model/ctpn
    CONFIG=$BASE_DIR/config/model.cfg

    echo "CTPN模型目录：$CTPN_MODEL"
    echo "CRNN模型目录：$CTPN_MODEL"
    echo "Config配置： $CONFIG"

    # "--runtime=nvidia":启动nvidia-docker，是一种特殊的docker，支持GPU资源管理
    docker run \
     --runtime=nvidia  \
     -e NVIDIA_VISIBLE_DEVICES=1 \
     -t --rm  \
     -p 8500:8500 \
     --mount type=bind,source=$CRNN_MODEL,target=/model/crnn \
     --mount type=bind,source=$CTPN_MODEL,target=/model/ctpn \
     --mount type=bind,source=$CONFIG,target=/model/model.cfg \
     tensorflow/serving:'$TF-VERSION'-gpu \
     --model_config_file=/model/model.cfg
fi