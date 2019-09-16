# 启动nvidia-docker，是一种特殊的docker，支持GPU资源管理

echo "启动Nvidia-Docker...."

CRNN_MODEL=./model/crnn
CTPN_MODEL=./model/ctpn
CONFIG=./config/model.cfg

echo "CTPN模型目录：$CTPN_MODEL"
echo "CRNN模型目录：$CTPN_MODEL"
echo "Config配置： $CONFIG"

docker run \
 --runtime=nvidia  \
 -e NVIDIA_VISIBLE_DEVICES=1 \
 -t --rm  \
 -p 8501:8501 \
 -p 8500:8500 \
 --mount type=bind,source=$CRNN,target=/model/crnn \
 --mount type=bind,source=$CTPN,target=/model/ctpn \
 --mount type=bind,source=$CONFIG,target=/model/model.cfg \
 tensorflow/serving:1.14.0-gpu \
 --model_config_file=/model/model.cfg

