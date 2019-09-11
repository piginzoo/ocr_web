docker run --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=1 -t --rm -p 8501:8501 -p 8500:8500 \
 --mount type=bind,source=/app/models/crnn/pb,target=/models/crnn \
 --mount type=bind,source=/app/models/ctpn/pb,target=/models/ctpn \
 --mount type=bind,source=/app/models/config/models.cfg,target=/models/models.cfg \
 tensorflow/serving:1.14.0-gpu --model_config_file=/models/models.cfg