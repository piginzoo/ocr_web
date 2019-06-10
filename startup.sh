Date=$(date +%Y%m%d%H%M)

if [ "$1" = "stop" ]; then
    echo "停止 OCR Web 服务"
    ps aux|grep python|grep gunicorn|awk '{print $2}'|xargs kill -9
    exit
fi

CUDA_VISIBLE_DEVICES=0 gunicorn --workers=1 --worker-class=gevent server:app --bind=0.0.0.0:8080 --timeout=300 \
>> ./logs/ocr_server_$Date.log 2>&1 &