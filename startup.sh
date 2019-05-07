CUDA_VISIBLE_DEVICES=1 gunicorn --workers=1 --worker-class=gevent server:app --bind=0.0.0.0:8080 --timeout=300
# --log-level=DEBUG