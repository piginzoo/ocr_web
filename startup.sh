CUDA_VISIBLE_DEVICES=0 gunicorn --workers=2 --worker-class=gevent server:app --bind=0.0.0.0:8080 --timeout=300
# --log-level=DEBUG