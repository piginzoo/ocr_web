From ocr.base

MAINTAINER piginzoo

RUN apt-get update
RUN apt-get install -y git
RUN apt-get install -y libgtk2.0-dev libsm6 libxext6 git
RUN apt-get install -y libevent-dev

RUN easy_install pip
RUN pip install gevent
RUN pip install gunicorn
RUN pip install opencv-contrib-python
RUN pip install tensorflow-serving-api
RUN pip install -U scikit-image

#RUN mkdir -p /usr/share/zoneinfo/Asia/
#RUN mkdir -p /usr/share/zoneinfo/
#ADD deploy/PRC /usr/share/zoneinfo/PRC
#RUN ln -s /usr/share/zoneinfo/PRC /usr/share/zoneinfo/Asia/Shanghai
#RUN rm /etc/localtime
#RUN ln -s /usr/share/zoneinfo/Asia/Shanghai /etc/localtime

#WORKDIR /home/
#RUN git clone from repo
#copy latest source code to image,and current path is /app/captcha,not /app/catpcha/deploy, so use "."
ADD ./ctpn/ctpn  /home/ocr/ctpn/ctpn
ADD ./ctpn/bin   /home/ocr/ctpn/bin
ADD ./crnn/crnn  /home/ocr/crnn/crnn
ADD ./crnn/bin   /home/ocr/crnn/bin
ADD ./ocr/bin    /home/ocr/ocr/bin
ADD ./ocr/server /home/ocr/ocr/server
ADD ./ocr/utils  /home/ocr/ocr/utils
ADD ./ocr/web    /home/ocr/ocr/web
ADD ./ocr/requirements.txt    /home/ocr/ocr/requirements.txt

# 把2个项目安装到python库中
RUN /home/ocr/ctpn/bin/install.sh
RUN /home/ocr/crnn/bin/install.sh

EXPOSE 8080

## 参考
WORKDIR /app
# Copy the current directory contents into the container at /app
ADD . /app
# Install any needed packages specified in requirements.txt
RUN pip install --trusted-host pypi.python.org -r requirements.txt




WORKDIR /home/captcha/web
CMD gunicorn -w 1 -k gevent api_server:app -b 0.0.0.0:8080