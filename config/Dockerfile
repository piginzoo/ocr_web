From registry.cn-hangzhou.aliyuncs.com/denverdino/tensorflow

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
ADD . /home/captcha/

EXPOSE 8080

## 参考
WORKDIR /app
# Copy the current directory contents into the container at /app
ADD . /app
# Install any needed packages specified in requirements.txt
RUN pip install --trusted-host pypi.python.org -r requirements.txt




WORKDIR /home/captcha/web
CMD gunicorn -w 1 -k gevent api_server:app -b 0.0.0.0:8080