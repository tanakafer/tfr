FROM  nvcr.io/nvidia/tensorflow:20.03-tf2-py3
MAINTAINER Fernando Rodríguez López

RUN addgroup -gid 1000 fernando
RUN adduser --disabled-password --gecos '' --uid 1000 --gid 1000 fernando


ENV TZ=Europe/Madrid
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt update \
    && apt install -y python3-opencv \
    && apt install -y graphviz
    ## apt install htop

VOLUME /workspace/TFM/datasets
VOLUME /mnt/data


RUN python -m pip install --upgrade pip
RUN pip install --upgrade pip
RUN pip install pandas matplotlib packaging sklearn opencv-contrib-python Pillow
RUN pip install pydot graphviz


USER fernando
