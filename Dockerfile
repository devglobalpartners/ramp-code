### BASE DOCKERFILE for the Ramp development and deployment environment
### updated March 29th, 2022

FROM tensorflow/tensorflow:2.8.0-gpu-jupyter

RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list

RUN apt-get update && apt-get install -y python3-opencv
RUN add-apt-repository ppa:ubuntugis/ppa && apt-get update
RUN apt-get update
RUN apt-get install -y gdal-bin
RUN apt-get install -y libgdal-dev
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal

# pip install dependencies.
COPY docker/pipped-requirements.txt pipped-requirements.txt
RUN pip install -r pipped-requirements.txt

# pip install solaris -- try with tmp-free build
COPY solaris /tmp/solaris
RUN pip install /tmp/solaris --use-feature=in-tree-build

### Add more libraries available via pip here
RUN pip install scikit-fmm --use-feature=in-tree-build

ENV RAMP_HOME=/tf

# pip install ramp
# comment out this whole block if you are going to do development on the ramp codebase
RUN mkdir /tmp/ramp-code
COPY setup.py /tmp/ramp-code/setup.py
COPY README.md /tmp/ramp-code/README.md
COPY ramp /tmp/ramp-code/ramp
RUN pip install /tmp/ramp-code --use-feature=in-tree-build

# matplotlib wants a writable tmp dir
RUN mkdir /tmp/matplotlib
RUN chmod a+w /tmp/matplotlib
ENV MPLCONFIGDIR=/tmp/matplotlib

