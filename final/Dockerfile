FROM xiilab/cuda
MAINTAINER Xiilab (sj.son@xiilab.com)

# Set up essential packages
 

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        wget \
	cmake \
        curl \
        git \
        libatlas-base-dev \
        libboost-all-dev \
        libgflags-dev \
        libgoogle-glog-dev \
        libhdf5-serial-dev \
        libleveldb-dev \
        liblmdb-dev \
        libopencv-dev \
        libprotobuf-dev \
        libsnappy-dev \
        protobuf-compiler \
        python-dev \
        python-numpy \
        python-setuptools \
        python-scipy \
        libcurl3-dev \
        libfreetype6-dev \
        libpng12-dev \
        libzmq3-dev \
        pkg-config \
        rsync \
        software-properties-common

RUN apt-get update && apt-get install -y \
        unzip \
        zip \
        zlib1g-dev \
        libssl-dev \
        python-zmq \ 
        python-opencv \
        libssl-dev \
        python-zmq \
        python-opencv \
        curl \
        libreadline-dev \
        git-core \
        libjpeg-dev \
        libpng-dev \
        ncurses-dev 

RUN apt-get install -y \
        imagemagick \
        gfortran \
        gnuplot \
        gnuplot-x11 \
        libgraphicsmagick1-dev \
        libfftw3-dev \
        sox \
        libsox-dev \
        python-tk \ 
        vim \
        python3-pip \
        python3-dev \
        liblapack-dev \ 
        libopenblas-dev \
        python-nose \
        libhdf5-dev \
        python-h5py \
        python-yaml \
        bc \
        python-pip \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENV LD_LIBRARY_PATH /usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH


# Set up openjdk
RUN add-apt-repository -y ppa:openjdk-r/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends openjdk-8-jdk openjdk-8-jre-headless && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*




# Set up jupyter notebook 

RUN pip --no-cache-dir install \
	ipykernel \ 
        jupyter \
        matplotlib \
        numpy \
        scipy \
        sklearn \
        pandas \
        && \
    python -m ipykernel.kernelspec

RUN pip install --upgrade pip
RUN pip3 install ipykernel

# Set up our notebook config.
COPY jupyter_notebook_config.py /root/.jupyter/


#############  python 3 -> jupyter kernel ###############

RUN pip3 install zmq 
RUN mkdir -p /root/.ipython/kernels/python3
COPY kernel.json /root/.ipython/kernels/python3



################# Tensorflow ###############

# Configure the build for our CUDA configuration.
ENV CI_BUILD_PYTHON python
ENV LD_LIBRARY_PATH /usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH
ENV TF_NEED_CUDA 1
ENV TF_CUDA_COMPUTE_CAPABILITIES=3.0,3.5,5.2,6.0,6.1

RUN pip install tensorflow-gpu
RUN pip3 install tensorflow-gpu




################# Caffe ####################


ENV CAFFE_ROOT=/opt/caffe
WORKDIR $CAFFE_ROOT

# FIXME: use ARG instead of ENV once DockerHub supports this
ENV CLONE_TAG=rc4

RUN git clone -b ${CLONE_TAG} --depth 1 https://github.com/BVLC/caffe.git . && \
    pip install --upgrade pip && \
    cd python && for req in $(cat requirements.txt) pydot; do pip install $req; done && cd .. && \
    git clone https://github.com/NVIDIA/nccl.git && cd nccl && make -j install && cd .. && rm -rf nccl && \
    mkdir build && cd build && \
    cmake -DUSE_CUDNN=1 -DUSE_NCCL=1 .. && \
    make -j"$(nproc)"

ENV PYCAFFE_ROOT $CAFFE_ROOT/python
ENV PYTHONPATH $PYCAFFE_ROOT:$PYTHONPATH
ENV PATH $CAFFE_ROOT/build/tools:$PYCAFFE_ROOT:$PATH
RUN echo "$CAFFE_ROOT/build/lib" >> /etc/ld.so.conf.d/caffe.conf && ldconfig

RUN mkdir /root/data


###################### DIGITS #####################

RUN git clone https://github.com/NVIDIA/DIGITS /root/digits


RUN pip install requests==2.9.1 \
	Flask==0.10.1 \
	Flask-SocketIO==2.6 \
	psutil==3.4.2 \
	scikit-fmm==0.0.9 \
	lmdb==0.87 \
	wtforms==2.1 \ 
  gevent==1.1.0 \
  Flask-WTF==0.12


####################### TORCH 7  ######################


# Run Torch7 installation scripts (dependencies only)
RUN git clone https://github.com/torch/distro.git /root/torch

RUN git clone https://github.com/xianyi/OpenBLAS.git /root/OpenBLAS
RUN apt-get update && cd /root/OpenBLAS && make NO_AFFINITY=1 USE_OPENMP=0


# Run Torch7 installation scripts
RUN cd /root/torch && \
# Run without nvcc to prevent timeouts
  sed -i 's/path_to_nvcc=$(which nvcc)/path_to_nvcc=$(which no_nvcc)/g' install.sh && \
  sed -i 's,path_to_nvcc=/usr/local/cuda/bin/nvcc,path_to_nvcc=,g' install.sh && \
  ./install.sh

# Export environment variables manually
ENV LUA_PATH='/root/.luarocks/share/lua/5.1/?.lua;/root/.luarocks/share/lua/5.1/?/init.lua;/root/torch/install/share/lua/5.1/?.lua;/root/torch/install/share/lua/5.1/?/init.lua;./?.lua;/root/torch/install/share/luajit-2.1.0-beta1/?.lua;/usr/local/share/lua/5.1/?.lua;/usr/local/share/lua/5.1/?/init.lua'
ENV LUA_CPATH='/root/.luarocks/lib/lua/5.1/?.so;/root/torch/install/lib/lua/5.1/?.so;./?.so;/usr/local/lib/lua/5.1/?.so;/usr/local/lib/lua/5.1/loadall.so'
ENV PATH=/root/torch/install/bin:$PATH
ENV LD_LIBRARY_PATH=/root/torch/install/lib:$LD_LIBRARY_PATH
ENV DYLD_LIBRARY_PATH=/root/torch/install/lib:$DYLD_LIBRARY_PATH
ENV LUA_CPATH='/root/torch/install/lib/?.so;'$LUA_CPATH

# Set ~/torch as working directory
WORKDIR /root/torch

# Restore Torch7 installation script
RUN sed -i 's/path_to_nvcc=$(which no_nvcc)/path_to_nvcc=$(which nvcc)/g' install.sh

# Install CUDA libraries
RUN luarocks install cutorch && \
  luarocks install cunn && \
  luarocks install cudnn

WORKDIR /root

######################### PYTORCH ########################

# PYTHON 2.7
RUN pip install http://download.pytorch.org/whl/cu80/torch-0.1.11.post5-cp27-none-linux_x86_64.whl 
RUN pip install torchvision

# PYTHON 3.5
RUN pip3 install http://download.pytorch.org/whl/cu80/torch-0.1.11.post5-cp35-cp35m-linux_x86_64.whl 
RUN pip3 install torchvision



######################### THEANO #########################


ENV CUDA_ROOT /usr/local/cuda/bin
# Install bleeding-edge Theano
RUN pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git
RUN pip install --upgrade pip six
# Set up .theanorc for CUDA
RUN echo "[global]\ndevice=gpu\nfloatX=float32\noptimizer_including=cudnn\n[lib]\ncnmem=0.1\n[nvcc]\nfastmath=True" > /root/.theanorc


# Install bleeding-edge Theano
RUN pip3 install --upgrade --no-deps git+git://github.com/Theano/Theano.git
RUN pip3 install --upgrade pip six


################ KERAS ######################


# Clone Keras repo and move into it
RUN cd /root && git clone https://github.com/fchollet/keras.git && cd keras && \
  # Install
  python setup.py install


RUN cd /root/keras && python3 setup.py install

#############################################

# Expose server port
EXPOSE 5000
# TensorBoard
EXPOSE 6006
# IPython
EXPOSE 8888

######## for display in docker container ########

RUN apt-get install -y libcanberra-gtk-module packagekit-gtk3-module
RUN pip install dlib

RUN ["/bin/bash"]
