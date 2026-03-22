### BASE DOCKERFILE for the Ramp development and deployment environment
### updated for TF 2.15 / RHEL 9.7 / Tesla T4, March 2026

FROM tensorflow/tensorflow:2.15.0-gpu-jupyter

RUN apt-get update && apt-get install -y \
    python3-opencv \
    gdal-bin \
    libgdal-dev
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal

# Remove apt-installed Python GDAL bindings (built for Python 3.10, not 3.11)
# and force pip to compile from source for the correct Python version
RUN apt-get remove -y python3-gdal && rm -f /usr/lib/python3/dist-packages/osgeo/_gdal*.so

# pip install dependencies (GDAL will compile from source against libgdal-dev)
COPY docker/pipped-requirements.txt pipped-requirements.txt
RUN pip install --no-binary GDAL -r pipped-requirements.txt

# pip install solaris -- try with tmp-free build
COPY solaris /tmp/solaris
RUN pip install /tmp/solaris

# Fix efficientnet for TF 2.15: keras.utils.generic_utils was removed
RUN sed -i 's/keras.utils.generic_utils.get_custom_objects/keras.utils.get_custom_objects/' \
    /usr/local/lib/python3.11/dist-packages/efficientnet/__init__.py

# Create user/group matching the host (kbhatt:def-cg)
ARG USER_ID=150857
ARG GROUP_ID=2766
RUN groupadd -g ${GROUP_ID} def-cg || true && \
    useradd -m -u ${USER_ID} -g ${GROUP_ID} -s /bin/bash kbhatt || true

ENV RAMP_HOME=/tf

# pip install ramp
# comment out this whole block if you are going to do development on the ramp codebase
RUN mkdir /tmp/ramp-code
COPY setup.py /tmp/ramp-code/setup.py
COPY README.md /tmp/ramp-code/README.md
COPY ramp /tmp/ramp-code/ramp
RUN pip install /tmp/ramp-code

# matplotlib wants a writable tmp dir
RUN mkdir -p /tmp/matplotlib && chmod a+w /tmp/matplotlib
ENV MPLCONFIGDIR=/tmp/matplotlib

# Pre-run ldconfig as root (so the cache is built) and patch bash.bashrc
# to skip the bare ldconfig call that fails as non-root
RUN ldconfig && sed -i 's/^ldconfig$/# ldconfig/' /etc/bash.bashrc

# Ensure the working directory is accessible
RUN chown -R ${USER_ID}:${GROUP_ID} /tf

USER kbhatt
