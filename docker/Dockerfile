FROM nvcr.io/nvidia/tensorflow:20.12-tf2-py3

# where external files are mapped to
ARG APP_DIR="/app"

# build directory
WORKDIR /tmp

# for vs code, see https://aka.ms/vscode-docker-python
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# use bash as a default shell
ENV SHELL=/bin/bash
ENV force_color_prompt=yes

# path to app modules
ENV PYTHONPATH=${APP_DIR}:${APP_DIR}/lib:${APP_DIR}/src

# apt setup
RUN apt-get update
ENV TZ=America/Los_Angles
ENV DEBIAN_FRONTEND=noninteractive

# node js (required by for jupyterlab_code_formatter and plotly)
RUN apt-get install -y nodejs npm
RUN npm install -g n && n lts

# pip
RUN pip install --upgrade pip

# jupyter lab
RUN pip install jupyterlab==v3.0.5

# jupyterlab_code_formatter extension
RUN jupyter labextension install @ryantam626/jupyterlab_code_formatter@v1.4.2 --no-build
RUN pip install jupyterlab_code_formatter==v1.4.2
RUN jupyter server extension enable --py jupyterlab_code_formatter
RUN pip install isort yapf black

# plotly jl extension
RUN pip install "ipywidgets>=7.5"
RUN jupyter labextension install jupyterlab-plotly@4.14.3 --no-build
RUN jupyter labextension install @jupyter-widgets/jupyterlab-manager plotlywidget@4.14.3 --no-build
RUN pip install plotly==4.14.3

# lsp jl extension
# https://github.com/krassowski/jupyterlab-lsp
RUN pip install jupyterlab-lsp
RUN pip install python-language-server[all]

# jl debugger extension
# RUN pip install xeus-python
# RUN jupyter labextension install @jupyterlab/debugger --no-build

# build jupyter lab assets
RUN jupyter lab build

# jupyter config saving
RUN echo "cp -r /root/.jupyter ${APP_DIR}/docker/res/root/ && \
    rm -rf ${APP_DIR}/docker/res/root/.jupyter/lab/workspaces && echo \"done\"" \
    >> /usr/bin/save-jupyter-config
RUN chmod +x /usr/bin/save-jupyter-config

# startup commands
ARG JUPYTER_TOKEN=""
RUN echo "tensorboard --logdir=${APP_DIR}/.tensorboard --bind_all &" >> /cmd
RUN echo "jupyter lab --allow-root --ip=0.0.0.0 --no-browser --NotebookApp.token='${JUPYTER_TOKEN}' &" >> /cmd
RUN echo "sleep infinity" >> /cmd
CMD ["sh", "/cmd"]

# kaggle cli
ENV KAGGLE_CONFIG_DIR ${APP_DIR}
RUN pip install kaggle

# python packages
RUN pip install pandas
RUN pip install numpy
RUN pip install matplotlib
RUN pip install tqdm
RUN pip install bert-for-tf2
RUN pip install prettytable
RUN pip install venn

# google cloud sdk
RUN curl https://sdk.cloud.google.com | bash -s -- --disable-prompts --install-dir=/opt
RUN pip uninstall crcmod; pip install --no-cache-dir -U crcmod
RUN echo ". /opt/google-cloud-sdk/completion.bash.inc" >> ~/.bashrc
RUN echo ". /opt/google-cloud-sdk/path.bash.inc" >> ~/.bashrc
ENV PATH="${PATH}:/opt/google-cloud-sdk/bin"

# cleanup
RUN rm -rf /var/lib/apt/lists/*
RUN rm -rf /tmp/*
RUN jupyter lab clean
RUN jlpm cache clean
RUN npm cache clean --force

# supress too many messages from tf
ENV TF_CPP_MIN_LOG_LEVEL=3

# copy external resources
COPY docker/res /

WORKDIR ${APP_DIR}