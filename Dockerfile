FROM pytorch/pytorch:2.7.0-cuda12.6-cudnn9-devel
RUN pip install -U "huggingface_hub[cli]"
RUN pip install --upgrade google-api-python-client
RUN apt-get -y update
RUN apt install -y vim
RUN apt install -y git
RUN apt install -y kmod
RUN pip install accelerate

# Build lm-eval from source
RUN git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness \
  && cd lm-evaluation-harness \
  && pip install -e . \
  && cd ..