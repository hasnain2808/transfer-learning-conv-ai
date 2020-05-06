
FROM pytorch/pytorch

########################################  BASE SYSTEM
# set noninteractive installation
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y apt-utils
RUN apt-get install -y --no-install-recommends \
    build-essential \
    pkg-config \
    tzdata \
    curl

######################################## PYTHON3
# RUN apt-get install -y \
#     python3 \
#     python3-pip
# RUN pip3 install torch
# set local timezone
RUN ln -fs /usr/share/zoneinfo/America/New_York /etc/localtime && \
    dpkg-reconfigure --frontend noninteractive tzdata

# transfer-learning-conv-ai
# ENV PYTHONPATH /usr/local/lib/python3.6 

RUN pip freeze
# RUN pip install torch
RUN pip install pytorch-ignite 
RUN pip install transformers==2.5.1
RUN pip install tensorboardX==1.8
RUN pip install tensorflow  
RUN pip install nltk
RUN pip install seaborn
RUN pip install matplotlib
RUN pip install flask
RUN pip install scikit-learn
RUN pip install xgboost

RUN pip install fuzzywuzzy>=0.16.0
RUN pip install scikit-learn>=0.19.1
RUN pip install torchtext==0.3.1


RUN curl https://personabaseddeprdete.s3.ap-south-1.amazonaws.com/trained_model.zip > mod.zip 

RUN apt-get install -y unzip
RUN unzip mod.zip
RUN curl https://personabaseddeprdete.s3.ap-south-1.amazonaws.com/dataset_cache_OpenAIGPTTokenizer > dataset_cache_OpenAIGPTTokenizer 

RUN apt-get install nohup

COPY . ./

RUN cd factual/KEQA_WSDM19 && main.sh && nohup sh -c python api.py && cd..

RUN cd transfer-learning-conv-ai && nohup sh -c interact_api.py && cd ..

RUN cd classifier && nohup sh -c classifier_api.py && cd ..

RUN cd UI

RUN main.py





RUN pip install pytorch_transformers

CMD ["bash"]


