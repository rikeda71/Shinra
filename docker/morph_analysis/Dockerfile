FROM python:3.7


RUN apt update && apt install -q -y \
    curl \
    make \
    gcc build-essential \
    mecab mecab-ipadic-utf8 libmecab-dev \
    libboost-all-dev \
    locales \
    git \
    xz-utils \
    file \
    && locale-gen ja_JP.UTF-8 \
    && pip3 install -q -U pip \
    && pip3 install -q six mecab-python3 flask python-dotenv\
    && git clone --depth 1 https://github.com/neologd/mecab-ipadic-neologd.git \
    && mkdir -p `mecab-config --dicdir` \
    && ./mecab-ipadic-neologd/bin/install-mecab-ipadic-neologd -n -y \
    && rm -rf mecab-ipadic-neologd \
    && curl -L -o jumanpp-1.02.tar.xz 'http://lotus.kuee.kyoto-u.ac.jp/nl-resource/jumanpp/jumanpp-1.02.tar.xz' \
    && tar xJvf jumanpp-1.02.tar.xz \
    && cd jumanpp-1.02\
    && ./configure && make && make install \
    && cd ../ \
    && rm -rf jumanpp-1.02* \
    && pip3 install -q pyknp \
    && pip3 install SudachiPy \
    && pip3 install https://object-storage.tyo2.conoha.io/v1/nc_2520839e1f9641b08211a5c85243124a/sudachi/SudachiDict_core-20190718.tar.gz \
    && mkdir -p /home \
    && echo "NEOLOGD_PATH=`mecab-config --dicdir`/mecab-ipadic-neologd" >> /home/.env \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ENV LANG=ja_JP.UTF-8
EXPOSE 5000
WORKDIR /app
