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
    && pip3 install -q -e git+git://github.com/WorksApplications/SudachiPy@develop#egg=SudachiPy \
    && wget -q https://object-storage.tyo2.conoha.io/v1/nc_2520839e1f9641b08211a5c85243124a/sudachi/sudachi-dictionary-20190531-full.zip \
    && unzip sudachi-dictionary-20190531-full.zip \
    && mv sudachi-dictionary-20190531/system_full.dic /src/sudachipy/resources/system.dic \
    && rm -rf sudachi-dictionary-20190531* \
    && mkdir -p /home \
    && echo "NEOLOGD_PATH=`mecab-config --dicdir`/mecab-ipadic-neologd" >> /home/.env \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ENV LANG=ja_JP.UTF-8
EXPOSE 5000
WORKDIR /app
