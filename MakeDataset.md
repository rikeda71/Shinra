# Datasetの作り方

以下は，森羅プロジェクトのデータを固有表現抽出データセットに変換するための手順を記したものである．

## コーパスの準備

### 1. 森羅プロジェクトのコーパスをダウンロードする

- [森羅プロジェクトのデータセットに関するページ](http://liat-aip.sakura.ne.jp/%E6%A3%AE%E7%BE%85/%E6%A3%AE%E7%BE%85wikipedia%E6%A7%8B%E9%80%A0%E5%8C%96%E3%83%97%E3%83%AD%E3%82%B8%E3%82%A7%E3%82%AF%E3%83%882019/%E6%A3%AE%E7%BE%852019%E3%83%87%E3%83%BC%E3%82%BF%E9%85%8D%E5%B8%83/)からデータをダウンロードしてくる
- ただし，メールアドレスの登録が必要．森羅プロジェクトの参加者数の把握に使用されるとのこと

### 2. 以下のスクリプトにより，コーパスを変形する

```sh
python scripts/make_annotated_corpus.py -ad アノテーションファイルがあるディレクトリ -hd WikipediaのHTMLファイルがあるディレクトリ -cls データセットを構築したいクラス(Person, Cityなど)
```

- 森羅プロジェクトのコーパスはアノテーションファイルとテキストファイル，HTMLファイルの3種が存在する．本スクリプトでは，アノテーションファイルとHTMLファイルのみを利用してコーパスを作成する

- スクリプトのオプションは以下の通り

```sh
python scripts/make_annotated_corpus.py --help
Usage: make_annotated_corpus.py [OPTIONS]

  make IE dataset from html and annotation files
   :param html_dir: [location of html files]
   :type html_dir: str
   :param annotation_dir: [location of annotation files]
   :type annotation_dir: str
   :param class_name: [class of making dataset]
   :type class_name: str
   :param out: [location of annotated files]
   :type out: str

Options:
  -hd, --html_dir TEXT
  -ad, --annotation_dir TEXT
  -cls, --class_name TEXT
  -o, --out TEXT
```

### 3. 以下のコマンドにより，形態素解析APIを実行する

```sh
docker build -f docker/morph_analysis/Dockerfile docker/morph_analysis -t morph-analysis-api
docker run -t --rm -d -p 5000:5000 -v $PWD/scripts/:/root morph-analysis-api:latest python /root/morph_analysis_api.py
```

- 次のステップで形態素解析を行うため，先に実行しておく．
- mecab, mecab-neologd, juman++, sudachiが使用可能
- 詳しくは，[ここ](https://github.com/s14t284/MorphAnalysisAPI)を参照

### 4. 以下のスクリプトにより，固有表現抽出データセットを構築する

```sh
python scripts/make_dataset_from_corpus.py -cp 変形したコーパスファイル
```

- スクリプトのオプションは以下の通り

```sh
python scripts/make_dataset_from_corpus.py --help                                                                                                          [develop-nested-ner-model]
Usage: make_dataset_from_corpus.py [OPTIONS]

  :param corpus_path: [a path of annotated corpus]
  :type corpus_path: str
  :param out_dir: [a storage directory of dataset]
  :type out_dir: str
  :param ksplit_num: [dataset split number. default=1 (no split) ]
  :type ksplit_num: int
  :param bioul: [NE label is BIOUL format (IOB2 -> BIOUL).]
  :type bioul: bool
  :param char_level: [make char level annotation.]
  :type char_level: bool
  :param morph_analysis: [morph analysis algorithm]
  :type morph_analysis: str
  :param sudachim: [sudachi morph analyser mode]
  :type sudachim: str

Options:
  -cp, --corpus_path TEXT
  -o, --out_dir TEXT
  -k, --ksplit_num INTEGER
  -b, --bioul
  -c, --char_level
  --MECAB
  --NEOLOGD
  --JUMANPP
  --SUDACHI
  --sudachim [A|B|C]
```

- `layered-bilstm-crf`はBIOULラベリングスキームに対応していない点に注意
- 通常の固有表現抽出データセットと違い，1つの単語に複数のラベルが振られている点に注意．
  - 以下にデータセットの例を示す．
  - 以下のように，ある系列に複数のラベルが振られることもある（複数の系列を包含する形でラベルが振られることもある）ため，その情報を落とさないようにしている

```
これ	名詞	代名詞	一般	*	*	*	これ	コレ	コレ	O	O	O
は	助詞	係助詞	*	*	*	*	は	ハ	ワ	O	O	O
、	記号	読点	*	*	*	*	、	、	、	O	O	O
アンジオテンシン	名詞	一般	*	*	*	*	*	B-種類	B-用途	O
変換	名詞	サ変接続	*	*	*	*	変換	ヘンカン	ヘンカン	I-種類	I-用途	O
酵素	名詞	一般	*	*	*	*	酵素	コウソ	コーソ	I-種類	I-用途	O
阻害	名詞	サ変接続	*	*	*	*	阻害	ソガイ	ソガイ	I-種類	I-用途	O
薬	名詞	接尾	一般	*	*	*	薬	ヤク	ヤク	I-種類	I-用途	O
で	助動詞	*	*	*	特殊・ダ	連用形	だ	デ	デ	O	O	O
あり	助動詞	*	*	*	五段・ラ行アル	連用形	ある	アリ	アリ	O	O	O
、	記号	読点	*	*	*	*	、	、	、	O	O	O
アンジオテンシン	名詞	一般	*	*	*	*	*	B-特性	O	O
I	名詞	一般	*	*	*	*	*	I-特性	O	O
の	助詞	連体化	*	*	*	*	の	ノ	ノ	I-特性	O	O
アンジオテンシン	名詞	一般	*	*	*	*	*	I-特性	O	O
II	名詞	一般	*	*	*	*	*	I-特性	O	O
へ	助詞	格助詞	一般	*	*	*	へ	ヘ	エ	I-特性	O	O
の	助詞	連体化	*	*	*	*	の	ノ	ノ	I-特性	O	O
変換	名詞	サ変接続	*	*	*	*	変換	ヘンカン	ヘンカン	I-特性	O	O
を	助詞	格助詞	一般	*	*	*	を	ヲ	ヲ	I-特性	O	O
阻害	名詞	サ変接続	*	*	*	*	阻害	ソガイ	ソガイ	I-特性	O	O
し	動詞	自立	*	*	サ変・スル	連用形	する	シ	シ	I-特性	O	O
、	記号	読点	*	*	*	*	、	、	、	I-特性	O	O
また	接続詞	*	*	*	*	*	また	マタ	マタ	I-特性	O	O
ブラジキニン	名詞	一般	*	*	*	*	*	I-特性	O	O
の	助詞	連体化	*	*	*	*	の	ノ	ノ	I-特性	O	O
薬理	名詞	一般	*	*	*	*	薬理	ヤクリ	ヤクリ	I-特性	O	O
作用	名詞	サ変接続	*	*	*	*	作用	サヨウ	サヨー	I-特性	O	O
の	助詞	連体化	*	*	*	*	の	ノ	ノ	I-特性	O	O
いくつ	名詞	代名詞	一般	*	*	*	いくつ	イクツ	イクツ	I-特性	O	O
か	助詞	副助詞／並立助詞／終助詞	*	*	*	*	か	カ	カ	I-特性	O	O
を	助詞	格助詞	一般	*	*	*	を	ヲ	ヲ	I-特性	O	O
増強	名詞	サ変接続	*	*	*	*	増強	ゾウキョウ	ゾーキョー	I-特性	O	O
する	動詞	自立	*	*	サ変・スル	基本形	する	スル	スル	I-特性	O	O
可能	名詞	形容動詞語幹	*	*	*	*	可能	カノウ	カノー	I-特性	O	O
性	名詞	接尾	一般	*	*	*	性	セイ	セイ	I-特性	O	O
が	助詞	格助詞	一般	*	*	*	が	ガ	ガ	I-特性	O	O
ある	動詞	自立	*	*	五段・ラ行	基本形	ある	アル	アル	I-特性	O	O
。	記号	句点	*	*	*	*	。	。	。	O	O	O
```
