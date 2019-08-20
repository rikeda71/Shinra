# 日本語クラス名 -> 英語クラス名

scikit-learnが日本語名クラス（正確にはダブルバイト文字）を渡すとエラーをはいてしまうため，日本語クラス名を英語クラス名に変換する必要がある
名前をここで変換前->変換後の定義をしておく

## Airport

```python
{
'滑走路数':'NOR',  # number of runways
'IATA（空港コード）':'IATA',
'ICAO（空港コード）':'ICAO',
'名称由来人地の地位職業名':'ORIGINNAME',
'年間発着回数データの年':'YOA',  # year of arrival
'滑走路の長さ':'LOR',  # length of runways
'標高':'ELEVATION',
'年間利用者数データの年':'YUD',  # year of user data
'国':'COUNTRY',
'母都市':'MCI',  # mother's city
'年間発着回数':'ART',  # arrival times
'開港年':'YOP',  # year of opening port
'総面積':'AREA',
'年間利用客数':'NUY',  # number of users in a year
'近隣空港':'NEAR',
'旧称':'OLDNAME',
'所在地':'LOC',
'運用時間':'OTIME',  # operation time
'座標・緯度':'LATITUDE',
'名前の謂れ':'REASON',
'座標・経度':'LONGITUDE',
'運営者':'OWNER',
'ふりがな':'PHONETIC',
'別名':'ALIAS'
}
```

## City

```python
{
 '種類': 'TYPE',
 '産業': 'INDUSTRY',
 '国内位置': 'PLACE',
 '地形': 'TERRAIN',
 '特産品': 'SPROD',  # special product
 '国': 'COUNTRY',
 '合併市区町村': 'MTOWN',  # merger towns,
 '温泉・鉱泉': 'SPRING',
 '旧称': 'OLDNAME',
 '友好市区町村': 'FTOWN', # friendly towns,
 '鉄道会社': 'RCOMPANY', # railway companies
 'ふりがな': 'PHONETIC',
 '人口': 'POPULATION',
 '観光地': 'TOURISTSPOT',
 '所在地': 'LOCATION',
 '施設': 'FACILITY',
 '人口データの年': 'YPD',  # year of population data
 '面積': 'AREA',
 '別名': 'ALIAS',
 '恒例行事': 'AEVENT',  # annual event
 '首長': 'EMILY',
 '人口密度': 'PDENSITY',  # population density
 '座標・緯度': 'LATITUDE',
 '座標・経度': 'LONGITUDE',
 '成立年': 'YESTABLISH', # year of establishment
 '地名の謂れ': 'REASON',
 '読み': 'READING'
}
```

## Company

```python
{
 '本拠地国' : 'HCOUNTRY',  # home country
 '創業国': 'FCOUNTRY',  # founding country
 '種類': 'TYPE',
 '業界': 'INDUSTRY',
 '従業員数（単体）': 'NEMPLOY',  # number of employ
 '別名': 'ALIAS',
 '代表者': 'REPRESENTATIVE',
 '従業員数（連結）': 'NCEMPLOY',  # number of concating employees
 '取扱商品': 'PRODUCT',
 '創業者': 'FOUNDER',
 '創業地': 'FPLACE',  # founding place
 '資本金': 'CAPITAL',
 '売上高データの年': 'YSD',  # year of sales data
 '主要株主': 'MHOLDERS',  # major shareholders
 '売上高（連結）データの年': 'NSCD',  # year of sales concating data
 '売上高（単体）': 'SALES',
 '解散年': 'YDISRUPTION',  # year of disruption
 '従業員数（単体）データの年': 'YNED',  # year of the number of employees data
 '社名使用開始年': 'NAMESTART',
 '資本金データの年': 'YCD',  # year of capital data
 '設立年': 'YESTABLISH',  # year of establish
 '従業員数（連結）データの年': 'YNCED',  # year of the number of concating employees data
 '売上高（連結）': 'SALES',  # concating sales
 '商品名': 'PRODUCTNAME',
 '買収・合併した会社': 'MERGEDCOM', # merged company
 'ふりがな': 'PHONETIC',
 '過去の社名': 'PASTNAME',
 '創業時の事業': 'FBUSINESS', # when founding bisiness
 '起源': 'ORIGIN',
 '子会社・合弁会社': 'SUBCOMPANY',
 '事業内容': 'CONTENT',  # (business content)
 '本拠地': 'HOMEBASE',
 '正式名称': 'FORMALNAME',
 'コーポレートスローガン': 'SLOGAN',
 '業界内地位・規模': 'STATUS'
}
```

## Compound

```python
{
 '沸点': 'BPOINT',  # boiling point
 '融点': 'MPOINT',  # melting point
 '商標名': 'TRADENAME',
 '原材料': 'METERIALS',
 '化学式': 'FORMULA',
 '生成化合物': 'PCOMPOUNDS',  # product compounds
 'CAS番号': 'CASNUM',
 '種類': 'TYPE',
 '読み': 'READING',
 '示性式': 'EQUATION',
 '密度': 'DENSITY',
 '用途': 'USAGE',
 '別称': 'ALIAS',
 '特性': 'CHARACTERISTIC',
 '製造方法': 'PROMETHOD'  # production method
}
```

## Person

```python
{
 '死因': 'DEATHCAUSE',
 '国籍': 'CITIZENSHIP',
 '家族': 'FAMILY',
 '両親': 'PARENTS',
 '時代': 'ERA',
 '師匠': 'MASTER',
 '職業': 'PROFESSION',
 '居住地': 'RESIDENCE',
 '別名': 'ALIAS',
 'ふりがな': 'PHONETIC',
 '所属組織': 'AFFILIATION',
 '生誕地': 'BIRTHPLACE',
 '学歴': 'EDUCATIONAL',
 '没地': 'MATH',
 '没年月日': 'DEATHDAY',
 '生年月日': 'BIRTHDAY',
 '本名': 'REALNAME',
 '称号': 'TITLE',
 '作品': 'ARTWORK',
 '異表記': 'VARIATION',
 '参加イベント': 'JOINEVENT',
 '受賞歴': 'AWARDS'
}
```
