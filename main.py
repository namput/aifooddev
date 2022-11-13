
import iplot as iplot
from bs4 import BeautifulSoup
import pandas as pd
import plotly.graph_objs as go
from pythainlp.corpus import thai_stopwords
from sklearn.naive_bayes import MultinomialNB
import requests
from langdetect import detect
import warnings
warnings.filterwarnings('ignore')
from pythainlp import sent_tokenize, word_tokenize
from nltk.sentiment.util import *
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer  # TF-IDF
from nltk.tokenize import word_tokenize
from pythainlp.util import normalize

# เพิ่มอาหาร
base_url = requests.get('https://nlovecooking.com/%e0%b8%aa%e0%b8%b9%e0%b8%95%e0%b8%a3%e0%b8%ad%e0%b8%b2%e0%b8%ab%e0%b8%b2%e0%b8%a3/%e0%b8%aa%e0%b8%b9%e0%b8%95%e0%b8%a3%e0%b8%ad%e0%b8%b2%e0%b8%ab%e0%b8%b2%e0%b8%a3%e0%b9%84%e0%b8%97%e0%b8%a2/', timeout=5)
base_url.encoding = 'utf-8'
base_url = BeautifulSoup(base_url.text, 'html.parser')
base_url.title.text

df1= pd.DataFrame(columns = ['ลิ้ง'])
for link in base_url.find_all('a'):
  df1 = df1.append({'ลิ้ง':link.get('href')}, ignore_index=True)
  # print(link.get('href'))

df1
df2 = df1.drop_duplicates()
df2 = df2.reset_index(drop=True)

df3 = df2.drop(df2.index[0:46])
df3 = df3.reset_index(drop=True)

df4 = df3.drop(df3.index[200:235])
df5 = df4

dfm = pd.DataFrame(columns=['ชื่ออาหาร', 'สูตรอาหาร', 'ประเภทอาหาร'])

i = 0
while i < 200:
      val = df5['ลิ้ง'].values[i]
      base_url2 = requests.get(val, timeout=10)
      base_url2.encoding = 'utf-8'
      base_url2 = BeautifulSoup(base_url2.text, 'html.parser')
      try:
          x = base_url2.find(['ol']).get_text()
          y = base_url2.title.get_text()
          z = base_url.title.text
      except AttributeError:
          pass
      dfm = dfm.append({'สูตรอาหาร': x, 'ชื่ออาหาร': y, 'ประเภทอาหาร': z}, ignore_index=True)
      # print(i)
      i += 1

dff = dfm
dff = dff.drop_duplicates()
dff = dff.reset_index(drop=True)

base_url3 = requests.get('https://nlovecooking.com/dessert/', timeout=5)
base_url3.encoding = 'utf-8'
base_url3 = BeautifulSoup(base_url3.text, 'html.parser')

dx= pd.DataFrame(columns = ['ลิ้ง'])
for link in base_url3.find_all('a', attrs={'href':re.compile("^https://")}):
  dx = dx.append({'ลิ้ง':link.get('href')}, ignore_index=True)
  # print(link.get('href'))

dqq = pd.DataFrame(columns=['ชื่ออาหาร', 'สูตรอาหาร', 'ประเภทอาหาร'])

i = 0
while i < dx.shape[0]:
      val = dx['ลิ้ง'].values[i]
      base_url4 = requests.get(val, timeout=10)
      base_url4.encoding = 'utf-8'
      base_url4 = BeautifulSoup(base_url4.text, 'html.parser')
      try:
          x = base_url4.find(['ol']).get_text()
          y = base_url4.title.get_text()
          z = base_url3.title.text
      except AttributeError:
          pass
      dqq = dqq.append({'สูตรอาหาร': x, 'ชื่ออาหาร': y, 'ประเภทอาหาร': z}, ignore_index=True)
      i += 1

dffx = dqq
dffx = dffx.drop_duplicates()
dffx = dffx.reset_index(drop=True)

dffx2 = dffx.drop(df2.index[0:6])
dffx3 = dffx2.drop(df2.index[61:71])
dffx3 = dffx3.reset_index(drop=True)

dffx2 = dffx.drop(df2.index[0:6])
dffx3 = dffx2.drop(df2.index[61:71])
dffx3 = dffx3.reset_index(drop=True)

menu = pd.concat([dff,dffx3])
menu = menu.drop_duplicates(subset=['ชื่ออาหาร'],keep='last')
menu=menu.reset_index(drop=True)
menu

menu['detect'] = menu['สูตรอาหาร'].apply(detect)
menu = menu[menu['detect'] == 'th']
menu = menu.reset_index(drop=True)

menu=menu.drop(['detect'], axis=1)

base_urj3 = requests.get('https://nlovecooking.com/%e0%b8%aa%e0%b8%b9%e0%b8%95%e0%b8%a3%e0%b8%ad%e0%b8%b2%e0%b8%ab%e0%b8%b2%e0%b8%a3/%e0%b8%aa%e0%b8%b9%e0%b8%95%e0%b8%a3%e0%b8%ad%e0%b8%b2%e0%b8%ab%e0%b8%b2%e0%b8%a3%e0%b8%8d%e0%b8%b5%e0%b9%88%e0%b8%9b%e0%b8%b8%e0%b9%88%e0%b8%99/', timeout=5)
base_urj3.encoding = 'utf-8'
base_urj3 = BeautifulSoup(base_urj3.text, 'html.parser')

dxj= pd.DataFrame(columns = ['ลิ้ง'])

for link in base_urj3.find_all('a', attrs={'href':re.compile("^https://")}):
  dxj = dxj.append({'ลิ้ง':link.get('href')}, ignore_index=True)

djj = pd.DataFrame(columns = ['ชื่ออาหาร','สูตรอาหาร','ประเภทอาหาร'])
i = 0
while i <  dxj.shape[0] :
    val1 = dxj['ลิ้ง'].values[i]
    base_urj4 = requests.get(val1, timeout=10)
    base_urj4.encoding = 'utf-8'
    base_urj4 = BeautifulSoup(base_urj4.text, 'html.parser')
    try:
        x1=base_urj4.find(['ol']).get_text()
        y1=base_urj4.title.get_text()
        z1=base_urj3.title.text
        djj = djj.append({'สูตรอาหาร':x1,'ชื่ออาหาร':y1,'ประเภทอาหาร':z1}, ignore_index=True)
    except AttributeError:
      pass
    # print(i)
    i += 1
djj2=djj
djj2=djj2.drop_duplicates()
djj2=djj2.reset_index(drop=True)

djj3=djj2.drop(djj2.index[0:6])
djj4=djj3.drop(djj2.index[77:88])
djj4=djj4.reset_index(drop=True)

menu2 = pd.concat([menu,djj4])
menu2 = menu2.drop_duplicates(subset=['ชื่ออาหาร'],keep='last')
menu2=menu2.reset_index(drop=True)

menu2['detect'] = menu2['สูตรอาหาร'].apply(detect)

menu2 = menu2[menu2['detect'] == 'th']
menu2=menu2.reset_index(drop=True)

menu2=menu2.drop(['detect'], axis=1)

base_urv3 = requests.get('https://nlovecooking.com/%e0%b8%aa%e0%b8%b9%e0%b8%95%e0%b8%a3%e0%b8%ad%e0%b8%b2%e0%b8%ab%e0%b8%b2%e0%b8%a3/%e0%b8%aa%e0%b8%b9%e0%b8%95%e0%b8%a3%e0%b8%ad%e0%b8%b2%e0%b8%ab%e0%b8%b2%e0%b8%a3%e0%b9%80%e0%b8%a7%e0%b8%b5%e0%b8%a2%e0%b8%94%e0%b8%99%e0%b8%b2%e0%b8%a1/', timeout=5)
base_urv3.encoding = 'utf-8'
base_urv3 = BeautifulSoup(base_urv3.text, 'html.parser')

dxv= pd.DataFrame(columns = ['ลิ้ง'])

for link in base_urv3.find_all('a', attrs={'href':re.compile("^https://")}):
  dxv = dxv.append({'ลิ้ง':link.get('href')}, ignore_index=True)

dvv = pd.DataFrame(columns = ['ชื่ออาหาร','สูตรอาหาร','ประเภทอาหาร'])


i = 0
while i <  dxv.shape[0] :
    val2 = dxv['ลิ้ง'].values[i]
    base_urv4 = requests.get(val2, timeout=10)
    base_urv4.encoding = 'utf-8'
    base_urv4 = BeautifulSoup(base_urv4.text, 'html.parser')
    try:
        x1=base_urv4.find(['ol']).get_text()
        y1=base_urv4.title.get_text()
        z1=base_urv3.title.text
        dvv = dvv.append({'สูตรอาหาร':x1,'ชื่ออาหาร':y1,'ประเภทอาหาร':z1}, ignore_index=True)
    except AttributeError:
      pass
    # print(i)
    i += 1

dvv2=dvv
dvv2=dvv2.drop_duplicates()
dvv2=dvv2.reset_index(drop=True)

dvv3=dvv2.drop(djj2.index[0:6])
dvv4=dvv3.drop(djj2.index[70:81])
dvv4=dvv4.reset_index(drop=True)

menu3 = pd.concat([menu2,dvv4])
menu3 = menu3.drop_duplicates(subset=['ชื่ออาหาร'],keep='last')
menu3=menu3.reset_index(drop=True)

menu3['detect'] = menu3['สูตรอาหาร'].apply(detect)

menu3 = menu3[menu3['detect'] == 'th']
menu3=menu3.reset_index(drop=True)

menu3=menu3.drop(['detect'], axis=1)

base_urk3 = requests.get('https://nlovecooking.com/%e0%b8%aa%e0%b8%b9%e0%b8%95%e0%b8%a3%e0%b8%ad%e0%b8%b2%e0%b8%ab%e0%b8%b2%e0%b8%a3/%e0%b8%aa%e0%b8%b9%e0%b8%95%e0%b8%a3%e0%b8%ad%e0%b8%b2%e0%b8%ab%e0%b8%b2%e0%b8%a3%e0%b9%80%e0%b8%81%e0%b8%b2%e0%b8%ab%e0%b8%a5%e0%b8%b5/', timeout=5)
base_urk3.encoding = 'utf-8'
base_urk3 = BeautifulSoup(base_urk3.text, 'html.parser')

dxk= pd.DataFrame(columns = ['ลิ้ง'])

for link in base_urk3.find_all('a', attrs={'href':re.compile("^https://")}):
  dxk = dxk.append({'ลิ้ง':link.get('href')}, ignore_index=True)

dkk = pd.DataFrame(columns = ['ชื่ออาหาร','สูตรอาหาร','ประเภทอาหาร'])

i = 0
while i <  dxk.shape[0] :
    val3 = dxk['ลิ้ง'].values[i]
    base_urk4 = requests.get(val3, timeout=10)
    base_urk4.encoding = 'utf-8'
    base_urk4 = BeautifulSoup(base_urk4.text, 'html.parser')
    try:
        x1=base_urk4.find(['ol']).get_text()
        y1=base_urk4.title.get_text()
        z1=base_urk3.title.text
        dkk = dkk.append({'สูตรอาหาร':x1,'ชื่ออาหาร':y1,'ประเภทอาหาร':z1}, ignore_index=True)
    except AttributeError:
      pass
    # print(i)
    i += 1

dkk2=dkk
dkk2=dkk2.drop_duplicates()
dkk2=dkk2.reset_index(drop=True)

dkk3=dkk2.drop(dkk2.index[0:6])
dkk4=dkk3.drop(dkk2.index[28:38])
dkk4=dkk4.reset_index(drop=True)

menu4 = pd.concat([menu3,dkk4])
menu4 = menu4.drop_duplicates(subset=['ชื่ออาหาร'],keep='last')
menu4=menu4.reset_index(drop=True)

menu4['detect'] = menu4['สูตรอาหาร'].apply(detect)

menu4 = menu4[menu4['detect'] == 'th']
menu4=menu4.reset_index(drop=True)

menu4=menu4.drop(['detect'], axis=1)
menu4=menu4.reset_index(drop=True)

base_urc3 = requests.get('https://nlovecooking.com/%E0%B8%AA%E0%B8%B9%E0%B8%95%E0%B8%A3%E0%B8%AD%E0%B8%B2%E0%B8%AB%E0%B8%B2%E0%B8%A3/%E0%B8%AD%E0%B8%B2%E0%B8%AB%E0%B8%B2%E0%B8%A3%E0%B8%88%E0%B8%B5%E0%B8%99/', timeout=5)
base_urc3.encoding = 'utf-8'
base_urc3 = BeautifulSoup(base_urc3.text, 'html.parser')

dxc= pd.DataFrame(columns = ['ลิ้ง'])

for link in base_urc3.find_all('a', attrs={'href':re.compile("^https://")}):
  dxc = dxc.append({'ลิ้ง':link.get('href')}, ignore_index=True)

dcc = pd.DataFrame(columns = ['ชื่ออาหาร','สูตรอาหาร','ประเภทอาหาร'])

i = 0
while i <  dxc.shape[0] :
    val3 = dxc['ลิ้ง'].values[i]
    base_urc4 = requests.get(val3, timeout=10)
    base_urc4.encoding = 'utf-8'
    base_urc4 = BeautifulSoup(base_urc4.text, 'html.parser')
    try:
        x1=base_urc4.find(['ol']).get_text()
        y1=base_urc4.title.get_text()
        z1=base_urc3.title.text
        dcc = dcc.append({'สูตรอาหาร':x1,'ชื่ออาหาร':y1,'ประเภทอาหาร':z1}, ignore_index=True)
    except AttributeError:
      pass
    # print(i)
    i += 1

dcc2=dcc
dcc2=dcc2.drop_duplicates()
dcc2=dcc2.reset_index(drop=True)

dcc3=dcc2.drop(dcc2.index[0:6])
dcc4=dcc3.drop(dcc2.index[151:158])
dcc4=dcc4.reset_index(drop=True)

menu5 = pd.concat([menu4,dcc4])
menu5 = menu5.drop_duplicates(subset=['ชื่ออาหาร'],keep='last')
menu5=menu5.reset_index(drop=True)

menu5['detect'] = menu5['สูตรอาหาร'].apply(detect)

menu5 = menu5[menu5['detect'] == 'th']
menu5=menu5.reset_index(drop=True)

menu5=menu5.drop(['detect'], axis=1)

menu5=menu5.to_csv( "./thai_menu.csv", index=False, encoding='utf-8-sig')

test_menu=pd.read_csv("./thai_menu.csv")
print(test_menu)

def regex_nodigits_new(s):
    '''use regex to clean string:
    get rid of punctuations, capitalized letters and numbers'''
    s = re.sub(r'[\d]','',str(s))#replace digits with empty str
    return s #only returning ingredients of first recipe?
menu_test=test_menu
menu_test2=test_menu
menu_test2_ingr=menu_test2['สูตรอาหาร'].apply(regex_nodigits_new)
menu_test2_ingr

def replace_text(text):
    lenx=len(text)
    i=0
    while i < lenx:
      text[i] = text[i].replace("\n", "")
      i += 1
    return text

menu_test2_ingr=replace_text(menu_test2_ingr)

menu_test2_ingr=menu_test2_ingr.apply(normalize)
stopwords = list(thai_stopwords())

def remove_stopword(text):
    lenx=len(text)
    i=0
    while i < lenx:

      list_word = word_tokenize(text[i], keep_whitespace=False)
      text[i] = [i for i in list_word if i not in stopwords]
      i += 1
    return text

menu_test3_ingr=remove_stopword(menu_test2_ingr)

from pythainlp.tokenize import sent_tokenize
def remove_stopword2(text):
    lenx=len(text)
    i=0
    while i < lenx:
      list_word = sent_tokenize(text[i], engine="whitespace+newline", keep_whitespace=False)
      text[i] = [i for i in list_word if i not in stopwords]
      i += 1
    return text

menu_test4_ingr=menu_test2['ชื่ออาหาร'].apply(regex_nodigits_new)
menu_test5_ingr=menu_test4_ingr.apply(normalize)
menu_test66_ingr=remove_stopword2(menu_test5_ingr)
menu_test_name=menu_test66_ingr

def remove_list(text):
    lenx=len(text)
    i=0
    while i < lenx:
        del text[i][1:]
        i += 1
    return text

menu_test_name2=remove_list(menu_test_name)

def remove_stopword3(text):
    lenx=len(text)
    i=0
    while i < lenx:
      list_word = word_tokenize(text[i], keep_whitespace=False)
      text[i] = [i for i in list_word if i not in stopwords]
      i += 1
    return text

menu_test44_ingr=menu_test2['ชื่ออาหาร'].apply(regex_nodigits_new)
menu_test55_ingr=menu_test44_ingr.apply(normalize)
menu_test77_ingr=remove_stopword3(menu_test55_ingr)

menu_test7_ingr=menu_test2['ประเภทอาหาร'].apply(regex_nodigits_new)

menu_test8_ingr=menu_test7_ingr.apply(normalize)
menu_test9_ingr=remove_stopword(menu_test8_ingr)
menu_test9_ingr22=menu_test2['ประเภทอาหาร'].apply(regex_nodigits_new)
menu_test9_ingr22=menu_test9_ingr22.apply(normalize)
menu_test9_ingr22=remove_stopword(menu_test9_ingr22)

def remove_list2(text):
    lenx=len(text)
    i=0
    while i < lenx:
        del text[i][2:]
        del text[i][0]
        i += 1
    return text

menu_test9_ingr22=remove_list2(menu_test9_ingr22)
menu_test_name22=menu_test_name2.apply(', '.join)
menu_test777_ingr=menu_test77_ingr.apply(', '.join)
menu_test33_ingr=menu_test3_ingr.apply(', '.join)
menu_test99_ingr=menu_test9_ingr.apply(', '.join)
menu_test99_ingr22=menu_test9_ingr22.apply(', '.join)
menu_test99_ingr2222= menu_test99_ingr22+','
menu_df = pd.DataFrame(list(zip(menu_test77_ingr,menu_test3_ingr,menu_test9_ingr,menu_test9_ingr22,menu_test_name2)),
               columns =['รายละเอียดอาหาร', 'สูตรอาหาร','ประเภทอาหาร','สัญชาติอาหาร','ชื่ออาหาร'])
menu_df2 = pd.DataFrame(list(zip(menu_test777_ingr,menu_test33_ingr,menu_test99_ingr,menu_test99_ingr22,menu_test_name22)),
               columns =['รายละเอียดอาหาร', 'สูตรอาหาร','ประเภทอาหาร','สัญชาติอาหาร','ชื่ออาหาร'])
menu_df3 = pd.DataFrame(list(zip(menu_test99_ingr2222+menu_test777_ingr+menu_test33_ingr,menu_test_name22)),
               columns =['รายละเอียดอาหาร','ชื่ออาหาร'])
menu_df3 = menu_df3.drop_duplicates(subset=['ชื่ออาหาร'],keep='last')
menu_df3=menu_df3.reset_index(drop=True)
menu_df4 = pd.DataFrame(list(zip(menu_test_name22,menu_test777_ingr+menu_test33_ingr+menu_test99_ingr,menu_test99_ingr22)),
               columns =['ชื่ออาหาร','รายละเอียดอาหาร','สัญชาติอาหาร'])
menu_df4
menu_df4 = menu_df4.drop_duplicates(subset=['ชื่ออาหาร'],keep='last')
menu_df4=menu_df4.reset_index(drop=True)
train=menu_df4
total = train.isnull().sum().sort_values(ascending = False)
percent = (train.isnull().sum()/train.isnull().count()*100).sort_values(ascending = False)
missing_train_data  = pd.concat([total, percent], axis=1, keys=['Total missing', 'Percent missing'])
print("             # of Rows, Columns:",train.shape)
print(missing_train_data.head())

color_theme = dict(color = ['rgba(221,160,221,1)','rgba(169,169,169,1)','rgba(255,160,122,1)','rgba(176,224,230,1)','rgba(169,169,169,1)','rgba(255,160,122,1)','rgba(176,224,230,1)',
                   'rgba(188,143,143,1)','rgba(221,160,221,1)','rgba(169,169,169,1)','rgba(255,160,122,1)','rgba(176,224,230,1)','rgba(189,183,107,1)','rgba(188,143,143,1)','rgba(221,160,221,1)','rgba(169,169,169,1)','rgba(255,160,122,1)','rgba(176,224,230,1)','rgba(169,169,169,1)','rgba(255,160,122,1)'])
temp = train['สัญชาติอาหาร'].value_counts()
trace = go.Bar(y=temp.index[::-1],x=(temp)[::-1],orientation = 'h',marker=color_theme)
layout = go.Layout(title = "เมนูแต่ละสัญชาติอาหาร",xaxis=dict(title='จำนวนเมนูอาหาร',tickfont=dict(size=14,)),
                   yaxis=dict(title='สัญชาติอาหาร',titlefont=dict(size=16),tickfont=dict(size=14)),margin=dict(l=200,))
data = [trace]
fig = go.Figure(data=data, layout=layout)
iplot(fig,filename='basic-bar')
menu_df3.describe()
train = menu_df3
len(train)
train.shape
data_train= train['รายละเอียดอาหาร']
target_train= train['ชื่ออาหาร']
print(data_train.shape)
print(target_train.shape)
# create a count vectorizer object
count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
count_vect.fit(train['รายละเอียดอาหาร'])

# transform the training and validation data using count vectorizer object
xtrain_count =  count_vect.transform(data_train)
# word level tf-idf
tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
tfidf_vect.fit(train['รายละเอียดอาหาร'])
xtrain_tfidf =  tfidf_vect.transform(data_train)
# ngram level tf-idf
tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(1,3), max_features=5000)
tfidf_vect_ngram.fit(train['รายละเอียดอาหาร'])
xtrain_tfidf_ngram =  tfidf_vect_ngram.transform(data_train)

nb_classifier = MultinomialNB()
nb_classifier.fit(xtrain_tfidf_ngram, target_train)
# creating an empty list
lst = []

# number of elements as input
n = str(input("ว่าไงเพื่อนเลิฟ วันนี้เธออยากทานอะไร จ๊ะ : "))

lst.append(n)  # adding the element

print(lst)
xtest_count =  count_vect.transform(lst)
xtest_tfidf =  tfidf_vect.transform(lst)
xtest_tfidf_ngram =  tfidf_vect_ngram.transform(lst)
nb_preds = nb_classifier.predict(xtest_tfidf_ngram)
print(nb_preds)
index_nb=menu_df3.loc[menu_df3['ชื่ออาหาร'] == nb_preds[0]].index[0]
nb_result=xtrain_tfidf_ngram[index_nb]
print(xtrain_tfidf_ngram)
