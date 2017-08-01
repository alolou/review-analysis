
# coding: utf-8

# In[8]:

import requests
import pickle
import pandas as pd
import json, codecs

def extract_reviews(url, head):
    i = 0
#     f = open(url+'.txt', 'w')
    
    stack = []
    metrics = []
    date = []
    headline = []
    id_review = []
    helpful_votes = []

    
    try:
        
        res=requests.get(url, headers=head) 

        res.status_code
        res.headers

        c =res.content

        # print(c)
        j = res.json()


        id_item = j['results'][0]['identifiers']['page_id']
        category_code = j['results'][0]['details']['category_name']
        name = j['results'][0]['details']['name']
        description_ext = j['results'][0]['details']['description']
        brand_name = j['results'][0]['details']['brand_name']
        
        
#         spl = descr_item.strip().split("<li>")
#         spl = list(filter(None, spl))
#         spl = list(filter(lambda el: '</li>' not in el, spl))

        done = False

        while not done:
            for each in j['results'][0]['children'][0]['results']:
                print(each['details']['comments'])
                i+=1
    #             writer = csv.writer(f)
                id_review.append(each['id'])
                stack.append(each['details']['comments'])
                headline.append(each['details']['headline'])
                helpful_votes.append(each['metrics']['helpful_votes'])
                metrics.append(each['metrics']['rating'])
                date.append(each['identifiers']['modified_date'])
    #             writer.writerows(each['details']['comments'])

            done = True
    #     print(j['results'][0]['children'])
        try:
            next_page = ('https://jet-readservices.powerreviews.com' + j['results'][0]['children'][0]['paging']['next_page_url'] or
        'https://jet-readservices.powerreviews.com' + j['results'][0]['children'][0]['paging']['next_page_url'])
            print (next_page)  
        except:
            pass


        def extract_next(next_page, head):

            res=requests.get(next_page, headers=head) 

            res.status_code
            res.headers

            c =res.content

            # # print(c)
            j = res.json()
            # print (j)

            done = False


            while not done:
                for each in j['results']:
    #                 print(each['details']['comments'])
                    id_review.append(each['id'])
                    stack.append(each['details']['comments'])
                    i+=1
                    headline.append(each['details']['headline'])
                    helpful_votes.append(each['metrics']['helpful_votes'])
                    metrics.append(each['metrics']['rating'])
                    date.append(each['identifiers']['modified_date'])
    #                

                done = True
#             try:
            next_page = 'https://jet-readservices.powerreviews.com' + j['paging']['next_page_url']
            print (next_page)
            extract_next(next_page, head)
#             except:
#                 pass
        try:
            extract_next(next_page, head)
        except:
            pass

        print (len(stack))

        df = pd.DataFrame(stack, columns = ['reviews'])
        df['headline'] = headline
        df['id_review'] = id_review
        df['id_item'] = id_item
        df['helpful_votes'] = helpful_votes
        df['metrics'] = metrics
        df['page_id'] = page_id
        df['category_code'] = category_code
        df['name'] = name
        df['description_ext'] = description_ext
        df['description_int'] = descr_item
        df['brand_name'] = brand_name
        df['date_modified'] = date
        
        if (len(df.index)>0):
            saved = 'C:\\Users\\liliya.akhtyamova\\Documents\\reviews\\parsed\\'+'parsed_tv'+page_id+'.csv'
            df.to_csv(saved, index = True)
#         
        return [df]

    except:
        pass
    print (i)


# In[4]:

import csv
import pandas as pd

with open('C:\\Users\\liliya.akhtyamova\\Documents\\reviews\\top categories\\tv.csv', 'r') as rskus:
# xl = pd.ExcelFile("C:\\Users\\liliya.akhtyamova\\Documents\\reviews\\top categories\\tv.xlsx")
# parsed = pd.io.excel.ExcelFile.parse(xl, "Sheet1", header = None)
# df = pd.DataFrame(data=parsed)
# for i in range(len(df.index)):
#     print (df.iloc[i,0])
# with open('C:\\Users\\liliya.akhtyamova\\Documents\\reviews\\top categories\\tv.xlsx', 'r') as rskus:
    rowreader = csv.reader(rskus, delimiter = '|')
   
#     next(rowreader, None) 
 
    for row in rowreader:
        
#         print(''.join(row[0]))
    url2 = 'https://readservices.powerreviews.com/g/49172/product/'+row[0]+'/q/product_detail?properties=true&localizations=true&features=true&review_images.paging.size=8'
    print(url2)
    head2 = {'Authorization': '3ff84632-35e9-49b7-8a3a-7638cdd208cf'} 
    reviews = extract_reviews(url2, head2, descr)
    i+=1
    print(reviews)
#         df = pd.DataFrame.from_records(reviews, columns = ['reviews'])
#         f = open(row[0]+'.txt', 'w')
#         for s in reviews:
#             f.write(s + '\n')
print (i)    
    


# In[14]:

import nltk, os, collections, csv, re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from pywsd.utils import lemmatize, lemmatize_sentence

def reviews_to_wb(id_review):
    cList = {
  "ain't": "am not",
  "aren't": "are not",
  "can't": "cannot",
  "can't've": "cannot have",
  "'cause": "because",
  "could've": "could have",
  "couldn't": "could not",
  "couldn't've": "could not have",
  "didn't": "did not",
  "doesn't": "does not",
  "don't": "do not",
  "hadn't": "had not",
  "hadn't've": "had not have",
  "hasn't": "has not",
  "haven't": "have not",
  "he'd": "he would",
  "he'd've": "he would have",
  "he'll": "he will",
  "he'll've": "he will have",
  "he's": "he is",
  "how'd": "how did",
  "how'd'y": "how do you",
  "how'll": "how will",
  "how's": "how is",
  "I'd": "I would",
  "I'd've": "I would have",
  "I'll": "I will",
  "I'll've": "I will have",
  "I'm": "I am",
  "I've": "I have",
  "isn't": "is not",
  "it'd": "it had",
  "it'd've": "it would have",
  "it'll": "it will",
  "it'll've": "it will have",
  "it's": "it is",
  "let's": "let us",
  "ma'am": "madam",
  "mayn't": "may not",
  "might've": "might have",
  "mightn't": "might not",
  "mightn't've": "might not have",
  "must've": "must have",
  "mustn't": "must not",
  "mustn't've": "must not have",
  "needn't": "need not",
  "needn't've": "need not have",
  "o'clock": "of the clock",
  "oughtn't": "ought not",
  "oughtn't've": "ought not have",
  "shan't": "shall not",
  "sha'n't": "shall not",
  "shan't've": "shall not have",
  "she'd": "she would",
  "she'd've": "she would have",
  "she'll": "she will",
  "she'll've": "she will have",
  "she's": "she is",
  "should've": "should have",
  "shouldn't": "should not",
  "shouldn't've": "should not have",
  "so've": "so have",
  "so's": "so is",
  "that'd": "that would",
  "that'd've": "that would have",
  "that's": "that is",
  "there'd": "there had",
  "there'd've": "there would have",
  "there's": "there is",
  "they'd": "they would",
  "they'd've": "they would have",
  "they'll": "they will",
  "they'll've": "they will have",
  "they're": "they are",
  "they've": "they have",
  "to've": "to have",
  "wasn't": "was not",
  "we'd": "we had",
  "we'd've": "we would have",
  "we'll": "we will",
  "we'll've": "we will have",
  "we're": "we are",
  "we've": "we have",
  "weren't": "were not",
  "what'll": "what will",
  "what'll've": "what will have",
  "what're": "what are",
  "what's": "what is",
  "what've": "what have",
  "when's": "when is",
  "when've": "when have",
  "where'd": "where did",
  "where's": "where is",
  "where've": "where have",
  "who'll": "who will",
  "who'll've": "who will have",
  "who's": "who is",
  "who've": "who have",
  "why's": "why is",
  "why've": "why have",
  "will've": "will have",
  "won't": "will not",
  "won't've": "will not have",
  "would've": "would have",
  "wouldn't": "would not",
  "wouldn't've": "would not have",
  "y'all": "you all",
  "y'alls": "you alls",
  "y'all'd": "you all would",
  "y'all'd've": "you all would have",
  "y'all're": "you all are",
  "y'all've": "you all have",
  "you'd": "you had",
  "you'd've": "you would have",
  "you'll": "you you will",
  "you'll've": "you you will have",
  "you're": "you are",
  "you've": "you have"
}

    c_re = re.compile('(%s)' % '|'.join(cList.keys()))

    def expandContractions(text, c_re=c_re):
        def replace(match):
            return cList[match.group(0)]
        return c_re.sub(replace, text)
#     dataset_path = 'Documents\\reviews\\parsed'
#     for filename in os.listdir(dataset_path):
#         with open(os.path.join(dataset_path,filename)) as infile:
#             reader = csv.reader(infile)
#     #         data = list(reader)
#             next(reader, None)
#     #         print (len(data))
#             items = filename.split('.')[0][6:]
#     #         print (item)
#             for row in reader:
    #             print (row)
    wordnet_lemmatizer = WordNetLemmatizer()
    words = collections.Counter()
    content_text = ''.join(row[1])
    content_text = expandContractions(content_text)
    f = [word for word in content_text.split() if word not in stopwords.words('english') or word == 'not' ]
    sent = (' '.join(f))
    content_text = re.sub(r'\b(?:not|never|no)\b[\w\s]+[^\w\s]', 
   lambda match: re.sub(r'(\s+)(\w+)', r'\1NEG_\2', match.group(0)), sent, flags=re.IGNORECASE)
    #             print (row[1])
    space_chars = u'«»“”’*…/.\\'
    for c in space_chars:
        
#         content_text = content_text.replace('it\'s', 'it is')
#         content_text = content_text.replace('don\'t', 'do not')
#         content_text = content_text.replace('doesn\'t', 'does not')
#         content_text = content_text.replace('hasn\'t', 'has not')
#         content_text = content_text.replace('haven\'t', 'have not')
        content_text = content_text.replace(c, ' ')
    
#         content_text = content_text.replace('\'', '')
#     tokens = nltk.tokenize.wordpunct_tokenize(content_text)
#     tokens = nltk.word_tokenize(content_text)
    tokens = lemmatize_sentence(content_text)
        

    #             tokens = nltk.tokenize.wordpunct_tokenize(row[1])
    #             headline = row[3]
    id_item = row[6]
    helpful_votes = row[4]
    metrics_item = row[5]

    for token in tokens:
        if len(token) > 2:
            token = token.lower()
#             word = wordnet_lemmatizer.lemmatize(token)
            if len(token) > 0:
                words[token] += 1

    def construct_bow(words):
        stop_words = set(stopwords.words('english'))
        return [
        (
        word.replace(' ', '_').replace(':', '_').replace('|', '_').replace('\t', '_') + 
        ('' if cnt == 1 else ':%g' % cnt)
        )
        for word, cnt in words.items() if not word in stop_words
        ]

    parts = (
    ['%d' % int(id_review)] + 
    ['|@words'] + construct_bow(words) +
    #         ['|@headline'] + construct_bow({headlines: 1 for headlines in headline}) +
    ['|@category %s' % str(id_item)] +
    ['|@votes %d' % int(helpful_votes)] +
    ['|@metrics %d' % int(metrics_item)] 
    )  
    #         ['|@author'] + construct_bow({author: 1} if author is not None else {}) +
    #         ['|@users'] + construct_bow(users) + 
    #         ['|@tags'] + construct_bow({tag: 1 for tag in tags}) +
    #         ['|@hubs'] + construct_bow({hub_id: 1 for hub_id in hubs}))
    return ' '.join(parts)


# In[16]:

dataset_path = 'Documents\\reviews\\parsed'
for filename in os.listdir(dataset_path):
    with open(os.path.join(dataset_path,filename)) as infile:
        reader = csv.reader(infile)
        next(reader, None)
        for row in reader:
            line = reviews_to_wb(row[3])
            print(line)
#             with open('Documents/reviews/wb/review_wb/' + str(row[3]), 'w') as fout:
#                 fout.write(line)
            with open('Documents/reviews/wb/review_corpus_7items_neg.txt', 'a') as fout1:
                fout1.write(line + "\n")


# In[129]:

import os
import artm
import glob

batches_folder = 'Documents/reviews/wb/review_wb/'
source_file = 'Documents/reviews/wb/review_corpus_multimodal.txt'
dict_name = batches_folder + '/dictionary.dict'

# batches_folder = 'Documents/reviews/wb_temp/review_wb/'
# source_file = 'Documents/reviews/wb_temp/reviews_temp.txt'
# dict_name = batches_folder + '/dictionary.dict'


# In[133]:

if not glob.glob(os.path.join(batches_folder, "*")):
    batch_vectorizer = artm.BatchVectorizer(data_path=source_file, 
                                            data_format="vowpal_wabbit", 
                                            target_folder=batches_folder,
                                            batch_size=100)
else:
    batch_vectorizer = artm.BatchVectorizer(data_path=batches_folder,
                                            data_format='batches')
dictionary = artm.Dictionary(name="dictionary")
dictionary.gather(batch_vectorizer.data_path)


# In[123]:

dictionary = artm.Dictionary()

if not os.path.isfile(dict_name):
    dictionary.gather(data_path=batch_vectorizer.data_path)
    dictionary.save(dictionary_path=dict_name)

dictionary.load(dictionary_path=dict_name)


# In[113]:

dict_name = os.path.join(batches_folder, "dict.txt")
dictionary = artm.Dictionary(name="dictionary")
if not os.path.exists(dict_name):
    dictionary.gather(batches_folder)
    dictionary.save_text(dict_name)
else:
    dictionary.load_text(dict_name)


# In[199]:

scores_list = []
scores_list.append(artm.PerplexityScore(name='PerplexityScore'))  # перплексия (перенормированное правдоподобие)
scores_list.append(artm.SparsityPhiScore(name='SparsityPhiScore', class_id="@words"))   # разреженность Phi
scores_list.append(artm.SparsityThetaScore(name='SparsityThetaScore'))   # разреженность Theta
scores_list.append(artm.TopTokensScore(name="top_words", 
                                          num_tokens=15, 
                                          class_id="@words"))  # для печати наиболее вероятных терминов темы


# In[205]:

T = 30   # количество тем
model_artm = artm.ARTM(num_topics=T,  # число тем
                           class_ids={"@words":1, 
                                  "@category":1, "@votes":1,"@metrics":1},   # число после названия модальностей - это их веса
                       num_document_passes=10,   # сколько делать проходов по документу
                       cache_theta=True,   # хранить или нет глоабльную матрицу Theta
                       reuse_theta=False,   # если Theta хранится, нужно ли ее вновь инициализировать при каждом проходе
                       theta_columns_naming="title",   # как именовать столбцы в матрице Theta
                       seed=789,   # random seed
                       scores=scores_list)  # метрики качества


# In[206]:

# model_artm.regularizers.add(artm.DecorrelatorPhiRegularizer(name='DecorrelatorPhi', tau=1.5e+10,
#                                                             class_ids=["@words"]))
model_artm.initialize(dictionary)
get_ipython().magic('time model_artm.fit_offline(batch_vectorizer=batch_vectorizer,                              num_collection_passes=20)')


# In[47]:

batch_vectorizer = artm.BatchVectorizer(data_path=REVIEW_DATA_PATH, data_format='vowpal_wabbit', collection_name='reviews', target_folder=BATCHES_FOLDER)


# In[46]:

batch_vectorizer = None
if len(glob.glob(os.path.join(BATCHES_FOLDER, '*.batch'))) < 1:
    batch_vectorizer = artm.BatchVectorizer(data_path=REVIEW_DATA_PATH, data_format='vowpal_wabbit', collection_name='reviews', target_folder=BATCHES_FOLDER)
else:
    batch_vectorizer = artm.BatchVectorizer(data_path=REVIEW_DATA_PATH, data_format='vowpal_wabbit')


# In[175]:

# dictionary = artm.Dictionary()

model_plsa = artm.ARTM(topic_names=['topic_{}'.format(i) for i in range(20)],
                       scores=[artm.PerplexityScore(name='PerplexityScore',
#                                                     use_unigram_document_model=False,
                                                    dictionary=dictionary)],
                       cache_theta=True)

model_artm = artm.ARTM(topic_names=['topic_{}'.format(i) for i in range(20)],
                       scores=[artm.PerplexityScore(name='PerplexityScore',
#                                                     use_unigram_document_model=False,
                                                    dictionary=dictionary)],
                       regularizers=[artm.SmoothSparseThetaRegularizer(name='SparseTheta', tau=-0.15),
                                     artm.SmoothSparsePhiRegularizer(name='SparsePhi', tau=-0.1)],
                       cache_theta=True)


# In[176]:

model_artm.regularizers.add(artm.DecorrelatorPhiRegularizer(name='DecorrelatorPhi', tau=1.5e+5))


# In[177]:

model_artm.scores.add(artm.SparsityPhiScore(name='SparsityPhiScore'))
model_artm.scores.add(artm.SparsityThetaScore(name='SparsityThetaScore'))
model_artm.scores.add(artm.TopicKernelScore(name='TopicKernelScore', probability_mass_threshold=0.3))


# In[188]:


model_plsa.initialize(dictionary=dictionary)
model_artm.initialize(dictionary=dictionary)


# In[179]:

model_plsa.num_document_passes = 1
model_artm.num_document_passes = 1

model_plsa.fit_offline(batch_vectorizer=batch_vectorizer, num_collection_passes=15)
model_artm.fit_offline(batch_vectorizer=batch_vectorizer, num_collection_passes=15)


# In[207]:

from matplotlib import pyplot as plt
get_ipython().magic('matplotlib inline')
import seaborn

plt.plot(model_artm.score_tracker["PerplexityScore"].value[1:])

# plt.plot(range(model_plsa.num_phi_updates), model_plsa.score_tracker['SparsityPhiScore'].value, 'b--',
#                  range(model_artm.num_phi_updates), model_artm.score_tracker['SparsityPhiScore'].value, 'r--', linewidth=2)
# plt.xlabel('Iterations count')
# plt.ylabel('PLSA Phi sp. (blue), ARTM Phi sp. (red)')
# plt.grid(True)
# plt.show()

# plt.plot(range(model_plsa.num_phi_updates), model_plsa.score_tracker['SparsityThetaScore'].value, 'b--',
#                  range(model_artm.num_phi_updates), model_artm.score_tracker['SparsityThetaScore'].value, 'r--', linewidth=2)
# plt.xlabel('Iterations count')
# plt.ylabel('PLSA Theta sp. (blue), ARTM Theta sp. (red)')
# plt.grid(True)
# plt.show()


# In[208]:

for topic_name in model_artm.topic_names:
    print (topic_name + ': ',)
    print (", ".join(model_artm.score_tracker["top_words"].last_tokens[topic_name]))


# In[151]:

print ("Phi sparsity:",model_artm.score_tracker["SparsityPhiScore"].last_value)
print ("Theta sparsity:", model_artm.score_tracker["SparsityThetaScore"].last_value)


# In[171]:

model_artm.regularizers.add(artm.DecorrelatorPhiRegularizer(name='DecorrelatorPhi', tau=1.5e+10))
#                                                             class_ids=["@words"]))


# In[172]:

model_artm.initialize(dictionary=dictionary)
model_artm.num_document_passes = 1

model_artm.fit_offline(batch_vectorizer=batch_vectorizer, 
                       num_collection_passes=15)


# In[173]:

plt.plot(model_artm.score_tracker["PerplexityScore"].value[1:])


# In[174]:

for topic_name in model_artm.topic_names:
#     print (topic_name + ': ')
    print (", ".join(model_artm.score_tracker["top_words"].last_tokens[topic_name]))


# In[9]:

url2 = 'https://readservices.powerreviews.com/g/49172/product/823fbb41b264471cbe41c5adbcdeb932/q/product_detail?properties=true&localizations=true&features=true&review_images.paging.size=8'
head2 = {'Authorization': '3ff84632-35e9-49b7-8a3a-7638cdd208cf'} 

# extract_reviews(url2, head2)

res=requests.get(url2, headers=head2) 

res.status_code
res.headers

c =res.content

# print(c)
j = res.json()

print (j['results'][0]['identifiers']['page_id'])

# next_page = ('https://jet-readservices.powerreviews.com' + j['results'][0]['children'][0]['paging']['next_page_url'] or
# 'https://jet-readservices.powerreviews.com' + j['results'][0]['children'][0]['paging']['next_page_url'])
# print (next_page)  

reviews = extract_reviews(url2, head2)
print(reviews)


# In[ ]:

res=requests.get(url2, headers=head2) 

res.status_code
res.headers

c =res.content

# print(c)
j = res.json()

