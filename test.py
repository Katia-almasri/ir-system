import ir_datasets
dataset = ir_datasets.load("antique/test")
dataset2 = ir_datasets.load("beir/quora/test")

###### IMPORTING PACKAGES #################################
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import numpy as np
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()
import string
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import save_npz
from scipy.sparse import load_npz
from flask import Flask, request, jsonify, render_template, json
import gzip
from scipy.sparse import csr_matrix, load_npz
from nltk.stem import WordNetLemmatizer
import re
from datetime import datetime
from dateutil import parser
import enchant
from nltk.corpus import wordnet
################# GLOBAL ARRAYS TO STORE THE CORPUS FROM THE ir-datasets WEBSITER ##################
documents = []
documents_keys = []

documents2 = []
documents_keys2 = []

d = enchant.DictWithPWL("en_US", "my_pwl.txt")
#### ASSIGN THE CORPUS IN DATASET 1 TO DOCUMENTS ARRAY #################
def assign_data_set_to_documents():
    for i, doc in enumerate(dataset.docs_iter()):
        documents.append(doc[1])
        documents_keys.append(doc[0])
        
#### ASSIGN THE CORPUS IN DATASET 2 TO DOCUMENTS ARRAY #################
def assign_data_set_to_documents2():
    for i, doc in enumerate(dataset2.docs_iter()):
        documents2.append(doc[1])
        documents_keys2.append(doc[0])
############ PREPROCESSING TEXT ###########      
def remove_whitespace(text):
	return " ".join(text.split()) 

def remove_stopwords(words):
    stop_words = set(stopwords.words("english"))
    filtered_text = [word for word in words if word not in stop_words]
    return filtered_text

def stem_words(words):
	stems = [stemmer.stem(word) for word in words]
	return stems

def stem_words(words):
	stems = [stemmer.stem(word) for word in words]
	return stems

def remove_lemma(words):
    tagged_words = nltk.pos_tag(words)
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = []
    for word, tag in tagged_words:
        lemma = lemmatizer.lemmatize(word)
        lemmatized_words.append(lemma)
    return lemmatized_words


def is_date_or_time(word):
    try:
        parser.parse(word)
        return True
    except:
        return False

def remove_punctuation(words):
    new_words = []
    # Remove punctuation
    for word in words:
        if  word.isalpha() or is_date_or_time(word):
            new_words.append(word)
    return new_words

def text_lowercase(text):
	return text.lower()

def format_dates(document):
    # Define a regular expression pattern to match different date formats
    date_pattern = re.compile(r'\d{1,2}[/-]\d{1,2}[/-]\d{4}|\d{4}[/-]\d{1,2}[/-]\d{1,2}')
    time_pattern = re.compile(r'\d{1,2}:\d{2}(:\d{2})?\s*(AM|PM|am|pm)?')
    # Define the desired output date format
    output_format = "%Y-%m-%d"
    output_time_format = "%I:%M %p"
    # Replace the dates in the document with the desired output format
    for match in date_pattern.findall(document):
        try:
            date_obj = parser.parse(match)
            standard_date_format = date_obj.strftime(output_format)
            document = document.replace(match, standard_date_format)
        except ValueError:
            continue
    try:
        document = time_pattern.sub(lambda match: datetime.strptime(match.group(), '%I:%M %p' if match.group(2) else '%H:%M').strftime(output_time_format), document)
    except ValueError:
        print("")
    return document

def process_shortcuts(document):
    matches = []
    shortcut_dict = {
    'p.p.s':'post postscript',
    'u.s.a': 'united states of america',
    'a.k.a': 'also known as',
    'm.a.d': 'Mutually Assured Destruction',
    'a.b.b': 'Asea Brown Boveri',
    's.c.o': 'Santa Cruz Operation',
    'e.t.c': 'etcetera',
    'm.i.t': 'Massachusetts Institute of Technology',
    'v.i.p': 'very important person',
    'us':'united states of america',
    'u.s.':'united states of america',
    'usa':'united states of america',
    'cobol':'common business oriented language',
    'rpm':'red hat package manager',
    'ap':'associated press',
    'gpa':'grade point average',
    'npr':'national public radio',
    'fema':'federal emergency',
    'crt':'cathode ray tube',
    'gm':'grandmaster',
    'fps':'frames per second',
    'pc':'personal computer',
    'pms':'premenstrual syndrome',
    'cia':'central intelligence agency',
    'aids':'acquired immune deficiency syndrome',
    'it\'s':'it is',
    'you\'ve':'you have',
    'what\'s':'what is',
    'that\'s':'that is',
    'who\'s':'who is',
    'don\'t':'do not',
    'haven\'t':'have not',
    'there\'s':'there is',
    'i\'d':'i would',
    'it\'ll':'it will',
    'i\'m':'i am',
    'here\'s':'here is',
    'you\'ll':'you will',
    'cant\'t':'can not',
    'didn\'t':'did not',
    'hadn\'t':'had not',
    'kv':'kilovolt',
    'cc':'cubic centimeter',
    'aoa':'american osteopathic association',
    'rbi':'reserve bank',
    'pls':'please',
    'dvd':'digital versatile disc',
    'bdu':'boise state university',
    'dvd':'digital versatile disc',
    'mac':'macintosh',
    'tv':'television',
    'cs':'computer science',
    'cse':'computer science engineering',
    'iit':'indian institutes of technology',
    'uk':'united kingdom',
    'eee':'electrical and electronics engineering',
    'ca':'california',
    'etc':'etcetera',
    'ip':'internet protocol',
    'bjp':'bharatiya janata party',
    'gdp':' gross domestic product',
    'un':'unitednations',
    'ctc':'cost to company',
    'atm':'automated teller machine',
    'pvt':'private',
    'iim':'indian institutes of management'
    
    }
    shortcut_pattern1 = re.compile(r'[A-Za-z]\.[A-Za-z]\.[A-Za-z]*')
    shortcut_pattern2 = re.compile(r'\b[A-Za-z]{2,3}\b')
    shortcut_pattern3 = re.compile(r'\w+\'\w+')
    
    matches.append(shortcut_pattern1.findall(document)) 
    matches.append(shortcut_pattern2.findall(document))
    matches.append(shortcut_pattern3.findall(document))
    
    for arr in matches:
        for match in arr:
            if match in shortcut_dict:
                document = document.replace(match, shortcut_dict[match])     
            
    return document

############### THE MAIN SERVICE FOR PROCESS ALL DOCUMENTS IN CHOOSEN CORPUS ###################
def text_processing(choosed_document):
    docs_array = []
    for i, document in enumerate(choosed_document):
        document = remove_whitespace(document)
        document = text_lowercase(document)
        document = process_shortcuts(document)
        document = format_dates(document)
        words = word_tokenize(document)
        words = remove_punctuation(words)
        words = remove_stopwords(words)
        words = remove_lemma(words)
        words = stem_words(words)
        docs_array.append(words)
    return docs_array

############# MAKING INVERTED INDEX(JUST AN ADDITION TO REPRESENT DATA)#############
def make_inverted_index(docs_array):
    inverted_index = defaultdict(list)
    for i, doc in enumerate(docs_array):
        for token in doc:
            if i not in inverted_index[token]:
                inverted_index[token].append(i)
    return dict(inverted_index)
########### STORE INVERTED INDEX(BECAUSE ITS TOO BIG) #####################
def store_inverted_index(inverted_index, file_name):
    json_data = json.dumps(inverted_index)
    with gzip.open(file_name, 'wt') as f:
        f.write(json_data) 
############ MAIN SERVICE FOR PROCESS THE INPUT QUERY ##################
def process_query(query):
    query_array = []
    query = remove_whitespace(query)
    query = text_lowercase(query)
    query = process_shortcuts(query)
    query = format_dates(query)
    # query = error_detection(query)
    words = word_tokenize(query)
    words = remove_punctuation(words)
    words = remove_stopwords(words)
    words = remove_lemma(words)
    words = stem_words(words)
    query_array = words
    return query_array
########### FOR THE EXTRA FEATURE WE HAVE TO CHECK IF AN EXPANDED WORD IS INCLUDED IN TH CORPUS
## ELSE NO NEED TO EXPAND IT IN THE QUERY 
def check_word_exist_in_doc(word, docs_array):
    for doc in docs_array:
        for term in doc:
            if word == term:
                return True
    return False
####### USING WORDNET LIBRARY FROM NLTK PACKAGE TO GET A LIST OF SIMILAR TERMS TO A WORD 
def expand_query(query_array, docs_array):
    synonyms = []
    for word in query_array:
        word_synonyms = set()
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                if check_word_exist_in_doc(lemma.name(), docs_array):
                        word_synonyms.add(lemma.name())
        word_synonyms = sorted(word_synonyms, key=lambda x: nltk.edit_distance(word, x))[:3]
        synonyms.append(word_synonyms)
    
    new_synonyms = []
    for syn in synonyms:
        for term in syn:
            if syn not in query_array:
                new_synonyms.append(term)
    expand_query = " ".join(query_array)+" "+" ".join(new_synonyms)
    return expand_query

######### MAIN SERVICE FOR MAKING TFIDF MATRIX
def make_tf_idf_values(docs_array):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([' '.join(doc) for doc in docs_array])
    return vectorizer.get_feature_names_out(), tfidf_matrix, vectorizer
######### STORE THE MATRIX IN A FILE BECAUSE OF ITS LARGE SIZE
def store_tfidf_vecotrizer(tfidf_matrix_file_name,vectirizer_file_name , tfidf, vecotrizer):
    save_npz(tfidf_matrix_file_name, tfidf)
    with open(vectirizer_file_name, 'wb') as f:
        pickle.dump(vecotrizer, f)
        
######## LOAD THE MATRIX (USED IN INLINE MODE) ##############
def load_vect_and_tfidf(tfidf_file, vectorizer_file):
    tfidf_matrix = load_npz(tfidf_file)
    with open(vectorizer_file, 'rb') as f:
        vectorizer = pickle.load(f)
    return tfidf_matrix, vectorizer

######## MATCHING & RANKING ##############
def get_query_result(query_array, vectorizer, tfidf, expand_query):
    q = " ".join(query_array)
    ex_q = " ".join(expand_query)
    
    query_vec = vectorizer.transform([q])
    expanded_query_vector = vectorizer.transform([ex_q])
    #### BIGGER WEIGHT FOR THE ORIGINAL QUERY
    alpha = 0.7
    beta = 0.3
    
    weighted_query_vector = alpha * query_vec + beta * expanded_query_vector
    cosine_similarities = cosine_similarity(weighted_query_vector, tfidf)
    sorted_doc_indices = np.argsort(cosine_similarities[0])[-10:]
    sorted_scores = cosine_similarities[0][sorted_doc_indices]
    return sorted_doc_indices, sorted_scores

######## SORTING THE DOCUMENTS AND GET TOP 10 DOCS
def get_documents(sorted_doc_indices, sorted_scores, dataset_type):
    ranked_doc = []
    cnt = len(sorted_scores)-1
    for idx in sorted_doc_indices:
        if dataset_type == 1:
            element = {
                "document": documents[idx],
                "socre": '{:.3f}'.format(sorted_scores[cnt]),
                "index": documents_keys[idx]
            }
        else:
             element = {
                "document": documents2[idx],
                "socre": '{:.3f}'.format(sorted_scores[cnt]),
                "index": documents_keys2[idx]
            }
        cnt-=1
        ranked_doc.append(element)
    return ranked_doc

import csv
########## SERVICE FOR STORING THE CLEANED CORPUS ###########
def store_docs_array_in_file(docs_array, file_name):
    with open(file_name, 'w', encoding='utf-8', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(docs_array)
######### READ IT FROM THE FILE       
def read_docs_array_from_file(file_name):
    with open(file_name, 'r') as file:
        reader = csv.reader(file)
        docs_array = []
        for row in reader:
            docs_array.append(row)
    return docs_array


################################## EXECUTION PART ##################################
                        #1. LOAD THE DATA FROM THE FILES  
                        #2. LOAD TFIDF MATRIX
                        #3. RUN THE SERVER(ONLINE MODE)   
                         
# assign_data_set_to_documents() 
# docs_array = text_processing(documents)

# assign_data_set_to_documents2()
# docs_array2 = text_processing(documents2)

docs_array2 = read_docs_array_from_file('docs_array2.csv')
docs_array = read_docs_array_from_file('docs_array.csv')

# inverted_index = make_inverted_index(docs_array)
# store_inverted_index(inverted_index, 'inverted_index.json.gz')

# inverted_index2 = make_inverted_index(docs_array2)
# store_inverted_index(inverted_index, 'inverted_index2.json.gz')

# feature_names, tfidf_matrix2, vectorizer2 = make_tf_idf_values(docs_array2)
# feature_names, tfidf_matrix, vectorizer = make_tf_idf_values(docs_array)

# store_tfidf_vecotrizer("tfidf_matrix2.npz", "vectorizer2.pkl", tfidf_matrix2, vectorizer2)
# store_tfidf_vecotrizer("tfidf_matrix.npz", "vectorizer.pkl", tfidf_matrix, vectorizer)

tfidf_matrix2, vectorizer2 = load_vect_and_tfidf("tfidf_matrix2.npz", "vectorizer2.pkl")
tfidf_matrix, vectorizer = load_vect_and_tfidf("tfidf_matrix.npz", "vectorizer.pkl")

app = Flask(__name__)
@app.route('/search', methods=['POST'])

def searchText():
    
    query = request.form['search']
    dataset = request.form['dataset']
    query_array = process_query(query)
    #expand the query
    
    sorted_docs_with_scores = []
    
    if dataset == 'dataset1':
        # antique dataset
        expanded_query = expand_query(query_array, docs_array)
        sorted_doc_indices, sorted_scores = get_query_result(query_array, vectorizer, tfidf_matrix, expanded_query)
        sorted_docs_with_scores = get_documents(sorted_doc_indices, sorted_scores, 1)
        
    else:   # dataset2
        expanded_query = expand_query(query_array, docs_array2)
        sorted_doc_indices, sorted_scores = get_query_result(query_array, vectorizer2, tfidf_matrix2, expanded_query)
        sorted_docs_with_scores = get_documents(sorted_doc_indices, sorted_scores, 2)
        
    return render_template('results.html', results=sorted_docs_with_scores)

if __name__ == '__main__':
    app.run()
################################# EVALUATION PART #################################
queries = []
queries2 = []
########## LOAD QUERIES FROM DATASET ###########
def get_queries():
    global queries
    counter = 0
    for query in dataset.queries_iter():
        if counter >=50:
            break
        counter+=1
        queries.append(query)
        

def get_queries2():
    global queries2
    counter = 0
    for query in dataset2.queries_iter():
        if counter >=50:
            break
        counter+=1
        queries2.append(query)
    
        
qrels = []
qrels2 = []
############## LOAD QRELS FROM DATASET ##############
def get_qrels():
    for qrel in dataset.qrels_iter():
         query_id = qrel[0]
         if query_id in [query[0] for query in queries]:
            qrels.append(qrel)
        
def get_qrels2():
    for qrel in dataset2.qrels_iter():
        query_id = qrel[0]
        if query_id in [query[0] for query in queries2]:
            qrels2.append(qrel)

########### GET THE RETRIEVED DOCUMENTS FOR ALL THE EXTRACTED QUERY
retrieved_docs = {}
retrieved_docs2 = {}
def get_retrieved_documents(datasets_type):
    if datasets_type==1:
        for query in queries:
            q = query.text
            retrieved_docs[query.query_id] = {}
            query_array = process_query(q)
            expanded_query = expand_query(query_array, docs_array)
            sorted_doc_indices, sorted_scores = get_query_result(query_array, vectorizer, tfidf_matrix, expanded_query)
            sorted_docs_with_scores = get_documents(sorted_doc_indices, sorted_scores, 1)
            retrieved_docs[query.query_id]  = sorted_docs_with_scores
    else:
        for query in queries2:
            q = query.text
            retrieved_docs2[query.query_id] = {}
            query_array = process_query(q)
            expanded_query = expand_query(query_array, docs_array2)
            sorted_doc_indices, sorted_scores = get_query_result(query_array, vectorizer2, tfidf_matrix2, expanded_query)
            sorted_docs_with_scores = get_documents(sorted_doc_indices, sorted_scores, 2)
            retrieved_docs2[query.query_id]  = sorted_docs_with_scores
########## CONVERT THE {(query_id, relevance, doc_id)}  => {query_id: [{doc_id, relevance}, ...]}      
def get_QRELS(datasets_type):
    retrievedRelevantDocs = {}
    if datasets_type==1:
        for qrel in qrels:
            if qrel.query_id not in retrievedRelevantDocs:
                retrievedRelevantDocs[qrel.query_id] = [] 
            retrievedRelevantDocs[qrel.query_id].append({'relevance': qrel.relevance, 'doc_id':qrel.doc_id})
    else:
        for qrel in qrels2:
            if qrel.query_id not in retrievedRelevantDocs:
                retrievedRelevantDocs[qrel.query_id] = [] 
            retrievedRelevantDocs[qrel.query_id].append({'relevance': qrel.relevance, 'doc_id':qrel.doc_id})
    return retrievedRelevantDocs

def get_relevance_non_relevance_docs(relevantDocs, datasets_type):
    qrels = {}
    threshold = 1
    for qrel in relevantDocs:
        # Retrieve the query ID and relevant documents
        query_id = qrel
        relevant_docs = relevantDocs[query_id]
        # Add the query ID to the qrels dictionary
        qrels[query_id] = []
        # Iterate over all the documents in the dataset and add them to the qrels dictionary
        if datasets_type==1:
            for doc_id in documents_keys:
                relevance = 2
                if doc_id in [d['doc_id'] for  d in relevant_docs]:
                    relevance = 1
                else:
                    relevance = 0

                # Add the document and relevance score to the qrels dictionary
                qrels[query_id].append({'doc_id':doc_id, 'relevance':relevance})
        else:
            for doc_id in documents_keys2:
                relevance = 2
                if doc_id in [d['doc_id'] for  d in relevant_docs]:
                        relevance = 1
                else:
                        relevance = 0

                # Add the document and relevance score to the qrels dictionary
                qrels[query_id].append({'doc_id':doc_id, 'relevance':relevance})
    return qrels

######### GET SET OF RELEVANT DOCUMENTS FOR A QUERY_ID
def getRelevance1(query_id, qrels_new):
    relevance1 = set()
    for doc in qrels_new.get(query_id):
        if(doc['relevance']==1):
            relevance1.add(doc['doc_id'])
    return relevance1

def getRetrievedDocs(retrieved):
    retrievedDocs = set()
    for doc in retrieved:
        retrievedDocs.add(doc['index'])
    return retrievedDocs
########### EVALUATION MEASURES ##############
def precission_at_10(relevance, retrieved):
    num_relevant_retrieved = len(set(relevance).intersection(retrieved))
    precision_at_10 = num_relevant_retrieved / 10
    return precision_at_10

def recall_values(relevance, retrieved):
    num_relevant_retrieved = len(relevance.intersection(retrieved))
    num_relevant_total = len(relevance)
    recall = num_relevant_retrieved / num_relevant_total
    return recall

def mean_avg_precision(relevance1, retrieved):
    precision_sum = 0.0
    num_relevant = len(relevance1)
    num_correct = 0
    for i, doc in enumerate(retrieved):
        if doc in relevance1:
            num_correct += 1
            precision = num_correct / (i + 1)
            precision_sum += precision

    ap  = precision_sum / num_relevant
    return ap

def mean_reciprocal_rank(relevance1, retrieved):
    rr = 0
    for i, doc in enumerate(retrieved):
        if doc in relevance1:
            rr = 1/(i+1)
            break
    return rr

############### MAIN SERVICE FOR CALCULATING THE EVALUATION FOR THE FIRST DATASET
def calc_evaluation(qrels_new):
    AP = []
    MRR = []
    
    for query in queries:
        
        relevance1 = getRelevance1(query.query_id, qrels_new)
        retrieved = getRetrievedDocs(retrieved_docs[query.query_id])
        #recall
        r = recall_values(relevance1, retrieved)
    #     #precission @ 10
        p = precission_at_10(relevance1, retrieved)
        with open('evaluation.txt', 'a') as f:
            f.write(f"{query.query_id}: precision@k:{p:.3f} recall:{r:.3f}\n")
        
        ap = mean_avg_precision(relevance1, retrieved)
        AP.append(ap)
        
        rr = mean_reciprocal_rank(relevance1, retrieved)
        MRR.append(rr)
    #MRR
    mean_MRR = sum(MRR) / len(MRR)
    #MAP
    MAP = sum(AP) / len(AP)
    with open('evaluation.txt', 'a') as f:
        f.write(f"{query.query_id}: MRR:{mean_MRR:.3f} MAP:{MAP:.3f}\n")
        
 ############### MAIN SERVICE FOR CALCULATING THE EVALUATION FOR THE SECOND DATASET   
def calc_evaluation2(qrels_new):
    AP = []
    MRR = []
    
    for query in queries2:
        
        relevance1 = getRelevance1(query.query_id, qrels_new)
        retrieved = getRetrievedDocs(retrieved_docs2[query.query_id])
        #recall
        r = recall_values(relevance1, retrieved)
    #     #precission @ 10
        p = precission_at_10(relevance1, retrieved)
        with open('evaluation2.txt', 'a') as f:
            f.write(f"{query.query_id}: precision@k:{p:.3f} recall:{r:.3f}\n")
        
        ap = mean_avg_precision(relevance1, retrieved)
        AP.append(ap)
        
        rr = mean_reciprocal_rank(relevance1, retrieved)
        MRR.append(rr)
    #MRR
    mean_MRR = sum(MRR) / len(MRR)
    #MAP
    MAP = sum(AP) / len(AP)
    with open('evaluation2.txt', 'a') as f:
        f.write(f"{query.query_id}: MRR:{mean_MRR:.3f} MAP:{MAP:.3f}\n")
        
##################### EXECUTION PART FOR EVALUATION ######################
get_queries()
get_qrels()
get_retrieved_documents(1)
relevantDocs = get_QRELS(1)
qrels_new = get_relevance_non_relevance_docs(relevantDocs, 1)

get_queries2()
get_qrels2()
get_retrieved_documents(2)
relevantDocs = get_QRELS(2)
qrels_new = get_relevance_non_relevance_docs(relevantDocs, 2)

calc_evaluation2(qrels_new)
calc_evaluation(qrels_new)
