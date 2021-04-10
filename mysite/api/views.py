from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from django.core.files.storage import FileSystemStorage
from django.views.decorators.csrf import csrf_exempt
from django.contrib import messages
import os
import pandas as pd
import re
import gensim 
from gensim.parsing.preprocessing import remove_stopwords
import sklearn
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec 
import gensim.downloader as api
import numpy
from zipfile import ZipFile
import csv
import glob
from gensim import corpora
from scipy import spatial

# Create your views here.


def clean_sentence(sentence, stopwords=False): 
    sentence = sentence.lower().strip()
    sentence = re.sub(r'[^a-z0-9\s]', '', sentence)    
    if stopwords:
         sentence = remove_stopwords(sentence)
    

    
    return sentence
                    
def get_cleaned_sentences(df,stopwords=False):    
    sents=df[["questions"]];
    cleaned_sentences=[]

    for index,row in df.iterrows():
        cleaned=clean_sentence(row["questions"],stopwords);
        cleaned_sentences.append(cleaned);
    return cleaned_sentences;


def retrieveAndPrintFAQAnswer(question_embedding,sentence_embeddings,FAQdf,sentences):
    max_sim=-1;
    index_sim=-1;
    for index,faq_embedding in enumerate(sentence_embeddings):
        #sim=cosine_similarity(embedding.reshape(1, -1),question_embedding.reshape(1, -1))[0][0];
        sim=cosine_similarity(faq_embedding,question_embedding)[0][0];
        #print(index, sim, sentences[index])
        if sim>max_sim:
            max_sim=sim;
            index_sim=index;
       
    #print("\n")
    #print("Question: ",question)
    #print("\n");
    #print("Retrieved: ",FAQdf.iloc[index_sim,0]) 
    #print(FAQdf.iloc[index_sim,1])
    return(FAQdf.iloc[index_sim,1])


def getWordVec(word,model):
        samp=model['model'];
        vec=[0]*len(samp);
        try:
                vec=model[word];
        except:
                vec=[0]*len(samp);
        return (vec)


def getPhraseEmbedding(phrase,embeddingmodel):       
        samp=getWordVec('model', embeddingmodel);
        vec=numpy.array([0]*len(samp));
        den=0;
        for word in phrase.split():
            #print(word)
            den=den+1;
            vec=vec+numpy.array(getWordVec(word,embeddingmodel));
        
        return vec.reshape(1, -1)


@csrf_exempt
def index(request):
    if request.method == 'POST':
        doc = request.FILES
        fileObj = doc['file']

        if not fileObj.name.endswith('.csv'):
            messages.error(request, 'this is not a csv file')

        else:
            fs = FileSystemStorage()
            filePathName = fs.save(fileObj.name, fileObj)
            filePathName = fs.url(filePathName)
            filePath = '.'+filePathName      
            return JsonResponse({'result': 'uploaded'})
            
            
def create_profile(file, v2w_model):
    csv = pd.read_csv(file, encoding= 'unicode_escape')
    my_dict = {};
    #final_dict = {};
    list_data = [];
    print(csv)
    i=0
    for ind in csv.index: 
        #print(csv['question'][ind]) 
        i=i+1
        key = "Q"+str(i)
        key_feedback = "Q"+str(i)+"_feedback"
        temp, feedback = model(csv['question'][ind], csv['answer'][ind], v2w_model)
        my_dict[key] = temp
        my_dict[key_feedback] = feedback
        list_data.append(temp)       
        
    #print(my_dict)
    return my_dict, list_data



def avg_feature_vector(sentence, model, num_features, index2word_set):
    words = sentence.split()
    feature_vec = numpy.zeros((num_features, ), dtype='float32')
    n_words = 0
    for word in words:
        if word in index2word_set:
            n_words += 1
            feature_vec = numpy.add(feature_vec, model[word])
    if (n_words > 0):
        feature_vec = numpy.divide(feature_vec, n_words)
    return feature_vec


def model(temp, temp2, v2w_model):
    temp = str(temp)
    temp2 = str(temp2)
    
    df = pd.read_csv("./Book1.csv", encoding= 'unicode_escape')
    df_json = df.to_dict()

    cleaned_sentences=get_cleaned_sentences(df,stopwords=True)
    cleaned_sentences_with_stopwords=get_cleaned_sentences(df,stopwords=False)
    sentences=cleaned_sentences_with_stopwords
    # Split it by white space 
    sentence_words = [[word for word in document.split() ]
            for document in sentences]


    dictionary = corpora.Dictionary(sentence_words)

    question_orig=temp
    question=clean_sentence(question_orig,stopwords=False);
    question_embedding = dictionary.doc2bow(question.split())
    cleaned_sentences=get_cleaned_sentences(df,stopwords=True)
    
    
    sent_embeddings=[];

    for sent in cleaned_sentences:
        sent_embeddings.append(getPhraseEmbedding(sent,v2w_model));

    question_embedding=getPhraseEmbedding(question,v2w_model);

    x = retrieveAndPrintFAQAnswer(question_embedding,sent_embeddings,df, cleaned_sentences);
    index2word_set = set(v2w_model.wv.index2word)
    s1_afv = avg_feature_vector(x, model=v2w_model, num_features=300, index2word_set=index2word_set)
    s2_afv = avg_feature_vector(temp2, model=v2w_model, num_features=300, index2word_set=index2word_set)
    sim = 1 - spatial.distance.cosine(s1_afv, s2_afv)
    feedback = ''
    if sim>=0.65:
        feedback = "Good, Keep it up!"
    elif sim>0.35 and sim<0.65:
        feedback = "Satisfactory, but need to work more"
    else:
        feedback = "Bad, work hard!"
    return sim, feedback
    
def calcData(list_data):
    correct_feedback = []
    improvement_feedback = []
    wrong_feedback = []
    for i in range(len(list_data)):
        if list_data[i]>0.65:
            list_data[i] = 1.0
            correct_feedback.append(i+1)
        elif list_data[i] > 0.35 and list_data[i] < 0.65:
            list_data[i] = 0.5
            improvement_feedback.append(i+1)
        else:
            list_data[i] = 0.0
            wrong_feedback.append(i+1)

    print("printing lists")
    print(correct_feedback)
    print(improvement_feedback)
    print(wrong_feedback)
    correct_ans = sum(list_data)
    total_qs = len(list_data)
    percentage = (correct_ans/total_qs) * 100

    if percentage > 90:
        grade = 'A'
    elif percentage > 80 and percentage < 90:
        grade = 'B'
    elif percentage > 70 and percentage < 80:
        grade = 'C'
    elif percentage > 60 and percentage < 70:
        grade = 'D'
    else:
        grade = 'F'
    return correct_ans, total_qs, percentage, grade, correct_feedback, improvement_feedback, wrong_feedback

@csrf_exempt
def score(request):
    if request.method=='POST':
        doc = request.FILES
        fileObj = doc['answers']
        fs = FileSystemStorage()
        filePathName = fs.save(fileObj.name, fileObj)
        filePathName = fs.url(filePathName)
        filePath = '.'+filePathName
        file = filePath
        temp, temp2 = os.path.splitext(fileObj.name)
        with ZipFile(file, 'r') as zip:
            x = zip.namelist()
            zip.extractall('./media/')
        x = x[0:]
        mypath='./media/'+temp
        final_db=pd.DataFrame()
        print(mypath)
        
        #Path for the files
        onlyfiles = [os.path.join(mypath, f) for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f))]
        i=0

        print(onlyfiles)
        v2w_model=None;
        try:
            v2w_model = gensim.models.KeyedVectors.load("./w2vecmodel.mod")
            print("Loaded w2v model")
        except:            
            v2w_model = api.load('word2vec-google-news-300')
            v2w_model.save("./w2vecmodel.mod")

        my_dict = {}
        while i < len(onlyfiles):
            file=onlyfiles[i]
            final_dict, list_data=create_profile(file, v2w_model)
            #print(final_dict)
            #final_db=final_db.append(dat)
            i+=1
            key_student = file.split(os.sep)
            key = key_student[-1]
            final_key = key.split('.')
            key = str(final_key[0])

            correct_ans, total_qs, percentage, grade, correct_feedback, improvement_feedback, wrong_feedback = calcData(list_data)
            my_dict[key] = final_dict
            print(my_dict)

            if len(improvement_feedback)==0 and len(wrong_feedback) == 0:
                improve = 'No need, you are doing just fine, keep it up!'
            elif len(improvement_feedback)!=0 and len(wrong_feedback) != 0:
                improve = key+" needs improvement in question no "+ ','.join(map(str, improvement_feedback)) +" but needs to work hard on question no "+ ','.join(map(str, wrong_feedback))
            elif len(improvement_feedback)==0 or len(wrong_feedback)==0:
                if len(improvement_feedback)==0:
                    improve = key+" needs to work on question no "+ ','.join(map(str, wrong_feedback))
                else:
                    improve = key+" needs improvement in question no "+ ','.join(map(str, improvement_feedback))
            
            
            my_dict[key].update({"Correct": correct_ans, "Total": total_qs, "Percentage": percentage, "Grade": grade, "Improvement_Feedback": improve})


               
            
            #print(final_db)
        #temp = model()
        
        return JsonResponse(my_dict)
        #print('im here!')
        






    
