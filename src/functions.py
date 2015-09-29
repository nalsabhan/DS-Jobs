import pymongo
import random
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import NMF
import pandas as pd 
from sklearn.feature_selection import SelectKBest, chi2

''' Returns the collection pointer of the given mongo DB and collection names'''
def init_db(db_name, db_col):
    
    client = pymongo.MongoClient()
    
    db = client[db_name]
    
    coll = db[db_col]

    return coll

'''extract the city name from the salary brief description'''
def get_city(string_arg):

    city = string_arg.split('(')[0].split(',')[0]
    return city.strip()

'''extract the state name from the salary brief description'''
def get_state(string_arg):

    state = string_arg.split('(')[0].split(',')[1]
    return state.strip()

'''extract the date from the salary brief description'''
def get_date(string_arg):
    return string_arg[string_arg.index("(") + 1:string_arg.rindex(")")]

'''to split the data such that the same job title isn't in both the training and test '''
def fair_split(df, p):
    
    
    df_grouped = df.groupby(['title', 'sal'])['desc'].apply(lambda x: ','.join(x)).reset_index()
    titles = df_grouped.title.unique()
    titles_num = titles.size
    test_p = p * titles_num

    rand_mask = random.sample(xrange(titles_num), int(test_p))
    
    test_titles = titles[rand_mask]
    
    X_test = df[df.title.isin(test_titles)]
    y_test = X_test.pop('sal')
    
    X_train = df[~df.title.isin(test_titles)]
    y_train = X_train.pop('sal')
    
    X_test.index = X_test.title.values
    X_train.index = X_train.title.values
    X_test = X_test.drop('title', 1)
    X_train = X_train.drop('title', 1)
    
    return X_train, X_test, y_train, y_test

'''To split the data and vectorize the text in 'desc' to a tfidf BOW matrix''' 
def split_and_vectorize(df, grams, max_feat):
    docs_train, docs_test, y_train, y_test = train_test_split(df['desc'], df['sal'], test_size = 0.1)

    
    #docs_train, docs_test, y_train, y_test = fair_split(df, 0.2)
    
    
    vect_model = TfidfVectorizer(stop_words='english', strip_accents = 'unicode',max_features=max_feat,  ngram_range = grams)
    
    X_train = vect_model.fit_transform(docs_train)
    X_test = vect_model.transform(docs_test)


    
    feature_words = vect_model.get_feature_names()
    return X_train, X_test, y_train, y_test, vect_model, feature_words

''' Returns the H and W from the NMF factorization of the training data, 
also returns W_test which corresponds to the NMF transformation applied to the testset
'''
def NMF_train(X_train, X_test, n):
    nmf_model = NMF(n_components=n)
    nmf_model.fit(X_train)
    
    H = nmf_model.components_;
    W = nmf_model.fit_transform(X_train)
    W_test = nmf_model.transform(X_test)
    
    return H, W, W_test

''' plot return the normalized matrix norm for (X_train - W*H)'''
def Kplot(k, n, step):
    mat_norm_train = []
    mat_norm_test = []
    for i in xrange(k, n, step):
        print i
        H, W, W_test = NMF_train(X_train, X_test, i)
        M_train = X_train - W.dot(H)
        mat_norm_train.append(np.linalg.norm(M_train) / np.sqrt(M_train.shape[0]*M_train.shape[1]) )
        M_test = X_test - W_test.dot(H)
        mat_norm_test.append(np.linalg.norm(M_test)/np.sqrt(M_test.shape[0]*M_test.shape[1]))
    return mat_norm_train, mat_norm_test

''' prints n_top_words corresponds to the topics in W or H'''
def describe_nmf_results(W, H, n_top_words = 15):
    for topic_num, topic in enumerate(H):
        print("Topic %d:" % topic_num)
        print(" ".join([feature_words[i] for i in topic.argsort()[:-n_top_words - 1:-1]]))

''' Applies randomforest to the dataset and returns the MSE for the test and training data'''
def randforest(W, W_test, y_train, depth, n_estm, sample_size):
    
    clf_1 = RandomForestRegressor(n_estimators = n_estm, max_depth=depth, min_samples_split = sample_size)

    clf_1.fit(W.toarray(), y_train)

    y_pred1 = clf_1.predict(W_test.toarray())
    y_train_pred = clf_1.predict(W.toarray())

    mse_test = mean_squared_error(y_test, y_pred1)
    mse_train = mean_squared_error(y_train, y_train_pred)
    
    return mse_test , mse_train

''' splits the data into test and train, vectorizes the text to tfidf BOW and
 uses chi2 to select features using quantile bins of the salary (slabel) '''
def split_vectorize_featSelection(df, grams, n_feat = 1000):
    
    df.index = df.title.values
    df = df.drop('title', 1)
    
    docs_train, docs_test, y_train, y_test = train_test_split(df[['desc', 'slabel']], df['sal'], test_size = 0.2)

    docs_train, docs_test = pd.DataFrame(docs_train), pd.DataFrame(docs_test)
    
    #docs_train, docs_test, y_train, y_test = fair_split(df, 0.2)
    
    #print pd.DataFrame(docs_test)[0]
    vect_model = TfidfVectorizer(stop_words='english', strip_accents = 'unicode',  ngram_range = grams)
    
    X_train = vect_model.fit_transform(docs_train[0])
    X_test = vect_model.transform(docs_test[0])
    
#     X_train = vect_model.fit_transform(docs_train.desc)
#     X_test = vect_model.transform(docs_test.desc)
    
    feature_words = vect_model.get_feature_names()
    
    ch2 = SelectKBest(chi2, k= n_feat)
    X_train_new_a = ch2.fit_transform(X_train, docs_train[1])
    X_test_new_a = ch2.transform(X_test)
    
    X_train_new_b = None
    X_test_new_b = None

    # lass = RandomizedLasso(alpha = 0.3)
    # X_train_new_b = lass.fit_transform(X_train.toarray(), y_train)
    # X_test_new_b = lass.transform(X_test.toarray())
    

    return X_train_new_a, X_train_new_b, X_test_new_a, X_test_new_b, y_train, y_test, vect_model, feature_words