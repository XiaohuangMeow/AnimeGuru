import re
import os
import pylab as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer, MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
import time
from sklearn.feature_extraction.text import TfidfVectorizer

def anime_sy_filter(data, myrequest):
    anime_data_selected = data.copy()
    if myrequest is not None:
        if myrequest == "before2015":
            anime_data_selected = anime_data_selected[anime_data_selected["StartYear"] <= 2015]
        else:
            try:
                myrequest = int(myrequest)
                anime_data_selected = anime_data_selected[anime_data_selected["StartYear"] == myrequest]
            except:
                return anime_data_selected
    return anime_data_selected


def anime_tag_filter(data, myrequest):
    anime_data_selected = data.copy()
    if myrequest is not None:
        if myrequest == '':
            return anime_data_selected
        pattern =  re.compile(r'{}'.format(myrequest), re.I) # 一行超人
        anime_data_selected["Tags1"] = anime_data_selected['Tags'].apply(
            lambda s: ''.join(set(re.findall(pattern, s))) if re.findall(pattern, s) else '')
        anime_data_selected = anime_data_selected[(
            anime_data_selected["Tags1"].notnull()) & (anime_data_selected["Tags1"] != "")]
        del anime_data_selected["Tags1"]
    return anime_data_selected


def anime_finished_filter(data, myrequest):
    anime_data_selected = data.copy()
    if myrequest is not None:
        if myrequest == '':
            return anime_data_selected

        if myrequest == "True":
            val = 1
        else:
            val = 0
        anime_data_selected = anime_data_selected[anime_data_selected["Finished"] == val]
    return anime_data_selected


def anime_warning_filter(data, myrequest):
    anime_data_selected = data.copy()
    if myrequest is not None:
        if myrequest == '':
            return anime_data_selected
        pattern = re.compile(r'{}'.format(myrequest), re.I)  # 一行超人
        anime_data_selected["Content Warning1"] = anime_data_selected['Content Warning'].apply(
            lambda s: ''.join(set(re.findall(pattern, s))) if re.findall(pattern, s) else '')
        anime_data_selected = anime_data_selected[anime_data_selected["Content Warning1"].isnull(
        ) | (anime_data_selected["Content Warning1"] == "")]
        del anime_data_selected["Content Warning1"]
    return anime_data_selected


def anime_Episodes_filter(data, myrequest):
    anime_data_selected = data.copy()
    if myrequest is not None:
        if myrequest == '':
            return anime_data_selected
        if myrequest == "Less than 100":
            anime_data_selected = anime_data_selected[anime_data_selected["Episodes"] <= 100]
        elif myrequest == "101-1000":
            anime_data_selected = anime_data_selected[(
                anime_data_selected["Episodes"] > 100) & (anime_data_selected["Episodes"] <= 1000)]
        else:
            anime_data_selected = anime_data_selected[anime_data_selected["Episodes"] > 1000]
    return anime_data_selected


def anime_filter(data, request):

    #Weighted Rank-WR
    #IMDB
    data["Rating Score_B"] = ((data["Number Votes"]/(data["Number Votes"]+816.))*data["Rating Score"]
                              + (816./(data["Number Votes"]+816.))*2.94)
    #filter
    data_selected = data.copy()
    data_selected = anime_sy_filter(data_selected, request['StartYear'])
    data_selected = anime_tag_filter(data_selected, request['Tags'])
    data_selected = anime_finished_filter(data_selected, request['Finished'])
    data_selected = anime_warning_filter(
        data_selected, request['Content Warning'])
    data_selected = anime_Episodes_filter(data_selected, request['Episodes'])

    #sort
    data_selected["Ranking"] = data_selected['Rating Score_B'].rank(
        ascending=False)
    data_selected["Hotness"] = data_selected['Number Votes'].rank(
        ascending=False)
    if request['sort'] is not None:
        data_selected = data_selected.sort_values(
            request['sort'], ascending=True)
    else:
        data_selected = data_selected.sort_values(
            'Ranking', ascending=True)

    del data_selected["Ranking"]
    del data_selected["Hotness"]

    datalist = data_selected.values.tolist()

    return datalist, data_selected


def recommender(anime_data, preference):
    pd.set_option("max_colwidth", None)
    pd.set_option('mode.chained_assignment', None)
    feature = ['Tags', 'Type', 'Episodes', 'Duration', 'Finished', 'StartYear']
    anime_metadata = anime_data[feature]

    def process_multilabel(series):
        # str -> list
        series = series.split(", ")
        if "Unknown" in series:
            series.remove("Unknown")
        return series

    def preprocessing_category(df, column, is_multilabel=False):
        # categorical feature preprocessing
        lb = LabelBinarizer()
        if is_multilabel:
            lb = MultiLabelBinarizer()
        lb.fit(df[column])
        expandedLabelData = lb.transform(df[column])
        labelClasses = lb.classes_
        category_df = pd.DataFrame(expandedLabelData, columns=labelClasses)
        del df[column]
        return pd.concat([df, category_df], axis=1), lb

    def get_nearest(df, preference, n_neighbors):

        model_knn = NearestNeighbors(metric='cosine', n_neighbors=n_neighbors)
        model_knn.fit(csr_matrix(df.astype(np.float64)))
        distances, indices = model_knn.kneighbors(
            preference, n_neighbors=n_neighbors)
        index = indices.flatten()[0]
        return index

    def get_recommended(vector, query_index, n_neighbors=10):
        model_knn = NearestNeighbors(metric='cosine', n_neighbors=n_neighbors)
        model_knn.fit(vector)

        distances, indices = model_knn.kneighbors(
            vector[query_index, :].reshape(1, -1), n_neighbors=n_neighbors)
        result = []
        for i in range(0, len(distances.flatten())):
            index = indices.flatten()[i]
            result.append(anime_data.iloc[index])

        return pd.DataFrame(result)

    def process_preference_category(preference, column, lb):

        expandedLabelData = lb.transform(preference[column])
        labelClasses = lb.classes_
        category_preference = pd.DataFrame(
            expandedLabelData, columns=labelClasses)
        del preference[column]
        return pd.concat([preference, category_preference], axis=1)

    def get_recommended(vector, query_index, n_neighbors=10):
        model_knn = NearestNeighbors(metric='cosine', n_neighbors=n_neighbors)
        model_knn.fit(vector)

        distances, indices = model_knn.kneighbors(
            vector[query_index, :].reshape(1, -1), n_neighbors=n_neighbors)
        result = []
        index_result = []
        for i in range(0, len(distances.flatten())):
            index = indices.flatten()[i]
            result.append(anime_data.iloc[index])
            index_result.append(index)
        return pd.DataFrame(result), index_result
    # Animedata preprocessing
    anime_metadata, lb_type = preprocessing_category(anime_metadata, "Type")

    anime_metadata["Tags"] = anime_metadata["Tags"].map(process_multilabel)
    anime_metadata, lb_tags = preprocessing_category(
        anime_metadata, "Tags", True)

    # --------------------add Episodes and Duration------------------------------------
    numeric_columns = ["StartYear", "Episodes", "Duration"]
    scaler = MinMaxScaler().fit(anime_metadata[numeric_columns])
    anime_metadata[numeric_columns] = scaler.transform(
        anime_metadata[numeric_columns])

    #Preference preprocessing
    preference["Tags"] = preference["Tags"].map(process_multilabel)
    preference = process_preference_category(preference, "Type", lb_type)
    preference = process_preference_category(preference, "Tags", lb_tags)
    preference[numeric_columns] = scaler.transform(preference[numeric_columns])

    index_result = get_nearest(anime_metadata.values, preference.values, 1)

    tfv = TfidfVectorizer(min_df=3,  max_features=None,
                          strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}',
                          ngram_range=(1, 3),
                          stop_words='english')

    tfv.fit(anime_data['Synopsis'])
    df_tf_idf = tfv.transform(anime_data['Synopsis'])

    anime_df = np.concatenate((anime_metadata.values, df_tf_idf.A), 1)

    return get_recommended(anime_df, index_result, n_neighbors=10)
