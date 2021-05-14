import argparse
import pickle
import numpy as np
import pandas as pd
import ast
from collections import Counter
from sklearn.metrics import mean_squared_log_error
import math
from zipfile import ZipFile


# # Parsing script arguments
# parser = argparse.ArgumentParser(description='Process input')
# parser.add_argument('tsv_path', type=str, help='tsv file path')
# args = parser.parse_args()
#
# # Reading input TSV
# data = pd.read_csv(args.tsv_path, sep="\t")

data = pd.read_csv("test.tsv", sep="\t")

#Declerations
input_target_class='logrevenue'

selected_cols_filename='f_selected_features.sav'
et_model_filename='f_tuned_et_best.sav'
rf_model_filename='f_tuned_rf_best.sav'

# specifying the zip file name
file_name = "tuned_rf.zip"

categorical_cols  =[]

original_lang_cols = [col for col in data.columns if col.startswith('original_languages_')]
genres_cols = [col for col in data.columns if col.startswith('genre_')]
prod_companies_cols = [col for col in data.columns if col.startswith('prod_companies_')]
has_cols = [col for col in data.columns if col.startswith('has_')]

numeric_cols = ['budget', 'logbudget', 'num_of_cast',
                  'num_of_crew','num_of_genres','num_of_langs','num_prod_companies','num_prod_countries','popularity',
                  'release_month','release_weekday','release_year','runtime','vote_average',
                  'vote_count'] + original_lang_cols + genres_cols + prod_companies_cols + has_cols + ['video']

wanted_cols = categorical_cols + numeric_cols + [input_target_class]
dict_columns = ['belongs_to_collection','genres','spoken_languages','production_companies',
                'production_countries','Keywords','cast','crew']
cols_to_drop = ['backdrop_path', 'belongs_to_collection', 'genres', 'original_language', 'homepage', 'imdb_id', 'original_title', 'overview', 'poster_path', 'production_companies', 'production_countries', 'release_date', 'spoken_languages', 'status', 'tagline', 'title', 'Keywords', 'cast', 'crew']

#Aux functions

def text_to_dict(df):
    for columns in dict_columns:
        df[columns] = df[columns].apply(lambda x: {} if pd.isna(x) else ast.literal_eval(x))
    return df

def times_transformations(df):
  #change data types- to be categorical
  df['release_date'] =  pd.to_datetime(df['release_date'])

  #add new datetime features - weekday, month, year
  df['release_weekday'] = df['release_date'].apply(lambda t: t.weekday())
  df['release_month'] = df['release_date'].apply(lambda t: t.month)
  df['release_year'] = df['release_date'].apply(lambda t: t.year)

  return df

def create_more_features(df):
  # has poster_path
  df['has_poster_path'] = 1
  df.loc[pd.isnull(df['poster_path']) ,"has_poster_path"] = 0

  # has collection
  df['has_collection'] = df['belongs_to_collection'].apply(lambda x: 1 if x!={} else 0)

  # num of genres
  df['num_of_genres'] = df['genres'].apply(lambda x: len(x) if x!={} else 0)

  # one hot encoding of genres
  list_of_genres = list(df['genres'].apply(lambda x: [i['name'] for i in x] if x!={} else []).values)
  top_genres =[m[0] for m in Counter([i for j in list_of_genres for i in j]).most_common(15)]
  df['all_genres'] = df['genres'].apply(lambda x: ' '.join(sorted([i['name'] for i in x ])) if x!= {} else '')
  for g in top_genres:
      df['genre_' + g] = df['all_genres'].apply(lambda x: 1 if g in x else 0)
  df = df.drop(columns=['all_genres'])

  # has home_page
  df['has_homepage'] = 1
  df.loc[pd.isnull(df['homepage']) ,"has_homepage"] = 0

  # num of production_companies
  df['num_prod_companies'] = df['production_companies'].apply(lambda x: len(x) if x!= {} else 0)

  # one hot encoding of production_companies
  List_of_companies = list(df['production_companies'].apply(lambda x: [i['name'] for i in x] if x!= {} else []))
  top_prod_companies = [m[0] for m in Counter(i for j in List_of_companies for i in j).most_common(30)]
  df['all_prod_companies'] = df['production_companies'].apply(lambda x: ' '.join(sorted([i['name'] for i in x ])) if x!= {} else '')
  for t in top_prod_companies:
      df['prod_companies_' + t] = df['all_prod_companies'].apply(lambda x: 1 if t in x else 0)
  df = df.drop(columns=['all_prod_companies'])

  # one hot encoding of original_language
  List_of_languages = df['original_language'].unique()
  top_languages = list(df['original_language'].value_counts().head(30).index)
  for t in top_languages:
    df['original_languages_' + t] = df['original_language'].apply(lambda x: 1 if t==x else 0)

  # num of production countries
  df['num_prod_countries'] = df['production_countries'].apply(lambda x: len(x) if x!= {} else 0)

  # num of spoken languages
  df['num_of_langs'] = df['spoken_languages'].apply(lambda x: len(x) if x!= {} else 0)

  # num of cast members
  df['num_of_cast'] = df['cast'].apply(lambda x: len(x) if x!={} else 0)

  # num of crew members
  df['num_of_crew'] = df['crew'].apply(lambda x: len(x) if x!= {} else 0)

  return df

def impute_using_annual_mean(df, col_to_impute):
    all_mean = df[col_to_impute].mean()

    df.loc[pd.isnull(df[col_to_impute]), col_to_impute] = 0

    run = df[(df[col_to_impute].notnull()) & (df[col_to_impute] != 0)]
    year_mean_df = run.groupby(['release_year'])[col_to_impute].agg('mean').to_frame()
    year_mean_df = year_mean_df.reset_index().rename(columns={col_to_impute: 'mean_val'})

    df = pd.merge(df, year_mean_df, on="release_year", how='left')
    df = df.fillna(value={"mean_val": all_mean})
    df.loc[df[col_to_impute] == 0, [col_to_impute]] = df['mean_val']

    df = df.drop(columns=['mean_val'])

    return df

# apply the min-max scaling in Pandas using the .min() and .max() methods
def min_max_scaling(df):
    # copy the dataframe
    df_norm = df.copy()
    # apply min-max scaling
    for column in df_norm.columns:
        if column not in ['revenue', 'logrevenue','id']:
            if (df_norm[column].max() - df_norm[column].min())!=0:
                df_norm[column] = (df_norm[column] - df_norm[column].min()) / (df_norm[column].max() - df_norm[column].min())
            else:
                df_norm[column]=0.0
    return df_norm



# Manipulations on data-

# convert nested columns
data = text_to_dict(data)

# create log revenue column
data['logrevenue'] = np.log1p(data['revenue'])

# create time transformations and times new features
data = times_transformations(data)

# add boolean features
data = create_more_features(data)

# impute uncertain data
data = impute_using_annual_mean(data, 'runtime')
data = impute_using_annual_mean(data, 'budget')

data['logbudget'] = np.log1p(data['budget'])

# impute cols with "num_" with -1 values
cols_with_num = [col for col in data.columns if col.startswith('num_')]
for col in cols_with_num:
  data.loc[data[col]==0 ,col] = -1

# feature encoding
data['video']=(pd.Categorical(data.video).codes).astype('int')

# drop not used columns
data = data.drop(columns=cols_to_drop)

# call the min_max_scaling function
data = min_max_scaling(data)

# complete missing data - if any
data = data.fillna(0)


#Load trained models
selected_cols = pickle.load(open(selected_cols_filename, 'rb'))

for col in selected_cols:
  if col not in data.columns:
    data[col] = 0.0

#####

x_data = data[selected_cols] # get_config('X')
y_data = data[input_target_class] # get_config('y')

with ZipFile(file_name, 'r') as zip_ref:
    zip_ref.extractall()
model_1 = pickle.load(open(et_model_filename, 'rb'))
model_2 = pickle.load(open(rf_model_filename, 'rb'))

# prediction code
pred1 = np.exp(model_1.predict(x_data))
pred2 = np.exp(model_2.predict(x_data))
avg_pred = np.mean(np.array([pred1, pred2]), axis=0)


prediction_df = pd.DataFrame(columns=['id', 'revenue'])
prediction_df['id'] = data['id']
prediction_df['revenue'] = avg_pred #data['revenue'].mean()

####
# export prediction results
prediction_df.to_csv("prediction.csv", index=False, header=False)
