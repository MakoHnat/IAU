import pandas as pd
import numpy as np
import seaborn as sns
import re
import matplotlib.pyplot as plt
import json
import scipy.stats as stats
import math

import category_encoders as ce

import statsmodels.api as sm
import statsmodels.stats as sm_stats
import statsmodels.stats.api as sms

import vizualizacia_funkcie as visual

from sklearn.experimental import enable_iterative_imputer 
from sklearn import impute 
from sklearn import preprocessing
from sklearn import pipeline
from sklearn import base
from sklearn import compose
from sklearn import feature_selection

from datetime import datetime
from datetime import date

import imblearn


#funckia, ktora mergne zaznamy, ktore su rovnake
def piece_datarows_together(data):
    
    data = data.copy().set_index("name")
    
    #toto nam vrati dataset, ktory obsahuje vsetky duplikaty, s ktorymi budeme pracovat
    #proste to vrati data, ktore maju index, ktory je v datasete viac ako raz pouzity
    duplicated = data[data.index.duplicated(keep=False)]
    
    index_values = duplicated.index.unique()
    
    #najprv vsetky hodnoty prenesieme do prveho vyskytu zaznamu daneho pacienta v datasete
    for idx in index_values:
        mini_dataset = duplicated.loc[idx] #toto vrati viacero zaznamov s rovnakych idx
        
        #zistim si, ktore atributy su nullove pre presne prvy zaznam a pre konkretne nullove atributy budem nadalej hladat
        #nenullovu hodnotu v ostatnych zaznamoch s rovnakym idx
        missing_mask = mini_dataset.iloc[0].isnull()
        attributes = mini_dataset.columns.values
        missing_attributes = attributes[missing_mask]
        
        #tu replacujem null hodnoty za nenullove
        for attr in missing_attributes:
            not_null = mini_dataset[attr][mini_dataset[attr].notnull()]
            
            if len(not_null) != 0:
                mini_dataset.iloc[0][attr] = not_null.values[0]
        
        
    #teraz uz mozme vymazat vsetky druhe, resp. ostatne zaznamy pacienta
    duplicated_mask = data.index.duplicated(keep="first")
    
    data = data.reset_index()
    duplicated_indices = data.index.values[duplicated_mask]
    
    
    return data.drop(index=duplicated_indices).reset_index(drop=True)


#funkcia, ktora joine obi dva dataframy, s ktorymi pracujeme + mergne riadky, kde su splittnute data
def one_proper_df(df1, df2, return_X_y=True):
    data = df1.drop(columns=["address"]).set_index("name").join(df2.set_index("name"), how="right").reset_index()
    data = piece_datarows_together(data)
    
    if return_X_y == True:
        X = data.drop(columns=["class"])
        y = data["class"]
        return X,y
    
    else:
        return data 
    


# Tu nizsie mame funkcie, ktore pouzivame na zmensenie poctu hodnot kategorickych atributov. Vyber atributov, ktore sa merguju, sme vybrali este pocas fazy analyzy, kedy malo pocetne hodnoty su mergnute do jednej hodnoty, aby hodnoty daneho atributu boli viac vyrovnane.

# # Prvotne preprocessing kroky - cez FunctionTransformer

def marital_status_categories(row):
    
    ms = row["marital-status"]
        
    if ms is not np.nan and ms not in ("Divorced", "Never-married", "Married-civ-spouse"):
        row["marital-status"] = "Other"
        
    return row

def relationship_categories(row):
    
    rel = row["relationship"]
        
    if rel is not np.nan and rel not in ("Not-in-family", "Husband", "Own-child"):
        row["relationship"] = "Other"
        
    return row

def occupation_categories(row):

    occ = row["occupation"]
    
    if occ is not np.nan and occ not in ("Craft-repair", "Prof-specialty", "Exec-managerial", 
                                         "Adm-clerical", "Sales", "Other-service", "Machine-op-inspct", 
                                         "Transport-moving"):
        
        row["occupation"] = "Other"
        
    return row

def workclass_categories(row):

    wc = row["workclass"]
    
    if wc is not np.nan and wc != "Private":
        row["workclass"] = "Non-private"
        
    return row

#oproti ostatnym funkciam v tejto bunke, tato funkcia sluzi na transformaciu spojiteho atributu hours-per-week na kategoricky
def categorize_hours(row):
    
    hour = row["hours-per-week"]
    
    if math.isnan(hour):
        row["hours-per-week-cat"] = math.nan
    elif hour <= 35:
        row["hours-per-week-cat"] = "<=35"
    elif hour <= 45:
        row["hours-per-week-cat"] = "35< hours <=45"
    elif hour > 45:
        row["hours-per-week-cat"] = ">45"        

    return row

def simplify_education(row):
        
    edu = row["education"]
        
    if edu is np.nan:
        row["simple-edu"] = edu
        
    elif re.match("^([0-9][a-zA-Z])|(1[0-2][a-zA-Z])", edu) or edu == "Preschool":
        row["simple-edu"] = "Attending-school"
        
    elif edu in ["Assoc-acdm", "Assoc-voc", "Prof-school"]:
        row["simple-edu"] = "Edu after HS, no uni"
        
    elif edu in ["Masters", "Doctorate"]:
        row["simple-edu"] = "Masters/Doctorate"
        
    else:
        row["simple-edu"] = row["education"]
    
    return row


# Tu su nejake cary-mary, kedy s atributu date_of_birth, chceme ziskat rok narodenia, ktory nasledne mozeme pouzit na imputaciu missing values, ci zlych hodnot atributu age - totiz age ma v sebe zle namerane hodnoty, ktore su bud zaporne, alebo velmi velke (v tisickach), a tak dane zle hodnoty rovno nastavime na np.nan, pricom ich nasledne imputujeme pomocou roku narodenia, co, ako som uz napisal, ziskavame pomocou tejto funkcie.

def date_formatting(data):    
    
    data = data.copy()
    
    import re
    dates = []

    for index,row in data.iterrows():
        dates.append(re.sub('\d', '*',  row['date_of_birth']))

    dates = list(set(dates))
    dates

    from datetime import datetime

    for index,row in data.iterrows():
        line = row['date_of_birth']
        if re.match(r"^\d{2}-\d{2}-\d{2}$", line):
            regex1 = line[0:2]
            regex2 = line[3:5]
            regex3 = line[6:8]

            verbose = False
            if (verbose == True):
                if (int(regex1) > 31):
                    print('Prvy udaj > 31: ',regex1)
                if (int(regex2) > 31):
                    print('Druhy udaj > 31: ',regex2)
                if (int(regex3) > 31):
                    print('Treti udaj > 31: ',regex3)

    data['date_of_birth'] = data['date_of_birth'].map(lambda x: x[:10])
    
    for index,row in data.iterrows():
        line = row['date_of_birth']
        dateObj = None
        if re.match(r"^\d{2}-", line):
            newDate = '19' + line
            dateObj = datetime.strptime(newDate,'%Y-%m-%d')
        elif re.match(r"^\d{4}-", line):
            dateObj = datetime.strptime(line,'%Y-%m-%d')
        elif re.match(r"^\d{4}/", line):
            dateObj = datetime.strptime(line,'%Y/%m/%d')
        elif re.match(r"^\d{2}/", line):
            dateObj = datetime.strptime(line,'%d/%m/%Y')
        data.at[index,'date_of_birth'] = dateObj.strftime('%d-%m-%Y')
    
    return data


# Tu nizsie mame rozne funckie, ktore aplikujeme v prvotnej faze pipelinu, kedy pouzivame triedu preprocessing.FunctionTransformer, ktory dovoluje aplikovanie custom funkcie na nas dataset. Teda pocas tejto prvotnej fazy aplikujeme na dataset jednoduche operacie, ktore opravuju nejake atributy, ako napriklad odstranovanie white spacov, ci ziskanie novych atributov z atributu medical_info, alebo odstranovanie useless atributov, vid nizsie.

def remove_useless_features(X):
    
    X = X.copy()
    
    useless_cols = ["name", "race", "pregnant", "capital-loss", "capital-gain", "fnlwgt", "native-country", "address"]
    
    return X.drop(columns=useless_cols)

def add_oxygen_features(X):
    X = X.copy()
    X = X.apply(get_oxygen_stats, axis=1)
    return X.drop(columns=["medical_info"])

#ziskavam 4 atributy o kysliku z atributu medical_info
def get_oxygen_stats(row):
    
    string = row["medical_info"]
    
    if string is np.nan:
        return row
    
    string = string.replace("\'", "\"")
    di = json.loads(string)
    
    for k in di.keys():
        row[k] = float(di[k])
        
    return row

def string_wrap_formatting(X):
    X = X.copy()
    return X.apply(string_formatting, axis=0)

#vymazem white spacy a "?" vymenim za np.nan pre vsetky atributy typu "O" - object - string
def string_formatting(col):
    
    if col.dtype == "O":
        col = col.apply(lambda row: row.strip() if row is not np.nan else row)
        col = col.apply(lambda row: np.nan if row is not np.nan and row == "?" else row)
    
    return col

#tato funkcia je wrapper, ktory aplikuje funkcie na zmensenie poctu hodnot kategorickych atributov
def bucket_cat_attr(X):
   
    X = X.copy()
    
    X = X.apply(marital_status_categories, axis=1)
    X = X.apply(relationship_categories, axis=1)
    X = X.apply(occupation_categories, axis=1)
    X = X.apply(workclass_categories, axis=1)
    
    X["hours-per-week-cat"] = 0
    X = X.apply(categorize_hours, axis=1)
    X = X.drop(columns=["hours-per-week"])
    
    return X

#atribut mean_glucose, je potrebne pretypovat
def repair_mean_glucose(X):
    
    X = X.copy()
    X["mean_glucose"] = pd.to_numeric(X['mean_glucose'], errors= 'coerce')
    return X

#tu najprv extraktujeme rok narodenia z atributu date_of_birth, nasledne nullujeme zle hodnoty v atribute age, a rovno aj
#imputujeme hodnoty v atribute age pomocou extrahovanych rokou narodenia
def prepare_age(X):
    X = X.copy()
    X = date_formatting(X)
    
    X = X.apply(make_bs_age_nan, axis=1)
    X = X.apply(calculate_age, axis=1)
    
    X = X.drop(columns=["date_of_birth"])
    
    return X
    
#zle hodnoty agu -> np.nan
def make_bs_age_nan(row):
    
    age = row["age"]
    
    if age is np.nan:
        return row
    
    if age <= 0 or age >= 100:
        row["age"] = np.nan
        
    return row

#imputovanie hodnot agu pomocou roku narodenia
def calculate_age(row):
    
    if row["age"] is np.nan or math.isnan(row["age"]):
    
        born = row["date_of_birth"]

        born = datetime.strptime(born, "%d-%m-%Y").date()
        today = date.today()
        
        row["age"] = today.year - born.year - ((today.month, today.day) < (born.month, born.day))
 
        
    return row


#tato funkcia je ekvivalentna s hociktorym inym transformatorom v scikit-learne, ci uz na imputaciu, transformaciu, scaling a ine veci
#jediny rozdiel je, ze to nevrati numpy array, ale DataFrame, a vdaka tomu si uchvoavam nazvy stlpcov
#toto je klucove, pokial chcem pouzivat napriklad ColumnTransformer v dalsom kroku pipelinu, pokial chcem referovat jednotlive atributy
#dataframu na zaklade ich mena
class KeepDataFrame(base.BaseEstimator, base.TransformerMixin):
    
    def __init__(self, transformation):
        self.transformation = transformation
    
    def fit(self, X, y=None):
        
        if self.transformation is not None:
            self.transformation.fit(X)
        return self
    
    def transform(self, X):
        
        if self.transformation is not None:
        
            X = X.copy()
            cols = X.columns
            indices = X.index

            X = self.transformation.transform(X)

            X = pd.DataFrame(X, columns=cols, index=indices)
        
        return X


#Toto je custom transformator, ktory sluzi na imputaciu kategorickych atributov prostrednictvom bud
#knn imputera alebo iterative imputera
#Najprv sa kategoricky atribut pretypuje na ciselny pomocou ce.OrdinalEncoder, nasledne dojde k imputacii
#a potom sa znova inverznou funkciou vrati do kategorickeho atributu.
#v tejto faze, kedy to uz pouzivame v pipeline, to nie je az take klucove transformovat naspat do kat. atributu
#no vyuzili sme funkciu, ktoru sme predtym vytvorili, kedze sme nasledne imputovany atribut znova analyzovali,
#co bol aj dovod, preco sme pouzili nan inverznu transformaciu
class CustomCatImputing(base.BaseEstimator, base.TransformerMixin):
    
    def __init__(self, imputer_type="knn"):
        self.ordinal_encoder = None
        self.imputer = None
        self.imputer_type = imputer_type
        
    def fit(self, X, y=None):
        
        X = X.copy()
        
        columns = X.columns.values
        indices = X.index

        #toto sme uz riesili v preprocessing notebooku - chceme, aby nam null hodnoty neinkrementovali encoding hodnoty v strede datasetu,
        #ale aby sme mali urcity range celociselnych hodnot, bez dier, ktore sa pouzije v imputerovi
        #je to klucove aj pri KNN imputerovi, aj pri Iterative imputerovi, lebo pri iterative pracujeme so ciselnymi hodnotami,
        #ktore su kludne aj desatinne, a teda nakoniec sa vysledok imputera rounduje
        #a pri knn sice pracujeme s celocislenymi cislami, no nakoniec imputuje sa priemer ziskany z danych
        #n-susedov, co znova moze byt desatinne cislo
        #takze, aby sme nahodou pri roundovani sa nedostali na encoding hodnotu, ktora patri null hodnote, tak 
        #feedujeme danemu ordinal encodingu hned na zaciatku null hodnoty
        null_values = pd.DataFrame(index=pd.Index([-1]), columns=columns, data=[[np.nan for i in range(len(columns))]])
        X = pd.concat([null_values,X])

        self.ordinal_encoder = ce.ordinal.OrdinalEncoder(handle_missing="return_nan", handle_unknown="return_nan")
        X = self.ordinal_encoder.fit_transform(X)
        
        X = X[1:]
        
        if self.imputer_type == "knn":
            self.imputer = impute.KNNImputer()
            X = self.imputer.fit(X)
        
        elif self.imputer_type == "iterative":

            self.imputer = impute.IterativeImputer(max_iter=20, random_state=42, initial_strategy="most_frequent", 
                                                  min_value=X.min(), max_value=X.max())


            try:
                X = self.imputer.fit(X)
            except (ValueError, np.linalg.LinAlgError):
                print("Jeden error bol trapnuty, kedy funkcii vadili NaNs. Tento error je ale divny, lebo mu to vadi",                   "len prvy krat, a potom to uz ide...")
                X = self.imputer.fit(X)
            
        return self
               

    def transform(self, X):
  
        X = X.copy()
        
        indices = X.index
        columns = X.columns
    
        X = self.ordinal_encoder.transform(X)
        X = self.imputer.transform(X).round()
        
        X = pd.DataFrame(data=X, columns=columns, index=indices)
        
        X = self.ordinal_encoder.inverse_transform(X)
        
        return X
    


#ColumnTransformer je sice fajn, ze dokaze konkretne transformacie aplikovat na nami vybrane atributy, no vysledkom daneho 
#ColumnTransformer triedy je numpy array, nie dataframe, co je zle, pokial chceme napriklad viackrat pouzivat 
#ColumnTransformer a podobne.

#Takze tato trieda sluzi ako wrapper okolo ColumnTransformer transformacie, kedy si uchovavame strukturu dataframu, teda
#index, ako aj mena stlpcov, a nasledne, potom, co sa vykona ColumnTransformer, dany output vlozime do dataframu
class WrapColumnTransformer(base.BaseEstimator, base.TransformerMixin):
    
    def __init__(self, column_transformer, keep_original_cols=True, custom_cols_names=None):
        self.column_transformer = column_transformer
        self.keep_original_cols = keep_original_cols
        self.custom_cols_names = custom_cols_names
    
    def fit(self, X, y=None):
        self.column_transformer.fit(X)
        return self
        
            
    def transform(self, X):
        indices = X.index
        
        columns = []
        
        for transf in self.column_transformer.transformers:
            columns += transf[2]
           

        X = X.copy()
        
        X = self.column_transformer.transform(X)

        if self.keep_original_cols == True:
            X = pd.DataFrame(X, columns=columns, index=indices)
        
        elif self.custom_cols_names is not None:
            X = pd.DataFrame(X, columns=self.custom_cols_names, index=indices)
            
        else:
            X = pd.DataFrame(X, index=indices)
        
        return X


# Tu uz mame jednotlive skupiny atributov, pre ktore patria rozlicne sposoby aplikacie imputovania missing values.

oxygen_attr = ["mean_oxygen", "std_oxygen", "kurtosis_oxygen", "skewness_oxygen"]
glucose_attr = ["mean_glucose", "std_glucose", "kurtosis_glucose", "skewness_glucose"]

vztahy_attr = ["relationship", "marital-status"]
work_attr = ["workclass", "occupation", "hours-per-week-cat", "income"]
edu_attr = ["education", "education-num"]

impute_col_transf = compose.ColumnTransformer(transformers=[
    ("oxygen_n_glucose_impute", KeepDataFrame(impute.IterativeImputer(max_iter=50)), oxygen_attr + glucose_attr),
    ("vztahy_impute", CustomCatImputing(imputer_type="knn"), vztahy_attr),
    ("work_impute", CustomCatImputing(imputer_type="knn"), work_attr),
    ("edu_impute", CustomCatImputing(imputer_type="knn"), edu_attr),
    ("sex_impute", KeepDataFrame(impute.SimpleImputer(strategy="most_frequent")), ["sex"]),
    ("age_impute", KeepDataFrame(impute.SimpleImputer()), ["age"])
])


#sluzi na vratenie casti datasetu, ktory je outliermi
def identify_outliers(a):
    q25 = a.quantile(0.25)
    q75 = a.quantile(0.75)
    
    iqr = q75-q25
        
    lower = q25 - 1.5 * iqr
    upper = q75 + 1.5 * iqr
    
    return a[(a > upper) | (a < lower)]

#odstranovanie outlierov pomocou takeho stylu, ze dany atribut, ktory riesime, rozdelime na 
#dvoje distribucie, podla target hodnoty, a jednotlivym distribuciach hladame outliery, ktore
#nasledne odstranime
def removing_outliers_per_class(data, column, clz="class"):

    data = data.copy()
    
    data_y0 = data[data[clz] == 0][column]
    data_y1 = data[data[clz] == 1][column]
        
    idx = identify_outliers(data_y0).index.values
    data = data.drop(index=idx)

    idx = identify_outliers(data_y1).index.values
    data = data.drop(index=idx)
    
    return data


# Ako mozno vidiet, tu mame zobrazene spojite atributy, ktorym boli aplikovane, resp. neaplikovane non-linear transforamacie, pricom sme pouzili:

power_transf_attr = ["mean_oxygen", "skewness_oxygen", "kurtosis_oxygen", "skewness_glucose"]
quant_transf_attr = ["age"]
other_attr = ["std_oxygen", "mean_glucose", "std_glucose", "kurtosis_glucose", "sex", "education"] + vztahy_attr + work_attr

non_linear_transf =  compose.ColumnTransformer(transformers=[
   ("power_transformer", KeepDataFrame(preprocessing.PowerTransformer()), power_transf_attr),
   ("quantile_transformer", KeepDataFrame(preprocessing.QuantileTransformer(output_distribution="normal")), quant_transf_attr),
   ("pass", "passthrough", other_attr)
])


#  # Outliers - resampling

# Tu sa sustredime na odstranenie outlierov zo spojitych atributov, kedy toto je cast pipelinu, kedy sa pouziva tzv. resampling, a kvoli ktoremu sme museli pouzit specialny pipeline od kniznice imblearn - nejaka podnoz scikit-learnu.
# Pre resampling je typicke, ze sa vykonava len pre trenovaci dataset, pri testovacom sa nepouziva.

class OutlierRemoval(base.BaseEstimator):
     
    def __init__(self, columns):
        self.columns = columns
        
    def fit_resample(self, X, y):
        return self.resample(X, y)
                
    def resample(self, X, y):
        
        X = X.copy()
        y = y.copy()
        
        data = X.join(y, how="left")
        clz = "class"
        
        
        for c in self.columns:
            
            data_y0 = data[data[clz] == 0][c]
            data_y1 = data[data[clz] == 1][c]

            idx = identify_outliers(data_y0).index.values
            data = data.drop(index=idx)

            idx = identify_outliers(data_y1).index.values
            data = data.drop(index=idx)
            
        #toto je specialne pre target atribut
        if data[clz].isnull().sum() > 0:
            idx = data[data[clz].isnull()].index.values
            data = data.drop(index=idx)

            
        X = data.drop(columns=["class"])
        y = data["class"]
            
        return X, y


outlier_columns = oxygen_attr + glucose_attr + ["age"]


scaling = pipeline.Pipeline(steps=[
    ("standard_scaler", preprocessing.StandardScaler())
])

onehot = pipeline.Pipeline(steps=[
    ("one_hot_enc", preprocessing.OneHotEncoder(handle_unknown="ignore"))
])

ord_mapping = [
    {"col": "education", "mapping": {
        "Attending-school": 1, 
        "HS-grad": 2,
        "Edu after HS, no uni": 3,
        "Some-college": 4,
        "Bachelors": 5,
        "Masters/Doctorate": 6}},
    
    {"col": "hours-per-week-cat", "mapping": {
        "<=35": 1,
        "35< hours <=45": 2,
        ">45": 3}},
    
    {"col": "income", "mapping": {
        "<=50K": 1,
        ">50K": 2}}
]


ordinal = pipeline.Pipeline(steps=[
    ("ordinal_enc", ce.OrdinalEncoder(mapping=ord_mapping, handle_unknown="return_nan")),
    ("impute_unknown", impute.SimpleImputer(strategy="most_frequent"))
])


scaling_attr = ["age"] + oxygen_attr + glucose_attr

onehot_attr = ["sex", "marital-status", "relationship", "occupation", "workclass"]

ordinal_attr = ["education", "hours-per-week-cat", "income"]

last_col_transf = compose.ColumnTransformer(transformers=[
    ("num_attr_scaling", scaling, scaling_attr),
    ("cat_attr_onehot_enc", onehot, onehot_attr),
    ("cat_attr_ordinal_enc", ordinal, ordinal_attr)
])


#tato trieda sa hra na klasifikator, aby mohla byt poslednym krokom v pipeline
#sluzi na to, aby sme vedeli z pipelinu dostat nove X a y, ktore uz mozme rovno hodit do nejakeho modelu
class Return_X_y(base.BaseEstimator, base.ClassifierMixin):
    
    def fit(self, X, y=None):
        
        return self
    
    def fit_predict(self, X, y=None):
        self.fit(X,y)
        return self.predict(X,y)
    
    def predict(self, X, y=None):
        
        if y is None:
            return X
        
        y = y.values
        return X,y


# Vytvaram nazvy pre stlpce, ktore bude mat dataframe, ktory pipeline vracia

custom_cols_names = scaling_attr.copy()
pocet_values = [2, 4, 4, 9, 2]

for col, pocet in zip(onehot_attr, pocet_values):
    for i in range(pocet):
        custom_cols_names.append(col+"_"+str(i))
    
custom_cols_names += ordinal_attr
custom_cols_names

def get_preprocessing_pipeline():

    MAIN_PIPELINE = imblearn.pipeline.Pipeline(steps=[
        #prvotne preprocessing stepy
        ("feature_removal", preprocessing.FunctionTransformer(remove_useless_features)),
        ("add_oxygen_attr", preprocessing.FunctionTransformer(add_oxygen_features)),
        ("mean_glucose_to_num", preprocessing.FunctionTransformer(repair_mean_glucose)),
        ("string_formatting", preprocessing.FunctionTransformer(string_wrap_formatting)),
        ("bucket_cat_attr", preprocessing.FunctionTransformer(bucket_cat_attr)),

        #imputacia
        ("imputation_stage",  WrapColumnTransformer(impute_col_transf)),

        #non-linear transformacie
        ("non_linear_transform", WrapColumnTransformer(non_linear_transf)),

        #resampling - outlier removal
        ("outlier_removal", OutlierRemoval(outlier_columns)),

        #scaling and encoding - tu uz nechceme, aby si wrapper okolo ColumnTransformer pamatal nazvy stlpcov, kedze ich tam bude ovela viac
        #kvoli OneHot encodingu; avsak si stale chceme pamatat index
        ("scaling_n_encoding_stage", WrapColumnTransformer(last_col_transf, keep_original_cols=False, custom_cols_names=custom_cols_names)),

        #vratenie datasetu po aplikovani krokov tohto pipelinu
        ("return_X_y", Return_X_y())

    ])
    
    return MAIN_PIPELINE