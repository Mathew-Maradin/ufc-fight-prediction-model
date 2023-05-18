#Import Cell
#used to import all the libraries and functions used
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import sys, warnings, os
from sklearn.dummy import DummyClassifier


#To ignore max-iteration warnings while cross validating scores
if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"

    #Setting columns and rows to display all the results
pd.set_option("display.max_columns", None, "display.max_rows", None)


#Reading the dataset
ufc_master_ds = pd.read_csv("../input/ultimate-ufc-dataset/ufc-master.csv")
#Seperating label from input
label = ufc_master_ds.Winner
#I have removed "B_Women's Featherweight_rank" because imputing with this feature in the dataset gives me a ton of errors in the baseline model.
X = ufc_master_ds.drop(['Winner',"B_Women's Featherweight_rank"], axis =1)


#Separating the features based on their data types
cat_col = [col for col in X.columns if X[col].dtypes == 'object']
num_col = [col for col in X.columns if col not in cat_col]


enc = LabelEncoder()
for i in X[cat_col]:
    #using astype(str) to avoid columns with 'float and str' to throw errors
    X[i] = enc.fit_transform(X[i].astype(str))



#Also encoding Label for Red to be 1 and Blue to be 0 
label = [1 if win == 'Red' else 0 for win in label]

X_train, X_valid, y_train, y_valid = train_test_split(X, label, random_state = 2, test_size = 0.3)

X_train.isnull().sum().sort_values(ascending=False)


#A DummyClassifier is used to be a baseline to compare a better model's performance later on
base_model = DummyClassifier(random_state=2)


base_model.fit(X_train,y_train)


preds = base_model.predict(X_valid)
accuracy_score(y_valid, preds)


ufc_master_ds.head()


#Encoding label so it is easier to find correlation
ufc_master_ds['Winner'] = [1 if winner == 'Red' else 0 for winner in ufc_master_ds.Winner]


num_corr_col = [col for col in ufc_master_ds.columns if ufc_master_ds[col].dtype == 'int64' or ufc_master_ds[col].dtype == 'float64']
corr_dict = {}
#Getting absolute values of correlation since we would need to inspect negative correlation too
for col in num_corr_col:
    corr_dict[col] = abs(ufc_master_ds[col].corr(ufc_master_ds['Winner']))



for w in sorted(corr_dict, key=corr_dict.get):
    print(w, corr_dict[w])


ufc_master_ds['B_kd_bout'].unique()


#Getting null values percentage
(ufc_master_ds['B_kd_bout'].isnull().sum()/ufc_master_ds.shape[0])*100


#For visualization purposes
ufc_master_ds['Winner'] = ['Red' if winner == 1 else 'Blue' for winner in ufc_master_ds.Winner]



sns.countplot(x=ufc_master_ds['B_kd_bout'], hue = ufc_master_ds['Winner'])

#Analysing "_odds" variables
sns.scatterplot(x="B_odds", y="R_odds", hue="Winner", data = ufc_master_ds)


#Just to be sure
ufc_master_ds["Winner"].loc[ufc_master_ds["B_odds"]>1].value_counts()



#Null values in _sig_str_pct_bout variables
[(ufc_master_ds[col].isnull().sum()/ufc_master_ds.shape[0])*100 for col in ['R_sig_str_pct_bout','B_sig_str_pct_bout']]
#Same number of missing values as _kd_bout variables


sns.scatterplot(x='R_sig_str_pct_bout',y='B_sig_str_pct_bout',hue = 'Winner', data=ufc_master_ds)


#Lets inspect _ev variables
sns.scatterplot(x='B_ev', y='R_ev',hue = 'Winner', data=ufc_master_ds)


fig, ax = plt.subplots(1,2, figsize=(15,7))
sns.scatterplot(x='B_ev', y='R_odds',hue = 'Winner', data=ufc_master_ds, ax=ax[0]);
sns.scatterplot(x='B_ev', y='B_odds',hue = 'Winner', data=ufc_master_ds, ax=ax[1]);
fig.show()


fig, ax = plt.subplots(1,2, figsize=(15,7))
sns.countplot(ufc_master_ds['Winner'], hue = ufc_master_ds['R_Stance'], ax=ax[0])
sns.countplot(ufc_master_ds['Winner'], hue = ufc_master_ds['B_Stance'], ax=ax[1])
ax[0].title.set_text('Stances of Red Players')
ax[1].title.set_text('Stances of Blue Players')
fig.show()


ufc_master_ds.head()


ufc_master_ds['draw_diff'] = (ufc_master_ds['B_draw']-ufc_master_ds['R_draw'])
ufc_master_ds['avg_sig_str_pct_diff'] = (ufc_master_ds['B_avg_SIG_STR_pct']-ufc_master_ds['R_avg_SIG_STR_pct'])
ufc_master_ds['avg_TD_pct_diff'] = (ufc_master_ds['B_avg_TD_pct']-ufc_master_ds['B_avg_TD_pct'])
ufc_master_ds['win_by_Decision_Majority_diff'] = (ufc_master_ds['B_win_by_Decision_Majority']-ufc_master_ds['R_win_by_Decision_Majority'])
ufc_master_ds['win_by_Decision_Split_diff'] = (ufc_master_ds['B_win_by_Decision_Split']-ufc_master_ds['R_win_by_Decision_Split'])
ufc_master_ds['win_by_Decision_Unanimous_diff'] = (ufc_master_ds['B_win_by_Decision_Unanimous']-ufc_master_ds['R_win_by_Decision_Unanimous'])
ufc_master_ds['win_by_TKO_Doctor_Stoppage_diff'] = (ufc_master_ds['B_win_by_TKO_Doctor_Stoppage']-ufc_master_ds['R_win_by_TKO_Doctor_Stoppage'])


ufc_master_ds['odds_diff'] = (ufc_master_ds['B_odds']-ufc_master_ds['R_odds'])
ufc_master_ds['ev_diff'] = (ufc_master_ds['B_ev']-ufc_master_ds['R_ev'])

ufc_master_ds['kd_bout_diff']=(ufc_master_ds['B_kd_bout']-ufc_master_ds['R_kd_bout'])
ufc_master_ds['sig_str_landed_bout_diff']=(ufc_master_ds['B_sig_str_landed_bout']-ufc_master_ds['R_sig_str_landed_bout'])
ufc_master_ds['sig_str_attempted_bout_diff']=(ufc_master_ds['B_sig_str_attempted_bout']-ufc_master_ds['R_sig_str_attempted_bout'])
ufc_master_ds['sig_str_attempted_bout_diff']=(ufc_master_ds['B_sig_str_attempted_bout']-ufc_master_ds['R_sig_str_attempted_bout'])
ufc_master_ds['sig_str_pct_bout_diff']=(ufc_master_ds['B_sig_str_pct_bout']-ufc_master_ds['R_sig_str_pct_bout'])
ufc_master_ds['tot_str_landed_bout_diff']=(ufc_master_ds['B_tot_str_landed_bout']-ufc_master_ds['R_tot_str_landed_bout'])
ufc_master_ds['tot_str_attempted_bout_diff']=(ufc_master_ds['B_tot_str_attempted_bout']-ufc_master_ds['R_tot_str_attempted_bout'])
ufc_master_ds['td_landed_bout_diff']=(ufc_master_ds['B_td_landed_bout']-ufc_master_ds['R_td_landed_bout'])
ufc_master_ds['td_attempted_bout_diff']=(ufc_master_ds['B_td_attempted_bout']-ufc_master_ds['R_td_attempted_bout'])
ufc_master_ds['td_pct_bout_diff']=(ufc_master_ds['B_td_pct_bout']-ufc_master_ds['R_td_pct_bout'])
ufc_master_ds['td_pct_bout_diff']=(ufc_master_ds['B_td_pct_bout']-ufc_master_ds['R_td_pct_bout'])
ufc_master_ds['sub_attempts_bout_diff']=(ufc_master_ds['B_sub_attempts_bout']-ufc_master_ds['R_sub_attempts_bout'])
ufc_master_ds['pass_bout_diff']=(ufc_master_ds['B_pass_bout']-ufc_master_ds['R_pass_bout'])
ufc_master_ds['rev_bout_diff']=(ufc_master_ds['B_rev_bout']-ufc_master_ds['R_rev_bout'])


#Dropping variables
var_drop = [
'B_odds',
'R_odds',
'B_ev',
'R_ev',
'R_kd_bout',
'B_kd_bout',
'R_sig_str_landed_bout',
'B_sig_str_landed_bout',
'R_sig_str_attempted_bout',
'B_sig_str_attempted_bout',
'R_sig_str_pct_bout',
'B_sig_str_pct_bout',
'R_tot_str_landed_bout',
'B_tot_str_landed_bout',
'R_tot_str_attempted_bout',
'B_tot_str_attempted_bout',
'R_td_landed_bout',
'B_td_landed_bout',
'R_td_attempted_bout',
'B_td_attempted_bout',
'R_td_pct_bout',
'B_td_pct_bout',
'R_sub_attempts_bout',
'B_sub_attempts_bout',
'R_pass_bout',
'B_pass_bout',
'R_rev_bout',
'B_rev_bout',
'B_current_lose_streak', 'R_current_lose_streak',
'B_current_win_streak', 'R_current_win_streak',
'B_longest_win_streak', 'R_longest_win_streak',
'B_wins', 'R_wins',
'B_losses', 'R_losses',
'B_total_rounds_fought', 'R_total_rounds_fought',
'B_total_title_bouts', 'R_total_title_bouts',
'B_win_by_KO/TKO', 'R_win_by_KO/TKO',
'B_win_by_Submission', 'R_win_by_Submission',
'B_Height_cms', 'R_Height_cms',
'B_Reach_cms', 'R_Reach_cms',
'B_age', 'R_age',
'B_avg_SIG_STR_landed', 'R_avg_SIG_STR_landed',
'B_avg_SUB_ATT', 'R_avg_SUB_ATT',
'B_avg_TD_landed', 'R_avg_TD_landed',
'B_draw','B_avg_SIG_STR_pct','B_avg_TD_pct','B_win_by_Decision_Majority','B_win_by_Decision_Split','B_win_by_Decision_Unanimous','B_win_by_TKO_Doctor_Stoppage',
'R_draw','R_avg_SIG_STR_pct','R_avg_TD_pct','R_win_by_Decision_Majority','R_win_by_Decision_Split','R_win_by_Decision_Unanimous','R_win_by_TKO_Doctor_Stoppage']
ufc_master_ds.drop(var_drop, axis=1, inplace = True)


comm_drop = [
'date','location','country','weight_class','gender','no_of_rounds','empty_arena','constant_1','finish','finish_details','finish_round','finish_round_time','total_fight_time_secs','B_Weight_lbs','R_Weight_lbs'
]
ufc_master_ds.drop(comm_drop, axis=1, inplace = True)


ufc_master_ds.B_Stance.unique()


#It has one spelling mistake
ufc_master_ds['B_Stance'].loc[ufc_master_ds['B_Stance']=='Switch '] = 'Switch'
#R_Stance doesn't have this error, so we're cool


stance = ['B_Stance', 'R_Stance']


for x in stance:
    ufc_master_ds[x] = [4 if st == 'Orthodox'
                           else 3 if st == 'Southpaw'
                           else 2 if st == 'Switch'
                           else 1 for st in ufc_master_ds[x]]
#using -1 and 1 for both red and blue so there is no misunderstanding that one variable is better than the other    
ufc_master_ds['better_rank'] = [-1 if rank == 'Red'
                               else 1 if rank == 'Blue'
                               else 0 for rank in ufc_master_ds['better_rank']]

ufc_master_ds['title_bout'] = [1 if tb==True else 0 for tb in ufc_master_ds['title_bout']]


ufc_master_ds['Stance_diff'] = (ufc_master_ds['B_Stance'] - ufc_master_ds['R_Stance'])
ufc_master_ds.drop(stance, axis = 1, inplace = True)

ufc_master_ds.head()


#Encoding label so it is easier to find correlation
ufc_master_ds['Winner'] = [1 if winner == 'Red' else 0 for winner in ufc_master_ds.Winner]



ufc_master_ds.loc[:,'B_match_weightclass_rank':'better_rank'].isnull().sum()



ufc_master_ds.drop(ufc_master_ds.loc[:,'B_match_weightclass_rank':'B_Pound-for-Pound_rank'], axis=1, inplace = True)

ufc_master_ds.sample(10)

label = ufc_master_ds.Winner
ufc_master_ds.drop(['Winner'], axis=1, inplace = True)


#Encoding the remaining categorical variables
cat_col = ['R_fighter', 'B_fighter']
enc = LabelEncoder()
for i in ufc_master_ds[cat_col]:
    ufc_master_ds[i] = enc.fit_transform(ufc_master_ds[i])

X_train, X_valid, y_train, y_valid = train_test_split(ufc_master_ds, label, test_size = 0.3, random_state=2)


#At this point all the null values are the ones that have been left empty by error/mistake and are not left empty deliberately.
#So, it would make sense to fill in these with mean rather than 0 or anything else
impute = SimpleImputer(strategy = 'mean')
impute.fit(X_train)
X_train = impute.transform(X_train)
X_valid = impute.transform(X_valid)

RF_model = RandomForestClassifier(random_state=2)


RF_model.fit(X_train, y_train)

preds = RF_model.predict(X_valid)
accuracy_score(y_valid, preds)

#Built a model after doing GridSearch but not putting the code here because the cell takes up a lot of time
RF_model = RandomForestClassifier(n_estimators = 350, max_depth = 12, random_state = 2)


RF_model.fit(X_train, y_train)
preds = RF_model.predict(X_valid)
accuracy_score(y_valid, preds)