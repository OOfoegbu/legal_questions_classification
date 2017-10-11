

```python
from IPython.display import HTML

HTML('''<script>
code_show=true; 
function code_toggle() {
 if (code_show){
 $('div.input').hide();
 } else {
 $('div.input').show();
 }
 code_show = !code_show
} 
$( document ).ready(code_toggle);
</script>
<form action="javascript:code_toggle()"><input type="submit" value="Click here to toggle on/off the raw code."></form>''')
```




<script>
code_show=true; 
function code_toggle() {
 if (code_show){
 $('div.input').hide();
 } else {
 $('div.input').show();
 }
 code_show = !code_show
} 
$( document ).ready(code_toggle);
</script>
<form action="javascript:code_toggle()"><input type="submit" value="Click here to toggle on/off the raw code."></form>



# ANALYSIS OF LEGAL CONCERNS AROUND THE UNITED STATES

# Table of Content

- Introduction
    - Project Goals
- Loading data and python packages 
- Data Cleaning
- Exploratory Analysis
- Recommendations
- Appendix

## INTRODUCTION

This project would make use of data collected from a free source website. The website contains questions asked by users from around the country, and answers are provided by attorneys. The entries are grouped into 18 legal forums.

The different legal forums are: 1) Auto accident, 2) Bankruptcy, 3) Business, 4) Collections & Debt, 5) Consumer & Lemon, 6) Child Custody, 7) Criminal Defense, 8) Divorce, 9) DUI & DWI, 10) Employment & Labor, 11) Immigration, 12) Insurance, 13) Landlord & Tenant, 14) Medical Malpractice, 15) Personal Injury, 16) Real Estate, 17)Traffic, and 18) Wills,Trust & Probate.

The data collected from the website are:
- Question title - A brief description of the question.
- Question detail - Detailed description of the question.
- State - The US state the user resides. These include DC, Puerto Rico, and US Virgin Islands.
- Date answered - The estimated date answered. This is calculated from the date it was retrieved from the website.
- Attorney - The name of the attorney or law firm who provided the first answer to the question.

There are 126597 questions in the dataset.

### Project Goals

Using the available data, below are the questions I propose to address:
 
1. What state has the most legal concerns? 
2. Countrywide, what are the most frequent legal concerns?
3. What states will certain types of lawyers be in high demand? 
4. Are there certain periods of the year when particular legal problems are prevalent?
5. What are the number of unique attorneys who answered questions in the various legal 	forums?
6. For each of the legal forums what attorney/firm answers most of the questions? 
7. For each of the legal forums what are the major questions?
8. What types of questions are left unanswered?

Based on my findings, I will make recommendations to a hypothetical law firm.

In addition to answering the above questions, I plan to build a classification model that takes the question title and detail, and determines what legal forum it belongs to. The classfication model will be presented in a separate report.

## LOADING DATA AND PYTHON PACKAGES

### Load required packages

The required packages for this project were loaded.


```python
## Load required packages
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import gensim
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
import string
from nltk.util import ngrams
from IPython.display import Image
from IPython.core.display import Image, display
% matplotlib inline
```

### Load data from postgres database

The data from the website were stored in a postgres database, and read into jupyter for exploration and analysis.


```python
## Load data from database
engine = create_engine('postgresql://postgres:Edoamen1@localhost:5433/LegalQuestions')

legal = pd.read_sql_query('select * from legal."legaldata"',con=engine)
```

## DATA CLEANING

Since the website did not keep track of the date each question was asked, I have to estimate that date using when it was answered and the date I collected the data. The assumption in this estimate is that the questions where answered around the same time the question was asked. 

Also, I merged the question title and detail columns inorder to have a full description of the question. Finally I filled empty state entries with 'unspecified state'.


```python
## Calculate the estimated date asked
year_five = legal[legal['date_answered'].str.contains('Answered 5 years')]
year_seven = legal[legal['date_answered'].str.contains('Answered 7 years')]
year_four = legal[legal['date_answered'].str.contains('Answered 4 years')]
year_six = legal[legal['date_answered'].str.contains('Answered 6 years')]
year_one = legal[legal['date_answered'].str.contains('Answered 1 year')]
year_eight = legal[legal['date_answered'].str.contains('Answered 8 years')]
year_two = legal[legal['date_answered'].str.contains('Answered 2 years')]
year_three = legal[legal['date_answered'].str.contains('Answered 3 years')]

month_one = legal[legal['date_answered'].str.contains('Answered 1 month')]
month_two = legal[legal['date_answered'].str.contains('Answered 2 months')]
month_three = legal[legal['date_answered'].str.contains('Answered 3 months')]
month_four = legal[legal['date_answered'].str.contains('Answered 4 months')]
month_five = legal[legal['date_answered'].str.contains('Answered 5 months')]
month_six = legal[legal['date_answered'].str.contains('Answered 6 months')]
month_seven = legal[legal['date_answered'].str.contains('Answered 7 months')]
month_eight = legal[legal['date_answered'].str.contains('Answered 8 months')]
month_nine = legal[legal['date_answered'].str.contains('Answered 9 months')]
month_ten = legal[legal['date_answered'].str.contains('Answered 10 months')]
month_eleven = legal[legal['date_answered'].str.contains('Answered 11 months')]
month_twelve = legal[legal['date_answered'].str.contains('Answered 12 months')]

day_1 = legal[legal['date_answered'].str.contains('Answered 1 day')]
day_2 = legal[legal['date_answered'].str.contains('Answered 2 days')]
day_4 = legal[legal['date_answered'].str.contains('Answered 4 days')]
day_5 = legal[legal['date_answered'].str.contains('Answered 5 days')]
day_6 = legal[legal['date_answered'].str.contains('Answered 6 days')]
day_7 = legal[legal['date_answered'].str.contains('Answered 7 days')]
day_8 = legal[legal['date_answered'].str.contains('Answered 8 days')]
day_9 = legal[legal['date_answered'].str.contains('Answered 9 days')]
day_10 = legal[legal['date_answered'].str.contains('Answered 10 days')]
day_11 = legal[legal['date_answered'].str.contains('Answered 11 days')]
day_15 = legal[legal['date_answered'].str.contains('Answered 15 days')]
day_16 = legal[legal['date_answered'].str.contains('Answered 16 days')]
day_17 = legal[legal['date_answered'].str.contains('Answered 17 days')]
day_18 = legal[legal['date_answered'].str.contains('Answered 18 days')]
day_19 = legal[legal['date_answered'].str.contains('Answered 19 days')]
day_20 = legal[legal['date_answered'].str.contains('Answered 20 days')]
day_21 = legal[legal['date_answered'].str.contains('Answered 21 days')]
day_22 = legal[legal['date_answered'].str.contains('Answered 22 days')]
day_23 = legal[legal['date_answered'].str.contains('Answered 23 days')]
day_24 = legal[legal['date_answered'].str.contains('Answered 24 days')]
day_25 = legal[legal['date_answered'].str.contains('Answered 25 days')]
day_26 = legal[legal['date_answered'].str.contains('Answered 26 days')]
day_27 = legal[legal['date_answered'].str.contains('Answered 27 days')]
day_28 = legal[legal['date_answered'].str.contains('Answered 28 days')]
day_29 = legal[legal['date_answered'].str.contains('Answered 29 days')]

today = legal[legal['date_answered'].str.contains('Answered  today')]
unans = legal[legal['date_answered'].str.contains('unanswered')]

year_five['est_date_asked'] = year_five['date_collected'].apply(lambda x: x - pd.DateOffset(years=5))
year_seven['est_date_asked'] = year_seven['date_collected'].apply(lambda x: x - pd.DateOffset(years=7))
year_four['est_date_asked'] = year_four['date_collected'].apply(lambda x: x - pd.DateOffset(years=4))
year_six['est_date_asked'] = year_six['date_collected'].apply(lambda x: x - pd.DateOffset(years=6))
year_one['est_date_asked'] = year_one['date_collected'].apply(lambda x: x - pd.DateOffset(years=1))
year_eight['est_date_asked'] = year_eight['date_collected'].apply(lambda x: x - pd.DateOffset(years=8))
year_two['est_date_asked'] = year_two['date_collected'].apply(lambda x: x - pd.DateOffset(years=2))
year_three['est_date_asked'] = year_three['date_collected'].apply(lambda x: x - pd.DateOffset(years=3))

month_one['est_date_asked'] = month_one['date_collected'].apply(lambda x: x - pd.DateOffset(months=1))
month_two['est_date_asked'] = month_two['date_collected'].apply(lambda x: x - pd.DateOffset(months=2))
month_three['est_date_asked'] = month_three['date_collected'].apply(lambda x: x - pd.DateOffset(months=3))
month_four['est_date_asked'] = month_four['date_collected'].apply(lambda x: x - pd.DateOffset(months=4))
month_five['est_date_asked'] = month_five['date_collected'].apply(lambda x: x - pd.DateOffset(months=5))
month_six['est_date_asked'] = month_six['date_collected'].apply(lambda x: x - pd.DateOffset(months=6))
month_seven['est_date_asked'] = month_seven['date_collected'].apply(lambda x: x - pd.DateOffset(months=7))
month_eight['est_date_asked'] = month_eight['date_collected'].apply(lambda x: x - pd.DateOffset(months=8))
month_nine['est_date_asked'] = month_nine['date_collected'].apply(lambda x: x - pd.DateOffset(months=9))
month_ten['est_date_asked'] = month_ten['date_collected'].apply(lambda x: x - pd.DateOffset(months=10))
month_eleven['est_date_asked'] = month_eleven['date_collected'].apply(lambda x: x - pd.DateOffset(months=11))
month_twelve['est_date_asked'] = month_twelve['date_collected'].apply(lambda x: x - pd.DateOffset(months=12))

day_1['est_date_asked'] = day_1['date_collected'].apply(lambda x: x - pd.DateOffset(days=1))
day_2['est_date_asked'] = day_2['date_collected'].apply(lambda x: x - pd.DateOffset(days=2))
day_4['est_date_asked'] = day_4['date_collected'].apply(lambda x: x - pd.DateOffset(days=4))
day_5['est_date_asked'] = day_5['date_collected'].apply(lambda x: x - pd.DateOffset(days=5))
day_6['est_date_asked'] = day_6['date_collected'].apply(lambda x: x - pd.DateOffset(days=6))
day_7['est_date_asked'] = day_7['date_collected'].apply(lambda x: x - pd.DateOffset(days=7))
day_8['est_date_asked'] = day_8['date_collected'].apply(lambda x: x - pd.DateOffset(days=8))
day_9['est_date_asked'] = day_9['date_collected'].apply(lambda x: x - pd.DateOffset(days=9))
day_10['est_date_asked'] = day_10['date_collected'].apply(lambda x: x - pd.DateOffset(days=10))
day_11['est_date_asked'] = day_11['date_collected'].apply(lambda x: x - pd.DateOffset(days=11))
day_15['est_date_asked'] = day_15['date_collected'].apply(lambda x: x - pd.DateOffset(days=15))
day_16['est_date_asked'] = day_16['date_collected'].apply(lambda x: x - pd.DateOffset(days=16))
day_17['est_date_asked'] = day_17['date_collected'].apply(lambda x: x - pd.DateOffset(days=17))
day_18['est_date_asked'] = day_18['date_collected'].apply(lambda x: x - pd.DateOffset(days=18))
day_19['est_date_asked'] = day_19['date_collected'].apply(lambda x: x - pd.DateOffset(days=19))
day_20['est_date_asked'] = day_20['date_collected'].apply(lambda x: x - pd.DateOffset(days=20))
day_21['est_date_asked'] = day_21['date_collected'].apply(lambda x: x - pd.DateOffset(days=21))
day_22['est_date_asked'] = day_22['date_collected'].apply(lambda x: x - pd.DateOffset(days=22))
day_23['est_date_asked'] = day_23['date_collected'].apply(lambda x: x - pd.DateOffset(days=23))
day_24['est_date_asked'] = day_24['date_collected'].apply(lambda x: x - pd.DateOffset(days=24))
day_25['est_date_asked'] = day_25['date_collected'].apply(lambda x: x - pd.DateOffset(days=25))
day_26['est_date_asked'] = day_26['date_collected'].apply(lambda x: x - pd.DateOffset(days=26))
day_27['est_date_asked'] = day_27['date_collected'].apply(lambda x: x - pd.DateOffset(days=27))
day_28['est_date_asked'] = day_28['date_collected'].apply(lambda x: x - pd.DateOffset(days=28))
day_29['est_date_asked'] = day_29['date_collected'].apply(lambda x: x - pd.DateOffset(days=29))

today['est_date_asked'] = today['date_collected'].apply(lambda x: x - pd.DateOffset(days=0))
 

legal = year_five.append(year_seven, ignore_index = True)
legal = legal.append(year_four, ignore_index = True)
legal = legal.append(year_six, ignore_index = True)
legal = legal.append(year_one, ignore_index = True)
legal = legal.append(year_eight, ignore_index = True)
legal = legal.append(year_two, ignore_index = True)
legal = legal.append(year_three, ignore_index = True)
legal = legal.append(month_one, ignore_index = True)
legal = legal.append(month_two, ignore_index = True)
legal = legal.append(month_three, ignore_index = True)
legal = legal.append(month_four, ignore_index = True)
legal = legal.append(month_five, ignore_index = True)
legal = legal.append(month_six, ignore_index = True)
legal = legal.append(month_seven, ignore_index = True)
legal = legal.append(month_eight, ignore_index = True)
legal = legal.append(month_nine, ignore_index = True)
legal = legal.append(month_ten, ignore_index = True)
legal = legal.append(month_eleven, ignore_index = True)
legal = legal.append(month_twelve, ignore_index = True)
legal = legal.append(day_1, ignore_index = True)
legal = legal.append(day_2, ignore_index = True)
legal = legal.append(day_4, ignore_index = True)
legal = legal.append(day_5, ignore_index = True)
legal = legal.append(day_6, ignore_index = True)
legal = legal.append(day_7, ignore_index = True)
legal = legal.append(day_8, ignore_index = True)
legal = legal.append(day_9, ignore_index = True)
legal = legal.append(day_10, ignore_index = True)
legal = legal.append(day_11, ignore_index = True)
legal = legal.append(day_15, ignore_index = True)
legal = legal.append(day_16, ignore_index = True)
legal = legal.append(day_17, ignore_index = True)
legal = legal.append(day_18, ignore_index = True)
legal = legal.append(day_19, ignore_index = True)
legal = legal.append(day_20, ignore_index = True)
legal = legal.append(day_21, ignore_index = True)
legal = legal.append(day_22, ignore_index = True)
legal = legal.append(day_23, ignore_index = True)
legal = legal.append(day_24, ignore_index = True)
legal = legal.append(day_25, ignore_index = True)
legal = legal.append(day_26, ignore_index = True)
legal = legal.append(day_27, ignore_index = True)
legal = legal.append(day_28, ignore_index = True)
legal = legal.append(day_29, ignore_index = True)
legal = legal.append(today, ignore_index = True)
legal = legal.append(unans, ignore_index = True)

## Shuffle the dataframe
legal = legal.sample(frac=1, random_state = 12).reset_index(drop=True)
legal['est_date_asked'] = pd.to_datetime(legal['est_date_asked']).dt.date
```


```python
## Replacing nan with space
legal = legal.replace(np.nan, '', regex=True)

## Merging the title and question text columns
legal['title and question'] = legal['titles'].str.cat(legal['questions'], sep=' ')
```


```python
## Fill empty state entries with 'Unspecified State'

state_empty = legal[legal['state'] == '']['state'].index.tolist()

for i in range(0,len(state_empty)):
    n = state_empty[i]
    legal['state'].iloc[n] = 'Unspecified State'

```

## EXPLORATORY ANALYSIS

### Question 1: What state has the most and least legal concerns

As shown in the bar chart below, a majority of the question on the free advice website originated from California, while the least where from US Virgin Islands.

The pie charts show that most of the questions from California relate to labor law, and a majority of questions from the US virgin Islands relate to labor, criminal and divorce law.


```python
## select top 5 states where most questions originated
top_5_states = legal['state'].value_counts()[0:5]
## select bottom 5 states where most questions originated
bottom_5_states = legal['state'].value_counts()[50:55]

## Plot bar chart of the  5 top and bottom states 
fig = plt.figure(figsize=(20, 25))
ax = fig.add_subplot(211)
top_5_states.plot(kind = 'barh', ax = ax)
ax.set_xlabel('No. of Questions', fontsize = 25)
ax.set_ylabel('State', fontsize = 25)
ax.set_title('Top 5 states with most legal questions', size = 25)
ax.tick_params(axis='x', labelsize=30)
ax.tick_params(axis='y', labelsize=20)
plt.gca().invert_yaxis()
ax1 = fig.add_subplot(212)
bottom_5_states.plot(kind = 'barh', ax = ax1)
ax1.set_xlabel('No. of Questions', fontsize = 25)
ax1.set_ylabel('State', fontsize = 25)
ax1.set_title('Bottom 5 states with most legal questions', size = 25)
ax1.tick_params(axis='x', labelsize=30)
ax1.tick_params(axis='y', labelsize=20)
plt.gca().invert_yaxis()
plt.show()
```


![png](legal_questions_analysis_files/legal_questions_analysis_18_0.png)



```python
top_5_states
```




    California      15728
    Texas            9908
    Florida          8779
    New York         6533
    Pennsylvania     5642
    Name: state, dtype: int64



### Distribution of Questions in California and US Virgin Islands


```python
california = legal[legal['state'] == 'California']
cali_category = california.groupby('category').size().reset_index(drop = False)
cali_category.columns = ['category', 'size']

islands = legal[legal['state'] == 'US Virgin Islands']
isl_category = islands.groupby('category').size().reset_index(drop = False)
isl_category.columns = ['category', 'size']

## Pie chart showing distribution of legal categories in California
sizes = cali_category['size']
labels = cali_category['category']
explode = [0.1] * 18
## Pie chart showing distribution of legal categories in US Virgin Islands
sizes_2 = isl_category['size']
labels_2 = isl_category['category']
explode_2 = [0.1] * 11

fig = plt.figure(figsize=(18, 8))
ax = fig.add_subplot(121)
patches, texts, autotexts = ax.pie(sizes, explode = explode, labels=labels,
        autopct='%1.1f%%', shadow=False, startangle=80)
for t in texts:
    t.set_fontsize(14)
for t in autotexts:
    t.set_fontsize(13)
ax.set_title('Distribution of legal categories in California', fontsize = 18)
ax1 = fig.add_subplot(122)
patches, texts, autotexts = ax1.pie(sizes_2, explode = explode_2, labels=labels_2,
        autopct='%1.1f%%', shadow=False, startangle=80)
for t in texts:
    t.set_fontsize(14)
for t in autotexts:
    t.set_fontsize(15)
ax1.set_title('Distribution of legal categories in US Virgin Islands', fontsize = 18)
plt.show()
```


![png](legal_questions_analysis_files/legal_questions_analysis_21_0.png)


### Question 2: Countrywide what are most frequent legal concerns?

According to the plot below, countrywide, most of the legal questions pertain to labor law, follwed by criminal defense and real estate.


```python
## select the top 5 legal categories with the most questions
top_category = legal['category'].value_counts()[0:17]

## make bar chart of the top 5 legal categories
fig = plt.figure(figsize=(14, 4))
ax = fig.add_subplot(121)
top_category.plot(kind = 'barh', ax = ax)
ax.set_xlabel('No. of Questions', fontsize = 20)
ax.set_ylabel('Legal Category', fontsize = 20)
ax.set_title('Top 5 legal categories with most legal questions', fontsize = 20)
ax.tick_params(axis='x', labelsize=15)
ax.tick_params(axis='y', labelsize=15)
plt.gca().invert_yaxis()

```


![png](legal_questions_analysis_files/legal_questions_analysis_23_0.png)



```python
top_category
```




    labor          19149
    criminal       14652
    estate         12943
    wills          11995
    divorce        11199
    tenant         10584
    business        9080
    auto            6905
    debt            5584
    injury          5499
    custody         4286
    bankruptcy      3405
    dui             2661
    insurance       2262
    immigration     2190
    consumer        2035
    medical         1729
    Name: category, dtype: int64



### Question 3: What states will certain types of lawyers be in high demand?

Since most of the questions originated from California, and most are related to labor law, we can expect a higher demand for lawyers in California than other states. We can also expect a high demand for lawyers specializing in labor law in all the states.


```python
## Remove entries with non specific state entries
state_category = legal[legal['state']!= 'All USA'] 
state_category = state_category[state_category['state'] != 'Unspecified State']
```

The first map below shows that after taking the sum of the number of questions in each category over the years, most states have more labor related legal questions than any other legal questions. 

The next three maps show that the major legal concerns of each state has changed over the years. This change in each states legal concerns could be random or as a result of political and economic changes, howevere; there is insufficient information to give a reason this change.

Not shown on the maps are:
- Alaska - mainly labor in 2009, tenant in 2013, labor in 2017, and labor all years combined
- Hawaii - mainly criminal in 2009, custody in 2013, labor in 2017, and criminal all years combined
- Puerto Rico - mainly divorce in 2009, labor in 2013, immigration in 2017, and labor all years combined
- US Virgin Islands - mainly custody in 2013, divorce in 2017, and criminal all years combined. No questions in 2009.

This information could be useful in deciding what states to base various legal specialities. 

### The major legal category that originated from each state


```python
## make dataframe of each state and its major legal category in the complete dataset
state_category_groupby = state_category.groupby(['category','state']).size().unstack()
major_category_per_state = state_category_groupby.idxmax(axis = 0).reset_index(drop = False)
major_category_per_state.columns = ['state','category']
```


```python
## remove rows with null est_date_asked
yearly_state_category = state_category[state_category['est_date_asked'] != '']

yearly_state_category['est_date_asked'] = pd.to_datetime(yearly_state_category['est_date_asked'])
## extract year from estimated date asked
yearly_state_category['year'] = pd.DatetimeIndex(yearly_state_category['est_date_asked']).year.astype(int) 
```


```python
## make dataframe of each state and its major legal category in year 2014
y2014_state_category = yearly_state_category[yearly_state_category['year'] == 2014]
## make dataframe of each state and its major legal category in year 2014
y2014_state_category_groupby = y2014_state_category.groupby(['category','state']).size().unstack()
major_y2014_category_per_state = y2014_state_category_groupby.idxmax(axis = 0).reset_index(drop = False)
major_y2014_category_per_state.columns = ['state','category']
major_y2014_category_per_state
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>state</th>
      <th>category</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Alabama</td>
      <td>wills</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Alaska</td>
      <td>criminal</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Arizona</td>
      <td>wills</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Arkansas</td>
      <td>criminal</td>
    </tr>
    <tr>
      <th>4</th>
      <td>California</td>
      <td>wills</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Colorado</td>
      <td>criminal</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Connecticut</td>
      <td>criminal</td>
    </tr>
    <tr>
      <th>7</th>
      <td>DC</td>
      <td>criminal</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Delaware</td>
      <td>auto</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Florida</td>
      <td>wills</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Georgia</td>
      <td>criminal</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Hawaii</td>
      <td>criminal</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Idaho</td>
      <td>criminal</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Illinois</td>
      <td>wills</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Indiana</td>
      <td>criminal</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Iowa</td>
      <td>criminal</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Kansas</td>
      <td>criminal</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Kentucky</td>
      <td>criminal</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Louisiana</td>
      <td>criminal</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Maine</td>
      <td>criminal</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Maryland</td>
      <td>wills</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Massachusetts</td>
      <td>wills</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Michigan</td>
      <td>criminal</td>
    </tr>
    <tr>
      <th>23</th>
      <td>Minnesota</td>
      <td>criminal</td>
    </tr>
    <tr>
      <th>24</th>
      <td>Mississippi</td>
      <td>criminal</td>
    </tr>
    <tr>
      <th>25</th>
      <td>Missouri</td>
      <td>criminal</td>
    </tr>
    <tr>
      <th>26</th>
      <td>Montana</td>
      <td>criminal</td>
    </tr>
    <tr>
      <th>27</th>
      <td>Nebraska</td>
      <td>criminal</td>
    </tr>
    <tr>
      <th>28</th>
      <td>Nevada</td>
      <td>criminal</td>
    </tr>
    <tr>
      <th>29</th>
      <td>New Hampshire</td>
      <td>auto</td>
    </tr>
    <tr>
      <th>30</th>
      <td>New Jersey</td>
      <td>wills</td>
    </tr>
    <tr>
      <th>31</th>
      <td>New Mexico</td>
      <td>injury</td>
    </tr>
    <tr>
      <th>32</th>
      <td>New York</td>
      <td>wills</td>
    </tr>
    <tr>
      <th>33</th>
      <td>North Carolina</td>
      <td>criminal</td>
    </tr>
    <tr>
      <th>34</th>
      <td>North Dakota</td>
      <td>criminal</td>
    </tr>
    <tr>
      <th>35</th>
      <td>Ohio</td>
      <td>criminal</td>
    </tr>
    <tr>
      <th>36</th>
      <td>Oklahoma</td>
      <td>wills</td>
    </tr>
    <tr>
      <th>37</th>
      <td>Oregon</td>
      <td>wills</td>
    </tr>
    <tr>
      <th>38</th>
      <td>Pennsylvania</td>
      <td>wills</td>
    </tr>
    <tr>
      <th>39</th>
      <td>Puerto Rico</td>
      <td>insurance</td>
    </tr>
    <tr>
      <th>40</th>
      <td>Rhode Island</td>
      <td>auto</td>
    </tr>
    <tr>
      <th>41</th>
      <td>South Carolina</td>
      <td>wills</td>
    </tr>
    <tr>
      <th>42</th>
      <td>South Dakota</td>
      <td>criminal</td>
    </tr>
    <tr>
      <th>43</th>
      <td>Tennessee</td>
      <td>wills</td>
    </tr>
    <tr>
      <th>44</th>
      <td>Texas</td>
      <td>wills</td>
    </tr>
    <tr>
      <th>45</th>
      <td>US Virgin Islands</td>
      <td>auto</td>
    </tr>
    <tr>
      <th>46</th>
      <td>Utah</td>
      <td>criminal</td>
    </tr>
    <tr>
      <th>47</th>
      <td>Vermont</td>
      <td>auto</td>
    </tr>
    <tr>
      <th>48</th>
      <td>Virginia</td>
      <td>criminal</td>
    </tr>
    <tr>
      <th>49</th>
      <td>Washington</td>
      <td>wills</td>
    </tr>
    <tr>
      <th>50</th>
      <td>West Virginia</td>
      <td>wills</td>
    </tr>
    <tr>
      <th>51</th>
      <td>Wisconsin</td>
      <td>criminal</td>
    </tr>
    <tr>
      <th>52</th>
      <td>Wyoming</td>
      <td>criminal</td>
    </tr>
  </tbody>
</table>
</div>




```python
## make dataframe of each state and its major legal category in year 2015
y2015_state_category = yearly_state_category[yearly_state_category['year'] == 2015]
## make dataframe of each state and its major legal category in year 2015
y2015_state_category_groupby = y2015_state_category.groupby(['category','state']).size().unstack()
major_y2015_category_per_state = y2015_state_category_groupby.idxmax(axis = 0).reset_index(drop = False)
major_y2015_category_per_state.columns = ['state','category']
major_y2015_category_per_state
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>state</th>
      <th>category</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Alabama</td>
      <td>wills</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Alaska</td>
      <td>wills</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Arizona</td>
      <td>criminal</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Arkansas</td>
      <td>wills</td>
    </tr>
    <tr>
      <th>4</th>
      <td>California</td>
      <td>wills</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Colorado</td>
      <td>business</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Connecticut</td>
      <td>auto</td>
    </tr>
    <tr>
      <th>7</th>
      <td>DC</td>
      <td>business</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Delaware</td>
      <td>business</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Florida</td>
      <td>wills</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Georgia</td>
      <td>wills</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Hawaii</td>
      <td>criminal</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Idaho</td>
      <td>criminal</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Illinois</td>
      <td>wills</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Indiana</td>
      <td>wills</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Iowa</td>
      <td>criminal</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Kansas</td>
      <td>wills</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Kentucky</td>
      <td>wills</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Louisiana</td>
      <td>wills</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Maine</td>
      <td>wills</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Maryland</td>
      <td>wills</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Massachusetts</td>
      <td>wills</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Michigan</td>
      <td>wills</td>
    </tr>
    <tr>
      <th>23</th>
      <td>Minnesota</td>
      <td>wills</td>
    </tr>
    <tr>
      <th>24</th>
      <td>Mississippi</td>
      <td>wills</td>
    </tr>
    <tr>
      <th>25</th>
      <td>Missouri</td>
      <td>wills</td>
    </tr>
    <tr>
      <th>26</th>
      <td>Montana</td>
      <td>wills</td>
    </tr>
    <tr>
      <th>27</th>
      <td>Nebraska</td>
      <td>wills</td>
    </tr>
    <tr>
      <th>28</th>
      <td>Nevada</td>
      <td>wills</td>
    </tr>
    <tr>
      <th>29</th>
      <td>New Hampshire</td>
      <td>wills</td>
    </tr>
    <tr>
      <th>30</th>
      <td>New Jersey</td>
      <td>wills</td>
    </tr>
    <tr>
      <th>31</th>
      <td>New Mexico</td>
      <td>criminal</td>
    </tr>
    <tr>
      <th>32</th>
      <td>New York</td>
      <td>wills</td>
    </tr>
    <tr>
      <th>33</th>
      <td>North Carolina</td>
      <td>wills</td>
    </tr>
    <tr>
      <th>34</th>
      <td>North Dakota</td>
      <td>business</td>
    </tr>
    <tr>
      <th>35</th>
      <td>Ohio</td>
      <td>wills</td>
    </tr>
    <tr>
      <th>36</th>
      <td>Oklahoma</td>
      <td>wills</td>
    </tr>
    <tr>
      <th>37</th>
      <td>Oregon</td>
      <td>wills</td>
    </tr>
    <tr>
      <th>38</th>
      <td>Pennsylvania</td>
      <td>wills</td>
    </tr>
    <tr>
      <th>39</th>
      <td>Puerto Rico</td>
      <td>business</td>
    </tr>
    <tr>
      <th>40</th>
      <td>Rhode Island</td>
      <td>auto</td>
    </tr>
    <tr>
      <th>41</th>
      <td>South Carolina</td>
      <td>wills</td>
    </tr>
    <tr>
      <th>42</th>
      <td>South Dakota</td>
      <td>wills</td>
    </tr>
    <tr>
      <th>43</th>
      <td>Tennessee</td>
      <td>wills</td>
    </tr>
    <tr>
      <th>44</th>
      <td>Texas</td>
      <td>wills</td>
    </tr>
    <tr>
      <th>45</th>
      <td>US Virgin Islands</td>
      <td>labor</td>
    </tr>
    <tr>
      <th>46</th>
      <td>Utah</td>
      <td>criminal</td>
    </tr>
    <tr>
      <th>47</th>
      <td>Vermont</td>
      <td>wills</td>
    </tr>
    <tr>
      <th>48</th>
      <td>Virginia</td>
      <td>wills</td>
    </tr>
    <tr>
      <th>49</th>
      <td>Washington</td>
      <td>wills</td>
    </tr>
    <tr>
      <th>50</th>
      <td>West Virginia</td>
      <td>wills</td>
    </tr>
    <tr>
      <th>51</th>
      <td>Wisconsin</td>
      <td>wills</td>
    </tr>
    <tr>
      <th>52</th>
      <td>Wyoming</td>
      <td>wills</td>
    </tr>
  </tbody>
</table>
</div>




```python
## make dataframe of each state and its major legal category in year 2016
y2016_state_category = yearly_state_category[yearly_state_category['year'] == 2016]
## make dataframe of each state and its major legal category in year 2016
y2016_state_category_groupby = y2016_state_category.groupby(['category','state']).size().unstack()
major_y2016_category_per_state = y2016_state_category_groupby.idxmax(axis = 0).reset_index(drop = False)
major_y2016_category_per_state.columns = ['state','category']
major_y2016_category_per_state
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>state</th>
      <th>category</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Alabama</td>
      <td>labor</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Alaska</td>
      <td>labor</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Arizona</td>
      <td>labor</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Arkansas</td>
      <td>estate</td>
    </tr>
    <tr>
      <th>4</th>
      <td>California</td>
      <td>labor</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Colorado</td>
      <td>labor</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Connecticut</td>
      <td>labor</td>
    </tr>
    <tr>
      <th>7</th>
      <td>DC</td>
      <td>labor</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Delaware</td>
      <td>labor</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Florida</td>
      <td>labor</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Georgia</td>
      <td>labor</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Hawaii</td>
      <td>labor</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Idaho</td>
      <td>labor</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Illinois</td>
      <td>labor</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Indiana</td>
      <td>labor</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Iowa</td>
      <td>labor</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Kansas</td>
      <td>labor</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Kentucky</td>
      <td>labor</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Louisiana</td>
      <td>labor</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Maine</td>
      <td>labor</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Maryland</td>
      <td>labor</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Massachusetts</td>
      <td>labor</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Michigan</td>
      <td>labor</td>
    </tr>
    <tr>
      <th>23</th>
      <td>Minnesota</td>
      <td>labor</td>
    </tr>
    <tr>
      <th>24</th>
      <td>Mississippi</td>
      <td>labor</td>
    </tr>
    <tr>
      <th>25</th>
      <td>Missouri</td>
      <td>labor</td>
    </tr>
    <tr>
      <th>26</th>
      <td>Montana</td>
      <td>labor</td>
    </tr>
    <tr>
      <th>27</th>
      <td>Nebraska</td>
      <td>labor</td>
    </tr>
    <tr>
      <th>28</th>
      <td>Nevada</td>
      <td>labor</td>
    </tr>
    <tr>
      <th>29</th>
      <td>New Hampshire</td>
      <td>labor</td>
    </tr>
    <tr>
      <th>30</th>
      <td>New Jersey</td>
      <td>labor</td>
    </tr>
    <tr>
      <th>31</th>
      <td>New Mexico</td>
      <td>labor</td>
    </tr>
    <tr>
      <th>32</th>
      <td>New York</td>
      <td>labor</td>
    </tr>
    <tr>
      <th>33</th>
      <td>North Carolina</td>
      <td>labor</td>
    </tr>
    <tr>
      <th>34</th>
      <td>North Dakota</td>
      <td>labor</td>
    </tr>
    <tr>
      <th>35</th>
      <td>Ohio</td>
      <td>labor</td>
    </tr>
    <tr>
      <th>36</th>
      <td>Oklahoma</td>
      <td>labor</td>
    </tr>
    <tr>
      <th>37</th>
      <td>Oregon</td>
      <td>labor</td>
    </tr>
    <tr>
      <th>38</th>
      <td>Pennsylvania</td>
      <td>labor</td>
    </tr>
    <tr>
      <th>39</th>
      <td>Puerto Rico</td>
      <td>labor</td>
    </tr>
    <tr>
      <th>40</th>
      <td>Rhode Island</td>
      <td>labor</td>
    </tr>
    <tr>
      <th>41</th>
      <td>South Carolina</td>
      <td>labor</td>
    </tr>
    <tr>
      <th>42</th>
      <td>South Dakota</td>
      <td>labor</td>
    </tr>
    <tr>
      <th>43</th>
      <td>Tennessee</td>
      <td>labor</td>
    </tr>
    <tr>
      <th>44</th>
      <td>Texas</td>
      <td>labor</td>
    </tr>
    <tr>
      <th>45</th>
      <td>US Virgin Islands</td>
      <td>wills</td>
    </tr>
    <tr>
      <th>46</th>
      <td>Utah</td>
      <td>labor</td>
    </tr>
    <tr>
      <th>47</th>
      <td>Vermont</td>
      <td>labor</td>
    </tr>
    <tr>
      <th>48</th>
      <td>Virginia</td>
      <td>labor</td>
    </tr>
    <tr>
      <th>49</th>
      <td>Washington</td>
      <td>labor</td>
    </tr>
    <tr>
      <th>50</th>
      <td>West Virginia</td>
      <td>labor</td>
    </tr>
    <tr>
      <th>51</th>
      <td>Wisconsin</td>
      <td>labor</td>
    </tr>
    <tr>
      <th>52</th>
      <td>Wyoming</td>
      <td>business</td>
    </tr>
  </tbody>
</table>
</div>




```python
## display map picture from local disk
display(Image(filename="/Users/oofoegbu/us_map.png", width = 600, height = 600))
display(Image(filename="/Users/oofoegbu/us_map_2009.png", width = 600, height = 600))
display(Image(filename="/Users/oofoegbu/us_map_2013.png", width = 600, height = 600))
display(Image(filename="/Users/oofoegbu/us_map_2017.png", width = 600, height = 600))
```


![png](legal_questions_analysis_files/legal_questions_analysis_34_0.png)



![png](legal_questions_analysis_files/legal_questions_analysis_34_1.png)



![png](legal_questions_analysis_files/legal_questions_analysis_34_2.png)



![png](legal_questions_analysis_files/legal_questions_analysis_34_3.png)


The pie chart below shows the state where most questions from each legal category originated. As shown, a majority of the questions about custody and traffic where from Texas, while the a majority of questions in the other legal categories where from California. 


```python
major_state_per_category = state_category_groupby.idxmax(axis = 1).reset_index(drop = False)
major_state_per_category.columns = ['category','state']
major_state_per_category['no'] = [1]*18
major_state_per_category['colors'] = ['blue', 'blue', 'blue', 'blue', 'blue', 'red', 'blue', 'blue', 'blue', 'blue', 'blue','blue','blue','blue','blue','blue',
              'red', 'blue']

## Pie chart showing distribution of legal categories in California
sizes = major_state_per_category['no']
labels = major_state_per_category['category']
explode = [0] * 18
colors = major_state_per_category['colors']

fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111)
patches, texts = ax.pie(sizes, explode = explode, labels=labels,
         shadow=False, startangle=80, colors = colors)
for t in texts:
    t.set_fontsize(14)

ax.set_title('Major state from which each legal category originated', fontsize = 18)
plt.show()

legend = pd.DataFrame(columns = ['col', 'state'])
legend.loc[0] = ['blue', 'California']
legend.loc[1] = ['red', 'Texas']
print(legend)
```


![png](legal_questions_analysis_files/legal_questions_analysis_36_0.png)


        col       state
    0  blue  California
    1   red       Texas


### Question 4: Are there certain periods of the year when particular legal problems are prevalent?

On the website, the specific date the questions where answered were not recorded. Rather the date answered was recorded as years, months, and days ago. Due to the non-specific date recorded, analysis of how the category of legal questions changed over months, day of the week, and weekday/weekend  the month was would be biased to the month I collected the data. The data was retrieved in July. That would explain the high number of questions recorded in July, as shown in the plot above. The most reliable information about the periods would be from the year. 

The months, days and weekday/weekend are encoded as below:
- Months: January - 1, December - 12
- Day of the week: Monday - 0, Sunday - 6
- Weekday or Weekend: Weekday - 0, Weekend - 1



```python
## subset relevant columns
category_date = legal[['category','state', 'est_date_asked']]
## remove rows with null est_date_asked
periods = category_date[category_date['est_date_asked'] != '']

periods['est_date_asked'] = pd.to_datetime(periods['est_date_asked'])

periods['year'] = pd.DatetimeIndex(periods['est_date_asked']).year.astype(int) 
periods['month'] = pd.DatetimeIndex(periods['est_date_asked']).month.astype(int)  
periods['day_of_week'] = pd.DatetimeIndex(periods['est_date_asked']).dayofweek.astype(int) 
periods['weekday'] = ((pd.DatetimeIndex(periods['est_date_asked']).dayofweek) // 5 == 1).astype(int)        
```


```python
## Plot bar charts shoqing no. of questions asked in each year, month, day of the week, weekend/weekday
fig = plt.figure(figsize=(12, 14))
ax = fig.add_subplot(221)
periods['year'].value_counts().sort_index().plot(kind = 'bar', ax = ax)
ax.set_xlabel('Est. Year Asked')
ax.set_ylabel('No. of Questions')
ax.set_title('No. of Questions Asked Each Year')
ax1 = fig.add_subplot(222)
periods['month'].value_counts().sort_index().plot(kind = 'bar', ax = ax1)
ax1.set_xlabel('Est. Month Asked')
ax1.set_ylabel('No. of Questions')
ax1.set_title('No. of Questions Asked Each Month')
ax2 = fig.add_subplot(223)
periods['day_of_week'].value_counts().sort_index().plot(kind = 'bar', ax = ax2)
ax2.set_xlabel('Est. Day of the Week Asked')
ax2.set_ylabel('No. of Questions')
ax2.set_title('No. of Questions Asked Each Day of the Week')
ax3 = fig.add_subplot(224)
periods['weekday'].value_counts().plot(kind = 'bar', ax = ax3)
ax3.set_xlabel('Est. Weekday or Weekend')
ax3.set_ylabel('No. of Questions')
ax3.set_title('No. of Questions Asked on Weekday and Weekends')
plt.show()
```


![png](legal_questions_analysis_files/legal_questions_analysis_40_0.png)


### Plots showing the distribution of questions from each legal category for the various years.


```python
## Subset based on estimated year asked
year_2009 = periods.loc[periods['year'] == 2009]
year_2010 = periods.loc[periods['year'] == 2010]
year_2011 = periods.loc[periods['year'] == 2011]
year_2012 = periods.loc[periods['year'] == 2012]
year_2013 = periods.loc[periods['year'] == 2013]
year_2014 = periods.loc[periods['year'] == 2014]
year_2015 = periods.loc[periods['year'] == 2015]
year_2016 = periods.loc[periods['year'] == 2016]
year_2017 = periods.loc[periods['year'] == 2017]

## For each year, group data by category, and count the number of questions in each group
year_2009_groupby = year_2009.groupby(['category','year']).size().reset_index(drop=False)
year_2009_groupby.columns = ['category','year','size']
year_2010_groupby = year_2010.groupby(['category','year']).size().reset_index(drop=False)
year_2010_groupby.columns = ['category','year','size']
year_2011_groupby = year_2011.groupby(['category','year']).size().reset_index(drop=False)
year_2011_groupby.columns = ['category','year','size']
year_2012_groupby = year_2012.groupby(['category','year']).size().reset_index(drop=False)
year_2012_groupby.columns = ['category','year','size']
year_2013_groupby = year_2013.groupby(['category','year']).size().reset_index(drop=False)
year_2013_groupby.columns = ['category','year','size']
year_2014_groupby = year_2014.groupby(['category','year']).size().reset_index(drop=False)
year_2014_groupby.columns = ['category','year','size']
year_2015_groupby = year_2015.groupby(['category','year']).size().reset_index(drop=False)
year_2015_groupby.columns = ['category','year','size']
year_2016_groupby = year_2016.groupby(['category','year']).size().reset_index(drop=False)
year_2016_groupby.columns = ['category','year','size']
year_2017_groupby = year_2017.groupby(['category','year']).size().reset_index(drop=False)
year_2017_groupby.columns = ['category','year','size']
```


```python
## percent of questions each year from the different legal categories
periods_groupby = periods.groupby(['category','year']).size().groupby(level = [1]).apply(lambda x: x/x.sum()).unstack()
```


```python
year_2009_frac = periods_groupby.iloc[:,0].reset_index(drop = False)
year_2009_frac.columns = ['category','percent']
year_2009_frac = year_2009_frac.fillna(0) ## fill NA with zero
year_2010_frac = periods_groupby.iloc[:,1].reset_index(drop = False)
year_2010_frac.columns = ['category','percent']
year_2010_frac = year_2010_frac.fillna(0)
year_2011_frac = periods_groupby.iloc[:,2].reset_index(drop = False)
year_2011_frac.columns = ['category','percent']
year_2011_frac = year_2011_frac.fillna(0)
year_2012_frac = periods_groupby.iloc[:,3].reset_index(drop = False)
year_2012_frac.columns = ['category','percent']
year_2012_frac = year_2012_frac.fillna(0)
year_2013_frac = periods_groupby.iloc[:,4].reset_index(drop = False)
year_2013_frac.columns = ['category','percent']
year_2013_frac = year_2013_frac.fillna(0)
year_2014_frac = periods_groupby.iloc[:,5].reset_index(drop = False)
year_2014_frac.columns = ['category','percent']
year_2014_frac = year_2014_frac.fillna(0)
year_2015_frac = periods_groupby.iloc[:,6].reset_index(drop = False)
year_2015_frac.columns = ['category','percent']
year_2015_frac = year_2015_frac.fillna(0)
year_2016_frac = periods_groupby.iloc[:,7].reset_index(drop = False)
year_2016_frac.columns = ['category','percent']
year_2016_frac = year_2016_frac.fillna(0)
year_2017_frac = periods_groupby.iloc[:,8].reset_index(drop = False)
year_2017_frac.columns = ['category','percent']
year_2017_frac = year_2017_frac.fillna(0)

## Plot the distribution of questions from each legal category in the different years
fig = plt.figure(figsize=(35, 80))
ax = fig.add_subplot(311)
ax.plot(year_2009_frac.index, year_2009_frac['percent'],marker = 'o', color = 'r', label = 'year 2009')
ax.plot(year_2010_frac.index, year_2010_frac['percent'],marker = 'o', color = 'b', label = 'year 2010')
ax.plot(year_2011_frac.index, year_2011_frac['percent'],marker = 'o', color = 'g', label = 'year 2011')
ax.legend(fontsize = 30)
ax.set_ylim([-0.1,0.4])
ax.set_xticks(year_2009_frac.index) # choose which x locations to have ticks
ax.set_xticklabels(year_2009_frac['category'], rotation = 90) # set the labels to display at those ticks
ax.tick_params(axis='x', labelsize=30)
ax.tick_params(axis='y', labelsize=30)
ax.set_xlabel('Legal Category', fontsize = 40)
ax.set_ylabel('Fraction of total questions', fontsize = 40)
ax1 = fig.add_subplot(312)
ax1.plot(year_2012_frac.index, year_2012_frac['percent'],marker = 'o', color = 'r', label = 'year 2012')
ax1.plot(year_2013_frac.index, year_2013_frac['percent'],marker = 'o', color = 'b', label = 'year 2013')
ax1.plot(year_2014_frac.index, year_2014_frac['percent'],marker = 'o', color = 'g', label = 'year 2014')
ax1.legend(fontsize = 30)
ax1.set_ylim([-0.1,0.4])
ax1.set_xticks(year_2012_frac.index) # choose which x locations to have ticks
ax1.set_xticklabels(year_2012_frac['category'], rotation = 90) # set the labels to display at those ticks
ax1.tick_params(axis='x', labelsize=30)
ax1.tick_params(axis='y', labelsize=30)
ax1.set_xlabel('Legal Category', fontsize = 40)
ax1.set_ylabel('Fraction of total questions', fontsize = 40)
ax2 = fig.add_subplot(313)
ax2.plot(year_2015_frac.index, year_2015_frac['percent'],marker = 'o', color = 'r', label = 'year 2015')
ax2.plot(year_2016_frac.index, year_2016_frac['percent'],marker = 'o', color = 'b', label = 'year 2016')
ax2.plot(year_2017_frac.index, year_2017_frac['percent'],marker = 'o', color = 'g', label = 'year 2017')
ax2.legend(fontsize = 30)
ax2.set_ylim([-0.1,0.4])
ax2.set_xticks(year_2015_frac.index) # choose which x locations to have ticks
ax2.set_xticklabels(year_2015_frac['category'], rotation = 90) # set the labels to display at those ticks
ax2.tick_params(axis='x', labelsize=30)
ax2.tick_params(axis='y', labelsize=30)
ax2.set_xlabel('Legal Category', fontsize = 40)
ax2.set_ylabel('Fraction of total questions', fontsize = 40)

plt.show()
```


![png](legal_questions_analysis_files/legal_questions_analysis_44_0.png)


The plots above show the percent of questions in each year that are from each legal category. For example, in 2009 about 25% of the questions related to criminal defense law. We can see from the plots that the percent distribution of questions from different legal categories varied randomly over the years.

### Question 5:  What are the number of unique attorneys who answered questions in the various legal forums? 

From the table below, labor and tenant law categories have a high ratio of questions to attorneys. It will be beneficial to increase the number of free advice contributing lawyers with specialities in labor and tenant law, in order to have a broader range of opinions.


```python
auto = legal[legal['category'] == 'auto']['attorney'] ## list of the attorneys that contributed to category
len_auto_attorneys = len(auto.value_counts()) ## no. of unique attorneys that answered questions from category
no_auto = len(auto) ## no. of questions in category

bankruptcy = legal[legal['category'] == 'bankruptcy']['attorney']
len_bankruptcy_attorneys = len(bankruptcy.value_counts()) 
no_bankruptcy = len(bankruptcy)

business = legal[legal['category'] == 'business']['attorney']
len_business_attorneys = len(business.value_counts())
no_business = len(business)

consumer = legal[legal['category'] == 'consumer']['attorney']
len_consumer_attorneys = len(consumer.value_counts())
no_consumer = len(consumer)

criminal = legal[legal['category'] == 'criminal']['attorney']
len_criminal_attorneys = len(criminal.value_counts())
no_criminal = len(criminal)

custody = legal[legal['category'] == 'custody']['attorney']
len_custody_attorneys = len(custody.value_counts())
no_custody = len(custody)

debt = legal[legal['category'] == 'debt']['attorney']
len_debt_attorneys = len(debt.value_counts())
no_debt = len(debt)

divorce = legal[legal['category'] == 'divorce']['attorney']
len_divorce_attorneys = len(divorce.value_counts())
no_divorce = len(divorce)

dui = legal[legal['category'] == 'dui']['attorney']
len_dui_attorneys = len(dui.value_counts())
no_dui = len(dui)

estate = legal[legal['category'] == 'estate']['attorney']
len_estate_attorneys = len(estate.value_counts())
no_estate = len(estate)

immigration = legal[legal['category'] == 'immigration']['attorney']
len_immigration_attorneys = len(immigration.value_counts())
no_immigration = len(immigration)

injury = legal[legal['category'] == 'injury']['attorney']
len_injury_attorneys = len(injury.value_counts())
no_injury = len(injury)

insurance = legal[legal['category'] == 'insurance']['attorney']
len_insurance_attorneys = len(insurance.value_counts())
no_insurance = len(insurance)

labor = legal[legal['category'] == 'labor']['attorney']
len_labor_attorneys = len(labor.value_counts())
no_labor = len(labor)

medical = legal[legal['category'] == 'medical']['attorney']
len_medical_attorneys = len(medical.value_counts())
no_medical = len(medical)

tenant = legal[legal['category'] == 'tenant']['attorney']
len_tenant_attorneys = len(tenant.value_counts())
no_tenant = len(tenant)

traffic = legal[legal['category'] == 'traffic']['attorney']
len_traffic_attorneys = len(traffic.value_counts())
no_traffic = len(traffic)

wills = legal[legal['category'] == 'wills']['attorney']
len_wills_attorneys = len(wills.value_counts())
no_wills = len(wills)


## Dataframe of the no. of unique attorneys, no. of questions and ratio of questions to attorney in each category
attorneys_per_category = pd.DataFrame(columns = ['category', 'no. of unique attorneys', 'no. of questions'])
attorneys_per_category.loc[0] = ['auto', len_auto_attorneys, no_auto]
attorneys_per_category.loc[1] = ['bankruptcy', len_bankruptcy_attorneys, no_bankruptcy]
attorneys_per_category.loc[2] = ['business', len_business_attorneys, no_business]
attorneys_per_category.loc[3] = ['consumer', len_consumer_attorneys, no_consumer]
attorneys_per_category.loc[4] = ['criminal', len_criminal_attorneys, no_criminal]
attorneys_per_category.loc[5] = ['custody', len_custody_attorneys, no_custody]
attorneys_per_category.loc[6] = ['debt', len_debt_attorneys, no_debt]
attorneys_per_category.loc[7] = ['divorce', len_divorce_attorneys, no_divorce]
attorneys_per_category.loc[8] = ['dui', len_dui_attorneys, no_dui]
attorneys_per_category.loc[9] = ['estate', len_estate_attorneys, no_estate]
attorneys_per_category.loc[10] = ['immigration', len_immigration_attorneys, no_immigration]
attorneys_per_category.loc[11] = ['injury', len_injury_attorneys, no_injury]
attorneys_per_category.loc[12] = ['insurance', len_insurance_attorneys, no_insurance]
attorneys_per_category.loc[13] = ['labor', len_labor_attorneys, no_labor]
attorneys_per_category.loc[14] = ['medical', len_medical_attorneys, no_medical]
attorneys_per_category.loc[15] = ['tenant', len_tenant_attorneys, no_tenant]
attorneys_per_category.loc[16] = ['traffic', len_traffic_attorneys, no_traffic]
attorneys_per_category.loc[17] = ['wills', len_wills_attorneys, no_wills]

attorneys_per_category['questions per attorney'] = round(attorneys_per_category['no. of questions']/attorneys_per_category['no. of unique attorneys'], 1)
```


```python
## bar chart showing the ratio of questions per attorney in each category
fig = plt.figure(figsize=(30, 14))
ax = fig.add_subplot(221)
attorneys_per_category['questions per attorney'].plot(kind = 'bar', ax = ax)
ax.set_xlabel('Legal Category', fontsize = 15)
ax.set_ylabel('Ratio of questions per attorney', fontsize = 15)
ax.set_xticklabels(attorneys_per_category['category'], rotation = 90)
plt.show()
```


![png](legal_questions_analysis_files/legal_questions_analysis_49_0.png)


### Question 6:  For each of the legal forums what attorney/firm answers most of the questions?

This question is to identify the named attorneys that answered the most questions from each legal category. This information could help the firm identify experts in the various legal categories, which may come in handy if the firm is seeking to increase its manpower.  

#### Table below shows name of attorney who answered the most questions in each category


```python
## Select rows with named attorneys
rows_named_attorney = legal[legal['attorney'].str.contains('^FreeAdvice')].index.tolist()
named_attorney = legal.drop(rows_named_attorney)
## Which named attorney answered the maximum no. of questions in each category
attorney = named_attorney.groupby(['category', 'attorney']).size().unstack()
max_attorney_category = attorney.idxmax(axis = 1).reset_index(drop = False)
max_attorney_category.columns = ['category', 'attorney']
print(max_attorney_category)

```

           category                                           attorney
    0          auto  SJZ,  Member, New York Bar / FreeAdvice Contri...
    1    bankruptcy  M.D.,  Member, California And New York Bar / F...
    2      business  SJZ,  Member, New York Bar / FreeAdvice Contri...
    3      consumer  SJZ,  Member, New York Bar / FreeAdvice Contri...
    4      criminal  M.D.,  Member, California And New York Bar / F...
    5       custody  M.T.G.,  Member, New York Bar / FreeAdvice Con...
    6          debt  SJZ,  Member, New York Bar / FreeAdvice Contri...
    7       divorce  M.T.G.,  Member, New York Bar / FreeAdvice Con...
    8           dui  M.D.,  Member, California And New York Bar / F...
    9        estate  SJZ,  Member, New York Bar / FreeAdvice Contri...
    10  immigration  SB Member California Bar / FreeAdvice Contribu...
    11       injury  SJZ,  Member, New York Bar / FreeAdvice Contri...
    12    insurance  SJZ,  Member, New York Bar / FreeAdvice Contri...
    13        labor  SJZ,  Member, New York Bar / FreeAdvice Contri...
    14      medical  SJZ,  Member, New York Bar / FreeAdvice Contri...
    15       tenant  SJZ,  Member, New York Bar / FreeAdvice Contri...
    16      traffic  SJZ,  Member, New York Bar / FreeAdvice Contri...
    17        wills  M.T.G.,  Member, New York Bar / FreeAdvice Con...


We can conclude from the table that SJZ answers most of the questions in most of the legal categories.

### Question 7: For each of the legal forums what are the major questions?

The goal is to highlight one major concern of users from each of the legal categories, this could help the firm tailor its practice to meet the needs of people.

To do this, I found the largest group of similar questions in each category. Using the most common words in those questions, I was able to determine the most common topic in each category.

The length of the largest similar group, a sample question from each group as well as the top 7 most common words in the groups are provided in the appendix.

The common topics table below shows the topic of the largest group of similar questions in each category.

### Common Topics Table


```python
## Dataframe of the common topic in each legal category
common_topics = pd.DataFrame(columns = ['Legal Category','Topic'])

common_topics.loc[0] = ['Auto', 'Car accident involving rental cars']
common_topics.loc[1] = ['Bankruptcy', 'Involves filing Chapter 7 or 13']
common_topics.loc[2] = ['Business', 'Business involing car']
common_topics.loc[3] = ['Consumer', 'Involves titles']
common_topics.loc[4] = ['Criminal', 'Sex involving minors']
common_topics.loc[5] = ['Custody', 'Involves the fathers name on birth certificate']
common_topics.loc[6] = ['Debt', 'Debt involving car loan']
common_topics.loc[7] = ['Divorce', 'Cars in divorce']
common_topics.loc[8] = ['Dui', 'Miranda rights during arrest']
common_topics.loc[9] = ['Estate', 'Fence around property']
common_topics.loc[10] = ['Immigration', 'Involving name change and green card']
common_topics.loc[11] = ['Injury', 'Injury involving dogs']
common_topics.loc[12] = ['Insurance', 'Beneficiary of life insurance']
common_topics.loc[13] = ['Labor', 'Cases involving vacation time']
common_topics.loc[14] = ['Medical', 'Medical case involving knee replacement']
common_topics.loc[15] = ['Tenant', 'Late rental fees']
common_topics.loc[16] = ['Traffic', 'Suspended license']
common_topics.loc[17] = ['Wills', 'Beneficiaries of life insurance policy']

print(common_topics)   
```

       Legal Category                                           Topic
    0            Auto              Car accident involving rental cars
    1      Bankruptcy                 Involves filing Chapter 7 or 13
    2        Business                           Business involing car
    3        Consumer                                 Involves titles
    4        Criminal                            Sex involving minors
    5         Custody  Involves the fathers name on birth certificate
    6            Debt                         Debt involving car loan
    7         Divorce                                 Cars in divorce
    8             Dui                    Miranda rights during arrest
    9          Estate                           Fence around property
    10    Immigration            Involving name change and green card
    11         Injury                           Injury involving dogs
    12      Insurance                   Beneficiary of life insurance
    13          Labor                   Cases involving vacation time
    14        Medical         Medical case involving knee replacement
    15         Tenant                                Late rental fees
    16        Traffic                               Suspended license
    17          Wills          Beneficiaries of life insurance policy


### Question 8: What types of questions are left unanswered?

There are only 9 unanswered out of over 120,000 questions in the dataset. This shows that most of the questions on this website get answered. Due to the small number of unanswered questions it is impossible to make any correlation involving the types of unanswered questions.

I checked if there where other questions that where similar to the unanswered questions, and found two questions similar to one of the unanswered questions in the estate category. So, the individual who asked the unanswered questions can get answers from the similar questions. The similar questions can be seen in the appendix.

#### Number of unanswered questions in the various legal categories


```python
## No. of unanswered questions 
unans_group = unans.groupby(['category']).size().reset_index()
unans_group.columns = ['category', 'no. of unanswered questions']
print(unans_group)
```

       category  no. of unanswered questions
    0      auto                            1
    1  business                            1
    2  criminal                            1
    3   divorce                            2
    4       dui                            1
    5    estate                            2
    6     wills                            1


## RECOMMENDATIONS

Based on the above analysis, these are my recommendations to the hypothetical law firm:
- They should consider increasing their workforce in states like California and Texas, as these are the states where most of the legal questions originated.
- Also, a large fraction of the workforce should consist of attorneys with expertise in labor law.
- The data shows that users of this website could benefit from more attorneys contributing to the legal and tenant forums. If the firm had some of its attorneys registered as contributors, these could serve as free advertisement and help bring in more cases for the firm.
- Finally if the law firm is considering an addition to its workforce, they should consider reaching out to SJZ, as this attorney appears to be well-versed in a wide array of legal areas.

## APPENDIX

### Largest group of similar questions in each legal category


```python
## Subsets of the different legal categories
auto_data = legal[legal['category'] == 'auto'].reset_index()
bankruptcy_data = legal[legal['category'] == 'bankruptcy'].reset_index()
business_data = legal[legal['category'] == 'business'].reset_index()
consumer_data = legal[legal['category'] == 'consumer'].reset_index()
criminal_data = legal[legal['category'] == 'criminal'].reset_index()
custody_data = legal[legal['category'] == 'custody'].reset_index()
debt_data = legal[legal['category'] == 'debt'].reset_index()
divorce_data = legal[legal['category'] == 'divorce'].reset_index()
dui_data = legal[legal['category'] == 'dui'].reset_index()
estate_data = legal[legal['category'] == 'estate'].reset_index()
immigration_data = legal[legal['category'] == 'immigration'].reset_index()
injury_data = legal[legal['category'] == 'injury'].reset_index()
insurance_data = legal[legal['category'] == 'insurance'].reset_index()
labor_data = legal[legal['category'] == 'labor'].reset_index()
medical_data = legal[legal['category'] == 'medical'].reset_index()
tenant_data = legal[legal['category'] == 'tenant'].reset_index()
traffic_data = legal[legal['category'] == 'traffic'].reset_index()
wills_data = legal[legal['category'] == 'wills'].reset_index()
```


```python
def similarity_function(data):
    doc = data['title and question'].tolist()
    gen_docs = [[w.lower() for w in word_tokenize(text)] 
            for text in doc]
    
    stop_words = set(stopwords.words('english'))
 
    filtered_sentence = [[] for i in range(len(gen_docs))]

    for i in range(len(gen_docs)):
        for w in gen_docs[i]:
            if w not in stop_words:
                filtered_sentence[i].append(w)
 
    dictionary = gensim.corpora.Dictionary(filtered_sentence)
    corpus = [dictionary.doc2bow(sent) for sent in filtered_sentence]
    tf_idf = gensim.models.TfidfModel(corpus)
    sims = gensim.similarities.Similarity('/Users/oofoegbu/',tf_idf[corpus],num_features=len(dictionary))
    similarity = pd.DataFrame(columns = ['similar','index'])

    for i in range(0,len(doc)):
        query_doc = [w.lower() for w in word_tokenize(doc[i])]
        query_sentence = [w for w in query_doc if not w in stop_words]
        query_doc_bow = dictionary.doc2bow(query_sentence)
        query_doc_tf_idf = tf_idf[query_doc_bow]
        sim = sims[query_doc_tf_idf]
        similar = np.where((sim >= 0.3))[0]
        similarity.loc[i] = [len(similar),similar]
    
    similar_questions = similarity['similar'].value_counts().reset_index()
    similar_questions.columns = ['no. of similar questions', 'index of similar questions']
   
    return similarity

```


```python
auto_sim_group = similarity_function(auto_data) 
bankruptcy_sim_group = similarity_function(bankruptcy_data)
business_sim_group = similarity_function(business_data)
consumer_sim_group = similarity_function(consumer_data)
criminal_sim_group = similarity_function(criminal_data)
custody_sim_group = similarity_function(custody_data)
debt_sim_group = similarity_function(debt_data)
divorce_sim_group = similarity_function(divorce_data)
dui_sim_group = similarity_function(dui_data)
estate_sim_group = similarity_function(estate_data)
immigration_sim_group = similarity_function(immigration_data)
injury_sim_group = similarity_function(injury_data)
insurance_sim_group = similarity_function(insurance_data)
labor_sim_group = similarity_function(labor_data)
medical_sim_group = similarity_function(medical_data)
tenant_sim_group = similarity_function(tenant_data)
traffic_sim_group = similarity_function(traffic_data)
wills_sim_group = similarity_function(wills_data)
```


```python
top_auto_group = auto_sim_group['similar'].value_counts().sort_index(ascending = False)[0:1,]
top_auto_group = top_auto_group.reset_index()
top_auto_group.columns = ['no. of similar questions','no. of groups this size']
top_bankruptcy_group = bankruptcy_sim_group['similar'].value_counts().sort_index(ascending = False)[0:1,]
top_bankruptcy_group = top_bankruptcy_group.reset_index()
top_bankruptcy_group.columns = ['no. of similar questions','no. of groups this size']
top_business_group = business_sim_group['similar'].value_counts().sort_index(ascending = False)[0:1,]
top_business_group = top_business_group.reset_index()
top_business_group.columns = ['no. of similar questions','no. of groups this size']
top_consumer_group = consumer_sim_group['similar'].value_counts().sort_index(ascending = False)[0:1,]
top_consumer_group = top_consumer_group.reset_index()
top_consumer_group.columns = ['no. of similar questions','no. of groups this size']
top_criminal_group = criminal_sim_group['similar'].value_counts().sort_index(ascending = False)[0:1,]
top_criminal_group = top_criminal_group.reset_index()
top_criminal_group.columns = ['no. of similar questions','no. of groups this size']
top_custody_group = custody_sim_group['similar'].value_counts().sort_index(ascending = False)[0:1,]
top_custody_group = top_custody_group.reset_index()
top_custody_group.columns = ['no. of similar questions','no. of groups this size']
top_debt_group = debt_sim_group['similar'].value_counts().sort_index(ascending = False)[0:1,]
top_debt_group = top_debt_group.reset_index()
top_debt_group.columns = ['no. of similar questions','no. of groups this size']
top_divorce_group = divorce_sim_group['similar'].value_counts().sort_index(ascending = False)[0:1,]
top_divorce_group = top_divorce_group.reset_index()
top_divorce_group.columns = ['no. of similar questions','no. of groups this size']
top_dui_group = dui_sim_group['similar'].value_counts().sort_index(ascending = False)[0:1,]
top_dui_group = top_dui_group.reset_index()
top_dui_group.columns = ['no. of similar questions','no. of groups this size']
top_estate_group = estate_sim_group['similar'].value_counts().sort_index(ascending = False)[0:1,]
top_estate_group = top_estate_group.reset_index()
top_estate_group.columns = ['no. of similar questions','no. of groups this size']
top_immigration_group = immigration_sim_group['similar'].value_counts().sort_index(ascending = False)[0:1,]
top_immigration_group = top_immigration_group.reset_index()
top_immigration_group.columns = ['no. of similar questions','no. of groups this size']
top_injury_group = injury_sim_group['similar'].value_counts().sort_index(ascending = False)[0:1,]
top_injury_group = top_injury_group.reset_index()
top_injury_group.columns = ['no. of similar questions','no. of groups this size']
top_insurance_group = insurance_sim_group['similar'].value_counts().sort_index(ascending = False)[0:1,]
top_insurance_group = top_insurance_group.reset_index()
top_insurance_group.columns = ['no. of similar questions','no. of groups this size']
top_labor_group = labor_sim_group['similar'].value_counts().sort_index(ascending = False)[0:1,]
top_labor_group = top_labor_group.reset_index()
top_labor_group.columns = ['no. of similar questions','no. of groups this size']
top_medical_group = medical_sim_group['similar'].value_counts().sort_index(ascending = False)[0:1,]
top_medical_group = top_medical_group.reset_index()
top_medical_group.columns = ['no. of similar questions','no. of groups this size']
top_tenant_group = tenant_sim_group['similar'].value_counts().sort_index(ascending = False)[0:1,]
top_tenant_group = top_tenant_group.reset_index()
top_tenant_group.columns = ['no. of similar questions','no. of groups this size']
top_traffic_group = traffic_sim_group['similar'].value_counts().sort_index(ascending = False)[0:1,]
top_traffic_group = top_traffic_group.reset_index()
top_traffic_group.columns = ['no. of similar questions','no. of groups this size']
top_wills_group = wills_sim_group['similar'].value_counts().sort_index(ascending = False)[0:1,]
top_wills_group = top_wills_group.reset_index()
top_wills_group.columns = ['no. of similar questions','no. of groups this size']
```


```python
def sim_index_function(sim_group, top_group):
    sim_questions_list = sim_group[sim_group['similar'] == top_group['no. of similar questions'].iloc[0]]
    sim_questions_list = sim_questions_list.reset_index()
    sim_questions_list.columns = ['top group index', 'no. of similar questions', 'doc index']
    
    top_index = sim_group['index'].iloc[sim_questions_list['top group index'].iloc[0]]
    
    return top_index
```


```python
top_auto_index = sim_index_function(auto_sim_group, top_auto_group)
top_bankruptcy_index = sim_index_function(bankruptcy_sim_group, top_bankruptcy_group)
top_business_index = sim_index_function(business_sim_group, top_business_group)
top_consumer_index = sim_index_function(consumer_sim_group, top_consumer_group)
top_criminal_index = sim_index_function(criminal_sim_group, top_criminal_group)
top_custody_index = sim_index_function(custody_sim_group, top_custody_group)
top_debt_index = sim_index_function(debt_sim_group, top_debt_group)
top_divorce_index = sim_index_function(divorce_sim_group, top_divorce_group)
top_dui_index = sim_index_function(dui_sim_group, top_dui_group)
top_estate_index = sim_index_function(estate_sim_group, top_estate_group)
top_immigration_index = sim_index_function(immigration_sim_group, top_immigration_group)
top_injury_index = sim_index_function(injury_sim_group, top_injury_group)
top_insurance_index = sim_index_function(insurance_sim_group, top_insurance_group)
top_labor_index = sim_index_function(labor_sim_group, top_labor_group)
top_medical_index = sim_index_function(medical_sim_group, top_medical_group)
top_tenant_index = sim_index_function(tenant_sim_group, top_tenant_group)
top_traffic_index = sim_index_function(traffic_sim_group, top_traffic_group)
top_wills_index = sim_index_function(wills_sim_group, top_wills_group)
```


```python
def topic_function(data, top_index):
    top_questions = data.iloc[top_index]
    top_doc = top_questions['title and question'].tolist()
    
    docs = [[w.lower() for w in word_tokenize(text)]
           for text in top_doc]
    
    ## Remove stopwords
    stop_words = set(stopwords.words('english'))
    
    filtered_sentence = [[] for i in range(len(docs))]
    
    for i in range(len(docs)):
        for w in docs[i]:
            if w not in stop_words:
                filtered_sentence[i].append(w)
    
    ## Remove punctuations 
    punct = set(string.punctuation)
    
    filtered_punct = [[] for i in range(len(filtered_sentence))]
    
    for m in range(len(filtered_sentence)):
        for w in filtered_sentence[m]:
            if w not in punct:
                filtered_punct[m].append(w)
    
    ## Word frequency count
    words = sum(filtered_punct,[])
    
    counts = Counter(words)
    top_words = sorted(counts, key=counts.get, reverse=True)[0:7]
    
    return top_words   
```

#### Auto Category


```python
## No. of questions in largest group of similar questions from auto category
print('No. of similar questions in largest auto group  = %s' % top_auto_group['no. of similar questions'].iloc[0])
print()
## Sample question from this group of similar questions from auto category
print('Sample Question: %s'  % auto_data['title and question'].iloc[top_auto_index[4]])
print()
## Most common words from group of similar questions in auto category 
print('Most common words: %s' % topic_function(auto_data, top_auto_index))
```

    No. of similar questions in largest auto group  = 68.0
    
    Sample Question: What to do if I was rear-ended in a rental car and the rental company wants me to pay? I had a rental car and was rear-ended. I didn't have the rental company's insurance and my own didn't cover the accident. The rental company contacted the party who hit me and they did not respond. Now the rental company is coming to me to pay the damages. What are my options?
    
    Most common words: ['car', 'rental', 'insurance', 'company', 'pay', 'accident', "'s"]


#### Bankruptcy Category


```python
## No. of questions in largest group of similar questions from bankruptcy category
print('No. of similar questions in largest bankruptcy group  = %s' % top_bankruptcy_group['no. of similar questions'].iloc[0])
print()
## Sample question from this group of similar questions from bankruptcy category
print('Sample Question: %s'  % bankruptcy_data['title and question'].iloc[top_bankruptcy_index[9]])
print()
## Most common words from group of similar questions in bankruptcy category 
print('Most common words: %s' %topic_function(bankruptcy_data, top_bankruptcy_index))
```

    No. of similar questions in largest bankruptcy group  = 49.0
    
    Sample Question: What to do about a car and bankruptcy? My mother is going to be retiring soon and thinking about filing bankruptcy. She has a 4 year old car with 30,000 miles. She owes $5,000. We had it appraised and it was appraised at $12,000. She is afraid of losing her car since there is quite a bit of equity in it. Can she sell me the car (for what she owes on it, $5,000) and put the car in my name and then I sell her my daughter's car (old car with lots of miles so it would be sold (or given to her) and we put that car in her name. Is that allowed? We do not want to do anything fraudulent but we want her to be able to keep her car. We just know they will take the car since it is worth quite a bit of money.
    
    Most common words: ['car', 'loan', 'bankruptcy', 'file', 'chapter', 'name', 'credit']


#### Business Category


```python
## No. of questions in largest group of similar questions from business category
print('No. of similar questions in largest business group  = %s' % top_business_group['no. of similar questions'].iloc[0])
print()
## Sample question from this group of similar questions from business category
print('Sample Question: %s'  % business_data['title and question'].iloc[top_business_index[12]])
print()
## Most common words from group of similar questions in business category 
print('Most common words: %s' %topic_function(business_data, top_business_index))
```

    No. of similar questions in largest business group  = 116.0
    
    Sample Question: Can?I sue someone for not making car payments on loan that I signed for? I signed for a car for someone else and the person said that they were paying the car note. However, the car got repossessed and was sold at an auction.?Nowthe repo is on my credit report. Can I sue the person for the remaining balance on the car after it was sold at auction and also to get the repo off of my credit report?
    
    Most common words: ['car', 'get', 'back', "n't", 'title', 'loan', 'name']


#### Consumer Category


```python
## No. of questions in largest group of similar questions from consumer category
print('No. of similar questions in largest consumer group  = %s' % top_consumer_group['no. of similar questions'].iloc[0])
print()
## Sample question from this group of similar questions from consumer category
print('Sample Question: %s'  % consumer_data['title and question'].iloc[top_consumer_index[12]])
print()
## Most common words from group of similar questions in consumer category 
print('Most common words: %s' %topic_function(consumer_data, top_consumer_index))
```

    No. of similar questions in largest consumer group  = 16.0
    
    Sample Question: What legal action can I take if someone intentionally falsified a truck title? A private owner advertised the truck with 74,769 miles. The odometer says 74,840 and that is what he wrote in the blanks. However, when I got home and pulled out the title, it says the truck had 165,100 miles when the truck was last titled. Am I up the creek or can I get my money back?
    
    Most common words: ['truck', 'warranty', 'bought', 'back', "n't", 'would', 'said']


#### Criminal Category


```python
## No. of questions in largest group of similar questions from criminal category
print('No. of similar questions in largest criminal group  = %s' % top_criminal_group['no. of similar questions'].iloc[0])
print()
## Sample question from this group of similar questions from criminal category
print('Sample Question: %s'  % criminal_data['title and question'].iloc[top_criminal_index[12]])
print()
## Most common words from group of similar questions in criminal category 
print('Most common words: %s' %topic_function(criminal_data, top_criminal_index))
```

    No. of similar questions in largest criminal group  = 128.0
    
    Sample Question: Laws of statutory rape. In the state of Tennessee, is it legal for a seventeen year old and a twenty one year old to live together if the seventeen year old is graduated?
    
    Most common words: ['old', 'year', '16', '18', '17', 'date', 'sex']


#### Custody Category


```python
## No. of questions in largest group of similar questions from custody category
print('No. of similar questions in largest custody group  = %s' % top_custody_group['no. of similar questions'].iloc[0])
print()
## Sample question from this group of similar questions from custody category
print('Sample Question: %s'  % custody_data['title and question'].iloc[top_custody_index[12]])
print()
## Most common words from group of similar questions in custody category 
print('Most common words: %s' %topic_function(custody_data, top_custody_index))
```

    No. of similar questions in largest custody group  = 92.0
    
    Sample Question: If my daughter is 15 years old and has not seen nor received any support from her biological father, how can she changeher last namw from his to mine? She wants my last name before she gets her learner's permit which is the name I received after marrying her stepfather. She says her stepdad is her father so how can we give her our name?
    
    Most common words: ['name', "'s", 'last', 'father', 'child', 'birth', 'certificate']


#### Debt Category


```python
## No. of questions in largest group of similar questions from debt category
print('No. of similar questions in largest debt group  = %s' % top_debt_group['no. of similar questions'].iloc[0])
print()
## Sample question from this group of similar questions from debt category
print('Sample Question: %s'  % debt_data['title and question'].iloc[top_debt_index[12]])
print()
## Most common words from group of similar questions in debt category 
print('Most common words: %s' %topic_function(debt_data, top_debt_index))
```

    No. of similar questions in largest debt group  = 41.0
    
    Sample Question: What is the statute of limitations regarding food stamp overpayment? My mother-in-law just received a notice from the state that she was overpaid in food stamps from over 20 years ago. Is there statute of limitations, or what route would she need to go to get this taken care of? In IL.
    
    Most common words: ['limitations', 'statute', 'debt', 'years', 'judgment', 'credit', 'card']


#### Divorce Category


```python
## No. of questions in largest group of similar questions from divorce category
print('No. of similar questions in largest divorce group  = %s' % top_divorce_group['no. of similar questions'].iloc[0])
print()
## Sample question from this group of similar questions from divorce category
print('Sample Question: %s'  % divorce_data['title and question'].iloc[top_divorce_index[2]])
print()
## Most common words from group of similar questions in divorce category 
print('Most common words: %s' %topic_function(divorce_data, top_divorce_index))
```

    No. of similar questions in largest divorce group  = 109.0
    
    Sample Question: What are a couple's rights to a car in a divorce? My wife and?I just split up.?However, prior to that we bought a newer car and she is the co-signer on the loan. She doesn't hold a driver's license and is listed on my insurance as a non-driver. Can she keep the car? Also, she is constantly driving it. Do?I have the right to take it back?
    
    Most common words: ['car', 'name', 'divorce', 'take', 'get', 'husband', 'payments']


#### Dui Category


```python
## No. of questions in largest group of similar questions from dui category
print('No. of similar questions in largest dui group  = %s' % top_dui_group['no. of similar questions'].iloc[0])
print()
## Sample question from this group of similar questions from dui category
print('Sample Question: %s'  % dui_data['title and question'].iloc[top_dui_index[12]])
print()
## Most common words from group of similar questions in dui category 
print('Most common words: %s' %topic_function(dui_data, top_dui_index))
```

    No. of similar questions in largest dui group  = 52.0
    
    Sample Question: What to do if I live in one state but got a DUI in another and lost my license in both states? I got the DUI 5 months ago and they just now revoked my license in my state of residence. They revoked my license in the state in which I got arrested in 4 months ago. Do I have to take my driver's test in both states? And is it even legal for them to revoke it 4 months after the fact in the state that I didn't get it in?
    
    Most common words: ['state', 'license', 'dui', 'another', 'get', 'one', 'suspended']


#### Estate Category


```python
## No. of questions in largest group of similar questions from estate category
print('No. of similar questions in largest estate group  = %s' % top_estate_group['no. of similar questions'].iloc[0])
print()
## Sample question from this group of similar questions from estate category
print('Sample Question: %s'  % estate_data['title and question'].iloc[top_estate_index[2]])
print()
## Most common words from group of similar questions in estate category 
print('Most common words: %s' %topic_function(estate_data, top_estate_index))
```

    No. of similar questions in largest estate group  = 127.0
    
    Sample Question: Can a neighbor who dislikes my fence make me take a down? I replaced my old chain link fence with a new chain link fence. I hired a fencing company, got the permit from my town and informed my neighbor about it. After the new fence was installed, the neighbor stated he does not like it. My property is about?2 feet higher than my neighbor's on an upward slope. The new fence does not cover the retaining wall and the neighbor does not like that. The old fence did cover the retraining wall. The town inspector is an old friend of my neighbors and according to the inspector the fence is on my property. Who is legally right?
    
    Most common words: ['fence', 'property', 'neighbor', 'line', 'put', 'years', "'s"]


#### Immigration Category


```python
## No. of questions in largest group of similar questions from immigration  category
print('No. of similar questions in largest immigration group  = %s' % top_immigration_group['no. of similar questions'].iloc[0])
print()
## Sample question from this group of similar questions from immigration category
print('Sample Question: %s'  % immigration_data['title and question'].iloc[top_immigration_index[10]])
print()
## Most common words from group of similar questions in immigration category 
print('Most common words: %s' %topic_function(immigration_data, top_immigration_index))
```

    No. of similar questions in largest immigration group  = 27.0
    
    Sample Question: If I have my green card and want to get married, do I use my maiden name or married name when filing for citizenship? I just want to know in what order I would need to do this. I have had my green card for almost 10 years now and I was going to apply for US citizenship and then I got engaged to an American. I figured I should wait until after the wedding so that I don't have to complicate the name change process. I tried doing a google search but it really didn't help me. I guess my main question is if I change my name do I do it normally and then apply for citizenship with that name? Will that affect my green card if I change my name without requesting a new green card if I'm already applying for citizenship? My green card doesn't expire for about another 21 months so I definitely have some time but I'm getting married in 7 months and I don't want to do something wrong.
    
    Most common words: ['name', 'change', 'card', 'last', 'green', 'married', "'s"]


#### Injury Category


```python
## No. of questions in largest group of similar questions from injury category
print('No. of similar questions in largest injury group  = %s' % top_injury_group['no. of similar questions'].iloc[0])
print()
## Sample question from this group of similar questions from injury category
print('Sample Question: %s'  % injury_data['title and question'].iloc[top_injury_index[70]])
print()
## Most common words from group of similar questions in injury category 
print('Most common words: %s' %topic_function(injury_data, top_injury_index))
```

    No. of similar questions in largest injury group  = 72.0
    
    Sample Question: Am I entitled to damages if?a neighbor's dog attacked me but did not bite me? I was attacked by a neighbor's dog but not bitten. I believe the dog owner is indeed guilty of negligence. How much money is typically awarded in cases such as this? And how often does the dog owner get away with a warning? I'm tired of dog owners having vicious dogs and not controlling them.
    
    Most common words: ['dog', "'s", 'bit', 'owner', 'pay', 'neighbor', 'got']


#### Insurance Category


```python
## No. of questions in largest group of similar questions from insurance category
print('No. of similar questions in largest insurance group  = %s' % top_insurance_group['no. of similar questions'].iloc[0])
print()
## Sample question from this group of similar questions from insurance category
print('Sample Question: %s'  % insurance_data['title and question'].iloc[top_insurance_index[1]])
print()
## Most common words from group of similar questions in insurance category 
print('Most common words: %s' %topic_function(insurance_data, top_insurance_index))
```

    No. of similar questions in largest insurance group  = 19.0
    
    Sample Question: If your roof suffered damage but the adjuster disputes the reason that you gave as the cause, what can you do to get your claim approved? I had a tree branch fall on roof which caused damage. However, the adjuster said this didn't happen. But that is how it happened. What are my legal rights here?
    
    Most common words: ['roof', 'insurance', 'damage', 'company', 'hail', 'claim', 'new']


#### Labor Category


```python
## No. of questions in largest group of similar questions from labor category
print('No. of similar questions in largest labor group  = %s' % top_labor_group['no. of similar questions'].iloc[0])
print()
## Sample question from this group of similar questions from labor category
print('Sample Question: %s'  % labor_data['title and question'].iloc[top_labor_index[12]])
print()
## Most common words from group of similar questions in labor category 
print('Most common words: %s' %topic_function(labor_data, top_labor_index))
```

    No. of similar questions in largest labor group  = 225.0
    
    Sample Question: Is it legal to layoff an employee during their vacation? My wife was laid off last week while she was on vacation.
    
    Most common words: ['vacation', 'time', 'pay', 'paid', 'employer', 'company', 'hours']


#### Medical Category


```python
## No. of questions in largest group of similar questions from medical category
print('No. of similar questions in largest medical group  = %s' % top_medical_group['no. of similar questions'].iloc[0])
print()
## Sample question from this group of similar questions from medical category
print('Sample Question: %s'  % medical_data['title and question'].iloc[top_medical_index[12]])
print()
## Most common words from group of similar questions in medical category 
print('Most common words: %s' %topic_function(medical_data, top_medical_index))
```

    No. of similar questions in largest medical group  = 15.0
    
    Sample Question: What can I do if a dentist cracked my tooth?  I got a tooth pulled and the dentist cracked the tooth next to the one I had pulled.
    
    Most common words: ['tooth', 'dentist', 'pulled', 'wrong', 'went', 'sue', 'cracked']


#### Tenant Category


```python
## No. of questions in largest group of similar questions from tenant category
print('No. of similar questions in largest tenant group  = %s' % top_tenant_group['no. of similar questions'].iloc[0])
print()
## Sample question from this group of similar questions from tenant category
print('Sample Question: %s'  % tenant_data['title and question'].iloc[top_tenant_index[10]])
print()
## Most common words from group of similar questions in tenant category 
print('Most common words: %s' %topic_function(tenant_data, top_tenant_index))
```

    No. of similar questions in largest tenant group  = 89.0
    
    Sample Question: If a renter is late with the rent payment, does the late fee get assessed the 15th or the 16th of the month? 
    
    Most common words: ['late', 'rent', 'fee', 'landlord', 'fees', 'pay', 'charge']


#### Traffic Category


```python
## No. of questions in largest group of similar questions from traffic category
print('No. of similar questions in largest traffic group  = %s' % top_traffic_group['no. of similar questions'].iloc[0])
print()
## Sample question from this group of similar questions from traffic category
print('Sample Question: %s'  % traffic_data['title and question'].iloc[top_traffic_index[2]])
print()
## Most common words from group of similar questions in traffic category 
print('Most common words: %s' %topic_function(traffic_data, top_traffic_index))
```

    No. of similar questions in largest traffic group  = 13.0
    
    Sample Question: I owe court fines in va my license was suspended can i get a license in another state 
    
    Most common words: ['license', 'state', 'suspended', 'another', 'get', 'one', "'s"]


#### Wills Category


```python
## No. of questions in largest group of similar questions from wills category
print('No. of similar questions in largest wills group  = %s' % top_wills_group['no. of similar questions'].iloc[0])
print()
## Sample question from this group of similar questions from wills category
print('Sample Question: %s'  % wills_data['title and question'].iloc[top_wills_index[4]])
print()
## Most common words from group of similar questions in wills category 
print('Most common words: %s' %topic_function(wills_data, top_wills_index))
```

    No. of similar questions in largest wills group  = 70.0
    
    Sample Question: Would I be responsible for the credit card debt of a parent upon their death? What happens if this person left no valuable assets or enough insurance money to pay their debt off?If this someone still owes for the home's mortgage and the other person that is on the loan is still living what happens then, would the debt collectors put a lien on the home? If there is a power of attorney or executor of the estate are they responsible?
    
    Most common words: ['credit', 'debt', 'card', 'responsible', "'s", 'house', 'spouse']


## Find questions similar to unanswered questions


```python
def similarity_unanswered_function(data, unanswered):
    doc = data['title and question'].tolist()
    gen_docs = [[w.lower() for w in word_tokenize(text)] 
            for text in doc]
    
    stop_words = set(stopwords.words('english'))
 
    filtered_sentence = [[] for i in range(len(gen_docs))]

    for i in range(len(gen_docs)):
        for w in gen_docs[i]:
            if w not in stop_words:
                filtered_sentence[i].append(w)
 
    dictionary = gensim.corpora.Dictionary(filtered_sentence)
    corpus = [dictionary.doc2bow(sent) for sent in filtered_sentence]
    tf_idf = gensim.models.TfidfModel(corpus)
    sims = gensim.similarities.Similarity('/Users/oofoegbu/',tf_idf[corpus],num_features=len(dictionary))
    
    similarity = pd.DataFrame(columns = ['similar','index'])
    
    query_doc = [w.lower() for w in word_tokenize(unanswered)]
    query_sentence = [w for w in query_doc if not w in stop_words]
    query_doc_bow = dictionary.doc2bow(query_sentence)
    query_doc_tf_idf = tf_idf[query_doc_bow]
    sim = sims[query_doc_tf_idf]
    similar = np.where((sim >= 0.5) & (sim < 0.9999))[0]
    similarity.loc[0] = [len(similar),similar]
    
    similar_questions = similarity['similar'].value_counts().reset_index()
    similar_questions.columns = ['no. of similar questions', 'index of similar questions']
   
    return similarity

```


```python
## Find question with nearest similarity to unanswered question from auto category
unans_auto = auto_data[auto_data['date_answered'] == 'unanswered'].index.tolist()
unanswered_auto = auto_data['title and question'].iloc[unans_auto[0]]

sim_unans_auto = similarity_unanswered_function(auto_data, unanswered_auto)['index'][0]
```


```python
## Find question with nearest similarity to unanswered question from business category
unans_business = business_data[business_data['date_answered'] == 'unanswered'].index.tolist()
unanswered_business = business_data['title and question'].iloc[unans_business[0]]

sim_unans_business = similarity_unanswered_function(business_data, unanswered_business)['index'][0]
```


```python
## Find question with nearest similarity to unanswered question from criminal category
unans_criminal = criminal_data[criminal_data['date_answered'] == 'unanswered'].index.tolist()
unanswered_criminal = criminal_data['title and question'].iloc[unans_criminal[0]]

sim_unans_criminal = similarity_unanswered_function(criminal_data, unanswered_criminal)['index'][0]
```


```python
## Find question with nearest similarity to unanswered question from divorce category
unans_divorce = divorce_data[divorce_data['date_answered'] == 'unanswered'].index.tolist()
unanswered_divorce1 = divorce_data['title and question'].iloc[unans_divorce[0]]
unanswered_divorce2 = divorce_data['title and question'].iloc[unans_divorce[1]]

sim_unans_divorce1 = similarity_unanswered_function(divorce_data, unanswered_divorce1)['index'][0]
sim_unans_divorce2 = similarity_unanswered_function(divorce_data, unanswered_divorce2)['index'][0]
```


```python
## Find question with nearest similarity to unanswered question from dui category
unans_dui = dui_data[dui_data['date_answered'] == 'unanswered'].index.tolist()
unanswered_dui = dui_data['title and question'].iloc[unans_dui[0]]

sim_unans_dui = similarity_unanswered_function(dui_data, unanswered_dui)['index'][0]
```


```python
## Find question with nearest similarity to unanswered question from estate category
unans_estate = estate_data[estate_data['date_answered'] == 'unanswered'].index.tolist()
unanswered_estate1 = estate_data['title and question'].iloc[unans_estate[0]]
unanswered_estate2 = estate_data['title and question'].iloc[unans_estate[1]]

sim_unans_estate1 = similarity_unanswered_function(estate_data, unanswered_estate1)['index'][0]
sim_unans_estate2 = similarity_unanswered_function(estate_data, unanswered_estate2)['index'][0]
```


```python
## Find question with nearest similarity to unanswered question from wills category
unans_wills = wills_data[wills_data['date_answered'] == 'unanswered'].index.tolist()
unanswered_wills = wills_data['title and question'].iloc[unans_wills[0]]

sim_unans_wills = similarity_unanswered_function(wills_data, unanswered_wills)['index'][0]
```

#### Questions similar to unanswered questions


```python
similarity = pd.DataFrame(columns = ['category of unanswered question','index of similar question'])
similarity.loc[0] = ['auto', sim_unans_auto]
similarity.loc[1] = ['business', sim_unans_business]
similarity.loc[2] = ['criminal', sim_unans_criminal]
similarity.loc[3] = ['divorce', sim_unans_divorce1]
similarity.loc[4] = ['divorce', sim_unans_divorce2]
similarity.loc[5] = ['dui', sim_unans_dui]
similarity.loc[6] = ['estate', sim_unans_estate1]
similarity.loc[7] = ['estate', sim_unans_estate2]
similarity.loc[8] = ['wills', sim_unans_wills]

print(similarity)
```

      category of unanswered question index of similar question
    0                            auto                        []
    1                        business                        []
    2                        criminal                        []
    3                         divorce                        []
    4                         divorce                        []
    5                             dui                        []
    6                          estate              [1766, 4791]
    7                          estate                        []
    8                           wills                        []


## Similar questions to unanswered question in estate category


```python
## Unanswered question from estate category
print('Unanswered Question: %s'  % unanswered_estate1)
print()
## Similar question to unanswered question
print('Similar Question: %s'  % estate_data['title and question'].iloc[1766])
print()
## Similar question to unanswered question
print('Similar Question: %s'  % estate_data['title and question'].iloc[4791])
print()
```

    Unanswered Question: How does a sibling buy out 2 other siblings if a home is part of an estate? 
    
    Similar Question: Can a real estate?inheritance be bought and sold?among beneficiaries? Right now the house is divided into 9 equal shares -?1 for each sibling. Could?1 sibling buy out 2 other siblings by paying them what their share of the house is worth so that those 2 siblings no longer have a legal share in the house? If so, then the one who bought them out would now own 3 of the 9 shares of the house. Can this be done if one sibling eventually wants to buy the house but can't get approved for a mortgage loan yet, so they would slowly buy the other siblings out of their shares over the course of a year or two?
    
    Similar Question: 3 siblings own property together, Undivided 1/3 interest.  Can two of the siblings make the other sibling buy them out or focre the sale of the home? Summit Co. OHIO.  3 siblings own a property together, Undivided 1/3 interest.  Property is paid off and was obtained through their father's death.  None of the chilkdren lived in the home for the past 10 years.  One sibling is living in home, (after the title transfer, about 3 months) not paying any form of rent to the other two.  The two non-resident sibilings would like to sell the home outright or have the other sibling buy them out.  Resident sibling is unable/unwilling to buy out other siblings and will not allow home to be put up for sale or vacate the home.
    

