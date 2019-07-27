# Winning is the Name of the Game: Predicting Individual NBA Player Win Shares

The primary purpose of this project is to solve the following question: "can we predict an individual player's win share per game?" More specifically, the statistic is win share per 48 minutes played (which is equivalent to 1 game). When scouting potential players, scouts often use what is called the 'eye test', which is a highly subjective measure of a player's talent. Players that often pass the eye test are very physically gifted which cause scouts to ignore glaring holes they may have when it comes to the stat line. However, analytics has revolutionized how players are evaluated not only in the NBA but in other professional sports as well. In summary, this project is able to take a wide-breadth of data from the past 10 NBA seasons and create ana algorithm that is able to predict an NBA player's win share to within a few percentage points.

## Table of Contents

- `Data`: folder containing compiled csv files of player statistics
    - [`df_all.csv`](https://github.com/Jearny58/Springboard-DS-Portfolio/blob/master/capstone_1/basketball/Individual_Player_Stats/Data/df_all.csv): compiles offensive, defensive and advanced statistics from the past 10 seasons for every NBA player (data set used for machine learning)
- `Data_Wrangling`: folder containing Jupyter Notebooks related to gathering/cleaning data from [basketball-reference.com](https://www.basketball-reference.com/)
    - [`data-wrangling-individual-player-stats.ipynb`](https://github.com/Jearny58/Springboard-DS-Portfolio/blob/master/capstone_1/basketball/Individual_Player_Stats/Data_Wrangling/data-wrangling-individual-player-stats.ipynb): Jupyter notebook detailing how data from past 10 seasons was scraped from [basketball-reference.com](https://www.basketball-reference.com/), combined and then cleaned to create `df_all.csv` file for machine learning
- `EDA`: folder containing notebooks utilized for exploratory data analysis of player statistics
    - [`Exploratory-Data-Analysis-WinShares.ipynb`](https://github.com/Jearny58/Springboard-DS-Portfolio/blob/master/capstone_1/basketball/Individual_Player_Stats/EDA/Exploratory-Data-Analysis-WinShares.ipynb) explores player statistics, with multiple graphs detailing relationship between multiple metrics and win shares per 48 minutes (our target variable)
    - `playground_nbs`: contains experimental Jupyter notebooks to test code/theories
- `Machine_Learning`: contains notebooks related to generation of machine learning models that predict win shares per 48 minutes
    - [`machine-learning.ipynb`](https://github.com/Jearny58/Springboard-DS-Portfolio/blob/master/capstone_1/basketball/Individual_Player_Stats/Machine_Learning/machine-learning.ipynb): Jupyter notebook detailing the development of multiple machine learning models, which utilize multiple linear regresssion, ridge regression, and random forest frameworks
- `Reports`: folder containing slide deck and milestone reports/final analysis of project
    - [`capstone1_slide_deck.pdf`](https://github.com/Jearny58/Springboard-DS-Portfolio/blob/master/capstone_1/basketball/Individual_Player_Stats/Reports/capstone1_slide_deck.pdf): slide deck detailing background of analytics in the NBA, mental biases and assessment of models created to predict win share
    - [`capstone1_comprehensive_report.pdf`](https://github.com/Jearny58/Springboard-DS-Portfolio/blob/master/capstone_1/basketball/Individual_Player_Stats/Reports/capstone1_comprehensive_report.pdf): report detailing entire workflow for project, from data wrangling and EDA to the development of machine learning models
   
