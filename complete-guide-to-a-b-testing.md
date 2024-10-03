# Complete Guide to A/B Testing

Welcome to this comprehensive guide on A/B testing! This notebook is designed to equip you with everything you need to know, from basic concepts to advanced applications in data science. Whether you're just starting out or you're a seasoned professional looking to sharpen your skills, this guide is for you!

A/B testing, also known as split testing, is a method of comparing two versions of a webpage, app, or other user experiences to determine which one performs better. It is widely used in marketing, product development, and UX design to make data-driven decisions that improve user engagement and conversion rates.

## What Will You Learn?

We'll dive into a variety of A/B testing techniques, each with a unique purpose in data analysis. By the end, you'll have a toolkit ready to tackle any data-driven challenge. Here's a peek at what we'll cover:

- **Binary A/B Testing:** Learn how to compare two versions of a binary outcome to determine which one performs better.
- **Continuous A/B Testing:** Discover how to compare means from continuous data and understand the statistical significance of their differences.

## Why This Guide?

- **Step-by-Step Tutorials:** Each section includes clear explanations followed by practical examples, ensuring you not only learn but also apply your knowledge.
- **Interactive Learning:** Engage with interactive code cells that allow you to see the effects of A/B tests in real-time.


### How to Use This Notebook

- **Run the Cells:** Follow along with the code examples by running the cells yourself. Play around with the parameters to see how the results change.
- **Explore Further:** After completing the guided sections, try applying the tests to your own datasets to reinforce your learning.

Get ready to unlock the full potential of A/B testing in data science. Let's dive in and turn data into decisions!


#### A/B Testing Workflow

## Step 1: Exploratory Data Analysis (EDA) and Problem Definition

### EDA and Dataset Check
Before diving into the A/B test, it's crucial to perform an exploratory data analysis (EDA) to understand the dataset's structure and characteristics. This step involves:

1. **Loading the Dataset**: We imported the dataset `Cookie_Cats_cleaned_v01.csv` and displayed the first few rows to get a sense of the data.
2. **Summary Statistics**: We generated summary statistics to understand the central tendency, dispersion, and distribution of the data.
3. **Missing Values**: We checked for any missing values and ensured they were appropriately handled.
4. **Data Types**: We verified the data types of each column to ensure correctness.
5. **Descriptive Statistics by Group**: We grouped the data by the `version` column and calculated descriptive statistics, including skewness and kurtosis, to understand the distribution and identify any potential outliers.
6. **Cross-tabulations**: We created cross-tabulations for `version` with `retention_1` and `retention_7` to observe the retention rates for different versions of the game.
7. **Plots**: We used various plots to visualize the distribution of data and retention rates, which helped us better understand the data.

### Define the Problem
Clearly defining the problem is a crucial step. For an A/B test, we typically compare two groups to determine if there is a significant difference between them.

- **Control Group**: The group that does not receive the treatment or intervention.
- **Treatment Group**: The group that receives the treatment or intervention.

### Define Null and Alternative Hypothesis
- **Null Hypothesis (H0)**: There is no difference between the control and treatment groups.
- **Alternative Hypothesis (H1)**: There is a significant difference between the control and treatment groups.

## Step 2: Data Preprocessing

Based on the findings from the EDA and the defined problem, preprocess the data to ensure it's ready for testing.

### Preprocessing Steps:
1. **Handling Missing Values**: Remove or impute missing values as identified in the EDA.
2. **Outlier Detection**: Identify and handle outliers that may skew the results.
3. **Data Transformation**: Normalize or standardize data if necessary, especially for continuous variables.
4. **Feature Engineering**: Create new features if needed to better capture the characteristics of the data.

## Step 3: Set the Probability of Type I and Type II Errors

### Type I Error (Alpha)
- **Definition**: The probability of rejecting the null hypothesis when it is true.
- **Common Value**: 0.05 (5%)

### Type II Error (Beta)
- **Definition**: The probability of failing to reject the null hypothesis when it is false.
- **Common Value**: 0.20 (20%)
- **Power of the Test**: 1 - Beta, often set to 0.80 (80%)

## Step 4: Calculate Sample Size and Choose Samples

### Calculate Sample Size
To determine the necessary sample size, use the formula that incorporates the desired significance level (alpha), power (1 - beta), and effect size.

#### For Binary Outcomes:
The sample size formula for binary outcomes is given by:

$$
n = \left( Z_{1-\alpha/2} + Z_{1-\beta} \right)^2 \frac{p_1(1 - p_1) + p_2(1 - p_2)}{(p_1 - p_2)^2}
$$

Where:
- $( Z_{1-\alpha/2} $) is the Z-value for the chosen significance level.
- $( Z_{1-\beta} $) is the Z-value for the chosen power.
- $( p_1 $) and $( p_2 $) are the proportions of success in the control and treatment groups, respectively.

#### For Continuous Outcomes:
The sample size formula for continuous outcomes is given by:

$$
n = \left( \frac{Z_{\alpha/2} + Z_{\beta}}{\mu_1 - \mu_2} \right)^2 \cdot 2\sigma^2
$$

Where:
- $( Z_{\alpha/2} $) is the Z-value for the chosen significance level.
- $( Z_{\beta} $) is the Z-value for the chosen power.
- $( \mu_1 $) and $( \mu_2 $) are the means of the control and treatment groups, respectively.
- $( \sigma $) is the standard deviation of the population.

### Choose Samples
- Randomly select samples for the control and treatment groups based on the calculated sample size.
- Ensure the samples are representative of the population.

## Step 5: Assign Users
- **Assign Users**: Randomly assign users to the control or treatment group.

## Step 6: Perform Statistical Test 
- For binary outcomes, use a Z-test.
- For continuous outcomes, a t-test can be used.

#### Note:
While we use a Z-test for binary outcomes in this guide, other tests like Chi-Square test or Fisher's Exact Test can also be used depending on the specific requirements and data characteristics.

## Step 7: Analyze the Test Results

### Practical Significance
- Beyond statistical significance, assess the practical significance of the results to determine if the observed effect is meaningful in a real-world context.

### Report Findings
- Summarize the results, including the test statistic, p-value, confidence intervals, and any practical implications.
- Visualize the results using graphs and charts for a clearer understanding.



# 1.1 EDA and Dataset Check

Before diving into the A/B test, it's crucial to perform an exploratory data analysis (EDA) to understand the dataset's structure and characteristics. This step involves:

1. **Loading the Dataset**: 
   - We imported the dataset `Cookie_Cats_cleaned_v01.csv` and displayed the first few rows to get a sense of the data.
   - This helps in identifying the columns and understanding the initial structure of the dataset.


2. **Summary Statistics**:
   - We generated summary statistics for the dataset to understand the central tendency (mean, median), dispersion (standard deviation, variance), and the overall distribution of the data.


3. **Missing Values**:
   - We checked for any missing values in the dataset and ensured they were appropriately handled.
   - Handling missing values is crucial for maintaining the integrity of the dataset and avoiding biases in the analysis.


4. **Data Types**:
   - We verified the data types of each column to ensure correctness.
   - Correct data types are essential for performing accurate calculations and analyses.


5. **Descriptive Statistics by Group**:
   - We grouped the data by the `version` column (indicating control or treatment groups) and calculated descriptive statistics for each group.
   - This includes measures such as mean, standard deviation, skewness, and kurtosis to understand the distribution within each group and identify any potential outliers.


6. **Cross-tabulations**:
   - We created cross-tabulations for `version` with `retention_1` and `retention_7` to observe the retention rates for different versions of the game.
   - Cross-tabulations help in understanding the relationship between categorical variables and provide a clear picture of how different groups perform with respect to retention.


7. **Plots**:
   - We used various plots to visualize the distribution of data and retention rates. These plots include bar plots, box plots, and heatmaps for correlation analysis.
   - Visualizations help in better understanding the data by providing a graphical representation of the distributions, relationships, and patterns.


By performing these EDA steps, we ensure a comprehensive understanding of the dataset, identify any potential issues or biases, and prepare the data for further analysis in the A/B testing workflow. This foundational step is essential for making informed decisions and deriving meaningful insights from the A/B test.



```python
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis

# Load the dataset
df = pd.read_csv('datasets/Cookie_Cats_cleaned_v01.csv')

# Check if user_id is not repetitive
if df['userid'].duplicated().any():
    print("Duplicates found in user_id. Dropping duplicates...")
    df = df.drop_duplicates(subset='userid')

# Set user_id as the index
df.set_index('userid', inplace=True)

# Display the first few rows of the dataset
print("First few rows of the dataset:")
print(df.head())

# Display the summary statistics of the dataset
print("\nSummary statistics of the dataset:")
print(df.describe())
print(df.describe(include='all'))

# Display the data types and non-null counts
print("\nData types and non-null counts:")
print(df.info())

# Check for missing values
print("\nMissing values in the dataset:")
print(df.isnull().sum())

```

    First few rows of the dataset:
            version  sum_gamerounds  retention_1  retention_7
    userid                                                   
    116     gate_30               3        False        False
    337     gate_30              38         True        False
    377     gate_40             165         True        False
    483     gate_40               1        False        False
    488     gate_40             179         True         True
    
    Summary statistics of the dataset:
           sum_gamerounds
    count    90189.000000
    mean        51.872457
    std        195.050858
    min          0.000000
    25%          5.000000
    50%         16.000000
    75%         51.000000
    max      49854.000000
            version  sum_gamerounds retention_1 retention_7
    count     90189    90189.000000       90189       90189
    unique        2             NaN           2           2
    top     gate_40             NaN       False       False
    freq      45489             NaN       50036       73408
    mean        NaN       51.872457         NaN         NaN
    std         NaN      195.050858         NaN         NaN
    min         NaN        0.000000         NaN         NaN
    25%         NaN        5.000000         NaN         NaN
    50%         NaN       16.000000         NaN         NaN
    75%         NaN       51.000000         NaN         NaN
    max         NaN    49854.000000         NaN         NaN
    
    Data types and non-null counts:
    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 90189 entries, 116 to 9999861
    Data columns (total 4 columns):
     #   Column          Non-Null Count  Dtype 
    ---  ------          --------------  ----- 
     0   version         90189 non-null  object
     1   sum_gamerounds  90189 non-null  int64 
     2   retention_1     90189 non-null  bool  
     3   retention_7     90189 non-null  bool  
    dtypes: bool(2), int64(1), object(1)
    memory usage: 2.2+ MB
    None
    
    Missing values in the dataset:
    version           0
    sum_gamerounds    0
    retention_1       0
    retention_7       0
    dtype: int64
    


```python
# Descriptive statistics with skewness and kurtosis for numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns
descriptive_stats = df[numeric_cols].describe().T
descriptive_stats['skewness'] = df[numeric_cols].skew()
descriptive_stats['kurtosis'] = df[numeric_cols].kurtosis()
print("\nDescriptive Statistics (including skewness and kurtosis):")
print(descriptive_stats)

# Percentage of each category in categorical features
version_counts = df['version'].value_counts(normalize=True) * 100
retention_1_counts = df['retention_1'].value_counts(normalize=True) * 100
retention_7_counts = df['retention_7'].value_counts(normalize=True) * 100

print("\nPercentage of each category in 'version':")
print(version_counts)
print("\nPercentage of each category in 'retention_1':")
print(retention_1_counts)
print("\nPercentage of each category in 'retention_7':")
print(retention_7_counts)

```

    
    Descriptive Statistics (including skewness and kurtosis):
                      count       mean         std  min  25%   50%   75%      max  \
    sum_gamerounds  90189.0  51.872457  195.050858  0.0  5.0  16.0  51.0  49854.0   
    
                      skewness      kurtosis  
    sum_gamerounds  185.436313  47130.369631  
    
    Percentage of each category in 'version':
    gate_40    50.437415
    gate_30    49.562585
    Name: version, dtype: float64
    
    Percentage of each category in 'retention_1':
    False    55.47905
    True     44.52095
    Name: retention_1, dtype: float64
    
    Percentage of each category in 'retention_7':
    False    81.393518
    True     18.606482
    Name: retention_7, dtype: float64
    


```python
# Cross-tabulation for version and retention_1
crosstab_retention_1 = pd.crosstab(df['version'], df['retention_1'], normalize='index')
# crosstab_retention_1.columns = ['Did Not Return', 'Returned']
# crosstab_retention_1 = crosstab_retention_1.reset_index()

# Cross-tabulation for version and retention_7
crosstab_retention_7 = pd.crosstab(df['version'], df['retention_7'], normalize='index')
crosstab_retention_7.columns = ['Did Not Return', 'Returned']
crosstab_retention_7 = crosstab_retention_7.reset_index()

# Cross-tabulation for version and retention_7
crosstab_retention_7_1 = pd.crosstab(df['retention_1'], df['retention_7'], normalize='index')
# crosstab_retention_7_1.columns = ['Did Not Return', 'Returned']
# crosstab_retention_7_1 = crosstab_retention_7_1.reset_index()

print("\nCross-tabulation for version and retention_1:")
print(crosstab_retention_1)

print("\nCross-tabulation for version and retention_7:")
print(crosstab_retention_7)

print("\nCross-tabulation for retention_1 and retention_7:")
print(crosstab_retention_7_1)
```

    
    Cross-tabulation for version and retention_1:
    retention_1     False      True
    version                        
    gate_30      0.551812  0.448188
    gate_40      0.557717  0.442283
    
    Cross-tabulation for version and retention_7:
       version  Did Not Return  Returned
    0  gate_30        0.809799  0.190201
    1  gate_40        0.818000  0.182000
    
    Cross-tabulation for retention_1 and retention_7:
    retention_7     False      True
    retention_1                    
    False        0.928072  0.071928
    True         0.671706  0.328294
    

## Looking at cross-tabulation for retention_1 and retention_7 seems like these 2 variables aren't independent
## Let's see with how much significance level we can reject these 2 being independent using Chi-Square Test for Independence
## Hypotheses for Chi-Square Test for Independence

- **Null Hypothesis (H0)**: No association between the categorical variables (independent).
- **Alternative Hypothesis (H1)**: An association exists between the categorical variables (not independent).

## To reject null hypothesis with significance level=a we need p(p_value) < a or chi_statistic to be not less than the value corresponding to the dof=1 and significance level=a pair at chi-square table.


```python
from scipy import stats
crosstab_retention_7_1 = pd.crosstab(df['retention_1'], df['retention_7'])
print(crosstab_retention_7_1)
chi2, p, dof, expected_frequencies  = stats.chi2_contingency(crosstab_retention_7_1)
print(f'\nexpected_frequencies:\n{expected_frequencies}\nchi_statistic: {chi2}\np_value: {p}\ndof: {dof}')
```

    retention_7  False   True
    retention_1              
    False        46437   3599
    True         26971  13182
    
    expected_frequencies:
    [[40726.06069476  9309.93930524]
     [32681.93930524  7471.06069476]]
    chi_statistic: 9665.803068737494
    p_value: 0.0
    dof: 1
    


```python
# Group by version and calculate descriptive statistics for numeric columns
grouped_stats = df.groupby('version')[numeric_cols].describe()

# Additional skewness and kurtosis by version
skewness_by_version = df.groupby('version')[numeric_cols].apply(lambda x: x.skew()).unstack()
kurtosis_by_version = df.groupby('version')[numeric_cols].apply(lambda x: x.kurtosis()).unstack()

# Combine descriptive statistics with skewness and kurtosis
grouped_stats = grouped_stats.unstack().T
grouped_stats['skewness'] = skewness_by_version.T.values.flatten()
grouped_stats['kurtosis'] = kurtosis_by_version.T.values.flatten()
grouped_stats = grouped_stats.reset_index()

print("\nDescriptive Statistics with Skewness and Kurtosis by Version:")
print(grouped_stats)

```

    
    Descriptive Statistics with Skewness and Kurtosis by Version:
               level_0 level_1  version                                        0
    0   sum_gamerounds   count  gate_30                                  44700.0
    1   sum_gamerounds   count  gate_40                                  45489.0
    2   sum_gamerounds    mean  gate_30                                52.456264
    3   sum_gamerounds    mean  gate_40                                51.298776
    4   sum_gamerounds     std  gate_30                               256.716423
    5   sum_gamerounds     std  gate_40                               103.294416
    6   sum_gamerounds     min  gate_30                                      0.0
    7   sum_gamerounds     min  gate_40                                      0.0
    8   sum_gamerounds     25%  gate_30                                      5.0
    9   sum_gamerounds     25%  gate_40                                      5.0
    10  sum_gamerounds     50%  gate_30                                     17.0
    11  sum_gamerounds     50%  gate_40                                     16.0
    12  sum_gamerounds     75%  gate_30                                     50.0
    13  sum_gamerounds     75%  gate_40                                     52.0
    14  sum_gamerounds     max  gate_30                                  49854.0
    15  sum_gamerounds     max  gate_40                                   2640.0
    16        skewness                   [163.70987062651201, 5.967287898713902]
    17        kurtosis                    [31688.38064342865, 63.10608364449687]
    


```python
import matplotlib.pyplot as plt
import seaborn as sns

# Set the style for seaborn
sns.set_style("darkgrid")

fig, axes = plt.subplots(4, 2, figsize=(18, 25))
fig.suptitle('Data Distributions and Analysis', fontsize=20)

# Distribution of the 'version' column
sns.barplot(x=version_counts.index, y=version_counts.values, ax=axes[0, 0], palette='viridis')
axes[0, 0].text(0, version_counts.values[0] + 1, f'{version_counts.values[0]:.1f}%', ha='center', va='bottom')
axes[0, 0].text(1, version_counts.values[1] + 1, f'{version_counts.values[1]:.1f}%', ha='center', va='bottom')
axes[0, 0].set_title('Distribution of Version')
axes[0, 0].set_ylabel('Percentage')
axes[0, 0].set_xlabel('Version')

# Distribution of 'retention_1'
sns.barplot(x=retention_1_counts.index, y=retention_1_counts.values, ax=axes[0, 1], palette='viridis')
axes[0, 1].text(0, retention_1_counts.values[0] + 1, f'{retention_1_counts.values[0]:.1f}%', ha='center', va='bottom')
axes[0, 1].text(1, retention_1_counts.values[1] + 1, f'{retention_1_counts.values[1]:.1f}%', ha='center', va='bottom')
axes[0, 1].set_title('Distribution of Retention 1 Day After Installation')
axes[0, 1].set_ylabel('Percentage')
axes[0, 1].set_xlabel('Retention 1')

# Distribution of 'retention_7'
sns.barplot(x=retention_7_counts.index, y=retention_7_counts.values, ax=axes[1, 0], palette='viridis')
axes[1, 0].text(0, retention_7_counts.values[0] + 1, f'{retention_7_counts.values[0]:.1f}%', ha='center', va='bottom')
axes[1, 0].text(1, retention_7_counts.values[1] + 1, f'{retention_7_counts.values[1]:.1f}%', ha='center', va='bottom')
axes[1, 0].set_title('Distribution of Retention 7 Days After Installation')
axes[1, 0].set_ylabel('Percentage')
axes[1, 0].set_xlabel('Retention 7')

# Distribution of 'sum_gamerounds' as a box plot
sns.boxplot(ax=axes[1, 1], y=df['sum_gamerounds'], color='blue')
axes[1, 1].set_title('Distribution of Sum of Game Rounds')
axes[1, 1].set_ylabel('Sum of Game Rounds')
axes[1, 1].set_yscale('log')

# Analysis of sum_gamerounds by version
sns.boxplot(ax=axes[2, 0], x='version', y='sum_gamerounds', data=df, palette='viridis')
axes[2, 0].set_title('Sum of Game Rounds by Version')
axes[2, 0].set_xlabel('Version')
axes[2, 0].set_ylabel('Sum of Game Rounds')
axes[2, 0].set_yscale('log')

# Analysis of retention_1 by version
retention_1_by_version = df.groupby(['version', 'retention_1']).size().reset_index(name='counts')
retention_1_by_version['percentage'] = retention_1_by_version['counts'] / retention_1_by_version.groupby('version')['counts'].transform('sum') * 100
sns.barplot(ax=axes[2, 1], x='version', y='percentage', hue='retention_1', data=retention_1_by_version, palette='viridis')
for p in axes[2, 1].patches:
    height = p.get_height()
    axes[2, 1].text(p.get_x() + p.get_width() / 2., height + 1, f'{height:.1f}%', ha='center', va='bottom')
axes[2, 1].set_title('Retention 1 Day After Installation by Version')
axes[2, 1].set_xlabel('Version')
axes[2, 1].set_ylabel('Retention 1 Rate (%)')

# Analysis of retention_7 by version
retention_7_by_version = df.groupby(['version', 'retention_7']).size().reset_index(name='counts')
retention_7_by_version['percentage'] = retention_7_by_version['counts'] / retention_7_by_version.groupby('version')['counts'].transform('sum') * 100
sns.barplot(ax=axes[3, 0], x='version', y='percentage', hue='retention_7', data=retention_7_by_version, palette='viridis')
for p in axes[3, 0].patches:
    height = p.get_height()
    axes[3, 0].text(p.get_x() + p.get_width() / 2., height + 1, f'{height:.1f}%', ha='center', va='bottom')
axes[3, 0].set_title('Retention 7 Days After Installation by Version')
axes[3, 0].set_xlabel('Version')
axes[3, 0].set_ylabel('Retention 7 Rate (%)')


# Adjust layout
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

```


    
![png](output_9_0.png)
    


## Dataset Description and Findings

The dataset used in this notebook is `Cookie_Cats_cleaned_v01.csv`. It contains information on user behavior and retention in the mobile game Cookie Cats, before and after an intervention. Below is a detailed description of each column, including the data types and a brief explanation.

## Columns:

1. **version** (object)
   - Indicates whether the user experienced the control or treatment version of the game.
   - Values: `gate_30` (control), `gate_40` (treatment)
   - **Distribution:**
     - gate_30 (Control): 49.56%
     - gate_40 (Treatment): 50.44%
   - The distribution of the `version` column is nearly equal, making it suitable for A/B testing as it ensures a balanced comparison between the control and treatment groups.


2. **sum_gamerounds** (int64)
   - Total number of game rounds played by the user.
   - **Descriptive Statistics:**
     - Mean: 51.87
     - Standard Deviation: 195.05
     - Min: 0
     - 25th Percentile: 5
     - Median: 16
     - 75th Percentile: 51
     - Max: 49,854
     - Kurtosis: 47,130.37
     - Skewness: 185.44
   - **Analysis Insight:**
     - The `sum_gamerounds` column is highly skewed and has a high kurtosis value, indicating that while most users play a small number of rounds, there are outliers who play a significantly larger number of rounds. This skewness and the presence of outliers can impact the analysis and should be taken into consideration during hypothesis testing.


3. **retention_1** (bool)
   - Indicates whether the user returned to play the game one day after installation.
   - Values: `True`, `False`
   - **Distribution:**
     - False: 55.48%
     - True: 44.52%
   - **Analysis Insight:**
     - The `retention_1` column shows that a smaller proportion of users return to play the game one day after installation.


4. **retention_7** (bool)
   - Indicates whether the user returned to play the game seven days after installation.
   - Values: `True`, `False`
   - **Distribution:**
     - False: 81.39%
     - True: 18.61%
   - **Analysis Insight:**
     - The `retention_7` column shows that even fewer users return to play the game seven days after installation.


## Dataset Overview:

- **Total Entries:** 90,189


# 1.2 Define the Problem

Clearly defining the problem is a crucial step in any A/B test. The objective is to compare two groups to determine if there is a significant difference between them after an intervention or treatment. In our case, we have a control group and a treatment group:

- **Control Group**: The group that does not receive the treatment or intervention.
- **Treatment Group**: The group that receives the treatment or intervention.

For this A/B test, we aim to answer the following three questions, two of which pertain to binary outcomes and one to a continuous outcome:

### Question 1: Has the average number of game sessions increased by 5 sessions?

#### **Objective**: To determine if the treatment has increased the average number of game sessions by 5 sessions.

- **Null Hypothesis (H0)**: The average number of game sessions in the treatment group has not increased by 5 sessions compared to the control group.

- **Alternative Hypothesis (H1)**: The average number of game sessions in the treatment group has increased by 5 sessions compared to the control group.

### Question 2: Has player retention increased by 2% after 1 day?

#### **Objective**: To determine if the treatment has increased player retention by 2% one day after installation.

- **Null Hypothesis (H0)**: The player retention rate one day after installation in the treatment group has not increased by 2% compared to the control group.

- **Alternative Hypothesis (H1)**: The player retention rate one day after installation in the treatment group has increased by 2% compared to the control group.

### Question 3: Has player retention increased by 5% after 7 days?

#### **Objective**: To determine if the treatment has increased player retention by 5% seven days after installation.

- **Null Hypothesis (H0)**: The player retention rate seven days after installation in the treatment group has not increased by 5% compared to the control group.

- **Alternative Hypothesis (H1)**: The player retention rate seven days after installation in the treatment group has increased by 5% compared to the control group.



# 2. Preprocessing Steps

Based on the findings from the EDA and the defined problem, preprocess the data to ensure it's ready for testing.

### Preprocessing Steps:
1. **Handling Missing Values**: 
   - Remove or impute missing values as identified in the EDA.
   - Ensuring there are no missing values is crucial for maintaining the integrity of the dataset and avoiding biases in the analysis.


2. **Outlier Detection**: 
   - Identify and handle outliers that may skew the results.
   - Outliers can significantly affect the results of statistical tests and may lead to incorrect conclusions. 


3. **Data Transformation**: 
   - Normalize or standardize data if necessary, especially for continuous variables.
   - This step is important to ensure that different scales of variables do not skew the analysis.


4. **Feature Engineering**: 
   - Create new features if needed to better capture the characteristics of the data.
   - This step involves adding new variables that can help in the analysis, such as interaction terms or derived metrics.


## Preprocessing in Our Case

In our case, the dataset has already been preprocessed and does not contain any missing values. Our focus will be on handling outliers, as we have identified some during the EDA. Data transformation and feature engineering are not necessary because we are not building a predictive model. Instead, we are conducting statistical tests to compare the control and treatment groups.

By ensuring the data is properly preprocessed, we can proceed with the A/B testing confidently, knowing that the dataset is clean and suitable for accurate and reliable analysis.

For more detailed information on data quality and data transformation, refer to the following guides:
- [Complete Guide to Data Quality A to Z](https://www.kaggle.com/code/matinmahmoudi/complete-guide-to-data-quality-a-to-z)
- [Complete Guide to Data Transformation A to Z](https://www.kaggle.com/code/matinmahmoudi/complete-guide-to-data-transformation-a-to-z)

## Removing Outliers Using the IQR Method

To ensure our analysis is not skewed by extreme values, we will use the Interquartile Range (IQR) method to remove outliers from our dataset. This method involves the following steps for each version:

1. **Calculate the IQR for each version.**
2. **Determine the bounds for outliers.**
3. **Filter out the outliers from the dataset.**


To ensure unbiased results in A/B testing, preprocessing steps like outlier handling, normalization, and missing value imputation should be performed **after** splitting the data. This practice maintains the independence of control and treatment groups and ensures fair and tailored processing for each group.



```python
# Separate control and treatment groups
control_group = df[df['version'] == 'gate_30']
treatment_group = df[df['version'] == 'gate_40']
control_group.hist()
treatment_group.hist()
# Define a function to remove outliers using the IQR method
def remove_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Filter the data
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

# Remove outliers for the 'sum_gamerounds' column in control and treatment groups
control_group = remove_outliers_iqr(control_group, 'sum_gamerounds')
treatment_group = remove_outliers_iqr(treatment_group, 'sum_gamerounds')

# Check the shape of the cleaned data
print(f"Original control group shape: {df[df['version'] == 'gate_30'].shape}")
print(f"Cleaned control group shape: {control_group.shape}")
print(f"Original treatment group shape: {df[df['version'] == 'gate_40'].shape}")
print(f"Cleaned treatment group shape: {treatment_group.shape}")
control_group.hist()
treatment_group.hist()
```

    Original control group shape: (44700, 4)
    Cleaned control group shape: (39493, 4)
    Original treatment group shape: (45489, 4)
    Cleaned treatment group shape: (40471, 4)
    




    array([[<AxesSubplot:title={'center':'sum_gamerounds'}>]], dtype=object)




    
![png](output_15_2.png)
    



    
![png](output_15_3.png)
    



    
![png](output_15_4.png)
    



    
![png](output_15_5.png)
    



```python
# Concatenate the cleaned control and treatment groups
df = pd.concat([control_group, treatment_group])

# Verify the concatenation by checking the new shape and by sampling the data
print(f"Shape of the concatenated dataframe: {df.shape}")
print(df.sample(5))  # Display a random sample of 5 rows to verify
```

    Shape of the concatenated dataframe: (79964, 4)
             version  sum_gamerounds  retention_1  retention_7
    userid                                                    
    5229081  gate_30               9        False        False
    534298   gate_30               9        False        False
    370381   gate_40               0        False        False
    7983850  gate_40               2        False        False
    471034   gate_30              14         True        False
    


```python
# Group by version and calculate descriptive statistics for numeric columns
grouped_stats = df.groupby('version')[numeric_cols].describe()

# Additional skewness and kurtosis by version
skewness_by_version = df.groupby('version')[numeric_cols].apply(lambda x: x.skew()).unstack()
kurtosis_by_version = df.groupby('version')[numeric_cols].apply(lambda x: x.kurtosis()).unstack()

# Combine descriptive statistics with skewness and kurtosis
grouped_stats = grouped_stats.unstack().T
grouped_stats['skewness'] = skewness_by_version.T.values.flatten()
grouped_stats['kurtosis'] = kurtosis_by_version.T.values.flatten()
grouped_stats = grouped_stats.reset_index()

print("\nDescriptive Statistics with Skewness and Kurtosis by Version:")
print(grouped_stats)
```

    
    Descriptive Statistics with Skewness and Kurtosis by Version:
               level_0 level_1  version                                         0
    0   sum_gamerounds   count  gate_30                                   39493.0
    1   sum_gamerounds   count  gate_40                                   40471.0
    2   sum_gamerounds    mean  gate_30                                 23.596536
    3   sum_gamerounds    mean  gate_40                                 24.245979
    4   sum_gamerounds     std  gate_30                                 26.584511
    5   sum_gamerounds     std  gate_40                                  28.17279
    6   sum_gamerounds     min  gate_30                                       0.0
    7   sum_gamerounds     min  gate_40                                       0.0
    8   sum_gamerounds     25%  gate_30                                       4.0
    9   sum_gamerounds     25%  gate_40                                       4.0
    10  sum_gamerounds     50%  gate_30                                      13.0
    11  sum_gamerounds     50%  gate_40                                      13.0
    12  sum_gamerounds     75%  gate_30                                      35.0
    13  sum_gamerounds     75%  gate_40                                      34.0
    14  sum_gamerounds     max  gate_30                                     117.0
    15  sum_gamerounds     max  gate_40                                     122.0
    16        skewness                    [1.5264981182659405, 1.558962382928122]
    17        kurtosis                   [1.7017429836991131, 1.7129695451814868]
    

# 3. Set the Probability of Type I and Type II Errors

In hypothesis testing, it is crucial to set the acceptable probabilities for Type I and Type II errors to ensure the robustness of our test results. Here’s a detailed explanation based on our dataset:

### Type I Error (Alpha)

**Definition**: Type I error, denoted by $\alpha$, is the probability of rejecting the null hypothesis when it is actually true. In other words, it represents the risk of concluding that there is an effect or difference when, in reality, there is none.

**Common Value**: The commonly accepted value for $\alpha$ is 0.05 (5%). This means that there is a 5% risk of committing a Type I error.

**Implication in Our Dataset**: For our A/B tests:
- **Question 1**: If we incorrectly conclude that the average number of game sessions has increased by 5 sessions when it has not, we commit a Type I error.
- **Question 2**: If we incorrectly conclude that player retention has increased by 2% after 1 day when it has not, we commit a Type I error.
- **Question 3**: If we incorrectly conclude that player retention has increased by 5% after 7 days when it has not, we commit a Type I error.

### Type II Error (Beta)

**Definition**: Type II error, denoted by $\beta$, is the probability of failing to reject the null hypothesis when it is actually false. In other words, it represents the risk of concluding that there is no effect or difference when, in reality, there is one.

**Common Value**: The commonly accepted value for $\beta$ is 0.20 (20%). This means that there is a 20% risk of committing a Type II error.

**Power of the Test**: The power of a test is defined as $1 - \beta$. A common power value is 0.80 (80%), indicating an 80% chance of correctly rejecting the null hypothesis when it is false.

**Implication in Our Dataset**: For our A/B tests:
- **Question 1**: If we fail to detect an actual increase of 5 sessions in the average number of game sessions, we commit a Type II error.
- **Question 2**: If we fail to detect an actual increase of 2% in player retention after 1 day, we commit a Type II error.
- **Question 3**: If we fail to detect an actual increase of 5% in player retention after 7 days, we commit a Type II error.

### Setting Alpha and Beta for Our A/B Tests

Based on standard practice and the need for robust conclusions, we will use the following values for our tests:

- **Alpha ($\alpha$)**: 0.05 (5%)
- **Beta ($\beta$)**: 0.20 (20%)
- **Power**: 0.80 (80%)

These values strike a balance between minimizing the risk of Type I and Type II errors, ensuring that our test results are both reliable and statistically significant.



```python
import scipy.stats as stats

# Define alpha, beta, and power
alpha = 0.05
beta = 0.20
power = 1 - beta
```

# 4.1 Calculate Sample Size and Choose Samples for Binary A/B Test

To calculate the sample size for a binary A/B test, we need to determine the proportions of success in both the control and treatment groups. The formula for calculating the required sample size for each group in a binary A/B test is:

$$
n = \left( \frac{Z_{\alpha/2} + Z_{\beta}}{p_1 - p_2} \right)^2 \left( p_1(1 - p_1) + p_2(1 - p_2) \right)
$$

Where:
- $Z_{\alpha/2}$ is the Z-score for the chosen significance level.
- $Z_{\beta}$ is the Z-score for the chosen power.
- $p_1$ is the proportion of success in the control group.
- $p_2$ is the proportion of success in the treatment group.

### Steps:
1. Define the baseline proportions ($p_1$) for retention rates in the control group.
2. Define the expected proportions ($p_2$) for retention rates in the treatment group, including the effect size.
3. Use the `statsmodels` library to calculate the required sample size.



### Method 1 ,First implementation approach:


```python
import statsmodels.stats.api as sms

# Recompute crosstabs for retention rates by version
crosstab_retention_1 = pd.crosstab(df['version'], df['retention_1'], normalize='index')
crosstab_retention_7 = pd.crosstab(df['version'], df['retention_7'], normalize='index')

# Extract baseline proportions from crosstab results
p1_retention_1 = crosstab_retention_1.loc['gate_30', True]
p1_retention_7 = crosstab_retention_7.loc['gate_30', True]

# Define expected improvements
p2_retention_1 = p1_retention_1 + 0.02  # 2% increase for retention after 1 day
p2_retention_7 = p1_retention_7 + 0.05  # 5% increase for retention after 7 days

# Define alpha, beta, and calculate power
alpha = 0.05
beta = 0.20
power = 1 - beta

# Calculate effect sizes and required sample sizes
effect_size_retention_1 = sms.proportion_effectsize(p1_retention_1, p2_retention_1)
n_retention_1 = sms.NormalIndPower().solve_power(effect_size=effect_size_retention_1, power=power, alpha=alpha, ratio=1)

effect_size_retention_7 = sms.proportion_effectsize(p1_retention_7, p2_retention_7)
n_retention_7 = sms.NormalIndPower().solve_power(effect_size=effect_size_retention_7, power=power, alpha=alpha, ratio=1)

required_sample_sizes = {
    "Required sample size for retention rate after 1 day test": int(n_retention_1),
    "Required sample size for retention rate after 7 days test": int(n_retention_7)
}

required_sample_sizes

```




    {'Required sample size for retention rate after 1 day test': 9396,
     'Required sample size for retention rate after 7 days test': 757}



### Method 1 ,Second implementation approach:


```python
import statsmodels.stats.api as sms

# Define alpha, beta, and power
alpha = 0.05
beta = 0.20
power = 1 - beta

# Baseline and expected proportions for retention rates
p1_retention_1 = control_group['retention_1'].mean()
p2_retention_1 = p1_retention_1 + 0.02  # 2% increase for retention after 1 day

p1_retention_7 = control_group['retention_7'].mean()
p2_retention_7 = p1_retention_7 + 0.05  # 5% increase for retention after 7 days

# Calculate sample sizes using statsmodels
effect_size_retention_1 = sms.proportion_effectsize(p1_retention_1, p2_retention_1)
n_retention_1 = sms.NormalIndPower().solve_power(effect_size=effect_size_retention_1, power=power, alpha=alpha, ratio=1)

effect_size_retention_7 = sms.proportion_effectsize(p1_retention_7, p2_retention_7)
n_retention_7 = sms.NormalIndPower().solve_power(effect_size=effect_size_retention_7, power=power, alpha=alpha, ratio=1)

print(f"Required sample size for retention rate after 1 day test: {int(n_retention_1)}")
print(f"Required sample size for retention rate after 7 days test: {int(n_retention_7)}")

```

    Required sample size for retention rate after 1 day test: 9396
    Required sample size for retention rate after 7 days test: 757
    

### Method 2 ,Implement from scratch:


```python
import scipy.stats as stats

# Define alpha, beta, and power
alpha = 0.05
beta = 0.20
power = 1 - beta

# Z-scores for the significance level (alpha) and power
Z_alpha = stats.norm.ppf(1 - alpha / 2)
Z_beta = stats.norm.ppf(power)

# Baseline and expected proportions for retention rates
p1_retention_1 = control_group['retention_1'].mean()
p2_retention_1 = p1_retention_1 + 0.02  # 2% increase for retention after 1 day

p1_retention_7 = control_group['retention_7'].mean()
p2_retention_7 = p1_retention_7 + 0.05  # 5% increase for retention after 7 days

# Function to calculate sample size for binary outcome
def calculate_sample_size_binary(p1, p2, Z_alpha, Z_beta):
    pooled_prob = (p1 + p2) / 2
    return int(((Z_alpha * (2 * pooled_prob * (1 - pooled_prob)) ** 0.5 + Z_beta * (p1 * (1 - p1) + p2 * (1 - p2)) ** 0.5) / (p1 - p2)) ** 2)

# Calculate sample sizes
n_retention_1 = calculate_sample_size_binary(p1_retention_1, p2_retention_1, Z_alpha, Z_beta)
n_retention_7 = calculate_sample_size_binary(p1_retention_7, p2_retention_7, Z_alpha, Z_beta)

print(f"Required sample size for retention rate after 1 day test: {n_retention_1}")
print(f"Required sample size for retention rate after 7 days test: {n_retention_7}")

```

    Required sample size for retention rate after 1 day test: 9397
    Required sample size for retention rate after 7 days test: 761
    

# 4.2 Calculate Sample Size and Choose Samples for Continuous A/B Test

To calculate the sample size for a continuous A/B test, we need to know the standard deviation of the population and the expected effect size. The formula for calculating the required sample size for each group in a continuous A/B test is:

$$
n = \left( \frac{Z_{\alpha/2} + Z_{\beta}}{\mu_1 - \mu_2} \right)^2 \cdot 2\sigma^2
$$

Where:
- $Z_{\alpha/2}$ is the Z-score for the chosen significance level.
- $Z_{\beta}$ is the Z-score for the chosen power.
- $\mu_1$ and $\mu_2$ are the means of the control and treatment groups, respectively.
- $\sigma$ is the standard deviation of the population.

### Steps:
1. Define the expected effect size ($\mu_1 - \mu_2$) for the average number of game sessions.
2. Calculate the standard deviation ($\sigma$) of the `sum_gamerounds` column.
3. Use the `statsmodels` library to calculate the required sample size.

Here's the Python code to perform these calculations:


### Method 1:


```python
import numpy as np

# Define effect size for the continuous outcome
effect_size_sessions = 5  # Difference in average number of sessions

# Calculate the standard deviation of the sum_gamerounds column
std_dev_sessions = pd.concat([control_group['sum_gamerounds'],treatment_group['sum_gamerounds']]).std()

# Calculate the required sample size for the continuous outcome using statsmodels
n_sessions = sms.NormalIndPower().solve_power(effect_size=effect_size_sessions / std_dev_sessions, power=power, alpha=alpha, ratio=1)

print(f"Required sample size for average number of game sessions test: {int(n_sessions)}")

```

    Required sample size for average number of game sessions test: 471
    

### Method 2: Implement from scratch:


```python
import numpy as np

# Define effect size for the continuous outcome
effect_size_sessions = 5  # Difference in average number of sessions

# Calculate the standard deviation of the sum_gamerounds column
std_dev_sessions = pd.concat([control_group['sum_gamerounds'],treatment_group['sum_gamerounds']]).std()

# Calculate the required sample size for the continuous outcome
n_sessions = int(((Z_alpha + Z_beta) * std_dev_sessions / effect_size_sessions) ** 2 * 2)

print(f"Required sample size for average number of game sessions test: {n_sessions}")

```

    Required sample size for average number of game sessions test: 471
    

# 5.1 Assign Users for Retention After 1 Day

To conduct the A/B test for retention after 1 day, we need to randomly assign users to the control and treatment groups based on the calculated sample size. This ensures that each group has an equal representation of users, and the test results are reliable.

### Steps:
1. Randomly select the required number of users for the control and treatment groups from the dataset.
2. Ensure that the selected users for both groups meet the sample size requirements.



```python
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Randomly select users for control and treatment groups based on the calculated sample size for retention after 1 day
control_users_retention_1   =   control_group.sample(n=int(n_retention_1), random_state=42)
treatment_users_retention_1 = treatment_group.sample(n=int(n_retention_1), random_state=42)

# Verify the sample sizes
print(f"Sample size for control group (retention 1 day): {len(control_users_retention_1)}")
print(f"Sample size for treatment group (retention 1 day): {len(treatment_users_retention_1)}")


```

    Sample size for control group (retention 1 day): 9397
    Sample size for treatment group (retention 1 day): 9397
    

# 5.2 Assign Users for Retention After 7 Days

To conduct the A/B test for retention after 7 days, we need to randomly assign users to the control and treatment groups based on the calculated sample size. This ensures that each group has an equal representation of users, and the test results are reliable.

### Steps:
1. Randomly select the required number of users for the control and treatment groups from the dataset.
2. Ensure that the selected users for both groups meet the sample size requirements.



```python
# Randomly select users for control and treatment groups based on the calculated sample size for retention after 7 days
control_users_retention_7 = control_group.sample(n=int(n_retention_7), random_state=43)
treatment_users_retention_7 = treatment_group.sample(n=int(n_retention_7), random_state=43)

# Verify the sample sizes
print(f"Sample size for control group (retention 7 days): {len(control_users_retention_7)}")
print(f"Sample size for treatment group (retention 7 days): {len(treatment_users_retention_7)}")


```

    Sample size for control group (retention 7 days): 761
    Sample size for treatment group (retention 7 days): 761
    

# 5.3 Assign Users for Sum of Game Rounds

To conduct the A/B test for the sum of game rounds, we need to randomly assign users to the control and treatment groups based on the calculated sample size. This ensures that each group has an equal representation of users, and the test results are reliable.

### Steps:
1. Randomly select the required number of users for the control and treatment groups from the dataset.
2. Ensure that the selected users for both groups meet the sample size requirements.



```python
# Randomly select users for control and treatment groups based on the calculated sample size for sum of game rounds
control_users_gamerounds = control_group.sample(n=int(n_sessions), random_state=42)
treatment_users_gamerounds = treatment_group.sample(n=int(n_sessions), random_state=42)

# Verify the sample sizes
print(f"Sample size for control group (sum of game rounds): {len(control_users_gamerounds)}")
print(f"Sample size for treatment group (sum of game rounds): {len(treatment_users_gamerounds)}")
```

    Sample size for control group (sum of game rounds): 471
    Sample size for treatment group (sum of game rounds): 471
    

# 6.1 Perform A/B Test for Retention After 1 Day

**Question**: Has player retention increased by 2% after 1 day?

To determine if the retention rate after 1 day has increased by 2% in the treatment group compared to the control group, we will perform a Z-test for proportions.

### Assumptions for the Z-Test:
1. **Independence**:
   - Observations in the sample must be independent of each other.
   - This is typically ensured by random sampling and proper experimental design.


2. **Normality**:
   - For a Z-test, we assume that the sampling distribution of the sample proportion is approximately normal.
   - This assumption holds if the sample size is large enough, usually with both \(np\) and \(n(1-p)\) greater than 5.


3. **Large Sample Size**:
   - The number of successes and failures in each group should be large enough to justify the use of the normal approximation to the binomial distribution.

### Steps:
1. Calculate the observed proportions of retention in both groups.
2. Perform a Z-test for proportions to compare the retention rates.
3. Determine if the difference is statistically significant.


### Method 1 ,First implementation approach:


```python
from statsmodels.stats.proportion import proportions_ztest

# Define alpha 
alpha = 0.05

# Calculate the observed proportions of retention in both groups
p_control = control_users_retention_1['retention_1'].mean()
p_treatment = treatment_users_retention_1['retention_1'].mean()
n_control = len(control_users_retention_1)
n_treatment = len(treatment_users_retention_1)

# Ensure np and n(1-p) are greater than 5 for both groups
assumptions_met = True
if n_control * p_control <= 5:
    print("Assumption not met: np > 5 for control group")
    assumptions_met = False
if n_control * (1 - p_control) <= 5:
    print("Assumption not met: n(1-p) > 5 for control group")
    assumptions_met = False
if n_treatment * p_treatment <= 5:
    print("Assumption not met: np > 5 for treatment group")
    assumptions_met = False
if n_treatment * (1 - p_treatment) <= 5:
    print("Assumption not met: n(1-p) > 5 for treatment group")
    assumptions_met = False

if assumptions_met:
    print("All assumptions are met for the Z-test.")

    # Calculate the number of successes (retained users) and the number of trials (total users) in both groups
    successes_retention_1 = [treatment_users_retention_1['retention_1'].sum(), control_users_retention_1['retention_1'].sum()]
    n_obs_retention_1 = [len(treatment_users_retention_1), len(control_users_retention_1)]

    # Perform a one-tailed Z-test for proportions to check if the treatment is better than control
    z_stat_retention_1, p_val_retention_1 = proportions_ztest(successes_retention_1, n_obs_retention_1, value=0, alternative='larger')
    
    # Print the results
    print(f"Z-statistic for retention after 1 day: {z_stat_retention_1}")
    print(f"P-value for retention after 1 day: {p_val_retention_1}")

    # Determine if the result is statistically significant
    if p_val_retention_1 < alpha:
        print("Reject the null hypothesis: The retention rate after 1 day has significantly increased.")
    else:
        print("Fail to reject the null hypothesis: The retention rate after 1 day has not significantly increased.")
else:
    print("Z-test assumptions not met, cannot perform the test.")

```

    All assumptions are met for the Z-test.
    Z-statistic for retention after 1 day: -0.6299651909770788
    P-value for retention after 1 day: 0.73564132058017
    Fail to reject the null hypothesis: The retention rate after 1 day has not significantly increased.
    

### Method 1 ,Second implementation approach


```python
from statsmodels.stats.proportion import proportions_ztest

# Define alpha 
alpha = 0.05

# Calculate the observed proportions of retention in both groups
p_control = control_users_retention_1['retention_1'].mean()
p_treatment = treatment_users_retention_1['retention_1'].mean()
n_control = len(control_users_retention_1)
n_treatment = len(treatment_users_retention_1)

# Ensure np and n(1-p) are greater than 5 for both groups
assumptions_met = True
if n_control * p_control <= 5:
    print("Assumption not met: np > 5 for control group")
    assumptions_met = False
if n_control * (1 - p_control) <= 5:
    print("Assumption not met: n(1-p) > 5 for control group")
    assumptions_met = False
if n_treatment * p_treatment <= 5:
    print("Assumption not met: np > 5 for treatment group")
    assumptions_met = False
if n_treatment * (1 - p_treatment) <= 5:
    print("Assumption not met: n(1-p) > 5 for treatment group")
    assumptions_met = False

if assumptions_met:
    print("All assumptions are met for the Z-test.")

    # Calculate the number of successes (retained users) and the number of trials (total users) in both groups
    successes_retention_1 = [control_users_retention_1['retention_1'].sum(), treatment_users_retention_1['retention_1'].sum()]
    n_obs_retention_1 = [len(control_users_retention_1), len(treatment_users_retention_1)]
    print(successes_retention_1, n_obs_retention_1)
    # Perform a one-tailed Z-test for proportions to check if the treatment is worse than control
    z_stat_retention_1, p_val_retention_1 = proportions_ztest(successes_retention_1, n_obs_retention_1, value=0, alternative='smaller')
    
    # Print the results
    print(f"Z-statistic for retention after 1 day: {z_stat_retention_1}")
    print(f"P-value for retention after 1 day: {p_val_retention_1}")

    # Determine if the result is statistically significant
    if p_val_retention_1 < alpha:
        print("Reject the null hypothesis: The retention rate after 1 day has significantly increased.")
    else:
        print("Fail to reject the null hypothesis: The retention rate after 1 day has not significantly increased.")
else:
    print("Z-test assumptions not met, cannot perform the test.")


```

    All assumptions are met for the Z-test.
    [3628, 3586] [9397, 9397]
    Z-statistic for retention after 1 day: 0.6299651909770788
    P-value for retention after 1 day: 0.73564132058017
    Fail to reject the null hypothesis: The retention rate after 1 day has not significantly increased.
    

# 6.2 Perform A/B Test for Retention After 7 Days

**Question**: Has player retention increased by 5% after 7 days?

To determine if the retention rate after 7 days has increased by 5% in the treatment group compared to the control group, we will perform a Z-test for proportions.

### Assumptions for the Z-Test:
1. **Independence**:
   - Observations in the sample must be independent of each other.
   - This is typically ensured by random sampling and proper experimental design.


2. **Normality**:
   - For a Z-test, we assume that the sampling distribution of the sample proportion is approximately normal.
   - This assumption holds if the sample size is large enough, usually with both \(np\) and \(n(1-p)\) greater than 5.


3. **Large Sample Size**:
   - The number of successes and failures in each group should be large enough to justify the use of the normal approximation to the binomial distribution.

### Steps:
1. Calculate the observed proportions of retention in both groups.
2. Perform a Z-test for proportions to compare the retention rates.
3. Determine if the difference is statistically significant.


### Method 1 ,First implementation approach:


```python
from statsmodels.stats.proportion import proportions_ztest

# Define alpha 
alpha = 0.05

# Check the assumptions
# Calculate the observed proportions of retention in both groups
p_control_7 = control_users_retention_7['retention_7'].mean()
p_treatment_7 = treatment_users_retention_7['retention_7'].mean()
n_control_7 = len(control_users_retention_7)
n_treatment_7 = len(treatment_users_retention_7)

# Ensure np and n(1-p) are greater than 5 for both groups
assumptions_met = True
if n_control_7 * p_control_7 <= 5:
    print("Assumption not met: np > 5 for control group")
    assumptions_met = False
if n_control_7 * (1 - p_control_7) <= 5:
    print("Assumption not met: n(1-p) > 5 for control group")
    assumptions_met = False
if n_treatment_7 * p_treatment_7 <= 5:
    print("Assumption not met: np > 5 for treatment group")
    assumptions_met = False
if n_treatment_7 * (1 - p_treatment_7) <= 5:
    print("Assumption not met: n(1-p) > 5 for treatment group")
    assumptions_met = False

if assumptions_met:
    print("All assumptions are met for the Z-test.")

    # Calculate the number of successes (retained users) and the number of trials (total users) in both groups
    successes_retention_7 = [treatment_users_retention_7['retention_7'].sum(), control_users_retention_7['retention_7'].sum()]
    n_obs_retention_7 = [len(treatment_users_retention_7), len(control_users_retention_7)]

    # Perform a one-tailed Z-test for proportions to check if the treatment is better than control
    z_stat_retention_7, p_val_retention_7 = proportions_ztest(successes_retention_7, n_obs_retention_7, value=0, alternative='larger')
    
    # Print the results
    print(f"Z-statistic for retention after 7 days: {z_stat_retention_7}")
    print(f"P-value for retention after 7 days: {p_val_retention_7}")

    # Determine if the result is statistically significant
    if p_val_retention_7 < alpha:
        print("Reject the null hypothesis: The retention rate after 7 days has significantly increased.")
    else:
        print("Fail to reject the null hypothesis: The retention rate after 7 days has not significantly increased.")
else:
    print("Z-test assumptions not met, cannot perform the test.")

```

    All assumptions are met for the Z-test.
    Z-statistic for retention after 7 days: -1.933298872613932
    P-value for retention after 7 days: 0.9734003046855899
    Fail to reject the null hypothesis: The retention rate after 7 days has not significantly increased.
    

### Method 1 ,Second implementation approach:


```python
from statsmodels.stats.proportion import proportions_ztest

# Define alpha 
alpha = 0.05

# Check the assumptions
# Calculate the observed proportions of retention in both groups
p_control_7 = control_users_retention_7['retention_7'].mean()
p_treatment_7 = treatment_users_retention_7['retention_7'].mean()
n_control_7 = len(control_users_retention_7)
n_treatment_7 = len(treatment_users_retention_7)

# Ensure np and n(1-p) are greater than 5 for both groups
assumptions_met = True
if n_control_7 * p_control_7 <= 5:
    print("Assumption not met: np > 5 for control group")
    assumptions_met = False
if n_control_7 * (1 - p_control_7) <= 5:
    print("Assumption not met: n(1-p) > 5 for control group")
    assumptions_met = False
if n_treatment_7 * p_treatment_7 <= 5:
    print("Assumption not met: np > 5 for treatment group")
    assumptions_met = False
if n_treatment_7 * (1 - p_treatment_7) <= 5:
    print("Assumption not met: n(1-p) > 5 for treatment group")
    assumptions_met = False

if assumptions_met:
    print("All assumptions are met for the Z-test.")

    # Calculate the number of successes (retained users) and the number of trials (total users) in both groups
    successes_retention_7 = [control_users_retention_7['retention_7'].sum(), treatment_users_retention_7['retention_7'].sum()]
    n_obs_retention_7 = [len(control_users_retention_7), len(treatment_users_retention_7)]

    # Perform a one-tailed Z-test for proportions to check if the treatment is worse than control
    z_stat_retention_7, p_val_retention_7 = proportions_ztest(successes_retention_7, n_obs_retention_7, value=0, alternative='smaller')
    
    # Print the results
    print(f"Z-statistic for retention after 7 days: {z_stat_retention_7}")
    print(f"P-value for retention after 7 days: {p_val_retention_7}")

    # Determine if the result is statistically significant
    if p_val_retention_7 < alpha:
        print("Reject the null hypothesis: The retention rate after 7 days has significantly increased.")
    else:
        print("Fail to reject the null hypothesis: The retention rate after 7 days has not significantly increased.")
else:
    print("Z-test assumptions not met, cannot perform the test.")

```

    All assumptions are met for the Z-test.
    Z-statistic for retention after 7 days: 1.933298872613932
    P-value for retention after 7 days: 0.9734003046855899
    Fail to reject the null hypothesis: The retention rate after 7 days has not significantly increased.
    

# 6.3 Perform A/B Test for Sum of Game Rounds

**Question**: Has the average number of game sessions increased by 5 sessions?

To determine if the average number of game sessions has increased by 5 sessions in the treatment group compared to the control group, we will perform a two-sample t-test.

### Assumptions for the T-Test:
1. **Independence**:
   - Observations in the sample must be independent of each other.
   - This is typically ensured by random sampling and proper experimental design.


2. **Normality**:
   - For a t-test, we assume that the samples are drawn from a normally distributed population.
   - This assumption can be relaxed if the sample size is large (Central Limit Theorem).
   

3. **Equal Variances**:
   - The t-test assumes that the variances of the two groups are equal. This can be tested using Levene’s test.

### Steps:
1. Calculate the means and standard deviations of the sum of game rounds in both groups.
2. Perform a two-sample t-test to compare the means.
3. Determine if the difference is statistically significant.


### Method 1 ,First implementation approach:


```python
from scipy.stats import ttest_ind, levene

# Define alpha
alpha = 0.05
nx, ny = len(treatment_users_gamerounds), len(control_users_gamerounds),

# Calculate the means and standard deviations of the sum of game rounds in both groups
mean_control_gamerounds = control_users_gamerounds['sum_gamerounds'].mean()
mean_treatment_gamerounds = treatment_users_gamerounds['sum_gamerounds'].mean()
std_control_gamerounds = control_users_gamerounds['sum_gamerounds'].std()
std_treatment_gamerounds = treatment_users_gamerounds['sum_gamerounds'].std()

print(f"Control group - Mean: {mean_control_gamerounds}, Std Dev: {std_control_gamerounds}")
print(f"Treatment group - Mean: {mean_treatment_gamerounds}, Std Dev: {std_treatment_gamerounds}")

# Perform Levene's test for equality of variances
stat, p_levene = levene(control_users_gamerounds['sum_gamerounds'], treatment_users_gamerounds['sum_gamerounds'])

# Check if variances are equal
if p_levene < alpha:
    print("Assumption not met: The variances of the two groups are not equal. Switching to Welch's t-test.")
    equal_var = False
else:
    print("Assumption met: The variances of the two groups are equal.")
    equal_var = True

# calculating T-statistic 
var = ((nx-1)*std_control_gamerounds**2 + (ny-1)*std_treatment_gamerounds**2)/(nx+ny-2)
t = ((mean_treatment_gamerounds - mean_control_gamerounds)-0)/np.sqrt((var/nx) + (var/ny))
print('T-statistic: ', t)

# Perform a one-tailed t-test directly comparing the two groups
t_stat_gamerounds, p_val_gamerounds = ttest_ind(treatment_users_gamerounds['sum_gamerounds'], control_users_gamerounds['sum_gamerounds'], equal_var=equal_var, alternative='greater')

# Print the results
print(f"T-statistic for sum of game rounds: {t_stat_gamerounds}")
print(f"P-value for sum of game rounds: {p_val_gamerounds}")

# Determine if the result is statistically significant
if p_val_gamerounds < alpha:
    print("Reject the null hypothesis: The average number of game sessions in the treatment group is significantly greater than in the control group.")
else:
    print("Fail to reject the null hypothesis: The average number of game sessions in the treatment group is not significantly greater than in the control group.")

```

    Control group - Mean: 21.6135881104034, Std Dev: 24.122456785190625
    Treatment group - Mean: 25.656050955414013, Std Dev: 28.61848384459519
    Assumption not met: The variances of the two groups are not equal. Switching to Welch's t-test.
    T-statistic:  2.3439658998783415
    T-statistic for sum of game rounds: 2.343965899878341
    P-value for sum of game rounds: 0.00964666595715196
    Reject the null hypothesis: The average number of game sessions in the treatment group is significantly greater than in the control group.
    

### Method 1 ,Second implementation approach:


```python
from scipy.stats import ttest_ind, levene

# Define alpha
alpha = 0.05

# Calculate the means and standard deviations of the sum of game rounds in both groups
mean_control_gamerounds = control_users_gamerounds['sum_gamerounds'].mean()
mean_treatment_gamerounds = treatment_users_gamerounds['sum_gamerounds'].mean()
std_control_gamerounds = control_users_gamerounds['sum_gamerounds'].std()
std_treatment_gamerounds = treatment_users_gamerounds['sum_gamerounds'].std()

print(f"Control group - Mean: {mean_control_gamerounds}, Std Dev: {std_control_gamerounds}")
print(f"Treatment group - Mean: {mean_treatment_gamerounds}, Std Dev: {std_treatment_gamerounds}")

# Perform Levene's test for equality of variances
stat, p_levene = levene(control_users_gamerounds['sum_gamerounds'], treatment_users_gamerounds['sum_gamerounds'])

# Check if variances are equal
if p_levene < alpha:
    print("Assumption not met: The variances of the two groups are not equal. Switching to Welch's t-test.")
    equal_var = False
else:
    print("Assumption met: The variances of the two groups are equal.")
    equal_var = True

# Perform a one-tailed t-test directly comparing the two groups
t_stat_gamerounds, p_val_gamerounds = ttest_ind(control_users_gamerounds['sum_gamerounds'], treatment_users_gamerounds['sum_gamerounds'], equal_var=equal_var, alternative='less')

# Print the results
print(f"T-statistic for sum of game rounds: {t_stat_gamerounds}")
print(f"P-value for sum of game rounds: {p_val_gamerounds}")

# Determine if the result is statistically significant
if p_val_gamerounds < alpha:
    print("Reject the null hypothesis: The average number of game sessions in the treatment group is significantly greater than in the control group.")
else:
    print("Fail to reject the null hypothesis: The average number of game sessions in the treatment group is not significantly greater than in the control group.")

```

    Control group - Mean: 21.6135881104034, Std Dev: 24.122456785190625
    Treatment group - Mean: 25.656050955414013, Std Dev: 28.61848384459519
    Assumption not met: The variances of the two groups are not equal. Switching to Welch's t-test.
    T-statistic for sum of game rounds: -2.343965899878341
    P-value for sum of game rounds: 0.00964666595715196
    Reject the null hypothesis: The average number of game sessions in the treatment group is significantly greater than in the control group.
    

# 7. Analyze the Test Results

### Practical Significance

Statistical significance alone does not guarantee that the observed effect is meaningful in a real-world context. Practical significance assesses whether the magnitude of the effect has real-world implications and value. Here’s how to assess practical significance:

1. **Effect Size**:
   - Measure the effect size to understand the magnitude of the difference between the control and treatment groups.
   - For retention rates, the effect size is the difference in proportions (e.g., a 2% increase in retention).
   - For the sum of game rounds, the effect size is the difference in means (e.g., an average increase of 5 game sessions).


2. **Real-World Impact**:
   - Consider the implications of the observed effect in a real-world context. For example:
     - A 2% increase in 1-day retention might lead to higher user engagement and potential revenue.
     - A 5% increase in 7-day retention indicates improved long-term engagement, which is valuable for customer retention strategies.
     - An average increase of 5 game sessions per user suggests enhanced user experience and satisfaction.


3. **Cost-Benefit Analysis**:
   - Evaluate the costs associated with implementing the treatment compared to the benefits derived from the observed effect.
   - Consider factors such as development costs, marketing expenses, and potential revenue gains.


### Report Findings

After analyzing the statistical and practical significance of the results, it is important to compile and report the findings in a clear and comprehensive manner. Here’s how to report the findings:

1. **Summary of Results**:
   - Provide a summary of the key findings from the A/B tests, including the observed differences between the control and treatment groups for each test (retention after 1 day, retention after 7 days, and sum of game rounds).


2. **Statistical Analysis Results**:
   - Present the Z-statistics and p-values for the retention tests, and the T-statistics and p-values for the sum of game rounds test.
   - Clearly state whether the null hypothesis was rejected or not for each test.


3. **Practical Implications**:
   - Discuss the practical significance of the results, including the effect sizes and their real-world impact.
   - Highlight the potential benefits and costs associated with implementing the changes based on the test results.


4. **Visualizations**:
   - Include visualizations such as bar charts, line graphs, or box plots to illustrate the differences between the control and treatment groups.
   - Use graphs to make the results more understandable and accessible to a broader audience.


5. **Recommendations**:
   - Provide recommendations based on the findings, such as whether to implement the treatment, conduct further testing, or consider alternative strategies.
   - Offer actionable insights and next steps based on the results of the A/B tests.
