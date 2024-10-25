**Important Notes on ANOVA Test**

- **ANOVA test** is used to check whether the means of **three or more groups** are different or not by using estimation parameters such as **the variance**.
- An **ANOVA table** is used to summarize the results of an ANOVA test.
- There are two types of ANOVA tests - **one way ANOVA** and **two way ANOVA**
- One way ANOVA has **only one** independent variable while a two way ANOVA has **two** independent variables.  


For **more statistical tests** check [this](https://www.kaggle.com/code/arthurarstamyan/statistical-tests) notebook.

# One way ANOVA

- One-way ANOVA is used to determine if there are **statistically significant** differences between the means of **three or more independent groups**. 
- It **extends the t-test** to more than two groups.
- The data in each group should be **normally distributed**.
- The variances of the groups should be approximately equal (**homoscedasticity**).

<img src="https://i.postimg.cc/pVQ02693/one-way-anova.jpg">


**Error** is also called **Within Groups** and **SSE** is also written as **SSW(Sum of Squares Within)**  

The hypothesis for a one way ANOVA test can be set up as follows:

**Null Hypothesis,** H0: μ1 = μ2 = μ3 = ... = μk (The means are equal).  
**Alternative Hypothesis,** H1: The means are not equal.

**Decision Rule:** If F test statistic > F critical value then reject the null hypothesis and conclude that the means of at least two groups are statistically significant.  

**So we need to calculate F test statistic.**  
**Step 1:** Calculate the mean for each group.  
**Step 2:** Calculate the total mean. This is done by adding all the means and dividing it by the total number of means.  
**Step 3:** Calculate the SSB.  
**Step 4:** Calculate the between groups degrees of freedom.  
**Step 5:** Calculate the SSE(SSW-Sum of Squares Within).  
**Step 6:** Calculate the degrees of freedom of errors.  
**Step 7:** Determine the MSB and the MSE.  
**Step 8:** Find the F test statistic.  
**Step 9:** Using the F table for the specified level of significance, α, find the critical value. This is given by F_critical(α, df1, df2).  
**Step 10:** If F > F_critical then reject the null hypothesis.


**Limitations of One Way ANOVA Test**  
The one way ANOVA is an **omnibus test statistic**. This implies that the test will determine whether the means of the various groups are statistically significant or not. However, it cannot **distinguish the specific groups** that have a statistically significant mean. Thus, to find the specific group with a different mean, a **post hoc test** needs to be conducted.

## Function(One Way ANOVA)


```python
import numpy as np
import pandas as pd

def one_way_anova(df):
    group_means = df.mean()
    total_mean = group_means.mean()
    n, k = df.shape
    N = k*n
    dfb = k - 1
    dfe = N - k
    SSB = ((group_means-total_mean)**2)@np.array([n]*k)
    SSE = ((df-group_means)**2).sum().sum()
    MSB = SSB/dfb
    MSE = SSE/dfe
    F = MSB/MSE
    anova = pd.DataFrame(columns=['Sum of Squares', 'Degrees of freedom', 'Mean Squares', 'F value'])
    anova.loc['Between Groups'] = [SSB, dfb, MSB, F]
    anova.loc['Error(Within Groups)'] = [SSE, dfe, MSE, '']
    anova.loc['Total'] = [SSB+SSE, dfb+dfe, '', '']
    return F, anova
```

## Example 1(One Way ANOVA)

Three types of fertilizers are used on **three groups of plants for 5 weeks**.
We want to check if there is a difference in the mean growth of each group. Using the data given below apply a **one way ANOVA test at 0.05 significant level**.  

### Data


```python
df = pd.DataFrame({'Fertilizer_1': [6,8,4,5,3,4], 'Fertilizer_2': [8,12,9,11,6,8], 'Fertilizer_3':[13,9,11,8,7,12]})
df
```

### Calculations


```python
F, anova = one_way_anova(df)
display(anova)
```

**F table of critical values of a=0.05** - https://statisticsbyjim.com/wp-content/uploads/2022/02/F-table_Alpha05.png  
F_critical(a, dfb, dfe) = F_critical(0.05, 2, 15) = 3.68


```python
F_critical = 3.68
```

### Conclusion


```python
if F>F_critical:
    print('The null hypothesis is rejected')
else:
    print("The null hypothesis can't be rejected")
```

**So it can be concluded that there is a difference in the mean growth of the plants.**

## Example 2(One Way ANOVA)

A trial was run to check the effects of **different diets**. Positive numbers indicate weight loss and negative numbers indicate weight gain. Check if there is an average difference in the weight of people following different diets at a **0.05 alpha level** using an ANOVA Table .

### Data


```python
df = pd.DataFrame({'Low Fat': [8,9,6,7,3], 'Low Calorie': [2,4,3,5,1], 
                   'Low Protein':[3,5,4,2,3], 'Low Carbohydrate': [2,2,-1,0,3]})
df
```

### Calculations


```python
F, anova = one_way_anova(df)
display(anova)
```

**F table of critical values of a=0.05** - https://statisticsbyjim.com/wp-content/uploads/2022/02/F-table_Alpha05.png  
F_critical(a, dfb, dfe) = F_critical(0.05, 3, 16) = 3.24


```python
F_critical = 3.24
```

### Conclusion


```python
if F>F_critical:
    print('The null hypothesis is rejected')
else:
    print("The null hypothesis can't be rejected")
```

**So there is an average difference in the weight of people following different diets**

## Example 3(One Way ANOVA)

Determine if there is a difference in the mean daily calcium intake for people with **normal bone density**, **osteopenia**, and **osteoporosis** at a **0.05 alpha level**.

### Data


```python
df = pd.DataFrame({'Normal Density': [1200,1000,980,900,750,800], 'Osteopenia': [1000,1100,700,800,500,700], 
                   'Osteoporosis':[890,650,1100,900,400,350]})
df
```

### Calculations


```python
F, anova = one_way_anova(df)
display(anova)
```

**F table of critical values of a=0.05** - https://statisticsbyjim.com/wp-content/uploads/2022/02/F-table_Alpha05.png   
F_critical(a, dfb, dfe) = F_critical(0.05, 2, 15) = 3.68


```python
F_critical = 3.68
```

### Conclusion


```python
if F>F_critical:
    print('The null hypothesis is rejected')
else:
    print("The null hypothesis can't be rejected")
```

**So there is not enough evidence to prove that the mean daily calcium intake of the three groups is different**

# Two way ANOVA

<img src='https://i.postimg.cc/59m3cQTV/two-way-anova.jpg' width=800 height=400>

In order to use a two-way ANOVA, the dataset should have **two independent variables**. Thus, it can be thought of as an **extension of a one way ANOVA** where only one variable affects the dependent variable. These variables are called **factors**, each with more than one level. For example, if one of the factors is **color**, the levels within the factor may be **light**, **neutral**, and **dark**. If the dataset fits these requirements and a person wants to know how the two factors affect the response variable, two-way ANOVA is likely a good method to use. A two way ANOVA test is used to check the main **effect of each independent variable** and to see if there is an **interaction effect** between them. To examine the main effect, each factor is considered separately as done in a **one way ANOVA**. Furthermore, to check the **interaction effect**, all factors are considered at the same time. 

There are **4** assumptions that must be met **before** using two-way ANOVA:  
- **Normality**: Observations from the sample population are normally distributed.
- **Sample Size**: The number of observations must be the same for each group.
- **Equal Variances**: The variances for each group are equal.
- **Independence**: Observations in each group are independent.

When performing a two-way ANOVA, there are **3 hypotheses** to test. The first two determine whether **each factor** has an effect on **the response variable**, and the third one decides if there is any **interaction** between the two factors.

- **Null Hypothesis #1**: The population means of the first factor are the same.
- **Null Hypothesis #2**: The population means of the second factor are the same.
- **Null Hypothesis #3**: There are no interaction effects between the factors.

## Function(Two Way ANOVA)


```python
import numpy as np
import pandas as pd

def two_way_anova(df):  
    n, n_factorB_groups = df.shape
    N = n*n_factorB_groups
    n_factorA_groups = len(df.index.unique())
    n_group_observations = n/n_factorA_groups

    total_mean = df.mean().mean()

    SSW = 0
    factorA_means = []
    for index in df.index.unique(): 
        factorA_means.append(df[df.index == index].mean().mean())
        for column in df.columns: 
            group = df[df.index == index][column]
            SSW += ((group-group.mean())**2).sum()

    factorB_means = []
    for column in df.columns: 
        factorB_means.append(df[column].mean())

    SSA = (((factorA_means-total_mean)**2)@np.array([n_group_observations*n_factorB_groups]*n_factorA_groups)).round(2)
    SSB = (((factorB_means-total_mean)**2)@np.array([n_group_observations*n_factorA_groups]*n_factorB_groups)).round(2)
    SST = (((df-total_mean)**2).sum().sum()).round(2)
    SSAB = (SST - (SSA+SSB+SSW)).round(2)

    dfA, dfB = n_factorA_groups-1, n_factorB_groups-1
    dfAB = dfA * dfB
    dfW = N - n_factorA_groups * n_factorB_groups
    dfTotal = N-1

    MSA, MSB, MSAB, MSW = SSA/dfA, SSB/dfB, SSAB/dfAB, SSW/dfW
    FA, FB, FAB = MSA/MSW, MSB/MSW, MSAB/MSW
    Fs = [FA, FB, FAB]

    anova = pd.DataFrame(columns=['Sum of Squares', 'Degrees of freedom', 'Mean Squares', 'F value'])
    anova.loc['Factor A'] = [SSA, dfA, MSA, FA]
    anova.loc['Factor B'] = [SSB, dfB, MSB, FB]
    anova.loc['Interaction effect'] = [SSAB, dfAB, MSAB, FAB]
    anova.loc['Within(Error))'] = [SSW, dfW, MSW, '']
    anova.loc['Total'] = [SST, dfTotal, '', '']
    
    return Fs, anova
```

## Example 1(Two Way ANOVA)

A farmer wants to see if there is a difference in the **average height** for two new **strains of hemp plants**. They believe there also may be some **interaction with different soil types** so they plant 5 hemp plants of each strain in 4 types of soil: **sandy**, **clay**, **loam** and **silt**. At **α = 0.01**, analyze the data shown, using a two-way ANOVA.

### Data


```python
df = pd.DataFrame({'Sandy':[60,53,58,62,57,36,41,54,65,53], 'Clay': [54,63,62,71,76,62,61,77,53,64],
                   'Loam': [80,82,62,88,71,68,72,71,82,86], 'Silt': [62,76,55,48,61,63,65,72,71,63]}, 
                  index=['Strain A']*5+['Strain B']*5)
df
```

### Calculations


```python
Fs, anova = two_way_anova(df)
display(anova)
```

**F table of critical values of a=0.01** - https://statisticsbyjim.com/wp-content/uploads/2022/02/F-table_Alpha01.png  
**F_criticals** = [**F_critical(**0.01, dfA(1), dfW(32)**)**, **F_critical(**0.01, dfB(3), dfW(32)**)**, **F_critical(**0.01, dfAB(3), dfW(32)**)**] =  [7.4993, 4.4594, 4.4594]


```python
F_criticals = [7.4993, 4.4594, 4.4594]
```

### Conclusions


```python
for i in range(3):
    if Fs[i]>F_criticals[i]:
        print(f'The Null Hypothesis #{i+1} is rejected')
    else:
        print(f"The Null Hypothesis #{i+1} can't be rejected")
```

**So**
- There is **no significant difference** in the mean plant height between the **two hemp strains**.
- The **type of soil** has a **significant effect** on the mean plant height.
- There is **no significant interaction** effect **between** the hemp strain and the soil type on the mean plant height.
