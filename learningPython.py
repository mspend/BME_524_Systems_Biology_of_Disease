##########################################################
## OncoMerge:  learningPython.py                        ##
##  ______     ______     __  __                        ##
## /\  __ \   /\  ___\   /\ \/\ \                       ##
## \ \  __ \  \ \___  \  \ \ \_\ \                      ##
##  \ \_\ \_\  \/\_____\  \ \_____\                     ##
##   \/_/\/_/   \/_____/   \/_____/                     ##
## @Developed by: Plaisier Lab                          ##
##   (https://plaisierlab.engineering.asu.edu/)         ##
##   Arizona State University                           ##
##   242 ISTB1, 550 E Orange St                         ##
##   Tempe, AZ  85281                                   ##
## @Author:  Chris Plaisier                             ##
## @License:  GNU GPLv3                                 ##
##                                                      ##
## If this program is used in your analysis please      ##
## mention who built it. Thanks. :-)                    ##
##########################################################

# Making variables and addition
a = 3
b = 2
print(a + b)
print(a * b)

## More fun with mathematical operators
# Import our first library
import math

# Use sqrt and factorial
a = 9
b = -9
# math.<function>(<parameter>)
print(math.sqrt(a))
print(math.sqrt(abs(b)))
print(math.factorial(a))


## Variable types
# Integer
a = int(9)
b = float(9)
c = str(9)
d = bool(9)
print(a + b) # Int + Float = Float
print(b + c) # Float + Str = ERROR
print(b + d) # Float + Bool = Float (note True == 1)


## Data structure types
# List
a = [0,1,2,3,4]
b = [3,4,5,6,7]
print(b[0]) # Access list elements via integer, python is zero based
print(a + b) # Joins the lists
a.append(b)
print(a) # Sticks b as the last element of a

# Set
a = [0,1,2,3,4]
b = [3,4,5,6,7]
set(a).intersection(b) # what values are in both lists
print(list(set(a + b))) # make the values in the combined list unique

# Dictionary
a = {'Bill':10, 'Sally':5, 'Trevor':23, 'Saheed':8}
print(a['Bill']+a['Saheed'])

# Nesting data structures
a = {'top':{'middle':['bottom']}}
print(a['top']['middle'][0])


## Control structures
# If statement
a = [0,1,2,3,4]
if a[2] % 2 == 0:
    print('Even number')

# If, else
if not a[2] % 2 == 0:
    print('Odd number')
else:
    print('Even number')

# If, elif, else
a = bool(0)
if type(a)==int:
    print('Integer')
elif type(a)==float:
    print('Float')
elif type(a)==bool:
    print('Bool')
else:
    print('Dunno')

# For loop
for i in range(5):
    print(i*2)

# While loop
i = 0
while i<5:
    print(i*2)
    i += 1


## Functions
# Define the function
def putABirdOnIt(it):
    return str(it)+'_Bird'

# Run the function
print(putABirdOnIt('Cat'))
print(putABirdOnIt('Small'))
print(putABirdOnIt('Big'))


## Importing modules
# Standard import
import scipy.stats

print(scipy.stats.pearsonr([0,2,4,6,8],[1,1.5,4,3,9]))

# Import sub-package component
from scipy import stats

print(stats.pearsonr([0,2,4,6,8],[1,1.5,4,3,9]))

# Rename imported module
import scipy.stats as st

print(st.pearsonr([0,2,4,6,8],[1,1.5,4,3,9])) # Parametric
print(st.spearmanr([0,2,4,6,8],[1,1.5,4,3,9])) # Non-parametric


## Running a Student's T-test
from scipy import stats
import numpy as np
rng = np.random.default_rng()

# Simulate some variables
rvs1 = stats.norm.rvs(loc=5, scale=10, size=500, random_state=rng)
rvs2 = stats.norm.rvs(loc=5, scale=10, size=500, random_state=rng)
print(stats.ttest_ind(rvs1, rvs2))

# Simulate variable with different mean
rvs3 = stats.norm.rvs(loc=3, scale=10, size=500, random_state=rng)
print(stats.ttest_ind(rvs1, rvs3))


## Calculating power for a T-test
# Estimate sample size via power analysis
from statsmodels.stats.power import TTestIndPower

# Parameters for power analysis
effect = 0.8 # Cohen’s d measure, 0.8 = larger effect size
alpha = 0.05 # alpha value
power = 0.8 # 1-beta, 0.8 = 80%

# Perform power analysis
analysis = TTestIndPower()
result = analysis.solve_power(effect, power=power, nobs1=None, ratio=1.0, alpha=alpha)
print('Sample Size: %.3f' % result) # Pretty formatting which rounds to 3 decimal digits

# Looping the power test
effects = [0.2,0.4,0.6,0.8] # Cohen’s d measure, 0.8 = larger effect size
alpha = 0.05 # alpha value
power = 0.8 # 1-beta, 0.8 = 80%
results = {}
for effect in effects:
    analysis = TTestIndPower()
    results[effect] = analysis.solve_power(effect, power=power, nobs1=None, ratio=1.0, alpha=alpha)

print(results)


## Multiple hypothesis correction
import statsmodels.stats.multitest as mt

# Simulate some p-values
pvalues = stats.uniform.rvs(loc=0, scale=1, size=500, random_state=rng)

# Correct p-values
# https://www.statsmodels.org/stable/generated/statsmodels.stats.multitest.multipletests.html
res1 = mt.multipletests(pvalues, method='fdr_bh')
