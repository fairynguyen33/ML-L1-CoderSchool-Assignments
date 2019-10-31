---
title: Unsupervised Learning
tags: Machine Learning, CoderSchool
---

# Unsupervised Learning Summary

Unsupervised learning is a type of machine learning algorithm used to draw inferences from datasets consisting of input data without labeled responses.

- **k-Means clustering**: partitions data into k distinct clusters based on distance to the centroid of a cluster.
- **Hierarchical clustering**: builds a multilevel hierarchy or clusters by creating a cluster tree.
- **PCA**: is a dimensionality-reduction method that is often used to reduce the dimensionality of large datasets, by transforming a large set of variables into a smaller one that still contains most of the information in the large set.

---

# k-Means Clustering

A cluster is defined by a centroid, which is a point (either imaginary or real) at the center of a cluster. Every point of the dataset is part of the cluster whose centroid is most closely located.

Given a set of observations $(x_1, x_2, …, x_n)$, where each observation is a $d$-dimensional real vector, k-Means clustering aims to partition the n observations into $k (≤ n)$ sets $S = {S_1, S_2, …, S_k}$ so as to minimize the within-cluster sum of squares.

Given an initial set of randomly defined k means $m_1^{(1)}$,…,$m_k^{(1)}$ (see below), the algorithm proceeds by alternating between 2 steps:

## Step 1: Assign Points

Assign each data point to the closest corresponding centroid, using the straight-line distance between the data point and the centroid (standard Euclidean distance).

![](https://i.imgur.com/acpbS9O.png)

where each observation $x_{p}$ is assigned to exactly one cluster $S^{(t)}$, even if it could be assigned to 2 or more of them.

## Step 2: Update Centroids

For each centroid, calculate the mean of the values of all the points belonging to it. The mean value becomes the new value of the centroid.

$$
{\displaystyle m_i^{(t+1)} = \frac{1}{\mid{S_i^{(t)}\mid}} \sum_{x_j \epsilon S_i^{(t)}}x_j}
$$

Once step 2 is complete, all of the centroids have new values that correspond to the means of all of their corresponding points. These new points are put through step 1 and 2 producing yet another set of centroid values. This process is repeated over and over until there is no change in the centroid values, meaning that they have been accurately grouped. Or, the process can be stopped when a previously determined maximum number of steps has been met.

![](https://i.imgur.com/dGITsti.gif)

## How to decide on k?
- Start with a random k, create centroids, and run the algorithm. Calculate a sum of squared distances between each point and its centroid. As an increase in clusters correlates with smaller groupings and distances, this sum will always decrease when k increases. An extreme example: is k = the number of data points, sum = 0
- k = 3 in the below graph is called the "elbow point" -- the ideal number of clusters -- which is the point at which increasing k will cause a very small decrease in the error sum, while decreasing k will sharply increase the error sum.
￼![](https://i.imgur.com/7yOxM8X.png)
*** *score = error sum*

## k-Means in scikit-learn

First, let's generate a two-dimensional dataset containing four distinct blobs. To emphasize that this is an unsupervised algorithm, we will leave the labels out of the visualization.

In:
```python=
from sklearn.datasets.samples_generator import make_blobs
X, y_true = make_blobs(n_samples=300, centers=4,
                       cluster_std=0.60, random_state=0)
plt.scatter(X[:, 0], X[:, 1], s=50);
```

Out:
![](https://i.imgur.com/VMpF4pn.png)

By eye, it is relatively easy to pick out the four clusters. The k-means algorithm does this automatically, and in Scikit-Learn uses the typical estimator API:

In:
```python=
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=4)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)
```

Let's visualize the results by plotting the data colored by these labels. We will also plot the cluster centers as determined by the k-means estimator:

In:
```python=
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')

centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);
```
Out:
![](https://i.imgur.com/0BCaWVs.png)

## Pros
- Widely used method for cluster analysis.
- Easy to understand.
- Trains quickly.

## Cons
- Euclidean distance is not ideal in many applications.
- Performance is generally not competitive with the best clustering methods.
- Small variations in the data can result in completely different clusters (high variance).
- Clusters are assumed to have a spherical shape and be evenly sized.

## Souces
> [1]https://www.naftaliharris.com/blog/visualizing-k-means-clustering
[2]https://in.mathworks.com/help/images/color-based-segmentation-using-k-means-clustering.html 
[3]https://mubaris.com/posts/kmeans-clustering/
[4]https://blog.easysol.net/machine-learning-algorithms-3/
[5]http://cs229.stanford.edu/notes/cs229-notes7a.pdf
[6]https://towardsdatascience.com/understanding-k-means-clustering-in-machine-learning-6a6e67336aa1
[7]https://en.wikipedia.org/wiki/K-means_clustering
[8]https://jakevdp.github.io/PythonDataScienceHandbook/05.11-k-means.html

---

# Hierarchical Clustering

An algorithm that groups similar objects into groups called clusters. The endpoint is a set of clusters, where each cluster is distinct from each other cluster, and the objects within each cluster are broadly similar to each other.

The total number of clusters is not predetermined before you start the tree creation.

All hierarchical clustering algorithms are monotonic -- they either increase or decrease (bottom up or top down).

HC can be performed with either a distance matrix or raw data. When raw data is provided, the software will automatically compute a distance matrix in the background. The distance matrix below shows the distance between 6 objects.
![](https://i.imgur.com/MPLKvvx.png)

## Bottom Up (Hierarchical Agglomerative Clustering)

### Step 1
Treat each document as a single cluster at the beginning of the algorithm.

### Step 2
Merge (agglomerate) 2 items at a time into a new cluster. How the pair merge involves calculating a dissimilarity between each merged pair and the other samples (by use of an appropriate measure of distance and a linkage criterion).

### Step 3
The pairing process continues until all items merge into a single cluster. The main output of HC is a dendrogram, which shows the hierarchical relationship between the clusters.
![](https://i.imgur.com/nIXCOUq.png)

![](https://i.imgur.com/ZI497rE.png)

### Measures of Distance (Similarity)
In the example above, the distance between two clusters has been computed based on length of the straight line drawn from one cluster to another (the Euclidean distance). Many other distance metrics have been developed.

The choice of distance metric should be made based on theoretical concerns from the domain of study. That is, a distance metric needs to define similarity in a way that is sensible for the field of study. *For example, if clustering crime sites in a city, city block distance may be appropriate (or, better yet, the time taken to travel between each location).* Where there is no theoretical justification for an alternative, **the Euclidean should generally be preferred**, as it is usually the appropriate measure of distance in the physical world.

### Linkage Criteria
After selecting a distance metric, it is necessary to determine from where distance is computed. It can be computed between:
- the two most similar parts of a cluster (single-linkage),
- the two least similar bits of a cluster (complete-linkage), 
- the center of the clusters (mean or average-linkage), 
- or some other criterion. Many linkage criteria have been developed.

As with distance metrics, the choice of linkage criteria should be made based on theoretical considerations from the domain of application. A key theoretical issue is what causes variation.

*For example, in archeology, we expect variation to occur through innovation and natural resources, so working out if two groups of artifacts are similar may make sense based on identifying the most similar members of the cluster.*

Where there are no clear theoretical justifications for choice of linkage criteria, **Ward’s method is the sensible default. This method works out which observations to group based on reducing the sum of squared distances of each observation from the average observation in a cluster.** This is often appropriate as this concept of distance matches the standard assumptions of how to compute differences between groups in statistics (e.g., ANOVA, MANOVA).

### Facts
- Account for the vast majority of hierarchical clustering algorithms.
- Have significant computational and storage requirements — esp for big data. These complex algorithms are about quadruple the size of the k-Means algorithm.
- Merging can’t be reversed, which can create a problem if you have noisy, high-dimensional data.

## Top Down (Divisive Clustering/ DIANA)

### Step 1
Data starts as one combined cluster.

### Step 2
The cluster splits into 2 distinct parts, according to some degree of similarity.

### Step 3
Clusters split into 2 again and again until the clusters only contain a single data point.

### Choosing Which Cluster to Split
Check the sums of squared errors of the clusters and choose the one with the largest value because this one is not very good so you wanna split it (you want to reduce the sum of squared errors overall).

### Splitting Criteria
One may use Ward’s criterion to chase for greater reduction in the difference in the SSE (sum of squared errors) criterion as a result of a split.

The initial cluster distances in Ward's minimum variance method are defined to be the squared Euclidean distance between points:

${\displaystyle d_{ij}=d(\{X_{i}\},\{X_{j}\})={\|X_{i}-X_{j}\|^{2}}.} d_{{ij}}=d(\{X_{i}\},\{X_{j}\})={\|X_{i}-X_{j}\|^{2}}$

For categorical data, Gini-index can be used.

### Handling the Noise

Use a threshold to determine the termination criterion (do not generate clusters that are too small because they contain mainly noises).

### Facts
- Very rarely used.
- Is conceptually more complex than bottom-up clustering since we need a second, flat clustering algorithm as a “subroutine”.
- More efficient when compared with HAC.

## HAC in scikit-learn
```python=
import matplotlib.pyplot as plt  
import pandas as pd  
%matplotlib inline
import numpy as np 

customer_data = pd.read_csv('D:\Datasets\customer_data.csv')  
customer_data.shape  
customer_data.head()  
data = customer_data.iloc[:, 3:5].values  

import scipy.cluster.hierarchy as shc

plt.figure(figsize=(10, 7))  
plt.title("Customer Dendograms")  
dend = shc.dendrogram(shc.linkage(data, method='ward')) 

from sklearn.cluster import AgglomerativeClustering

cluster = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')  
cluster.fit_predict(data) 

plt.figure(figsize=(10, 7))  
plt.scatter(data[:,0], data[:,1], c=cluster.labels_, cmap='rainbow')  
```


## Pros
- The math is easy to understand and to do.
- Straightforward to program.
- Its main output, the dendrogram, is also the most appealing of the outputs.

## Cons
- Rarely provides the best solution.

The scatterplot below shows data simulated to be in two clusters. The simplest hierarchical cluster analysis algorithm, single-linkage, has been used to extract two clusters. One observation — shown in a red filled circle — has been allocated into one cluster, with the remaining 199 observations allocated to other clusters.

It is obvious when you look at this plot that the solution is poor. It is relatively straightforward to modify the assumptions of hierarchical cluster analysis to get a better solution (e.g., changing single-linkage to complete-linkage). However, in real-world applications the data is typically in high dimensions and cannot be visualized on a plot like this, which means that poor solutions may be found without it being obvious that they are poor.

![](https://i.imgur.com/GCgS8bP.png)

- Arbitrary decisions: when using HC it is necessary to specify both the distance metric and the linkage criteria. There is rarely any strong theoretical basis for such decisions. A core principle of science is that findings are not the result of arbitrary decisions, which makes the technique of dubious relevance in modern research.
- Missing data: most HC software does not work with valued are missing in the data.
- Data types: with many types of data, it is difficult to determine how to compute a distance matrix. There is no straightforward formula that can compute a distance where the variables are both numeric and qualitative. For ex: how can one compute the distance between a 45-year-old man, a 10-year-old girl, and a 46-year-old woman? Formulas have been developed, but they involve arbitrary decisions.
- Misinterpretation of the dendrogram: many users believe that such dendrograms can be used to select the number of clusters. However, this is true only when the ultrametric tree inequality holds, which is rarely, if ever, the case in practice.
- There are better alternatives: more modern techniques, such as latent class analysis, address all the issues with hierarchical cluster analysis.

## Sources
> [1]https://www.youtube.com/watch?v=OcoE7JlbXvY
[2]https://www.statisticshowto.datasciencecentral.com/hierarchical-clustering/
[3]https://www.statisticshowto.datasciencecentral.com/find-outliers/
[4]https://www.displayr.com/strengths-weaknesses-hierarchical-clustering/
[5]https://www.displayr.com/what-is-hierarchical-clustering/
[6]https://en.wikipedia.org/wiki/Hierarchical_clustering#Metric
[7]https://stackabuse.com/hierarchical-clustering-with-python-and-scikit-learn/

---

# Principal Component Analysis

Is a **dimensionality-reduction method** that is often used to reduce the dimensionality of large datasets, by transforming a large set of variables into a smaller one that still contains most of the information in the large set.

The main goal of a PCA analysis is to identify patterns in data; PCA aims to detect the correlation between variables. If a strong correlation between variables exists, the attempt to reduce the dimensionality only makes sense.

Reducing the number of variables of a dataset naturally comes at the expense of accuracy, but the trick in dimensionality reduction is to trade a little accuracy for simplicity.

## PCA Vs. LDA

Both Linear Discriminant Analysis (LDA) and PCA are linear transformation methods. PCA yields the directions (principal components) that maximize the variance of the data, whereas LDA also aims to find the directions that maximize the separation (or discrimination) between different classes, which can be useful in pattern classification problem (PCA "ignores" class labels).

In other words, PCA projects the entire dataset onto a different feature (sub)space, and LDA tries to determine a suitable feature (sub)space in order to distinguish between patterns that belong to different classes.

## Prepare the Iris Dataset

For the following tutorial, we will be working with the famous "Iris" dataset that has been deposited on the UCI machine learning repository (https://archive.ics.uci.edu/ml/datasets/Iris).

The Iris dataset contains measurements for 150 Iris flowers from three different species.

The three classes in the Iris dataset are:

Iris-setosa (n=50)
Iris-versicolor (n=50)
Iris-virginica (n=50)

And the four features of in Iris dataset are:

sepal length in cm
sepal width in cm
petal length in cm
petal width in cm

## Load the Dataset

In:
```python=
import pandas as pd

df = pd.read_csv(
    filepath_or_buffer='https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', 
    header=None, 
    sep=',')

df.columns=['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid', 'class']
df.dropna(how="all", inplace=True) # drops the empty line at file-end

df.tail()
```
Out:

![](https://i.imgur.com/tOLvpId.png)

In:
```python=
# split data table into data X and class labels y

X = df.iloc[:,0:4].values
y = df.iloc[:,4].values
```
Our iris dataset is now stored in form of a 150×4 matrix where the columns are the different features, and every row represents a separate flower sample. Each sample row 𝐱 can be pictured as a 4-dimensional vector
![](https://i.imgur.com/TeZ0eDh.png)

## Exploratory Visualization

To get a feeling for how the 3 different flower classes are distributes along the 4 different features, let us visualize them via histograms.

In:
```python=
import plotly.plotly as py

# plotting histograms
data = []

legend = {0:False, 1:False, 2:False, 3:True}

colors = {'Iris-setosa': '#0D76BF', 
          'Iris-versicolor': '#00cc96', 
          'Iris-virginica': '#EF553B'}

for col in range(4):
    for key in colors:
        trace = dict(
            type='histogram',
            x=list(X[y==key, col]),
            opacity=0.75,
            xaxis='x%s' %(col+1),
            marker=dict(color=colors[key]),
            name=key,
            showlegend=legend[col]
        )
        data.append(trace)

layout = dict(
    barmode='overlay',
    xaxis=dict(domain=[0, 0.25], title='sepal length (cm)'),
    xaxis2=dict(domain=[0.3, 0.5], title='sepal width (cm)'),
    xaxis3=dict(domain=[0.55, 0.75], title='petal length (cm)'),
    xaxis4=dict(domain=[0.8, 1], title='petal width (cm)'),
    yaxis=dict(title='count'),
    title='Distribution of the different Iris flower features'
)

fig = dict(data=data, layout=layout)
py.iplot(fig, filename='exploratory-vis-histogram')
```
Out:
![](https://i.imgur.com/XxSmqlY.png)

## Step 1: Standardization 
To standardize the range of the continuous initial variables so that each one of them contributes equally to the analysis.

It is critical to perform standardization prior to PCA because the latter is quite sensitive regarding the variances of the initial variables. That is, if there are large differences between the ranges of initial variables, those variables with larger ranges will dominate over those with small ranges *(for example, a variable that ranges between 0 and 100 will dominate over a variable that ranges between 0 and 1)*, which will lead to biased results. So, transforming the data to comparable scales can prevent this problem.

$${\displaystyle z = \frac{value - mean}{standard \hspace{0.2cm} deviation}}$$

Once the standardization is done, all the variables will be transformed to the same range **[0,1]**.

In:
```python=
from sklearn.preprocessing import StandardScaler
X_std = StandardScaler().fit_transform(X)
```

## Step 2: Compute Covariance Matrix or Correlation Matrix

### Covariance Matrix
A covariance matrix $Σ$ is a $𝑑×𝑑$ matrix where each element represents the covariance between two features. The covariance between two features is calculated as follows: 

$${\displaystyle 𝜎_{𝑗𝑘} = \frac{1}{n-1} \sum_{i=1}^N(x_{ij}-\overline{x}_{j}) (x_{ik}-\overline{x}_{k})}$$

We can summarize the calculation of the covariance matrix via the following matrix equation: 

$${\displaystyle ∑ = \frac{1}{n-1}\Big((X - \overline{x})^T (X - \overline{x})\Big)}$$ 
where $\overline{x}$ is the mean vector ${\displaystyle \overline{x} = \sum_{k=1}^{n}x_i}$

The mean vector is a 𝑑-dimensional vector where each value in this vector represents the sample mean of a feature column in the dataset.

The sign of the covariance matters:
- if **positive** then : the 2 variables increase or decrease together
- if **negative** then : 1 increases when the other decreases

In:
```python=
import numpy as np
mean_vec = np.mean(X_std, axis=0)
cov_mat = (X_std - mean_vec).T.dot((X_std - mean_vec)) / (X_std.shape[0]-1)
print('Covariance matrix \n%s' %cov_mat)
```

**OR** use the numpy ```cov``` function

```python=
cov_mat = np.cov(X_std.T)
print('NumPy covariance matrix \n%s' %cov_mat)
```

Out:

```python=
NumPy covariance matrix: 
[[ 1.00671141 -0.11010327  0.87760486  0.82344326]
 [-0.11010327  1.00671141 -0.42333835 -0.358937  ]
 [ 0.87760486 -0.42333835  1.00671141  0.96921855]
 [ 0.82344326 -0.358937    0.96921855  1.00671141]]
```

### Correlation Matrix

In the field of "Finance," the correlation matrix is typically used instead of the covariance matrix. However, the eigendecomposition of the covariance matrix (if the input data was **standardized**) yields the **same** results as a eigendecomposition on the correlation matrix, since the correlation matrix can be understood as the **normalized covariance matrix**.

In:
```python=
cor_mat1 = np.corrcoef(X_std.T)
```

## Step 3: Eigendecomposition -- Computing Eigenvectors and Eigenvalues

The **first** principal component accounts for the largest possible variance in the data set. 

*For example, let’s assume that the scatter plot of our data set is as shown below, can we guess the first principal component?* 

Yes, it’s approximately **the line that matches the purple marks** because it **goes through the origin** ***and*** it’s the line in which **the projection of the blue points (red dots)** is the **most spread out**. 

Or mathematically speaking, it’s the line that **maximizes the variance** (the average of the squared distances from the projected points (red dots) to the origin). **The larger the variance carried by a line, the larger the dispersion of the data points along it**, and **the larger the dispersion along a line, the more the information it has**.

![](https://i.imgur.com/7Q4XkSu.gif)

The **second** principal component is calculated in the same way, with the condition that it is uncorrelated with (i.e., **perpendicular** to) the first principal component and that it accounts for **the next highest variance**.

This continues until a total of **p** principal components have been calculated, **equal to the original number of variables**.

![](https://i.imgur.com/0OHARVP.png)

Now that we understood what we mean by principal components, let’s go back to **eigenvectors and eigenvalues**. They **always come in pairs**, and their number is equal to **the number of dimensions of the data**.

*For example, for a 3-dimensional dataset, there are 3 variables, therefore there are 3 eigenvectors with 3 corresponding eigenvalues.*

It is eigenvectors and eigenvalues who are behind all the magic explained above, because the eigenvectors of the Covariance Matrix or the Correlation Matrix are actually **the directions of the axes where there is the most variance (most information) and that we call Principal Components**. And eigenvalues are simply **the coefficients attached to eigenvectors, which give the amount of variance carried in each Principal Component**.

### Eigendecomposition on the Covariance Matrix

In: 
```python=
eig_vals, eig_vecs = np.linalg.eig(cov_mat)

print('Eigenvectors \n%s' %eig_vecs)
print('\nEigenvalues \n%s' %eig_vals)
```
Out:
```python=
Eigenvectors 
[[ 0.52237162 -0.37231836 -0.72101681  0.26199559]
 [-0.26335492 -0.92555649  0.24203288 -0.12413481]
 [ 0.58125401 -0.02109478  0.14089226 -0.80115427]
 [ 0.56561105 -0.06541577  0.6338014   0.52354627]]

Eigenvalues 
[2.93035378 0.92740362 0.14834223 0.02074601]
```

### Eigendecomposition on the Correlation Matrix

In:
```python=
eig_vals, eig_vecs = np.linalg.eig(cor_mat1)

print('Eigenvectors \n%s' %eig_vecs)
print('\nEigenvalues \n%s' %eig_vals)
```
Out:
```python=
Eigenvectors 
[[ 0.52237162 -0.37231836 -0.72101681  0.26199559]
 [-0.26335492 -0.92555649  0.24203288 -0.12413481]
 [ 0.58125401 -0.02109478  0.14089226 -0.80115427]
 [ 0.56561105 -0.06541577  0.6338014   0.52354627]]

Eigenvalues 
[2.91081808 0.92122093 0.14735328 0.02060771]
```

### Singular Vector Decomposition (SVD)

While the eigendecomposition of the covariance or correlation matrix may be more intuitiuve, most PCA implementations perform a SVD to **improve the computational efficiency**. So, let us perform an SVD to confirm that the results are indeed the **same**:

In:
```python=
u,s,v = np.linalg.svd(X_std.T)
u
```
Out:
```python=
array([[-0.52237162, -0.37231836,  0.72101681,  0.26199559],
       [ 0.26335492, -0.92555649, -0.24203288, -0.12413481],
       [-0.58125401, -0.02109478, -0.14089226, -0.80115427],
       [-0.56561105, -0.06541577, -0.6338014 ,  0.52354627]])
```
## Step 4: Selecting Principal Components

The typical goal of a PCA is to reduce the dimensionality of the original feature space by projecting it onto a smaller subspace, where the eigenvectors will form the axes. However, the eigenvectors only define the directions of the new axis, since they have all **the same unit length 1**, which can confirmed by the following two lines of code:

In:
```python=
for ev in eig_vecs:
    np.testing.assert_array_almost_equal(1.0, np.linalg.norm(ev))
print('Everything ok!')
```
Out:
```python=
Everything ok!
```
We rank the eigenvalues from highest to lowest in order choose the top 𝑘 eigenvectors (the lowest eigenvalues bear the least information about the distribution of the data and can be dropped).

In:
```python=
# Make a list of (eigenvalue, eigenvector) tuples
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

# Sort the (eigenvalue, eigenvector) tuples from high to low
eig_pairs.sort()
eig_pairs.reverse()

# Visually confirm that the list is correctly sorted by decreasing eigenvalues
print('Eigenvalues in descending order:')
for i in eig_pairs:
    print(i[0])
```
Out:
```python=
Eigenvalues in descending order:
2.910818083752054
0.9212209307072242
0.14735327830509573
0.020607707235625678
```

After sorting the eigenpairs, the next question is "how many principal components are we going to choose for our new feature subspace?" A useful measure is the so-called "explained variance," which can be calculated from the eigenvalues. The "explained variance" tells us how much information (variance) can be attributed to each of the principal components (which is basically the percentage of variance accounted for by each component).

In:
```python=
tot = sum(eig_vals)
var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)

trace1 = dict(
    type='bar',
    x=['PC %s' %i for i in range(1,5)],
    y=var_exp,
    name='Individual'
)

trace2 = dict(
    type='scatter',
    x=['PC %s' %i for i in range(1,5)], 
    y=cum_var_exp,
    name='Cumulative'
)

data = [trace1, trace2]

layout=dict(
    title='Explained variance by different principal components',
    yaxis=dict(
        title='Explained variance in percent'
    ),
    annotations=list([
        dict(
            x=1.16,
            y=1.05,
            xref='paper',
            yref='paper',
            text='Explained Variance',
            showarrow=False,
        )
    ])
)

fig = dict(data=data, layout=layout)
py.iplot(fig, filename='selecting-principal-components')
```
Out:
![](https://i.imgur.com/OlgtvWI.png)

*The plot above clearly shows that most of the variance (72.77%) can be explained by the first principal component alone. The second principal component still bears some information (23.03%) while the third and fourth principal components can safely be dropped without losing to much information. Together, the first two principal components contain 95.8% of the information.*

It's about time to get to the really interesting part: The construction of the projection matrix -- a matrix of our concatenated top k eigenvectors -- that will be used to transform the Iris data onto the new feature subspace.

Here, we are reducing the 4-dimensional feature space to a 2-dimensional feature subspace, by choosing the "top 2" eigenvectors with the highest eigenvalues to construct our 𝑑×𝑘-dimensional eigenvector matrix 𝐖.

In:
```python=
matrix_w = np.hstack((eig_pairs[0][1].reshape(4,1), 
                      eig_pairs[1][1].reshape(4,1)))

print('Matrix W:\n', matrix_w)
```
Out:
```python=
Matrix W:
 [[ 0.52237162 -0.37231836]
 [-0.26335492 -0.92555649]
 [ 0.58125401 -0.02109478]
 [ 0.56561105 -0.06541577]]
```
## Step 5: Projection Onto the New Feature Space

In this last step we will use the 4×2-dimensional projection matrix 𝐖 to transform our samples onto the new subspace via the equation
𝐘=𝐗×𝐖, where 𝐘 is a 150×2 matrix of our transformed samples.

In:
```python=
Y = X_std.dot(matrix_w)
```
In:
```python=
data = []

for name, col in zip(('Iris-setosa', 'Iris-versicolor', 'Iris-virginica'), colors.values()):
    trace = dict(
        type='scatter',
        x=Y[y==name,0],
        y=Y[y==name,1],
        mode='markers',
        name=name,
        marker=dict(
            color=col,
            size=12,
            line=dict(
                color='rgba(217, 217, 217, 0.14)',
                width=0.5),
            opacity=0.8)
    )
    data.append(trace)

layout = dict(
    showlegend=True,
    scene=dict(
        xaxis=dict(title='PC1'),
        yaxis=dict(title='PC2')
    )
)

fig = dict(data=data, layout=layout)
py.iplot(fig, filename='projection-matrix')
```
Out:
![](https://i.imgur.com/8idSJtT.png)

## PCA in scikit-learn

For educational purposes, we went a long way to apply the PCA to the Iris dataset. But luckily, there is already implementation in scikit-learn.

In:
```python=
from sklearn.decomposition import PCA as sklearnPCA
sklearn_pca = sklearnPCA(n_components=2)
Y_sklearn = sklearn_pca.fit_transform(X_std)
```
In:
```python=
data = []

for name, col in zip(('Iris-setosa', 'Iris-versicolor', 'Iris-virginica'), colors.values()):

    trace = dict(
        type='scatter',
        x=Y_sklearn[y==name,0],
        y=Y_sklearn[y==name,1],
        mode='markers',
        name=name,
        marker=dict(
            color=col,
            size=12,
            line=dict(
                color='rgba(217, 217, 217, 0.14)',
                width=0.5),
            opacity=0.8)
    )
    data.append(trace)

layout = dict(
        xaxis=dict(title='PC1', showline=False),
        yaxis=dict(title='PC2', showline=False)
)
fig = dict(data=data, layout=layout)
py.iplot(fig, filename='pca-scikitlearn')
```
Out:
![](https://i.imgur.com/Hd9sQ32.png)

## Pros
- Reflects our intuitions about the data.
- Allows estimating probabilities in high-dimensional data (no need to assume independence, etc.).
- Dramatic reduction in sized data --> faster processing, smaller storage.

## Cons
- Too expensive for many applications (Twitter, web).
- Disastrous for tasks with fine-grained classes.
- Understand assumptions behind the methods (linearity, etc.). There may be better ways to deal with sparseness.

## Sources
> [1]https://towardsdatascience.com/a-step-by-step-explanation-of-principal-component-analysis-b836fb9c97e2
[2]https://www.quora.com/What-is-an-intuitive-explanation-for-PCA?redirected_qid=2314001
[3]https://towardsdatascience.com/a-one-stop-shop-for-principal-component-analysis-5582fb7e0a9c
[4]https://media.ed.ac.uk/media/Pros+and+cons+of+dimensionality+reduction/1_xo8l1cfm
[5]https://plot.ly/ipython-notebooks/principal-component-analysis/
[6]https://machinelearningmastery.com/singular-value-decomposition-for-machine-learning/

---

# Comparisons

## Sources
> [1]https://stats.stackexchange.com/questions/183236/what-is-the-relation-between-k-means-clustering-and-pca
[2]https://datascience.stackexchange.com/questions/27501/k-means-vs-hierarchical-clustering