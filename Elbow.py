from ucimlrepo import fetch_ucirepo 

import pandas as pd
from sklearn import cluster
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# fetch dataset 
st.subtitle('Original Data')
lenses = fetch_ucirepo(id=58) 
st.write(lenses)
# Extract features

st.subtitle('Extracted Data')
data = lenses.data.features
st.write(data)
#y = lenses.data.targets 
numerical_data = data

numerical_data = data.select_dtypes(include=['int', 'float'])

st.write(numerical_data)
sse = []


#trying to create clusters through k means
for k in range(1, 7):
    kmeans = cluster.KMeans(n_clusters=k, n_init=10, random_state=42)
    kmeans.fit(data)
    sse.append(kmeans.inertia_)

st.write("Sum of Squared Errors (SSE) for each k:")
for i, sse_value in enumerate(sse, 1):
    st.write(f"k = {i}: SSE = {sse_value}")


#plot the graph to find k
plt.figure(figsize=(8, 6))
plt.plot(range(1, 7), sse, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Sum of Squared Errors (SSE)')
plt.title('Elbow Method for Optimal k')
plt.grid(True)
plt.show()
st.pyplot(plt)

#package for finding the elbow point of graph
from kneed import KneeLocator

# Using KneeLocator to find the elbow point
kneedle = KneeLocator(range(1, 7), sse, curve='convex', direction='decreasing')
elbow_point = kneedle.elbow

# Display the elbow point in Streamlit
st.write(f"Elbow point found at k = {elbow_point}")

# Set k to the elbow point
k = elbow_point

# Create clusters using the optimal k found by kneed
kmeans = cluster.KMeans(n_clusters=k, random_state=0).fit(data)

# Create the cluster labels
labels = kmeans.labels_

# Display the clusters in Streamlit
st.write("Cluster labels for each instance:")
clusters = pd.DataFrame(labels, columns=['Cluster ID'])
st.write(clusters)

# Plot the Elbow Method graph to visualize the elbow point
plt.figure(figsize=(8, 6))
plt.plot(range(1, 7), sse, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Sum of Squared Errors (SSE)')
plt.title('Elbow Method for Optimal k')
plt.grid(True)

# Mark the elbow point on the plot
plt.axvline(x=k, color='r', linestyle='--')
st.pyplot(plt)
