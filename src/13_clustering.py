from sklearn.preprocessing import scale
from sklearn.metrics import pairwise_distances
from sklearn.datasets import load_iris
from pyclustertend import vat, ivat, hopkins # from github git@github.com:lachhebo/pyclustertend.git
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

USArrests = pd.read_csv("data/USArrests.csv", index_col=0)

# Escalar los datos
USArrests_scaled = scale(USArrests)

dist_cor = pairwise_distances(USArrests_scaled, metric='correlation')

# Imprimir la matriz de distancias
print(np.round(dist_cor[0:7, 0:7], 1))


plt.clf()
plt.figure(figsize=(4, 4))  
vat(dist_cor, figure_size = (4, 4))
plt.title('Gráfico VAT (Visual Assessment of Cluster Tendency)')
plt.show()

plt.clf()
ivat(dist_cor, figure_size = (4, 4))
plt.title('Gráfico VAT mejorado')
plt.show()



## Tendencia de factibilidad
iris = pd.DataFrame(load_iris().data, columns=load_iris().feature_names)

# Generar un DataFrame aleatorio con las mismas dimensiones que iris
random_df = pd.DataFrame(
  np.random.uniform(
    iris.min(axis=0), 
    iris.max(axis=0), 
    size=iris.shape),
    columns=load_iris().feature_names
    )

df = iris.copy()



import patchworklib as pw
from plotnine import *
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def plot_pca(data, title):
  # Escala los datos utilizando StandardScaler
  scaler = StandardScaler()
  data_scaled = scaler.fit_transform(data)

  # Realiza el análisis de componentes principales (PCA)
  pca = PCA(n_components=2)
  principal_components = pca.fit_transform(data_scaled)
  df_pca = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])

  return(
      ggplot(df_pca, aes(x='PC1', y='PC2'))
      + geom_point()
      + labs(title=title)
      + theme_classic()
  )


from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage
import seaborn as sns
import matplotlib.pyplot as plt


scaler = StandardScaler()
random_df_scaled = pd.DataFrame(
  data = scaler.fit_transform(random_df), 
  columns = random_df.columns).iloc[:,:2]

# K-means clustering
kmeans = KMeans(n_clusters=3, random_state=12345)
km_res2 = kmeans.fit_predict(random_df_scaled)

# Crear un DataFrame con los resultados del clustering
random_df_scaled['cluster'] = km_res2

# Elegir una paleta de colores con tres colores contrastantes
palette = sns.color_palette("husl", 3)

# Gráfico de puntos coloreados por clúster utilizando seaborn
plt.clf()

# Scatter plot con tres colores contrastantes
plt.subplot(1, 2, 1)

sns.scatterplot(
  data=random_df_scaled, 
  x='sepal length (cm)', 
  y='sepal width (cm)', 
  hue='cluster', 
  palette=palette
  )

plt.title("K-Means Clustering")
# Dendrograma jerárquico utilizando scipy
plt.subplot(1, 2, 2)

linkage_matrix = linkage(random_df_scaled, method='ward')
dendro = dendrogram(linkage_matrix, no_labels=True)
plt.title("Dendrograma")

# Ajustar el diseño y mostrar la figura
plt.tight_layout()
plt.show()




##############################
plt.figure(figsize=(4, 4))  
# Original data
plt.clf()
ivat(pairwise_distances(df, metric='correlation'), figure_size = (4, 4))
plt.title('Gráfico VAT - Datos reales')
plt.show()


# Random data
plt.clf()
ivat(pairwise_distances(random_df, metric='correlation'), figure_size = (4, 4))
plt.title('Gráfico VAT - Datos aleatorios')
plt.show()


###########  K - means ##########

from sklearn.cluster import KMeans

USArrests = pd.read_csv("data/USArrests.csv", index_col=0)
USArrests.head()

# Escalar los datos
USArrests_scaled = pd.DataFrame(data = scale(USArrests), columns = USArrests.columns)
USArrests_scaled.head()


kmeans = KMeans(
    n_clusters=2,
    init='k-means++',
    n_init=25,
    max_iter=300,
    tol=1e-4,
    random_state=42
)

# Ajustar el modelo a los datos
kmeans.fit(USArrests_scaled)

# Obtener las etiquetas de los clústeres asignadas a cada punto de datos
labels = kmeans.labels_
labels

# Obtener las coordenadas de los centroides finales
centroids = kmeans.cluster_centers_
centroids

# Obtener la inercia (suma de las distancias cuadradas de cada punto al centroide de su clúster)
inertia = kmeans.inertia_
inertia

# Realiza el análisis de componentes principales (PCA)
pca = PCA(n_components=2)
principal_components = pca.fit_transform(USArrests_scaled)
df_pca = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])

df_pca["cluster2"] = kmeans.labels_+1

plt.clf()
sns.scatterplot(x="PC1", y="PC2", hue="cluster2", data=df_pca, palette="rainbow")
plt.title("Visualización de los resultados de k-means")
plt.show()



k3 = KMeans(n_clusters=3, n_init = 25, random_state=333)
k3.fit(USArrests_scaled)
df_pca["cluster3"] = k3.labels_ + 1

k4 = KMeans(n_clusters=4, n_init = 25, random_state=444)
k4.fit(USArrests_scaled)

df_pca["cluster4"] = k4.labels_ + 1

k5 = KMeans(n_clusters=5, n_init = 25, random_state=555)
k5.fit(USArrests_scaled)

df_pca["cluster5"] = k5.labels_ + 1
df_pca.head()


from plydata.tidy import pivot_longer
from plydata.one_table_verbs import select
from plotnine import *

colores_por_valor = {1: 'red', 2: 'green', 3: 'blue', 4: 'purple', 5: 'yellow'}

(
  df_pca >>
  pivot_longer(
    cols=select(startswith='cluster'),
    names_to = "n_cluster",
    values_to='value'
    ) >>
  ggplot(aes(x = "PC1", y = "PC2", fill = "factor(value)", color = "factor(value)")) +
  geom_point() +
  scale_color_manual(values=colores_por_valor) +
  facet_wrap("~n_cluster", ncol=2)
  )



from yellowbrick.cluster import KElbowVisualizer

# Configurar el modelo KMeans
model = KMeans(n_init = 25)

# Crear visualizador de codo para determinar el número óptimo de clústeres (K)
plt.clf()
visualizer = KElbowVisualizer(model, k=(1, 10), metric='distortion')

# Ajustar el modelo y visualizar el codo
visualizer.fit(USArrests_scaled)
visualizer.show()