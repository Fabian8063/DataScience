import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

pd.set_option('display.max_columns', 15)


def prepare_data():
    """
    Loads data and splits it into 2 DataFrames.

    Returns
    -------
    df: DataFrame
        DataFrame with all features.
    df_numeric: DataFrame
        DataFrame with only numeric features that concern customer spending.
    """
    df = pd.read_csv('Wholesale customers data.csv')
    df_numeric = df[['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen']]
    return df, df_numeric


def visualise_data(df):
    """
    Visualises the data.

    Parameters
    -------
    df: DataFrame
        DataFrame with data.
    """

    plt.figure(1, figsize=(12, 6))
    n = 0
    for x in ['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen']:
        n += 1
        plt.subplot(2, 3, n)
        plt.subplots_adjust(hspace=0.5, wspace=0.5)
        sns.histplot(df[x], bins=20)
        plt.title('Histplot of {}'.format(x))
    plt.show()

    df.loc[:, 'Channel_str'] = df['Channel'].replace([1, 2], ['Horeca', 'Retail'])

    plt.figure(2, figsize=(12, 6))
    for channel in ['Horeca', 'Retail']:
        plt.scatter(x='Fresh', y='Frozen', data=df[df['Channel_str'] == channel],
                    s=200, alpha=0.5, label=channel)
    plt.xlabel('Fresh'), plt.ylabel('Frozen')
    plt.title('Fresh vs Frozen concerning Channel')
    plt.legend()
    plt.show()

    plt.figure(3, figsize=(12, 6))
    for channel in ['Horeca', 'Retail']:
        plt.scatter(x='Grocery', y='Detergents_Paper', data=df[df['Channel_str'] == channel],
                    s=200, alpha=0.5, label=channel)
    plt.xlabel('Grocery'), plt.ylabel('Detergents_Paper')
    plt.title('Grocery vs Detergents_Paper concerning Channel')
    plt.legend()
    plt.show()

    plt.figure(4, figsize=(12, 6))
    for channel in ['Horeca', 'Retail']:
        plt.scatter(x='Frozen', y='Delicassen', data=df[df['Channel_str'] == channel],
                    s=200, alpha=0.5, label=channel)
    plt.xlabel('Frozen'), plt.ylabel('Delicassen')
    plt.title('Frozen vs Delicassen concerning Channel')
    plt.legend()
    plt.show()

    plt.figure(5, figsize=(12, 6))
    for channel in ['Horeca', 'Retail']:
        plt.scatter(x='Milk', y='Grocery', data=df[df['Channel_str'] == channel],
                    s=200, alpha=0.5, label=channel)
    plt.xlabel('Milk'), plt.ylabel('Grocery')
    plt.title('Frozen vs Delicassen concerning Channel')
    plt.legend()
    plt.show()

    df.drop(columns='Channel_str', inplace=True)

    plt.figure(5, figsize=(15, 6))
    sns.heatmap(df.corr(), annot=True)
    plt.title('Correlation between variables')
    plt.show()


def _build_model(df, use_pca, n_components, use_kmeans, n_clusters):
    """
    Builds the model.

    Parameters
    ----------
    df: DataFrame
        The DataFrame with numeric features.
    use_pca: bool
        Whether to use PCA or not.
    n_components: int
        Number of components for PCA.
    use_kmeans: bool
        Whether to use Kmeans or not.
    n_clusters: int
        Number of clusters for Kmeans.

    Returns
    -------
    pipe: Pipeline
        Pipeline with transformers and estimator.
    """
    scaler = StandardScaler()
    pipe_list = [('scaler', scaler)]
    if use_pca:
        pca = PCA(n_components=n_components)
        pipe_list.append(('pca', pca))
    if use_kmeans:
        kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=30, tol=0, random_state=0)
        pipe_list.append(('kmeans', kmeans))
    pipe = Pipeline(steps=pipe_list)
    pipe.fit(df)
    return pipe


def get_number_of_clusters(df, use_pca, n_components):
    """
    Determinates the correct number of clusters for Kmeans.

    Parameters
    ----------
    df: DataFrame
        The DataFrame with numerical features.
    use_pca: bool
        Whether to use PCA or not.
    n_components: int
        Number of components for PCA.
    """
    n_clusters = 10
    cluster_with_distances = []
    for i in range(n_clusters):
        pipe = _build_model(df, use_pca, n_components, use_kmeans=True, n_clusters=i + 1)
        cluster_with_distances.append(pipe.named_steps['kmeans'].inertia_)
    plt.figure(6, figsize=(12, 6))
    plt.plot(range(1, 11), cluster_with_distances, 'o')
    plt.plot(range(1, 11), cluster_with_distances, '-', alpha=0.5)
    plt.title('The Elbow Criterion')
    plt.xlabel('number of cluster')
    plt.ylabel('Sum of squared distances of samples to their closest cluster center')
    plt.show()


def get_number_of_components(df):
    """
    Determinates the correct number of components for PCA.

    Parameters
    ----------
    df: DataFrame
        The DataFrame with numerical features.
    """
    n_components = 6  # since there a 6 numeric features
    pipe = _build_model(df, use_pca=True, n_components=n_components, use_kmeans=False, n_clusters=99)
    explained_variances = pipe.named_steps['pca'].explained_variance_ratio_
    plt.figure(7, figsize=(12, 6))
    plt.plot(range(1, 7), np.cumsum(explained_variances), 'o')
    plt.plot(range(1, 7), np.cumsum(explained_variances), '-', alpha=0.5)
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance')
    plt.show()


def get_clusters_with_all_features(df, n_clusters):
    """
    Calculates the clusters for the dataset with all numerical features.

    Parameters
    ----------
    df: DataFrame
        The DataFrame with numerical features.
    n_clusters: int
        The number of clusters for Kmeans.
    """
    pipe = _build_model(df, use_pca=False, n_components=0, use_kmeans=True, n_clusters=n_clusters)
    labels = pipe.named_steps['kmeans'].labels_
    df.loc[:, 'labels'] = labels
    print(df.groupby('labels').agg(
        {'Fresh': 'mean', 'Milk': 'mean', 'Grocery': 'mean', 'Frozen': 'mean', 'Detergents_Paper': 'mean',
         'Delicassen': 'mean'}))
    print(pipe.named_steps['scaler'].inverse_transform(pipe.named_steps['kmeans'].cluster_centers_))
    # cluster 1: low spending behaviour in general
    # cluster 2: high spending in detergents_paper, milk, grocery
    # cluster 3: high spending in fresh, rest low
    # cluster 4: high spending in everything except detergents_paper, extremely high in delicassen
    # cluster 5: medium spending in general, low in frozen, high in detergents and paper


def get_clusters_with_pca(df_full, df, n_clusters, n_components):
    """
    Calculates the clusters for the dataset that was reduced with PCA.

    Parameters
    ----------
    df_full: DataFrame
        The DataFrame with all features.
    df: DataFrame
        The DataFrame with reduced features.
    n_clusters: int
        Number of clusters for Kmeans.
    n_components: int
        Number of components for PCA.
    """
    pipe = _build_model(df, use_pca=True, n_components=2, use_kmeans=True, n_clusters=n_clusters)
    df.loc[:, ['PC-1', 'PC-2']] = pipe.named_steps['pca'].transform(df)
    labels = pipe.named_steps['kmeans'].labels_
    df.loc[:, 'labels'] = labels
    df_centers = df.groupby('labels').agg({'PC-1': 'mean', 'PC-2': 'mean'})
    print(df_centers)

    df.loc[:, 'Channel'] = df_full['Channel']
    df.loc[:, 'Channel_str'] = df['Channel'].replace([1, 2], ['Horeca', 'Retail'])
    plt.figure(8, figsize=(12, 6))
    for channel in ['Horeca', 'Retail']:
        plt.scatter(x='PC-1', y='PC-2', data=df[df['Channel_str'] == channel],
                    s=200, alpha=0.5, label=channel)
    plt.xlabel('PC-1'), plt.ylabel('PC-2')
    plt.title('PC-1 vs PC-2 concerning Channel')
    plt.legend()
    plt.show()

    plt.figure(9, figsize=(12, 6))
    plt.scatter(x='PC-1', y='PC-2', data=df[df['labels'] == 0], s=100, c='red', label='Cluster 1')
    plt.scatter(x='PC-1', y='PC-2', data=df[df['labels'] == 1], s=100, c='blue', label='Cluster 2')
    plt.scatter(x='PC-1', y='PC-2', data=df[df['labels'] == 2], s=100, c='green', label='Cluster 3')
    plt.scatter(x='PC-1', y='PC-2', data=df[df['labels'] == 3], s=100, c='cyan', label='Cluster 4')
    plt.scatter(x='PC-1', y='PC-2', data=df[df['labels'] == 4], s=100, c='magenta', label='Cluster 5')
    plt.scatter(df_centers.iloc[:, 0], df_centers.iloc[:, 1],
                s=100, c='yellow', label='Centroids')
    plt.title('Clusters of customers')
    plt.xlabel('Spending in PC-1')
    plt.ylabel('Spending in PC-2')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    df, df_numeric = prepare_data()
    visualise_data(df)
    # cluster without pca
    get_number_of_clusters(df_numeric, use_pca=False, n_components=0)  # suggests 5 cluster
    get_clusters_with_all_features(df_numeric, n_clusters=5)
    # cluster with pca
    get_number_of_components(df_numeric)  # suggests 2 components
    get_number_of_clusters(df_numeric, use_pca=True, n_components=2)  # suggests 5 cluster
    get_clusters_with_pca(df, df_numeric, n_clusters=5, n_components=2)
