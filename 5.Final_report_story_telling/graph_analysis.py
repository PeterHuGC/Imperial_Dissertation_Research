import pandas as pd
import numpy as np
from itertools import combinations
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering


def get_filtered_graph_data(boardex_file_path):

    boardex_data = pd.read_csv(boardex_file_path, index_col = 0)
    filtered_data_df = boardex_data[['boardid', 'companyid', 'directorid', 'overlapyearstart', 'overlapyearend' ]].drop_duplicates()
    filtered_data_df['overlapyearend'].replace("Curr", 2024)
    filtered_data_df['overlapyearend'] = pd.to_numeric(filtered_data_df['overlapyearend'], errors='coerce')

    return filtered_data_df, boardex_data

def create_adjacency_matrix_on_interlock_df(simplified_boardex_df, year):

    graph_simplified_df = simplified_boardex_df[ (year >= simplified_boardex_df['overlapyearstart']) & (year <= simplified_boardex_df['overlapyearend'])]

    graph_simplified_df['company_pair'] = graph_simplified_df.apply(
    lambda row: '-'.join(sorted([str(row['boardid']), str(row['companyid'])])),
    axis=1
    )

    graph_simplified_df = graph_simplified_df.drop_duplicates(subset=['company_pair', 'directorid'])

    # create director pairs and remove duplicates
    director_pairs = graph_simplified_df.groupby('company_pair')['directorid'].apply(
        lambda x: pd.DataFrame(combinations(x.unique(), 2), columns=['Director1', 'Director2']) if len(x) > 1 else pd.DataFrame(columns=['Director1', 'Director2'])
    ).reset_index(drop=True)
    
    director_pairs = director_pairs.drop_duplicates().reset_index(drop=True)

    # create 0-1 adjacency matrix
    if not director_pairs.empty:
        directors = np.unique(director_pairs[['Director1', 'Director2']])
        adj_matrix = pd.DataFrame(0, index=directors, columns=directors, dtype=int)

        for _, row in director_pairs.iterrows():
            adj_matrix.at[row['Director1'], row['Director2']] = 1
            adj_matrix.at[row['Director2'], row['Director1']] = 1

        # no -self loops
        np.fill_diagonal(adj_matrix.values, 0)

        return adj_matrix
    else:
        return None

def create_adjacency_matrices_by_year(filtered_data_df, year_lst):
    """
    Create adjacency matrix with graph details here
    """
    boardex_interlock_adj_dict = {}
    
    for year in year_lst:
    
        # create a sorted unique company identifier
        boardex_interlock_adj_dict[year] = create_adjacency_matrix_on_interlock_df(filtered_data_df, year)

    return boardex_interlock_adj_dict

def create_graph_statistics_df_by_year(boardex_interlock_adj_dict, year_lst):

    yearly_graph_stats_df_lst = []

    for year in year_lst:
    
        year_adj_matrix = boardex_interlock_adj_dict[year]
    
        G = nx.from_pandas_adjacency(year_adj_matrix)
    
        graph_density = nx.density(G) # create as a graph statistic detail
    
        # get the dictionary of director ids with relevant statistics
        clustering_dict = nx.clustering(G)
        degree_centrality_dict = nx.degree_centrality(G)
        betweenness_centrality_dict = nx.betweenness_centrality(G)
    
        df1 = pd.DataFrame.from_dict(clustering_dict, orient='index', columns=['local_clustering_coef'])
        df2 = pd.DataFrame.from_dict(degree_centrality_dict, orient='index', columns=['degree_centrality'])
        df3 = pd.DataFrame.from_dict(betweenness_centrality_dict, orient='index', columns=['betweenness_centrality'])
    
        # 
        combined_graph_df = pd.concat([df1, df2, df3], axis=1)
    
        # index is the director id, for workflow
        combined_graph_df.index.name = 'directorid'
    
        final_graph_df = combined_graph_df.reset_index()
    
        final_graph_df["graph_density"] = graph_density
    
        final_graph_df["year"] = year
    
        yearly_graph_stats_df_lst.append(final_graph_df)

    graph_stat_df_final = pd.concat(yearly_graph_stats_df_lst, axis = 0)

    return graph_stat_df_final

def create_company_level_statistics(boardex_data, boardex_gvkey_df, graph_stat_df_final):
    
    board_level_stats_df = pd.merge(boardex_data[["boardid", 'boardname', "directorid", 'directorname']], graph_stat_df_final, how = "inner", left_on="directorid", right_on = "directorid" )
    board_level_stats_df.sort_values("year", inplace=True)

    board_director_details_df = board_level_stats_df[['boardid', 'boardname', 'local_clustering_coef', 'degree_centrality', 'betweenness_centrality',
       'graph_density', 'year']].groupby(["boardid","boardname", "year"]).mean().reset_index()

    boarded_graph_df_with_gvkey = pd.merge(boardex_gvkey_df, board_director_details_df,
         left_on = "companyid", right_on = "boardid"
         )

    boarded_graph_df_with_gvkey = boarded_graph_df_with_gvkey.drop_duplicates(["gvkey", "year"])

    return boarded_graph_df_with_gvkey

# add graph details

def plot_graphs_from_dict(year_matrix_dict):

    num_plots = len(year_matrix_dict)
    num_cols = 3
    
    # create a figure with subplots
    num_rows = (num_plots + num_cols - 1) // num_cols
    
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))
    
    # flatten axes array and handle case where there is only one subplot
    if not isinstance(axes, np.ndarray):
        axes = [axes]  
    else:
        axes = axes.flatten()  # 
    

    for ax, (year, adj_matrix) in zip(axes, year_matrix_dict.items()):
        
        G = nx.from_pandas_adjacency(adj_matrix)
        pos = nx.spring_layout(G)
        nx.draw(G, pos, ax=ax, with_labels=False, 
                node_color='skyblue', edge_color='gray', node_size=5, font_size=10)
        
        ax.set_title(str(year))
    for ax in axes[num_plots:]:
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()

# clustering plots - showing statistics

def plot_spectral_clustering_from_dict(year_matrix_dict, n_clusters=5, colour_map = None):
    num_plots = len(year_matrix_dict)
    num_cols = 3
    num_rows = (num_plots + num_cols - 1) // num_cols
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))

    if num_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for ax, (year, adj_matrix) in zip(axes, year_matrix_dict.items()):
        G = nx.from_pandas_adjacency(adj_matrix)
        A = nx.to_numpy_array(G)
        sc = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', random_state=42)
        labels = sc.fit_predict(A)

        pos = nx.spring_layout(G)
        if colour_map is None:
            color_map = plt.get_cmap('viridis') 
            node_colors = [color_map(labels[i]) for i in range(len(G.nodes()))]
        else:
            node_colors = [colour_map[label] for label in sc.labels_]
        

        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, node_size=5)
        nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.5)
        ax.set_title(f"Spectral Clustering {year}:\n Clusters: {n_clusters}")
        ax.axis('off')

    for ax in axes[num_plots:]:
        ax.axis('off')

    plt.tight_layout()
    plt.show()

def plot_louvain_communities_from_dict(year_matrix_dict):
    num_plots = len(year_matrix_dict)
    num_cols = 3
    num_rows = (num_plots + num_cols - 1) // num_cols
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))

    if num_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for ax, (year, adj_matrix) in zip(axes, year_matrix_dict.items()):
        G = nx.from_pandas_adjacency(adj_matrix)
        communities = nx.community.louvain_communities(G, seed=123)
        num_communities = len(communities)

        cmap = plt.get_cmap('viridis') 

        colors = [cmap(i / num_communities) for i in range(num_communities)]

        # color map for communities
        color_map = []
        for node in G:
            for idx, community in enumerate(communities):
                if node in community:
                    color_map.append(colors[idx])
                    break

        pos = nx.spring_layout(G)  # positions for all nodes
        nx.draw(G, pos, node_color=color_map, with_labels=False, node_size=5, ax = ax)
        
        ax.set_title(f"Louvain Communities {year}\nCommunities No: {num_communities}")
        ax.axis('off')

    for ax in axes[num_plots:]:
        ax.axis('off')

    plt.tight_layout()
    plt.show()


import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# visualising the graph statistics below

def plot_network_statistics(data):
    # Group by year and calculate mean
    yearly_data = data.groupby('year').mean()
    
    metrics = ['local_clustering_coef', 'degree_centrality', 'betweenness_centrality', 'graph_density']
    
    # plot with 4 columns in one row
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    axes = axes.flatten()
    
    # plot each metric
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        ax.plot(yearly_data.index.astype(int), yearly_data[metric], marker='o', linestyle='-')
        ax.set_title(metric.replace('_', ' ').title())
        ax.set_xlabel('Year')
        if metric == 'graph_density':
            ax.set_ylabel(metric.replace('_', ' ').title())
        else: 
            ax.set_ylabel('Average ' + metric.replace('_', ' ').title())

        ax.xaxis.set_major_locator(MaxNLocator(integer=True)) 

    plt.tight_layout()
    plt.show()