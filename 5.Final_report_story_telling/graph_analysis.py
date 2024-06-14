import pandas as pd
import numpy as np
from itertools import combinations
import networkx as nx

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

def create_adjacency_matrices_by_year(boardex_interlock_adj_dict, year_lst):
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

def create_company_level_statistics(boardex_data, graph_stat_df_final, boardex_gvkey_df):
    
    board_level_stats_df = pd.merge(boardex_data[["boardid", 'boardname', "directorid", 'directorname']], graph_stat_df_final, how = "inner", left_on="directorid", right_on = "directorid" )
    board_level_stats_df.sort_values("year", inplace=True)

    board_director_details_df = board_level_stats_df[['boardid', 'boardname', 'local_clustering_coef', 'degree_centrality', 'betweenness_centrality',
       'graph_density', 'year']].groupby(["boardid","boardname", "year"]).mean().reset_index()

    boarded_graph_df_with_gvkey = pd.merge(boardex_gvkey_df, board_director_details_df,
         left_on = "companyid", right_on = "boardid"
         )

    boarded_graph_df_with_gvkey = boarded_graph_df_with_gvkey.drop_duplicates(["gvkey", "year"])

    return boarded_graph_df_with_gvkey