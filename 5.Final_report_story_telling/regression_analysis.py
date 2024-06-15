# add the regression analysis code here for later
import pandas as pd
import numpy as np

# a. preprocessing the data for panel regression via pandas

def get_combined_financial_graph_ar_dataset(ar_key_df, sec_10k_df, boardex_graph_stats_df, compustat_data_df):

    # 1. filter the boardex data to take only relevant fields
    boardex_graph_stats_df = boardex_graph_stats_df[[
    "gvkey", "year", "local_clustering_coef", "degree_centrality", "betweenness_centrality", "graph_density"
    ]]

    # 2. preprocess annual report data so that it can be linked to boardex
    # merge with the ar_key_df that contains the data for annual reports
    sec_10k_df["risk_topic_allocation"] = sec_10k_df["topic_distribution"].apply(
        lambda x : np.argmax(np.array([val for idx, val in x])))

    annual_report_key_features = sec_10k_df[["cik", "report_year", 
                           "risk_sentiment", "business_overview_sentiment",
                           "risk_topic_allocation"]]

    annual_report_key_features_final_df = pd.merge(ar_key_df[["gvkey", "cik"]], 
                                                   annual_report_key_features, how = "inner", on = "cik")
    

    # 3. combine the graph statistics and annual report features
    # into a single dataframe

    graph_ar_features = pd.merge(boardex_graph_stats_df, annual_report_key_features_final_df, 
                             how = "inner", left_on= ["gvkey", "year"], right_on=["gvkey", "report_year"])
    

    # 4. combine the combined graph statistics and annual report features
    # with compustat data

    compustat_data_df["fyear"] = compustat_data_df["fyear"].astype(int)

    final_merged_data_df = pd.merge(compustat_data_df, graph_ar_features, 
            how = "inner", left_on=["gvkey","fyear"], 
            right_on=["gvkey","year"])
    
    return final_merged_data_df




# b. commands to run to implement panel regression workflow

def run_regression_preprocess(final_merged_data_df:pd.DataFrame, stata):
    """ 
    Run final regression given the dataframe

    Documentation to run Stata with Python:
    https://www.stata.com/python/pystata18/stata.html#pystata.stata.run
    """
    
    stata.pdataframe_to_data(final_merged_data_df)

    # detect and drop duplicates
    # stata.run("duplicates list gvkey fyear")
    stata.run("duplicates drop gvkey fyear, force", quietly = True)
    
    # create create panel based on these company id and financial year
    stata.run("""
              xtset gvkey fyear
              tab risk_topic_allocation, gen (risktopic)
              """, 
              quietly = True)


    # # run descriptive statistics
    # stata.run(
    # """
    # sum epsfi sale rdipa local_clustering_coef degree_centrality betweenness_centrality graph_density risk_sentiment business_overview_sentiment risktopic2 risktopic3 risktopic4
    # """      
    # )

    # # correlation matrix
    # stata.run(
    # """
    # cor epsfi sale rdipa local_clustering_coef degree_centrality betweenness_centrality graph_density risk_sentiment business_overview_sentiment risktopic2 risktopic3 risktopic4
    # """
    # )