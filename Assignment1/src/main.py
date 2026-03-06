"""
orders.csv:
 - order_id
 - user_id
 - order_number
 - order_dow
 - order_hour_of_day
 - days_since_prior_order
products.csv:
 - product_id
 - product_name
 - aisle_id
 - department_id

order_products.csv: 
 - order_id
 - product_id
 - add_to_cart_order
 - reordered

aisles.csv:
 - aisle_id 
 - aisle
departments.csv:
 - department_id
 - department
"""

import os, ast
import pandas as pd
from tabulate import tabulate
from apyori import apriori

DATA_PATH = "../data/"
PROCESSED_PATH = "../processed_data/"

SAMPLED_ORDERS = PROCESSED_PATH + "sampled_orders.csv"
FILTERED_ORDER_PRODUCTS = PROCESSED_PATH + "filtered_order_products.csv"
TRANSACTIONS = PROCESSED_PATH + "transactions.csv"

def combine_to_set(dataframe):
    """
    function to group columns together for the transaction items.
    
    for example: (product_name, order_dow)
        input dataframe: [['Sweet Corn On The Cob' 1]
                          ['Extra Long Grain Enriched Rice' 1]
                          ['Beef Franks' 1]
                          ['French Baguette Bread' 1]]
        flatten values : ['Sweet Corn On The Cob' 1 
                          'Extra Long Grain Enriched Rice' 1
                          'Beef Franks' 1 
                          'French Baguette Bread' 1]
        finally convert to set to remove duplicates.
    """
    flat = dataframe.values.flatten()
    return set(flat)    # Remove duplicates

def set_elements_to_str(inputSet):
    """
    Turn all elements in the set into str
    """
    newSet = set()
    for element in inputSet:
        newSet.add(str(element))
    return newSet

def createTransactionsCSV(file1: str, file2: str, 
                          mergeOnColumn: str,  
                          groupByColumn: str | None = None,
                          groupBySelection: str | list[str] | None = None,
                          outputfile: str = "transactions.csv",
                          keepColumns1: list[str] | None = None, keepColumns2: list[str] | None = None,
                          keepColumnsMerged: list[str] | None = None,
                          newColumnNames: list[str] | None = None
                          ):
    if os.path.exists(outputfile):
        print("Skipped: create transactions")
        return 
        
    print("CREATING TRANSACTIONS CSV...")
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    print("\tLOADED FILES.")
    
    if keepColumns1:
        df1 = df1[keepColumns1]
        print(f"\t - Kept only columns: {keepColumns1} for file {file1}")
    if keepColumns2:
        df2 = df2[keepColumns2]
        print(f"\t - Kept only columns: {keepColumns2} for file {file2}")
    
    transactions = pd.merge(df1, df2, on=mergeOnColumn)
    print(f"\tMERGED DATAFRAMES (on {mergeOnColumn})")
    
    if keepColumnsMerged:
        transactions = transactions[keepColumnsMerged]
        print(f"\t - Kept only columns: {keepColumnsMerged} for merged transactions")
            
    if groupByColumn and groupBySelection:
        transactions = transactions.groupby(groupByColumn)[groupBySelection].apply(combine_to_set).reset_index()
        print(f"\t - Grouped by {groupByColumn} and selected {groupBySelection}")
    
    if newColumnNames:
        transactions.columns = newColumnNames
        print(f"\t - Renamed columns to {newColumnNames}")
    
    transactions.to_csv(outputfile)
    print(f"\tSAVED TRANSACTIONS TO: {outputfile}")

def sampleOrders(frac):
    if os.path.exists(SAMPLED_ORDERS): 
        print("Skipped: Sample orders")
        return
    orders = pd.read_csv(DATA_PATH + "orders.csv")
    sampledOrders = orders.sample(frac=frac)
    sampledOrders.to_csv(SAMPLED_ORDERS, index=False)
    print("Sampled orders")
        
def filterOrderProducts(keep_order_columns: list[str] = []):
    """
    keep_order_columns: Specify which columns from orders.csv to add to our filtered_order_products output
    """
    if os.path.exists(FILTERED_ORDER_PRODUCTS):
        print("Skipped: Filter order_products.csv")
        return
    
    if "order_id" not in keep_order_columns:
        keep_order_columns.append("order_id")
    
    sampled_orders = pd.read_csv(SAMPLED_ORDERS)
    order_products = pd.read_csv(DATA_PATH + "order_products.csv")
    
    # Keep only selected columns (or only id's)
    sampled_orders_columns = sampled_orders[keep_order_columns]
    
    filtered_order_products = pd.merge(order_products, sampled_orders_columns, on="order_id", how="inner")
    
    filtered_order_products.to_csv(FILTERED_ORDER_PRODUCTS, index=False)
    print("Filtered order_products.csv")

if __name__ == "__main__":
    # Remove files if they need to be reprocessed (comment out steps that dont have to be redone)
    # if os.path.exists(SAMPLED_ORDERS):
    #     os.remove(SAMPLED_ORDERS)
    # if os.path.exists(FILTERED_ORDER_PRODUCTS):
    #     os.remove(FILTERED_ORDER_PRODUCTS)
    # if os.path.exists(TRANSACTIONS):
    #     os.remove(TRANSACTIONS)
    
    # Task 1A: Preprocess/ prune data
    ## Sample orders
    sampleOrders(0.25)
    ## Filter order_products.csv based on sampled orders
    filterOrderProducts(keep_order_columns=["order_dow"])
    
    # Task 1B: Construct transactions
    createTransactionsCSV(
        file1=FILTERED_ORDER_PRODUCTS, 
        file2=DATA_PATH + "products.csv", 
        mergeOnColumn="product_id",     # Merge files on this column
        groupByColumn="order_id",       # Group merged datafframe on this column
        # groupBySelection="product_name",    # Select after grouping
        groupBySelection=["product_name", "order_dow"],      
        outputfile=TRANSACTIONS,
        # keepColumns1=["order_id", "product_id"],        # Keep only these columns from file 1
        keepColumns1=["order_id", "product_id", "order_dow"],  # This keeps the order_dow column for Task 2B purchase timing analysis
        keepColumns2=["product_id", "product_name"],    # Keep only these columns from file 2
        # keepColumnsMerged=["order_id", "product_name"], # After merge, keep only these columns
        keepColumnsMerged=["order_id", "product_name", "order_dow"], # This keeps order_dow as well 
        newColumnNames=["order_id", "items"]    # Rename columns directly after merge (before grouping)
    )
    
    # Task 2A: Explore dataset with apriori to mine association rules
    print("Loading transactions to run apriori")
    # Load transactions and turn dataframe to series by selecting items column
    transactions = pd.read_csv(TRANSACTIONS, index_col="order_id")["items"]
    # transactions = pd.read_csv(PROCESSED_PATH + "transactions_2a_data.csv", index_col="order_id")["items"]
    
    ## Since loading the file from csv, each itemset is actually a string (see transactions.csv)
    ### Example entry: 378,"{'Whole Vitamin D Milk', 'Skim Milk'}"
    ## Turn the string back into set using ast.literal_eval (https://docs.python.org/3/library/ast.html#ast.literal_eval)
    ## python built in eval() function is also possible to use here (https://docs.python.org/3/library/functions.html#eval), but the documentation links to ast.literal_eval as a "safer" way to evaluate these strings
    
    transactions = transactions.apply(ast.literal_eval)
    
    # The apriori algorithm crashes if not all elements are the same type, so turn every item into a string
    transactions = transactions.apply(set_elements_to_str)

    # Mine association rules with apriori
    min_supports = [0.005, 0.01, 0.05, 0.1, 0.2]
    min_confindences = [0.05, 0.1, 0.2, 0.3]
    results = []
    resulting_rules = {}
    
    print("Mining association rules with apriori...")
    for minsup in min_supports:
        # We store the result for each minsup value, to later create a table for each minsup value
        minsupResult = []
        for mincon in min_confindences:
            print(f"\t- Mining with: min. support: {minsup} | min. confidence: {mincon}")
            rules = (list(
                apriori(
                    transactions=transactions,
                    min_support=minsup,
                    min_confidence=mincon,
                )
            ))
            minsupResult.append([minsup, mincon, len(rules)])
            resulting_rules[(minsup,mincon)] = rules
        results.append(minsupResult)
            

    print("--- Results ---")

    # Create a table for each minsup value
    for r in results:
        table = tabulate(
            r,
            headers = ["Min. Support", "Min. Confidence", "Rule count"],
            tablefmt="fancy_grid"
        )
        print(table + "\n")
    
    # Write mined rules to txt file
    with open(PROCESSED_PATH + "mined_rules.txt", "w") as f:
        for key in resulting_rules.keys():
            f.write(f"min. sup={key[0]}, min. conf={key[1]} || Rules={resulting_rules[key]}\n\n")
    