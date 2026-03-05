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
    
def createTransactionsCSV(file1: str, file2: str, 
                          mergeOnColumn: str,  
                          groupByColumn: str | None = None,
                          groupBySelection: str | None = None,
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
    
    if newColumnNames:
        transactions.columns = newColumnNames
        print(f"\t - Renamed columns to {newColumnNames}")
            
    if groupByColumn and groupBySelection:
        transactions = transactions.groupby(groupByColumn)[groupBySelection].apply(set)
        print(f"\t - Grouped by {groupByColumn} and selected {groupBySelection}")
    
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
        
def filterOrderProducts():
    if os.path.exists(FILTERED_ORDER_PRODUCTS):
        print("Skipped: Filter order_products.csv")
        return
    
    ## Get order ids
    sampled_orders = pd.read_csv(SAMPLED_ORDERS)
    sampled_orders_ids = sampled_orders["order_id"]
    
    ## Get order_products
    order_products = pd.read_csv(DATA_PATH + "order_products.csv")
    
    ## Filter orders in order_products based on ids
    filtered_order_products = order_products[order_products["order_id"].isin(sampled_orders_ids)]
    
    ## Save
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
    filterOrderProducts()
    
    # Task 1B: Construct transactions
    createTransactionsCSV(
        file1=FILTERED_ORDER_PRODUCTS, 
        file2=DATA_PATH + "products.csv", 
        mergeOnColumn="product_id",     # Merge files on this column
        groupByColumn="order_id",       # Group merged datafframe on this column
        groupBySelection="items",       # Select after grouping
        outputfile=TRANSACTIONS,
        keepColumns1=["order_id", "product_id"],        # Keep only these columns from file 1
        keepColumns2=["product_id", "product_name"],    # Keep only these columns from file 2
        keepColumnsMerged=["order_id", "product_name"], # After merge, keep only these columns
        newColumnNames=["order_id", "items"]    # Rename columns directly after merge (before grouping)
    )
    
    # Task 2A: Explore dataset with apriori to mine association rules
    print("Loading transactions to run apriori")
    # Load transactions and turn dataframe to series by selecting items column
    transactions = pd.read_csv(TRANSACTIONS, index_col="order_id")["items"]
    
    ## Since loading the file from csv, each itemset is actually a string (see transactions.csv)
    ### Example entry: 378,"{'Whole Vitamin D Milk', 'Skim Milk'}"
    ## Turn the string back into set using ast.literal_eval (https://docs.python.org/3/library/ast.html#ast.literal_eval)
    ## python built in eval() function is also possible to use here (https://docs.python.org/3/library/functions.html#eval), but the documentation links to ast.literal_eval as a "safer" way to evaluate these strings
    transactions = transactions.apply(ast.literal_eval)
    
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
    
    ## GENERATE FULL TRANSACTIONS (not filtered)
    # createTransactionsCSV(
    #     file1=DATA_PATH + "order_products.csv", 
    #     file2=DATA_PATH + "products.csv", 
    #     mergeOnColumn="product_id",     # Merge files on this column
    #     groupByColumn="order_id",       # Group merged datafframe on this column
    #     groupBySelection="items",       # Select after grouping
    #     outputfile=PROCESSED_PATH + "transactions_FULL.csv",
    #     keepColumns1=["order_id", "product_id"],        # Keep only these columns from file 1
    #     keepColumns2=["product_id", "product_name"],    # Keep only these columns from file 2
    #     keepColumnsMerged=["order_id", "product_name"], # After merge, keep only these columns
    #     newColumnNames=["order_id", "items"]    # Rename columns directly after merge (before grouping)
    #     )
    
    
# All underneath have min_support = 0.1 and min_confidence = 0.2
# FULL OUTPUT of apriori on the full set
# [RelationRecord(items=frozenset({'Organic Baby Spinach', 'Bag of Organic Bananas'}), support=0.015722263912760083, ordered_statistics=[OrderedStatistic(items_base=frozenset({'Organic Baby Spinach'}), items_add=frozenset({'Bag of Organic Bananas'}), confidence=0.20900657515742635, lift=1.7708286228508512)]), RelationRecord(items=frozenset({'Bag of Organic Bananas', 'Organic Hass Avocado'}), support=0.019354271845617697, ordered_statistics=[OrderedStatistic(items_base=frozenset({'Organic Hass Avocado'}), items_add=frozenset({'Bag of Organic Bananas'}), confidence=0.2931993824617321, lift=2.4841604063142833)]), RelationRecord(items=frozenset({'Organic Raspberries', 'Bag of Organic Bananas'}), support=0.012636566397187398, ordered_statistics=[OrderedStatistic(items_base=frozenset({'Organic Raspberries'}), items_add=frozenset({'Bag of Organic Bananas'}), confidence=0.2965084886012216, lift=2.512197131299829)]), RelationRecord(items=frozenset({'Organic Strawberries', 'Bag of Organic Bananas'}), support=0.019336639288385853, ordered_statistics=[OrderedStatistic(items_base=frozenset({'Organic Strawberries'}), items_add=frozenset({'Bag of Organic Bananas'}), confidence=0.23478737340198927, lift=1.9892589541312347)]), RelationRecord(items=frozenset({'Banana', 'Large Lemon'}), support=0.012862203358374553, ordered_statistics=[OrderedStatistic(items_base=frozenset({'Large Lemon'}), items_add=frozenset({'Banana'}), confidence=0.2676625702771282, lift=1.8229952841403647)]), RelationRecord(items=frozenset({'Banana', 'Organic Avocado'}), support=0.016619731190170715, ordered_statistics=[OrderedStatistic(items_base=frozenset({'Organic Avocado'}), items_add=frozenset({'Banana'}), confidence=0.30186620635747785, lift=2.055949287422828)]), RelationRecord(items=frozenset({'Banana', 'Organic Baby Spinach'}), support=0.015957464294818747, ordered_statistics=[OrderedStatistic(items_base=frozenset({'Organic Baby Spinach'}), items_add=frozenset({'Banana'}), confidence=0.21213325122663435, lift=1.444796394935324)]), RelationRecord(items=frozenset({'Banana', 'Organic Fuji Apple'}), support=0.010505716684254396, ordered_statistics=[OrderedStatistic(items_base=frozenset({'Organic Fuji Apple'}), items_add=frozenset({'Banana'}), confidence=0.3784409348792645, lift=2.577484176798708)]), RelationRecord(items=frozenset({'Banana', 'Organic Strawberries'}), support=0.01743232310734671, ordered_statistics=[OrderedStatistic(items_base=frozenset({'Organic Strawberries'}), items_add=frozenset({'Banana'}), confidence=0.21166497929798206, lift=1.4416070901448015)]), RelationRecord(items=frozenset({'Banana', 'Strawberries'}), support=0.012904641038491873, ordered_statistics=[OrderedStatistic(items_base=frozenset({'Strawberries'}), items_add=frozenset({'Banana'}), confidence=0.28893572886346147, lift=1.967882437176007)]), RelationRecord(items=frozenset({'Organic Strawberries', 'Organic Raspberries'}), support=0.01061928230710356, ordered_statistics=[OrderedStatistic(items_base=frozenset({'Organic Raspberries'}), items_add=frozenset({'Organic Strawberries'}), confidence=0.24917428104598083, lift=3.0254985932976215)])]

# Apriori output of 50% sample:
# [RelationRecord(items=frozenset({'Bag of Organic Bananas', 'Organic Baby Spinach'}), support=0.01571006981589211, ordered_statistics=[OrderedStatistic(items_base=frozenset({'Organic Baby Spinach'}), items_add=frozenset({'Bag of Organic Bananas'}), confidence=0.20917149404799795, lift=1.7713877428727274)]), RelationRecord(items=frozenset({'Bag of Organic Bananas', 'Organic Hass Avocado'}), support=0.01930963425843474, ordered_statistics=[OrderedStatistic(items_base=frozenset({'Organic Hass Avocado'}), items_add=frozenset({'Bag of Organic Bananas'}), confidence=0.2935111417955869, lift=2.4856256888138724)]), RelationRecord(items=frozenset({'Organic Raspberries', 'Bag of Organic Bananas'}), support=0.012644194738647597, ordered_statistics=[OrderedStatistic(items_base=frozenset({'Organic Raspberries'}), items_add=frozenset({'Bag of Organic Bananas'}), confidence=0.2969903703079817, lift=2.5150898505992028)]), RelationRecord(items=frozenset({'Bag of Organic Bananas', 'Organic Strawberries'}), support=0.01930604843963095, ordered_statistics=[OrderedStatistic(items_base=frozenset({'Organic Strawberries'}), items_add=frozenset({'Bag of Organic Bananas'}), confidence=0.23451520167261958, lift=1.9860132263762456)]), RelationRecord(items=frozenset({'Banana', 'Large Lemon'}), support=0.012962137339250732, ordered_statistics=[OrderedStatistic(items_base=frozenset({'Large Lemon'}), items_add=frozenset({'Banana'}), confidence=0.2691243439093695, lift=1.8292973150088712)]), RelationRecord(items=frozenset({'Banana', 'Organic Avocado'}), support=0.01658620487695263, ordered_statistics=[OrderedStatistic(items_base=frozenset({'Organic Avocado'}), items_add=frozenset({'Banana'}), confidence=0.3010805181279698, lift=2.04651045465605)]), RelationRecord(items=frozenset({'Banana', 'Organic Baby Spinach'}), support=0.015903704031297025, ordered_statistics=[OrderedStatistic(items_base=frozenset({'Organic Baby Spinach'}), items_add=frozenset({'Banana'}), confidence=0.21174963396778915, lift=1.4393088014432214stics=[OrderedStatistic(items_base=frozenset({'Organic Fuji Apple'}), items_add=frozenset({'Banana'}), confidence=0.3762031625988312, lift=2.5571355799445707)]), RelationRecord(items=frozenset({'Banana', 'Organic Strawberries'}), support=0.017512541401266273, ordered_statistics=[OrderedStatistic(items_base=frozenset({'Organic Strawberries'}), items_add=frozenset({'Banana'}), confidence=0.2127290414379882, lift=1.4459660492771005)]), RelationRecord(items=frozenset({'Banana', 'Strawberries'}), support=0.012993214435550286, ordered_statistics=[OrderedStatistic(items_base=frozenset({'Strawberries'}), items_add=frozenset({'Banana'}), confidence=0.2908105939004816, lift=1.9767035225079295)]), RelationRecord(items=frozenset({'Organic Raspberries', 'Organic Strawberries'}), support=0.010631355116784142, ordered_statistics=[OrderedStatistic(items_base=frozenset({'Organic Raspberries'}), items_add=frozenset({'Organic Strawberries'}), confidence=0.2497122322356046, lift=3.0333143877666697)])]    

# Apriori output of 25% sample:
# [RelationRecord(items=frozenset({'Organic Baby Spinach', 'Bag of Organic Bananas'}), support=0.015599092911970144, ordered_statistics=[OrderedStatistic(items_base=frozenset({'Organic Baby Spinach'}), items_add=frozenset({'Bag of Organic Bananas'}), confidence=0.20773039145454258, lift=1.7670634260133653)]), RelationRecord(items=frozenset({'Organic Hass Avocado', 'Bag of Organic Bananas'}), support=0.019236769341667834, ordered_statistics=[OrderedStatistic(items_base=frozenset({'Organic Hass Avocado'}), items_add=frozenset({'Bag of Organic Bananas'}), confidence=0.29248609545966775, lift=2.4880398014237244)]), RelationRecord(items=frozenset({'Organic Raspberries', 'Bag of Organic Bananas'}), support=0.012578255469365456, ordered_statistics=[OrderedStatistic(items_base=frozenset({'Organic Raspberries'}), items_add=frozenset({'Bag of Organic Bananas'}), confidence=0.29412422429697543, lift=2.5019745826333275)]), RelationRecord(items=frozenset({'Bag of Organic Bananas', 'Organic Strawberries'}), support=0.019142330814574136, ordered_statistics=[OrderedStatistic(items_base=frozenset({'Organic Strawberries'}), items_add=frozenset({'Bag of Organic Bananas'}), confidence=0.23282492693778442, lift=1.980530678131527)]), RelationRecord(items=frozenset({'Banana', 'Large Lemon'}), support=0.01295361872895306, ordered_statistics=[OrderedStatistic(items_base=frozenset({'Large Lemon'}), items_add=frozenset({'Banana'}), confidence=0.2667914122513295, lift=1.8146549408121164)]), RelationRecord(items=frozenset({'Banana', 'Organic Avocado'}), support=0.016521960543822465, ordered_statistics=[OrderedStatistic(items_base=frozenset({'Organic Avocado'}), items_add=frozenset({'Banana'}), confidence=0.2991493690612757, lift=2.034746456143346)]), RelationRecord(items=frozenset({'Banana', 'Organic Baby Spinach'}), support=0.016205173079520827, ordered_statistics=[OrderedStatistic(items_base=frozenset({'Organic Baby Spinach'}), items_add=frozenset({'Banana'}), confidence=0.2158014550201379, lift=1.4678327659880868)]), RelationRecord(items=frozenset({'Banana', 'Organic Fuji Apple'}), support=0.010522125512388781, ordered_statistics=[OrderedStatistic(items_base=frozenset({'Organic Fuji Apple'}), items_add=frozenset({'Banana'}), confidence=0.37699160527668324, lift=2.5642117689888844)]), RelationRecord(items=frozenset({'Banana', 'Organic Strawberries'}), support=0.01750101312217357, ordered_statistics=[OrderedStatistic(items_base=frozenset({'Organic Strawberries'}), items_add=frozenset({'Banana'}), confidence=0.21286185788853831, lift=1.4478382901020745)]), RelationRecord(items=frozenset({'Strawberries', 'Banana'}), support=0.012869939021401683, ordered_statistics=[OrderedStatistic(items_base=frozenset({'Strawberries'}), items_add=frozenset({'Banana'}), confidence=0.2893386008761321, lift=1.968015826359949)]), RelationRecord(items=frozenset({'Organic Raspberries', 'Organic Strawberries'}), support=0.010541252302686238, ordered_statistics=[OrderedStatistic(items_base=frozenset({'Organic Raspberries'}), items_add=frozenset({'Organic Strawberries'}), confidence=0.2464918656007156, lift=2.9980388049479827)])]