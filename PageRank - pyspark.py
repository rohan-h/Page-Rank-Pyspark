from pyspark import SparkContext, SparkConf
import re
import numpy as np


# This function generates inward links from dataset [used to generate M]
def inLinks(urls):
    parts = re.split(r'\t+', urls)
    return int(parts[1]), int(parts[0]) ## Now gives in links


# This function generates outward links from dataset [used to calculate degrees]
def outLinks(urls):
    parts = re.split(r'\t+', urls)
    return int(parts[0]), int(parts[1])


# Driver Function
def page_rank(dataset_path, iterations=40, beta = 0.8):
    # Degree count function calculated from outward links
    def getDegree(links):
        deg = np.zeros(n)
        sortedLinks = links.sortByKey().collect()
        for col, line in enumerate(sortedLinks):
            deg[col] = 1 / len(line[1])
        return deg

    def create_M(links, n):
        def updateRow(rows):
            row = rows[1]
            index = rows[0]
            link = sortedLinks[index]
            # Assign 1/deg to proper nodes
            for i in link[1]:
                row[i - 1] = degrees[i - 1]
            return (rows[0], row)

        # inward links are distinct grouped sorted
        sortedLinks = links.sortByKey().collect()

        # Temp an RDD initialized with array of zero used for creation of M
        temp = sc.parallelize(np.zeros(shape=(n, n)))
        temp = temp.zipWithIndex()

        # the zipwithindex is flipped left to right
        flipTemp = temp.map(lambda r: (r[1], r[0]))

        # Calling updateRow Function on each temp row
        temp2 = flipTemp.map(lambda row: updateRow(row))
        return temp2

    # Function for Matrix Multiplication
    def matrixMult(rows):
        row = rows[1]
        product = np.dot(row, R.T) * beta
        return row[0], round(product, 3)

    # Gathering Data in RDD
    lines = sc.textFile(dataset_path)

    # Parsing in ward Links
    inlinks = lines.map(lambda urls: inLinks(urls))

    # Making them Distinct
    distinctInLinks = inlinks.distinct()

    # Grouping them according to the inward nodes
    distinctInLinks = distinctInLinks.groupByKey()

    # Counting the maximum nodes
    n = distinctInLinks.count()

    # Parsing out ward Links
    outlinks = lines.map(lambda urls: outLinks(urls))

    # Making them Distinct
    distinctOutLinks = outlinks.distinct()

    # Grouping them according to the outward nodes
    distinctOutLinks = distinctOutLinks.groupByKey()

    # Calculating degrees based on outward links
    degrees = getDegree(distinctOutLinks)

    # Creating M
    M = create_M(distinctInLinks, n)

    # Initializing R
    R = np.full(n, (1 / n))
    beta = beta

    oneMinusBetabyN = (1 - beta) / n

    # Starting Iterations
    for i in range(iterations):
        print("\n Iteration", i)
        product = M.map(matrixMult)
        BMR = np.array(product.map(lambda p: p[1]).collect())
        R = oneMinusBetabyN + BMR
        print("\nTop 5 Nodes per Iteration:\n")
        l = sorted(enumerate(R), key=lambda i: i[1], reverse=True)
        for rank in l[:5]:
            print("\t", rank[0] + 1, ": ", round(rank[1], 4))
        print("--------------------------------------------------------")

    print("\nLowest 5 Nodes per Iteration:\n")
    l = sorted(enumerate(R), key=lambda i: i[1])
    for rank in l[:5]:
        print("\t", rank[0] + 1, ": ", round(rank[1], 4))


# Intializing Spark
conf = SparkConf().setMaster("local").setAppName("PageRank")
sc = SparkContext(conf=conf)
dataset_path = 'graph-full.txt'  # path to the dataset
iterations = 40  # number of iterations
beta = 0.8

page_rank(dataset_path, iterations, beta)