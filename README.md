# Introduction

This is the third of four projects meant to showcase my knowledge of GitHub and various data science tools, such as Python. For this project, I worked with a dataset documenting over 500,000 transactions over the course of 1 year, for an online retail store based in the United Kingdom. This project will focus on cleaning, analyzing, and visualing the dataset using Python.

From this dataset, I wanted to know a few things:
- Which products are most successful?
- How do prices impact the success of a product?
- Does the Success of a product correlate with how often orders are cancelled?
- How much of overall revenue is contributed by low spending and high spending customers?
- What months of the year have the most revenue?
- How important is the customer base outside of the UK to revenue?


# Tools Used

- **Python**:
    - **Anoconda**: The data science oriented Python distribution used.
    - **Jupyter notebooks**: Used to separate the analysis process into multiple code blocks capable of being run separately.
    - **Pandas**: The Python library used for data manipulation, analysis, and cleaning.
    - **Matplotlib**: The Python library used for creating data visualizations.
- **Visual Studio Code**: Used to write and manage my Jupyter notebooks.
- **Git & GitHub**: Used to document and share my python code and analysis.


# Dataset

![Dataset variables](assets/Dataset.png)
*Dataset sourced from the [UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/dataset/352/online+retail)

The dataset, as shown above is a single table with 8 columns. Each row documents a single order from a customers with the ID of the product orders, the amount ordered, the date it was ordered, the ID of the customer making the order, the unit price of item, and the country it was ordered from.

One thing of note regarding how the transactions are documented, is that not all rows are completed transactions. Some rows are cancelled orders, represented by a 'C' in fron of the InvoiceNo. Additionally, some others rows are documenting instances of damaged stock. Both the cancelled orders and damaged stock utilize negative quantity numbers. 

# Cleaning

When starting the cleaning process of the data I had a few goals in mind:
1. First, I wanted to remove all damage instances since there weren't enough to get meaningful analysis out of it. 
2. Second, I wanted to change how the cancelled orders to documented since the 'C' in the InvoiceNo would mess up grouping analysis. 
3. Third, I needed to ensure that the Date column was correctly recognized by Python as a date time datatype. 

```python
## Data loading
df = pd.read_csv('../dataset/online_retail.csv')

## Data cleaning
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df['CustomerID'] = df['CustomerID'].astype('Int64')

df['Cancelled'] = df['InvoiceNo'].str.startswith('C')
df['InvoiceNo'] = df['InvoiceNo'].mask(df['Cancelled'],df['InvoiceNo'].str[1:])

df = df[(df.StockCode.str.len() <= 6) & (df.StockCode.str.isnumeric())]
```

After importing the necessary libraries and functions, I started by reading in the dataset from the csv with `pd.read_csv()`. Once the dataframe is created I use `pd.to_datetime()` to ensure the InvoiceDate is recognized as a date. Next I add a new column called Cancelled which is a boolean column for whether the row was documneting a cancelled transaction. I then use that column with `.mask()` to remove the 'C' from the ID of cancelled transactions. Following that, I drop all the transactions for products not following the StockCode 6 number format since a small selection of items had longer stock codes from color variants or were not customer purchases.

```python
df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

# Removed rows documenting product damage
df = df.drop(index=df[(df.Quantity < 0) & (~df.Cancelled)].index)

# Data separation by cancelled
df_full = df.copy()
df = df[~df.Cancelled].copy()
```
Next, I created a new column TotalPrice for the total cost of a transaction, this will make later analysis easier by doing the calculation here. Lastly, I drop all transactions representing damage instances with `.drop()` and create a separate dataframe without any cancelled transactions since only one of my analysis questions involves cancelled products.

# Analysis 

### Top Products



### Product Price Band



### Cancellation Rate



### Customer Revenue Band



### Monthly Revenue



### Country Revenue


## Conclusion