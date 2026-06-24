# Introduction

This is the third of four projects meant to showcase my knowledge of GitHub and various data science tools, such as Python. For this project, I worked with a dataset documenting over 500,000 transactions over the course of 1 year, for an online retail store based in the United Kingdom. This project will focus on cleaning, analyzing, and visualing the dataset using Python.

From this dataset, I wanted to know a few things:
- Which products are most successful?
- How do prices impact the success of a product?
- Does the success of a product correlate with how often orders are cancelled?
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
<p align="center">
  <img src="assets/Dataset.png" width="800">
</p>

*Dataset sourced from the [UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/dataset/352/online+retail)

Top 4 rows before cleaning:

| InvoiceNo | StockCode |                         Description | Quantity |    InvoiceDate | UnitPrice | CustomerID |        Country |
|----------:|----------:|------------------------------------:|---------:|---------------:|----------:|-----------:|---------------:|
|    536365 |    85123A |  WHITE HANGING HEART T-LIGHT HOLDER |        6 | 12/1/2010 8:26 |      2.55 |    17850.0 | United Kingdom |
|    536365 |     71053 |                 WHITE METAL LANTERN |        6 | 12/1/2010 8:26 |      3.39 |    17850.0 | United Kingdom |
|    536365 |    84406B |      CREAM CUPID HEARTS COAT HANGER |        8 | 12/1/2010 8:26 |      2.75 |    17850.0 | United Kingdom |
|    536365 |    84029G | KNITTED UNION FLAG HOT WATER BOTTLE |        6 | 12/1/2010 8:26 |      3.39 |    17850.0 | United Kingdom |

The dataset, as shown above is a single table with 8 columns. Each row documents a single order from a customers with the ID of the product orders, the amount ordered, the date it was ordered, the ID of the customer making the order, the unit price of item, and the country it was ordered from.

One thing of note regarding how the transactions are documented, is that not all rows are completed transactions. Some rows are cancelled orders, represented by a 'C' in fron of the InvoiceNo. Additionally, some others rows are documenting instances of damaged stock. Both the cancelled orders and damaged stock utilize negative quantity numbers. 

# Cleaning

When starting the cleaning process of the data I had a few goals in mind:
1. First, I wanted to remove all damage instances since there weren't enough to get meaningful analysis out of it. 
2. Second, I wanted to change how the cancelled orders to documented since the 'C' in the InvoiceNo would mess up grouping analysis. 
3. Third, I needed to ensure that the Date column was correctly recognized by Python as a date time datatype. 

```python
## Imports
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.patches import Patch

from scipy.stats import pearsonr

## Plotting Constants
COLORS = ['salmon', 'skyblue', 'lightgreen', 'khaki', 'plum']
EDGECOLOR = 'dimgrey'

## Data loading
df = pd.read_csv('../dataset/online_retail.csv')
```
Before cleaning the data I first needed to do my imports, constant initialization, and data loading.

```python
## Data cleaning
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df['CustomerID'] = df['CustomerID'].astype('Int64')

df['Cancelled'] = df['InvoiceNo'].str.startswith('C')
df['InvoiceNo'] = df['InvoiceNo'].mask(df['Cancelled'],df['InvoiceNo'].str[1:])

df = df[(df.StockCode.str.len() <= 6) & (df.StockCode.str.isnumeric())]
```

After importing the necessary libraries and functions, I started by reading in the dataset from the csv with `.read_csv()`. Once the dataframe is created I use `.to_datetime()` to ensure the InvoiceDate is recognized as a date. Next I add a new column called Cancelled which is a boolean column for whether the row was documneting a cancelled transaction. I then use that column with `.mask()` to remove the 'C' from the ID of cancelled transactions. Following that, I drop all the transactions for products not following the StockCode 6 number format since a small selection of products had longer stock codes from color variants or were not customer purchases.

```python
df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

# Removed rows documenting product damage
df = df.drop(index=df[(df.Quantity < 0) & (~df.Cancelled)].index)

# Data separation by cancelled
df_full = df.copy()
df = df[~df.Cancelled].copy()
```
Next, I created a new column TotalPrice for the total cost of a transaction, this will make later analysis easier by doing the calculation here. Lastly, I drop all transactions representing damage instances with `.drop()` and create a separate dataframe without any cancelled transactions since only one of my analysis questions involves cancelled products.

Top 4 rows after cleaning:

| InvoiceNo | StockCode |                       Description | Quantity |         InvoiceDate | UnitPrice | CustomerID |        Country | Cancelled | TotalPrice |
|----------:|----------:|----------------------------------:|---------:|--------------------:|----------:|-----------:|---------------:|----------:|-----------:|
|    536365 |     71053 |               WHITE METAL LANTERN |        6 | 2010-12-01 08:26:00 |      3.39 |      17850 | United Kingdom |     False |      20.34 |
|    536365 |     22752 |      SET 7 BABUSHKA NESTING BOXES |        2 | 2010-12-01 08:26:00 |      7.65 |      17850 | United Kingdom |     False |      15.30 |
|    536365 |     21730 | GLASS STAR FROSTED T-LIGHT HOLDER |        6 | 2010-12-01 08:26:00 |      4.25 |      17850 | United Kingdom |     False |      25.50 |
|    536366 |     22633 |            HAND WARMER UNION JACK |        6 | 2010-12-01 08:28:00 |      1.85 |      17850 | United Kingdom |     False |      11.10 |
|

# Product Analysis Code

Before starting analysis on the products, I needed to create a grouped dataset for the products sold. This grouped dataframe is used in most of the product analysis and has the following details:
- The total number of a product sold
- The total revenue generated by the product
- The average unit price
- The price band the product falls into

```python
## Dataset grouping
products_group = df.groupby('StockCode',observed=True)[['Quantity','TotalPrice', 'UnitPrice']].agg(
    Quantity=('Quantity','sum'),
    TotalRevenue=('TotalPrice','sum'),
    UnitPrice=('UnitPrice','mean')
)
product_details = df_full.drop_duplicates('StockCode')[['StockCode','Description']]
products_group = products_group.merge(product_details, on='StockCode')

## Product price band creation
bands = ['£0-1', '£1-3', '£3-5', '£5-10', '£10+']
color_map = dict(zip(bands, COLORS))

products_group['PriceBand'] = pd.cut(
    products_group['UnitPrice'], 
    bins=[0,1,3,5,10,float('inf')], 
    labels=bands
)
```
I start by using `.groupby()` to group by the StockCode and create the first 3 columns of Quantity, TotalRevenue, and UnitPrice. I then use `.cut()` to create a price band column which classifies the products based on what price range they fall into of (£0-1], (£1-3], (£3-5], (£5-10], and (£10+). I also create a color map for the color bands which will be used later to ensure consistent coloring for bands. In some places the color map isn't necessary because of the natural ordering of the bands.


### Top Products
<p align="center">
  <img src="assets/1_TopProducts.png" width="800">
</p>

```python
### Setup
# Calculation
top_quantity = products_group.nlargest(10, 'Quantity')
top_totalprice = products_group.nlargest(10, 'TotalRevenue')

# Figure creation
fig, ax = plt.subplots(1, 2, figsize=(14,6))

```
For the first part of the product analysis I want to create bar charts to visualize the top 10 products by quantity sold and revenue generated. Since I have a grouped dataset, creating the bar plot is relatively simple. Using the `.nlargest()` function I can filter the dataset for the 10 top products by the passed column, in this case Quantity and TotalRevenue. After getting the top products I use `.subplots()` to allow me to put the 2 planned bar charts next to eachother in the same single image.

```python
## Quantity bar
bars1 = ax[0].barh(top_quantity['Description'], top_quantity['Quantity'], color=top_quantity['PriceBand'].map(color_map), edgecolor=EDGECOLOR)

ax[0].invert_yaxis()
ax[0].grid(axis='x', linestyle='--', alpha=0.4)
ax[0].set_axisbelow(True)

ax[0].xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x/1000:.0f}K'))
ax[0].bar_label(bars1, labels=[f'{x/1000:.1f}K' for x in top_quantity['Quantity']], padding=-40, fontsize='12')

ax[0].set_title('Top 10 Products by Quantity', fontsize=14, weight='bold', pad=12)
ax[0].set_xlabel('Quantity')
ax[0].set_ylabel('Product')
```

Using the top products filtered dataset, I create a horizonal bar with `.barh()` using the color map to set the bar colors. Following the bar creation there are still many customizations needed to make the chart intuitive and insightful. I start by inverting the y axis with `.invert_yaxis()` to make the order of the bars more intuitive. Next, I turn on the grid lines with `.grid()`, but only for the axis that is needed and set them to be drawn below the bars with `.set_axisbelow()`. Then I use f string formatting with `.set_major_formatter()` and `.bar_label()` to add bar labels and customize the bar labels and axis ticks to use K to represent 1000, this makes understanding the numbers at a glance much easier. I then finish by adding titles and axis labels with `.set_title()`, `.set_xlabel()`, and `.set_ylabel()`. Since the process for creating the second bar plot is very similiar I won't showcase it here, the only notable differeneces are the labels.

```python
legend_handles = [Patch(facecolor=color, edgecolor='dimgrey', label=band) for band, color in color_map.items()]
ax[1].legend(handles=legend_handles, title='Price Band', bbox_to_anchor=(1.02, 0.5), loc='center left')
```
To create the legend I used `Patch()` with the color map to create the legend handles. This was necessary because neither of the bar plots have all the bands in their data.

```python
## Finish
fig.tight_layout()
plt.savefig('../assets/1_TopProducts.png', dpi=200)
```
Once I have both bar charts and the legend created I can finish by calling `.tight_layout()` to clean the visual placements and saving the figure with `.savefig()` to file so it can be displayed in the README.


### Product Price Band
<p align="center">
  <img src="assets/2_PriceBandAnalysis.png" width="800">
</p>

```python
## Setup
# Calculations
price_grouped = products_group.groupby('PriceBand', observed=True)['TotalRevenue'].agg(
    TotalRevenue ='sum',
    BandCounts = 'count'
)

# Figure creation
fig, ax = plt.subplots(1, 2, figsize=(14,6))

colors = ['salmon', 'skyblue', 'lightgreen', 'khaki', 'plum']
```
For the second part of the price band analysis I want to create a pie chart and bar chart to visualize the breakdown of unique products and total revenue by the price bands. Utilizing price band column created earlier, I can group by the bands to get the total revenue of all products in the price bands in the count of how many unique products are in each of the bands.

```python
## Band distribution pie chart
ax[0].pie(
    x=price_grouped['BandCounts'],
    labels=price_grouped.index,
    startangle=90,
    counterclock=False,
    autopct='%1.1f%%',
    pctdistance=0.75,
    colors=COLORS,
    wedgeprops={'edgecolor':EDGECOLOR, 'width':0.5},
    textprops={'size':'large'}
)
ax[0].set_title('Distribution of Products Across Price Bands', weight='bold', fontsize=14,pad=10)

```
With the new price band data I create a pie chart with `.pie()` to visualize the breakdown of how many products are in each price band. I make sure to add percent labels with `autopct=` so the viewer can more easily compare the band distribution with the next total revenue bar chart.

```python
## Revenue distribution bar chart
bars1 = ax[1].bar(height=price_grouped['TotalRevenue'], x=price_grouped.index, color=COLORS, edgecolor=EDGECOLOR)

ax[1].grid(axis='y', linestyle='--', alpha=0.4)
ax[1].set_axisbelow(True)

ax[1].yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'£{y/1000000:.0f}M'))
ax[1].bar_label(bars1, labels=[f'£{y/1000000:.2f}M' for y in price_grouped['TotalRevenue']], padding=-20, fontsize='12')
ax[1].bar_label(bars1, labels=[f'{y*100:.1f}%' for y in price_grouped['TotalRevenue']/price_grouped['TotalRevenue'].sum()], padding=-35, fontsize='12')

ax[1].set_ylabel('Total Revenue')
ax[1].set_xlabel('Price Band')
ax[1].set_title('Revenue Contribution by Price Band', weight='bold', fontsize=14,pad=10)

ax[1].legend(bars1, bands, title='Price Band', bbox_to_anchor=(1.02, 0.5), loc='center left')



## Finish
fig.tight_layout()
plt.savefig('../assets/2_PriceBandAnalysis.png', dpi=200)
```
The process for creating and finishing the second bar plot is similar to the already showcased bar plots. The most notable difference is the additonal bar label for the percentage breakdown of revenue to help the viewer make comparisons between the two plots. 


### Cancellation Rate
<p align="center">
  <img src="assets/3_CancellationRates.png" width="500">
</p>

```python
## Setup
# Calculation
cancelled_group = (df_full.groupby(['StockCode', 'Cancelled'], observed=True)['StockCode'].count().unstack(fill_value=0))
cancelled_group = cancelled_group.rename(columns={False: 'NotCancelled', True: 'Cancelled'})
cancelled_group = cancelled_group[(cancelled_group.Cancelled + cancelled_group.NotCancelled) > 20]
cancelled_group['PercentCancelled'] = (cancelled_group['Cancelled'] / (cancelled_group['NotCancelled'] + cancelled_group['Cancelled']))


# Figure creation
fig, ax = plt.subplots(figsize=(7,6))
```
For the last part of the product analysis I want to chart the cancellation rate of a product versus the totals sales made to see if there is a correlation between the two values. To start I will need the cancellation rates. I calculate this by grouping the full dataframe (the one previously worked with had cancelled orders exlcuded) by StockCode and Cancelled, counting the number of orders cancelled and not cancelled. I then unstack it with `.unstack()` so now I have a dataframe with rows for each product and 2 columns for the cancelled counts and not cancelled counts. After some filtering to only get products with sufficient cancelled data I calculate the percent cancelled and save it to a new column.

```python
## Cancellation percentage scatterplot
x = cancelled_group['Cancelled'] + cancelled_group['NotCancelled']
y = cancelled_group['PercentCancelled']

ax.scatter(x=x, y=y, alpha=0.5, s=8)

ax.set_xscale('log')
ax.set_ylim((0.005,0.25))

ax.grid(axis="both", linestyle='--', alpha=0.4)
ax.set_axisbelow(True)

ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y*100:.0f}%'))

ax.set_title('Cancellation Rates of Products by Orders Made', fontsize=14, weight='bold', pad=10)
ax.set_ylabel('Canellation Rate')
ax.set_xlabel('Orders Made (Logarithmic)')
```
Now that I have the two values needed of orders made and cancellation percentage, I can create the scatterplot with `.scatter()`. Since there is a large variety in the number of orders for each product I set the x scale to logarithmic with `.set_xscale()`.

```python
## Line of best fit
x_log = np.log10(x)

m, b = np.polyfit(x_log, y, 1)
x_line_log = np.linspace(x_log.min(), x_log.max(), 200)
y_line = m * x_line_log + b
plt.plot(10**x_line_log,y_line, color='orangered', linewidth=2, label='Logarithmic fit')
plt.legend()
```
Using the log of the x values I create a line of best fit for the chart with `.polyfit()` and plot the line with `.plot()`.

```python
## Finish
r, p = pearsonr(x_log, y)
plt.text(0.02, 0.98, f"r = {r:.2f}\np = {p:.3g}", transform=plt.gca().transAxes, va='top', fontsize=12);

plt.tight_layout()
plt.savefig('../assets/3_CancellationRates.png', dpi=200)
```
To finish the plot I wanted to some statistical analysis to see if there is a statistically significant relationship. I use `pearsonr()` to get the r and p values and put them onto the chart with `.text()`.

# Product Analysis Takeaways

### Top Products & Price Band
From the two bar charts of the top 10 products by quantity sold and total revenue we can notice a couple patterns. One pattern is that the top products by quantity sold are all from the two lowest price bands, clearly showing that lower priced products sell better. Another pattern is that accross the two charts, the £1-3 band is the top performer with it having both the top 3 products by quantity sold and half of both the top 10 by quantity and revenue. Interestingly there isn't a single product from the lowest price band in the top 10 by revenue, showing that although they sell well, that doesn't make up for low revenue generated per sale.

Looking at the second chart we can get some more insights regarding the overall performances of the different price bands. Products in the £1-3 band by far are the most popular with 44.5% of all unique products offered having a price in that band. The most interesting insights however come from comparing the distribution of prodcuts by band to the distribution of revenue by band. Products in the £1-3 and £3-5 price bands have a roughly equal representation in the product distribution and revenue distribution. Products in the £0-1 price band make up 17.7% of unique products falling into that band but with those products up less than half of that percentage of overall revenue. Conversly the revenue of products in the £5-10 and £10+ outperforms their percentage of the unique products within their band. 

From these two charts it is clear that low price products within the £0-1 band are much less profitable compared to products of higher prices. Products within that band make many sales but bring in very little revenue compared to the percentage of unique products within the band. Conversly, products in the £1-3 band make up the core of the business with the majority of revenue and unique products within the band. Lastly, products of the higher price bands out perform the lower bands with regard to revenue generated per unique product listed. I would recommened a shift in focus away from lower priced products between £0-1 and toward the more revenue generating higher priced products.

### Cancellation Rates

Based on the graph of cancellation rates and the p-value being greater than 0.05, there is no statistically significant relationship between cancellation rates and the total number of orders made for a product. This tells us that products which have been ordered more do not have statistically different cancellation rates. 

### Questions and Answers
To answer the questions posed at the start:

- Which products are most successful?

    - The most successful products are those within the £1-3 price band with those products making up the largest portion of unique offered products and revenue.

- How do prices impact the success of a product?

    - The products in the lowest price band  of £0-1 make up a small portion of the overall revenue. On the other hand, products in the highest price bands of £5-10 and £10+ make more revenue relative to the unique number of products in the price bands. 

- Does the success of a product correlate with how often orders are cancelled?

    - No, the success of a product does not correlate with how often orders are cancelled.

# Customer Analysis Code

### Customer Revenue Band
<p align="center">
  <img src="assets/4_CustomerRevenueByBand.png" width="500">
</p>


### Monthly Revenue
<p align="center">
  <img src="assets/5_MonthlyRevenue.png" width="800">
</p>


### Country Revenue
<p align="center">
  <img src="assets/6_UKvsOther.png" width="800">
</p>
## Customer Analysis Takeaways

# Conclusion