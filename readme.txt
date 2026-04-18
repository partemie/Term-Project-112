[Project Name]

Multi-asset Portfolio Optimization Simulator. This project takes multiple assets as inputs and uses Monte Carlo simulation to generate an optimal portfolio.

[Project Description]

​​This project is an interactive portfolio optimization and analysis tool built with a graphical interface. It allows users to input a portfolio of up to 5 assets (stocks), assign weights manually or upload via CSV, and analyze performance using real historical market data. It also allows the user to compare their portfolio with an optimal (max Sharpe ratio) portfolio, an equal-weight portfolio, and a benchmark (S&P 500 via SPY).

[Project’s Functionality]

Data input: 

Users can:
	1	Enter tickers (e.g., AAPL, MSFT)
	2	Assign weights
	3	Set investment amount and time period
	4	Upload a CSV file 

Market Data Processing

	1	Downloads historical price data using Yahoo Finance
	2	Computes log returns, correlation matrix, portfolio returns

Portfolio Analytics

Calculates:
	1	Expected return
	2	Volatility (risk)
	3	Variance
	4	Cumulative growth over time

Efficient Frontier
	1	Generates around 8000 random portfolios (using Monte Carlo simulation)
	2	Computes risk (volatility) and return
	3	Builds the Efficient Frontier
	4	Identifies optimal portfolio (max Sharpe ratio)

Backtesting

Simulates historical performance of user portfolio, optimal portfolio, equal-weight portfolio, and benchmark (SPY). Outputs cumulative returns for comparison.

Visualizations 
	1	The app includes multiple interactive screens:
	2	Weights: portfolio allocation (donut charts)
	3	Summary: key metrics (return, volatility, Sharpe)
	4	Growth: historical and 5-year projected performance
	5	Annual Returns: bar chart of yearly performance
	6	Correlation Heatmap: shows diversification relationships
	7	Efficient Frontier: risk-return scatter and optimal point
	8	Table: annual returns table


[How to Run]

Run the file 'main.py' using a python3 interpreter in an environment where all the libraries listed below are installed. Install required libraries listed below by running 'pip install cmu-graphics yfinance pandas numpy' in your terminal. Save the file in the same folder with cmu-graphics and then run the file. To use the app, enter tickers, weights, dates, and investment amount. Users can also use the file ‘portfolio.csv’ to see an example of data input that the app requires. Next, click "Optimise Portfolio" and navigate through results using the bottom tabs.

[Libraries]

The app uses the following non-built-in libraries:

1. cmu-graphics: charts, buttons, layout, interaction

2. yfinance: historical prices, benchmark (SPY), company names

3. numpy: matrix operations, portfolio math, Monte Carlo simulation

4. pandas: time series handling, grouping (annual returns), date parsing

All of them can be installed using pip by running:

pip install cmu-graphics
pip install yfinance 
pip install numpy
pip install pandas

[Shortcut Commands]

Caps Lock + Shift: to use the upper case in the CSV box (Mac)

The app doesn’t support any shortcut commands.