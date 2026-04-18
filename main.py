# Multi-asset portfolio optimization simulator

from cmu_graphics import *
import yfinance as yf
import pandas as pd
import numpy as np

def load_market_data(tickers, start='2020-01-01', end=None):
    # Download adjusted close price for selected tickers
    data = yf.download(tickers, start=start, end=end)['Close']
    if isinstance(data, pd.Series):
        data = data.to_frame()
    # Fill missing values
    data = data.ffill().bfill()
    return data

def load_portfolio_from_csv(file_path):
    """Parse a CSV with columns Ticker and Weight (max 5 rows).
    Reads the file directly with open() — no extra libraries needed.
    Returns (tickers, weights_normalised) or (None, error_message).
    """
    try:
        with open(file_path, 'r') as f:
            raw = f.read()
        # Clean and split into non-empty lines
        lines = [l.strip() for l in raw.splitlines() if l.strip()]
        if len(lines) < 2:
            return None, "CSV is empty or has no data rows"

        # Parse header: find ticker and weight column indices
        header = [h.strip().lower() for h in lines[0].split(',')]
        if 'ticker' not in header or 'weight' not in header:
            return None, "CSV must have 'Ticker' and 'Weight' columns"

        ti = header.index('ticker')
        wi = header.index('weight')

        tickers, weights = [], []
        # Parse each row
        for line in lines[1:]:
            parts = line.split(',')
            if len(parts) <= max(ti, wi):
                continue
            ticker = parts[ti].strip().upper()
            weight_str = parts[wi].strip()
            if not ticker or not weight_str:
                continue
            try:
                w = float(weight_str)
                tickers.append(ticker)
                weights.append(w)
            except ValueError:
                continue  # skip invalid numeric values

        # Validate
        if len(tickers) == 0:
            return None, "No valid rows found in CSV"
        if len(tickers) > 5:
            return None, f"CSV has {len(tickers)} tickers — maximum is 5"

        # Normalize weights to sum to 1
        weights = np.array(weights, dtype=float)
        if weights.sum() == 0:
            return None, "Weights sum to zero"

        weights = weights / weights.sum()
        return tickers, weights

    except FileNotFoundError:
        return None, f"File not found: {file_path}"
    except Exception as e:
        return None, f"CSV read error: {e}"

class PortfolioAppModel:
    def __init__(self):
        self.efficient_frontier = []
        self.optimal_weights = None
        self.tickers = ['AAPL', 'AMZN', 'MSFT']
        self.weights = [0.33, 0.33, 0.34]
        self.investment = 10000
        self.duration = 1
        self.price_data = None
        self.returns_df = None
        self.corr_matrix = None
        self.portfolio_returns = None
        self.equal_cumulative = None
        self.benchmark_cumulative = None
        self.using_csv = False
        self.optimal_returns = None
        self.optimal_cumulative = None
        self.cumulative_returns = []
        self.expected_return = 0
        self.volatility = 0
        self.variance = 0
        self.start_date = '2020-01-01'
        self.end_date = None
        self.all_portfolios = [] # stores (vol, ret) for every generated 
                                 # portfolio

    def calculate_portfolio(self):
        # Ensure there are tickers to process
        if len(self.tickers) == 0:
            return
        
        # Load historical price data for selected assets
        self.price_data = load_market_data(
            self.tickers, 
            start=self.start_date, 
            end=self.end_date
        )

        # Compute daily log returns
        self.returns_df = np.log(self.price_data / 
                                 self.price_data.shift(1)).dropna()

        # Correlation matrix for asset returns
        self.corr_matrix = self.returns_df.corr().values

        # Covert weights to NumPy array
        weights_array = np.array(self.weights)

        # Ensure weights match number of assets
        if len(weights_array) != self.returns_df.shape[1]:
            return
        
        # Compute portfolio daily returns (weighted sum of asset returns)
        portfolio_returns = self.returns_df.values @ weights_array
        if len(portfolio_returns) == 0:
            return
        
        # Compute risk/return statistics
        self.expected_return = np.mean(portfolio_returns)
        self.variance = np.var(portfolio_returns)
        self.volatility = np.sqrt(self.variance)

        # Build cumulative portfolio over time
        value = self.investment
        self.cumulative_returns = []
        for r in portfolio_returns:
            value *= (1 + r)
            self.cumulative_returns.append(value)
        
        self.generate_efficient_frontier()
        self.run_backtest()

    def generate_efficient_frontier(self, num_portfolios=8000):
        # Check if there is returns data
        if self.returns_df is None:
            return
        
        returns = self.returns_df.values
        # Annualized returns
        mu = np.mean(returns, axis=0) * 252 # Assumption of 252 trading days
        n = len(mu)
        # Annualized coveriance matrix
        if returns.shape[1] == 1:
            sigma = np.array([[np.var(returns[:, 0]) * 252]])
        else:
            sigma = np.cov(returns.T) * 252

        # Limit extreme concentration in generated portfolios
        max_weight = max(1.0 / n + 0.15, 0.6)
        results = []

        # Monte Carlo simulation of random portfolios
        for _ in range(num_portfolios):
            weights = np.random.random(n)
            weights = weights / np.sum(weights)

            # Skip oprtfolios with too many assets
            if n >= 4 and np.any(weights > max_weight):
                continue

            # Portfolio expected return
            port_return = np.dot(mu, weights)

            # Portfolio volatility
            port_vol = np.sqrt(np.dot(weights.T, np.dot(sigma, weights)))
            if port_vol <= 0:
                continue
            results.append((port_vol, port_return, weights))
        
        # If there are no valid portfolios
        if not results:
            self.efficient_frontier = []
            self.optimal_weights = None
            return
        
        # Sort portfolios by associated risk
        results = sorted(results, key=lambda x: x[0])
        
        # Build efficient frontier
        efficient = []
        max_return = -np.inf

        for vol, ret, w in results:
            if ret > max_return:
                efficient.append((vol, ret, w))
                max_return = ret
        self.efficient_frontier = efficient

        # Select maximum Sharpe ratio portfolio
        best = max(results, key=lambda x: x[1] / x[0])
        self.optimal_weights = best[2]

        # Store all (vol, ret) pairs, subsampled to 400
        step = max(1, len(results) // 400)
        self.all_portfolios = [(v, r) for v, r, _ in results[::step]]

        # Store user portfolio as a point
        w_user = np.array(self.weights)
        user_ret = float(np.dot(mu, w_user))
        user_vol = float(np.sqrt(np.dot(w_user.T, np.dot(sigma, w_user))))
        self.user_frontier_point = (user_vol, user_ret)
        
        # Store optimal portfolio point for plotting
        opt_ret = float(best[1])
        opt_vol = float(best[0])
        self.optimal_frontier_point = (opt_vol, opt_ret)

    def run_backtest(self):
        # Make sure that we have returns data before running backtest
        if self.returns_df is None:
            return
        
        # Convert user's weights into NumPy array
        w1 = np.array(self.weights)

        # Check if number of weights matches the number of assets
        if len(w1) != self.returns_df.shape[1]:
            return
        
        # Download benchmark data (S&P 500: SPY)
        try:
            benchmark = yf.download("SPY", start=self.start_date, 
                                    end=self.end_date)['Close']
            # Compute daily returns
            benchmark_returns = benchmark.pct_change().dropna()
        except:
            self.benchmark_cumulative = None
            return
        
        # Align dates between portfolio returns and benchmark returns
        common_idx = self.returns_df.index.intersection(benchmark_returns.index)
        aligned_df = self.returns_df.loc[common_idx]

        # Convert aligned returns to NumPy arrays
        aligned_returns = aligned_df.values
        benchmark_returns = benchmark_returns.loc[common_idx].values

        # Compute user's portfolio returns
        user_returns = aligned_returns @ w1

        # Store portfolio returns and compute cumulative performance
        self.portfolio_returns = user_returns
        self.user_cumulative = np.cumprod(1 + user_returns)

        # Equal-weight portfolio for comparison
        n = len(self.tickers)
        equal_weights = np.ones(n) / n
        equal_returns = aligned_returns @ equal_weights

        # Compute cumulative returns for equal-weight portfolio and benchmark
        self.equal_cumulative = np.cumprod(1 + equal_returns)
        self.benchmark_cumulative = np.cumprod(1 + benchmark_returns)

        # Compute performance of the portfolio with optimized weights 
        if self.optimal_weights is not None:
            w2 = np.array(self.optimal_weights)
            optimal_returns = aligned_returns @ w2

            # Store optimized portfolio returns and cumulative performance
            self.optimal_returns = optimal_returns
            self.optimal_cumulative = np.cumprod(1 + optimal_returns)
        else:
            # If there is no optimized portfolio
            self.optimal_returns = None
            self.optimal_cumulative = None

def createRow(y):
    # Create one input row for user's ticker, its weight, and bounds
    return {
        'ticker': createBox('', 100, y, 80, 30),
        'weight': createBox('', 200, y, 80, 30),
        'min': createBox('0', 300, y, 60, 30),
        'max': createBox('100', 380, y, 60, 30)
    }

def createBox(text, x, y, w, h):
    # Creates an input box
    return {'text': text, 'x': x, 'y': y, 'w': w, 'h': h}

# Background 
# Off-white graph paper color
BG_COLOR      = rgb(248, 249, 252)
GRID_COLOR    = rgb(218, 224, 236)   # subtle blue-grey lines
GRID_MAJOR    = rgb(200, 210, 228)   # slightly stronger every 5th line
GRID_MINOR    = 20                   # px between minor grid lines
GRID_MAJOR_N  = 5                    # every N lines is a major line

def drawBackground():
    """Off-white graph-paper background used on every screen."""
    drawRect(0, 0, 600, 600, fill=BG_COLOR, border=None)
    # Vertical lines

    x = 0
    col = 0

    while x <= 600:
        lineColor = GRID_MAJOR if col % GRID_MAJOR_N == 0 else GRID_COLOR
        drawLine(x, 0, x, 600, fill=lineColor, lineWidth=0.6, opacity=60)
        x += GRID_MINOR
        col += 1

    # Horizontal lines
    y = 0
    row = 0

    while y <= 600:
        lineColor = GRID_MAJOR if row % GRID_MAJOR_N == 0 else GRID_COLOR
        drawLine(0, y, 600, y, fill=lineColor, lineWidth=0.6, opacity=60)
        y += GRID_MINOR
        row += 1

# Shared header helpers
HEADER_H  = 46
HEADER_BG = rgb(25, 55, 120)

def drawScreenHeader(title):
    drawRect(0, 0, 600, HEADER_H, fill=HEADER_BG, border=None)
    drawLabel(title, 300, HEADER_H // 2, size=14, bold=True, fill='white')

def drawBackButton():
    drawRect(12, 10, 72, 26,
             fill=rgb(50, 80, 150), border=rgb(180, 200, 240), borderWidth=0.8)
    drawLabel("← Back", 48, 23, size=10, fill='white')

def onAppStart(app):
    # Set maximum number of drawable objects
    app.setMaxShapeCount(20000)
    # Initializes portfolio model
    app.model = PortfolioAppModel()

    # App starts on input screen (not part of navigation)
    app.screen = 'input'
    app.result_screens = ['weights', 'summary', 'growth',
                          'annual', 'correlation',
                          'frontier', 'table']

    app.graph_type = 'line'
    app.active_box = None
    app.status_message = ''  # for user feedback

    # Input fields for user's feedback
    app.start_year_box = createBox('2020-01-01', 160, 96, 100, 26)
    app.end_year_box   = createBox('2025-01-01', 160, 140, 100, 26)
    app.amount_box = createBox('10000', 420, 120, 100, 30)

    # Toggles
    app.use_historical = True
    app.use_benchmark = True

    app.ticker_names = {}   # populated lazily after optimise runs
    
    # CSV import state
    app.csv_loaded = False
    app.csv_path_box = createBox('', 312, 218, 256, 26)

    app.asset_rows = [createRow(334 + i * 36) for i in range(5)]

def inside(box, x, y):
    # Check if a mouse click was inside the box
    return (box['x'] <= x <= box['x'] + box['w'] and
            box['y'] <= y <= box['y'] + box['h'])

def processInput(app):
    # Extract tickers and weights from input grid
    tickers = []
    weights = []

    for row in app.asset_rows:
        ticker = row['ticker']['text'].strip().upper()
        weight = row['weight']['text'].strip()

        if ticker != '' and weight != '':
            try:
                w = float(weight)

                if w < 0 or w > 100:
                    app.status_message = 'Weights must be 0–100'
                    return False
                tickers.append(ticker)
                weights.append(w / 100)

            except:
                app.status_message = 'Invalid weight value'
                return False
    
    # Ensure there is at least one valid asset
    if len(tickers) == 0:
        app.status_message = 'Enter at least one ticker and weight'
        return False
    
    # Normalize weights to sum to 1
    total = sum(weights)
    if total == 0:
        app.status_message = 'Total weight cannot be zero'
        return False
    weights = [w / total for w in weights]

    try:
        investment = float(app.amount_box['text'])
        if investment <= 0:
            app.status_message = 'Investment must be positive'
            return False
        
    except:
        app.status_message = 'Invalid investment amount'
        return False
    
    # Parse dates
    start_text = app.start_year_box['text'].strip()
    end_text   = app.end_year_box['text'].strip()
    
    try:
        # Accept YYYY or YYYY-MM-DD
        if len(start_text) == 4:
            start_text = start_text + '-01-01'
        if len(end_text) == 4:
            end_text = end_text + '-12-31'
        start_dt = pd.to_datetime(start_text)
        end_dt   = pd.to_datetime(end_text)

        if end_dt <= start_dt:
            app.status_message = 'End date must be after start date'
            return False
        
        # Analysis window should not be too short
        if (end_dt - start_dt).days < 180:
            app.status_message = 'Period must be at least 6 months'
            return False
        
    except:
        app.status_message = 'Invalid date — use YYYY or YYYY-MM-DD'
        return False

    app.model.tickers = tickers
    app.model.weights = weights
    app.model.investment = investment
    app.model.start_date = start_text
    app.model.end_date   = end_text
    app.ticker_names = {}
    app.status_message = 'Loading data...'
    app.model.calculate_portfolio()
    app.status_message = 'Done!'
    return True

# Navigation 

def drawNavigation(app):
    # Bottom navigation bar for switching between result screens
    labels = ["Weights", "Summary", "Growth",
              "Annual", "Correlation",
              "Frontier", "Table"]
    
    # Draw each naviagtion tab
    for i, label in enumerate(labels):
        x = 20 + i * 82
        isActive   = app.screen == app.result_screens[i]
        fillColor  = rgb(25, 55, 120) if isActive else rgb(235, 238, 245)
        textColor  = 'white'          if isActive else rgb(60, 70, 100)
        borderCol  = rgb(25, 55, 120) if isActive else rgb(180, 190, 215)
        drawRect(x, 560, 74, 30, fill=fillColor, border=borderCol)
        drawLabel(label, x + 37, 575, fill=textColor, size=11)

# Input screen

CHAR_W = 7   # approximate pixel width per character at size=12

def drawBox(box, isActive=False):
    borderColor = rgb(25, 55, 120) if isActive else rgb(160, 170, 200)
    borderWidth = 2 if isActive else 1

    # Panel-style box: slightly off-white fill
    drawRect(box['x'], box['y'], box['w'], box['h'],
             fill=rgb(252, 253, 255), border=borderColor,
             borderWidth=borderWidth)
    
    # How many characters fit
    maxChars = max(1, int((box['w'] - 10) / CHAR_W))
    visibleText = box['text'][-maxChars:]
    drawLabel(visibleText,
              box['x'] + 5, box['y'] + box['h'] / 2,
              size=12, align='left', fill=rgb(20, 30, 60))
    
    # Blinking cursor when active
    if isActive:
        cursorX = box['x'] + 5 + len(visibleText) * CHAR_W
        cursorX = min(cursorX, box['x'] + box['w'] - 4)
        drawLine(cursorX, box['y'] + 5,
                 cursorX, box['y'] + box['h'] - 5,
                 fill=rgb(25, 55, 120), lineWidth=1.5)

def sumSafeWeights(app):
    # Computes total weight from all assets
    total = 0
    for row in app.asset_rows:
        try:
            total += float(row['weight']['text'])
        except:
            pass
    return total

def drawInputScreen(app):
    drawLabel("Portfolio Optimization Simulator", 300, 30, size=18, bold=True,
              fill=rgb(20, 40, 100))

    # Left panel card
    drawRect(10, 60, 280, 210, fill=rgb(252, 253, 255),
             border=rgb(200, 210, 230), borderWidth=1)
    drawRect(10, 60, 280, 24, fill=HEADER_BG, border=None)
    drawLabel("Date Range", 150, 72, bold=True, fill='white', size=11)

    drawLabel("Start Date:", 22, 96, size=10, fill=rgb(60,70,100), align='left')
    drawLabel("YYYY or YYYY-MM-DD", 22, 109, size=8, fill='gray', align='left')
    drawBox(app.start_year_box, app.active_box is app.start_year_box)

    drawLabel("End Date:", 22, 140, size=10, fill=rgb(60,70,100), align='left')
    drawLabel("YYYY or YYYY-MM-DD", 22, 153, size=8, fill='gray', align='left')
    drawBox(app.end_year_box, app.active_box is app.end_year_box)

    drawLabel("Use Historical", 22, 196, size=10, fill=rgb(60,70,100), 
              align='left')
    hFill = rgb(180, 230, 180) if app.use_historical else rgb(220, 225, 235)
    drawRect(160, 185, 52, 22, fill=hFill, border=rgb(150,170,150))
    drawLabel("Yes" if app.use_historical else "No", 186, 196, size=10,
              fill=rgb(30,90,30) if app.use_historical else rgb(80,80,80))

    drawLabel("Benchmark", 22, 226, size=10, fill=rgb(60,70,100), align='left')
    bFill = rgb(180, 230, 180) if app.use_benchmark else rgb(220, 225, 235)
    drawRect(160, 215, 52, 22, fill=bFill, border=rgb(150,170,150))
    drawLabel("Yes" if app.use_benchmark else "No", 186, 226, size=10,
              fill=rgb(30,90,30) if app.use_benchmark else rgb(80,80,80))

    # Right panel card
    drawRect(300, 60, 290, 210, fill=rgb(252, 253, 255),
             border=rgb(200, 210, 230), borderWidth=1)
    drawRect(300, 60, 290, 24, fill=HEADER_BG, border=None)
    drawLabel("Investment & Import", 445, 72, bold=True, fill='white', size=11)

    drawLabel("Amount ($):", 312, 101, size=10, fill=rgb(60,70,100), 
              align='left')
    drawBox(app.amount_box, app.active_box is app.amount_box)

    csvLoaded = getattr(app, 'csv_loaded', False)
    # "CSV path:" label first, then button below it, then box
    drawLabel("CSV path:", 340, 185, size=9, fill='gray')   # centred
    drawBox(app.csv_path_box, app.active_box is app.csv_path_box)   # y=218
    drawLabel("Users/path/to/portfolio.csv", 445, 250, size=8,
              fill=rgb(180,180,200))
    btnFill  = rgb(180, 230, 180) if csvLoaded else HEADER_BG
    btnLabel = "CSV Loaded" if csvLoaded else "Upload CSV"
    drawRect(380, 172, 180, 26, fill=btnFill, border=rgb(180,200,240))  
    # centred in panel
    drawLabel(btnLabel, 470, 185, size=10,
              fill=rgb(30,100,30) if csvLoaded else 'white')

    # Asset table card
    drawRect(10, 282, 580, 244, fill=rgb(252, 253, 255),
             border=rgb(200, 210, 230), borderWidth=1)
    drawRect(10, 282, 580, 24, fill=HEADER_BG, border=None)
    drawLabel("Portfolio Assets", 300, 294, bold=True, fill='white', size=11)

    drawLabel("Ticker", 140, 322, size=11, bold=True, fill=rgb(40,50,90))
    drawLabel("Weight %", 240, 322, size=11, bold=True, fill=rgb(40,50,90))
    drawLabel("Min %", 330, 322, size=11, bold=True, fill=rgb(40,50,90))
    drawLabel("Max %", 410, 322, size=11, bold=True, fill=rgb(40,50,90))

    for row in app.asset_rows:
        for key in row:
            drawBox(row[key], app.active_box is row[key])

    total = sumSafeWeights(app)
    color = rgb(30, 130, 50) if abs(total - 100) < 0.1 else rgb(180, 40, 40)
    drawLabel(f"Total: {pythonRound(total, 1)}%  (must sum to 100)",
              255, 540, fill=color, size=11)

    if app.status_message:
        drawLabel(app.status_message, 255, 560, fill=rgb(25,55,120), size=11)

    # Optimize button — styled like a proper CTA
    drawRect(360, 525, 150, 36, fill=HEADER_BG, border=rgb(100,140,220),
             borderWidth=1)
    drawLabel("Optimize Portfolio", 435, 541, bold=True, fill='white', size=12)

# Result screens 

import math

# Palette: 5 distinct colours used consistently across both donuts
DONUT_COLORS = [
    rgb(70,  130, 210),   # blue
    rgb(255, 160,  50),   # amber
    rgb(80,  190, 130),   # teal-green
    rgb(220,  80,  90),   # red
    rgb(150,  90, 200),   # purple
]

def getTickerName(ticker):
    """Return company short-name from yfinance, fall back to ticker itself."""
    try:
        # Fetch tickers from Yahoo Finance
        info = yf.Ticker(ticker).info
        return info.get('shortName') or info.get('longName') or ticker
    except:
        return ticker

def ensureTickerNames(app):
    if not hasattr(app, 'ticker_names') or app.ticker_names is None:
        app.ticker_names = {}
    for t in app.model.tickers:
        if t not in app.ticker_names:
            app.ticker_names[t] = getTickerName(t)

# Donut chart 
def drawDonut(cx, cy, weights, r=68, hole=36):
    """Draw a donut chart centred at (cx, cy).
    weights: list of floats that sum to 1.
    r      : outer radius
    hole   : inner hole radius (drawn as white circle on top)
    """
    if not weights:
        return
    total = sum(weights)
    if total == 0:
        return
    
    start = 90.0 # start from top (12 o'clock)

    for i, w in enumerate(weights):
        angle = 360.0 * w / total
        color = DONUT_COLORS[i % len(DONUT_COLORS)]
        
        drawArc(cx, cy, r * 2, r * 2, start, angle,
                fill=color, border='white', borderWidth=1.5)
        start += angle

    # Draw the inner hole
    drawCircle(cx, cy, hole, fill='white', border='white')

# Allocation table
def drawAllocationTable(tickers, weights, names, x, y, rowH=22):
    """Draw a compact 3-column table at top-left (x, y).
    Returns the y-coordinate just below the last row.
    """

    colW = [44, 110, 62]   # Ticker | Name | Alloc%
    headers = ["Ticker", "Name", "Alloc %"]
    tableW = sum(colW)

    # Header row background
    drawRect(x, y, tableW, rowH, fill=rgb(235, 240, 252), 
             border=rgb(200, 210, 225))
    cx_off = 0
    for h, w in zip(headers, colW):
        drawLabel(h, x + cx_off + w // 2, y + rowH // 2,
                  size=10, bold=True, fill=rgb(40, 60, 110))
        cx_off += w

    # Data rows: one rect per row (not per cell) to minimise shape count
    for i, (ticker, weight) in enumerate(zip(tickers, weights)):
        ry = y + rowH * (i + 1)

        rowFill = rgb(248, 250, 255) if i % 2 == 0 else rgb(252, 253, 255)
        drawRect(x, ry, tableW, rowH, fill=rowFill, border=rgb(215, 225, 240))

        swatchColor = DONUT_COLORS[i % len(DONUT_COLORS)]
        drawRect(x + 4, ry + 7, 8, 8, fill=swatchColor, border=None)
        drawLabel(ticker, x + 16, ry + rowH // 2,
                  size=10, bold=True, fill=rgb(25, 45, 90), align='left')

        # Company name
        name = names.get(ticker, ticker)
        if len(name) > 14:
            name = name[:13] + '…'

        drawLabel(name, x + colW[0] + 4, ry + rowH // 2,
                  size=9, fill=rgb(60, 70, 90), align='left')

        # Allocation percentage
        pct = f"{pythonRound(weight * 100, 1)}%"
        drawLabel(pct, x + colW[0] + colW[1] + colW[2] // 2, ry + rowH // 2,
                  size=10, bold=True, fill=rgb(30, 110, 60))

    return y + rowH * (len(tickers) + 1)

# Section panel (title bar, table, donut)
def drawPortfolioPanel(title, subtitle, tickers, weights, names,
                       panelY, panelH, appWidth=600):
    """Draw one half-page panel with title, table on the left, / 
    donut on right."""
    # Panel background
    drawRect(10, panelY, appWidth - 20, panelH,
             fill=rgb(248, 250, 254), border=rgb(210, 220, 235), borderWidth=1)

    # Title bar
    titleBarH = 28
    drawRect(10, panelY, appWidth - 20, titleBarH,
             fill=rgb(35, 70, 150), border=None)
    drawLabel(title, appWidth // 2, panelY + titleBarH // 2,
              size=12, bold=True, fill='white')

    contentY = panelY + titleBarH + 8

    drawAllocationTable(tickers, weights, names, 18, contentY + 2)

    # Right: donut
    donutCX = 460
    donutCY = panelY + panelH // 2 + 4
    drawDonut(donutCX, donutCY, weights, r=62, hole=30)

    # Subtitle (Sharpe / user label) below donut
    drawLabel(subtitle, donutCX, donutCY + 72, size=9,
              fill=rgb(100, 110, 130), italic=True)

    # Percentage labels around donut for each slice
    start = 90.0

    total_w = sum(weights)

    for i, w in enumerate(weights):
        angle = 360.0 * w / total_w
        mid_angle = start + angle / 2

        # Only label if slice is big enough to read without overlap
        if (w / total_w) >= 0.05:
            rad = math.radians(mid_angle)

            # Place label so that it clears the donut edge and the subtitle
            lx = donutCX + 86 * math.cos(rad)
            ly = donutCY - 86 * math.sin(rad)

            # Don't draw if label would overlap the subtitle 
            # area below the donut
            if ly < donutCY + 58:
                pct = f"{pythonRound(w * 100, 0)}%"
                drawLabel(pct, lx, ly, size=9, bold=True,
                        fill=DONUT_COLORS[i % len(DONUT_COLORS)])
        start += angle

# Main weights screen
def drawWeightsScreen(app):
    m = app.model

    # Page title
    drawScreenHeader("Portfolio Optimization Results")
    drawBackButton()

    # Ensure company names are fetched
    ensureTickerNames(app)
    names = app.ticker_names

    panelGap = 8
    panel1Y = 50
    panel1H = 220

    # Panel 1: user's portfolio
    drawPortfolioPanel(
        title="Provided Portfolio",
        subtitle="Your allocation",
        tickers=m.tickers,
        weights=m.weights,
        names=names,
        panelY=panel1Y,
        panelH=panel1H
    )

    # Panel 2: optimal portfolio 
    panel2Y = panel1Y + panel1H + panelGap

    if m.optimal_weights is not None:
        opt_weights = list(m.optimal_weights)

        # Compute Sharpe for subtitle
        if m.optimal_returns is not None and len(m.optimal_returns) > 0:
            ann_r = np.mean(m.optimal_returns) * 252
            ann_v = np.std(m.optimal_returns) * np.sqrt(252)
            sharpe = ann_r / ann_v if ann_v > 0 else 0
            subtitle = f"Sharpe ratio: {pythonRound(sharpe, 2)}"
        else:
            subtitle = "Max Sharpe allocation"

        drawPortfolioPanel(
            title="Maximum Sharpe Ratio Portfolio",
            subtitle=subtitle,
            tickers=m.tickers,
            weights=opt_weights,
            names=names,
            panelY=panel2Y,
            panelH=panel1H
        )
    else:
        drawRect(10, panel2Y, 580, panel1H,
                 fill=rgb(250, 245, 240), border=rgb(220, 200, 190))
        drawLabel("Optimal portfolio could not be computed.",
                  300, panel2Y + panel1H // 2, fill=rgb(160, 80, 60), size=12)

def computeMetrics(returns):
    # Annualized return and volatility
    ann_return = np.mean(returns) * 252
    vol = np.std(returns) * np.sqrt(252)
    sharpe = ann_return / vol if vol > 0 else 0
    cumulative = np.cumprod(1 + returns)
    return [cumulative[0], cumulative[-1], ann_return, vol, sharpe]

# Display information about the metric to a user
def drawInfoBox(lines, x, y, w, lineH=17, size=11, center=False, bullets=False):
    padding = 10
    boxH    = padding * 2 + len(lines) * lineH

    drawRect(x, y, w, boxH, fill=rgb(240, 244, 255),
             border=rgb(180, 198, 235), borderWidth=1)
    
    for i, (text, bold) in enumerate(lines):
        ty = y + padding + i * lineH + lineH // 2
        if center:
            label = ("• " + text) if bullets else text
            drawLabel(label, x + w // 2, ty,
                      size=size, bold=bold, fill=rgb(40, 55, 100), 
                      align='center')
        else:
            if bullets:
                drawCircle(x + padding + 4, ty, 2,
                           fill=rgb(70, 100, 170), border=None)
                drawLabel(text, x + padding + 14, ty,
                          size=size, bold=bold, fill=rgb(40, 55, 100), 
                          align='left')
            else:
                drawLabel(text, x + padding, ty,
                          size=size, bold=bold, fill=rgb(40, 55, 100), 
                          align='left')
    
    return y + boxH + 6

def drawSummaryScreen(app):
    m = app.model
    # Screen title
    drawScreenHeader("Performance Summary")
    drawBackButton()
    
    if m.portfolio_returns is None:
        drawLabel("Run Optimize first", 300, 200, fill='gray')
        return
    
    # Compute metrics for user's portfolio
    p = computeMetrics(m.portfolio_returns)
    rows = ["Start Value", "End Value", "Ann. Return", "Volatility", "Sharpe"]
    
    drawLabel("Metric", 150, 90, bold=True)
    drawLabel("Your Portfolio", 300, 90, bold=True)
    
    # Compute optimal metrics
    if m.optimal_returns is not None:
        o = computeMetrics(m.optimal_returns)
        drawLabel("Optimal Portfolio", 450, 90, bold=True)
    
    for i, label in enumerate(rows):
        y = 100 + i * 45
        rowFill = rgb(248, 250, 255) if i % 2 == 0 else rgb(252, 253, 255)
        drawRect(60, y - 15, 480, 38, fill=rowFill, border=rgb(210, 220, 238))
        drawLabel(label, 150, y + 4, size=11)
        drawLabel(f"{pythonRound(p[i], 3)}", 300, y + 4, size=11)
        if m.optimal_returns is not None:
            drawLabel(f"{pythonRound(o[i], 3)}", 450, y + 4, size=11)

    # Instruction box below table
    summary_lines = [
        (
            "Start Value:   baseline cumulative value at the start of the " 
            "period.", 
         False
        ),
        (
            "End Value:     final cumulative value of the investment over the "
        "period.", 
        False
        ),
        (
            "Ann. Return:   average yearly growth rate (CAGR).", 
            False
        ),
        (
            "Volatility:    risk measured as annualised standard deviation of "
        "returns.", 
        False
        ),
        (
            "Sharpe:        risk-adjusted return  (higher = better return per "
        "unit of risk).", 
        False
        ),
    ]
    drawInfoBox(summary_lines, x=10, y=365, w=580, size=12, center=True)

def _fmt_dollars(v):
    """Format a dollar value compactly: $1.23M, $456K, $1,234."""
    if v >= 1_000_000:
        return f"${pythonRound(v/1_000_000, 2)}M"
    if v >= 1_000:
        return f"${pythonRound(v/1_000, 1)}K"
    return f"${pythonRound(v, 0)}"

def _project_series(ann_return, ann_vol, start_value, years=5, 
                    steps_per_year=12):
    """
    Monte-Carlo-free deterministic projection:
      central path  = start * (1 + ann_return)^t
      upper band    = start * (1 + ann_return + ann_vol)^t
      lower band    = start * (1 + ann_return - ann_vol)^t   (floored at 0)
    Returns (central, upper, lower) as lists of length years*steps_per_year + 1.
    """
    # Total number of time steps
    n      = years * steps_per_year
    r_mo   = ann_return / steps_per_year
    v_mo   = ann_vol    / steps_per_year
    centre, upper, lower = [start_value], [start_value], [start_value]
    
    for _ in range(n):
        centre.append(centre[-1] * (1 + r_mo))
        # Upper assumes return + volatility
        upper.append( upper[-1]  * (1 + r_mo + v_mo))
        # Lower assumes return - volatility
        lower.append( max(lower[-1] * (1 + r_mo - v_mo), 0))
    return centre, upper, lower

def drawGrowthScreen(app):
    m = app.model
    investment = m.investment

    # title
    drawScreenHeader("Portfolio Growth & 5-Year Projection")
    drawBackButton()

    # Require computed results
    if not m.cumulative_returns:
        drawLabel("Run Optimize first", 300, 300, fill='gray')
        return

    # Chart geometry
    left, bottom = 58, 460
    hist_w  = 240          # historical portion width
    proj_w  = 170          # projection portion width
    height  = 340          # total chart height
    divX    = left + hist_w   # x where history ends / projection begins

    # Convert cumulative returns to actual dollar values
    def scale(cum):
        """Convert a cumulative-return series (starting at 1) to / 
        dollar values."""
        if cum is None or len(cum) < 2:
            return None
        base = cum[0]
        return [investment * v / base for v in cum]

    # Historical series
    user_cum = getattr(m, 'user_cumulative', None)
    hist_user = scale(user_cum if user_cum is not None else
                      (m.cumulative_returns if m.cumulative_returns else None))
    hist_opt  = scale(m.optimal_cumulative)
    hist_spy  = scale(m.benchmark_cumulative)

    if hist_user is None:
        drawLabel("Not enough data", 300, 300, fill='gray')
        return

    # Compute annualized return and volatility
    def ann_stats(returns_arr):
        if returns_arr is None or len(returns_arr) < 20:
            return None, None
        r = np.mean(returns_arr)  * 252
        v = np.std(returns_arr)   * np.sqrt(252)
        return float(r), float(v)

    # Benchmark returns derived from price series
    ar_user, av_user = ann_stats(m.portfolio_returns)
    ar_opt,  av_opt  = ann_stats(m.optimal_returns)
    ar_spy,  av_spy  = ann_stats(
        (m.benchmark_cumulative[1:] / m.benchmark_cumulative[:-1] - 1)
        if m.benchmark_cumulative is not None and 
                            len(m.benchmark_cumulative) > 1
        else None
    )

    # End values of historical series is a projection start values
    end_user = hist_user[-1]
    end_opt  = hist_opt[-1]  if hist_opt  else None
    end_spy  = hist_spy[-1]  if hist_spy  else None

    # Build projections
    proj_user_c, proj_user_u, proj_user_l = (
        _project_series(ar_user, av_user, end_user) if ar_user is not None
        else ([end_user], [end_user], [end_user])
    )
    proj_opt_c = proj_opt_u = proj_opt_l = None
    if end_opt is not None and ar_opt is not None:
        proj_opt_c, proj_opt_u, proj_opt_l = _project_series(
            ar_opt, 
            av_opt, 
            end_opt)

    proj_spy_c = proj_spy_u = proj_spy_l = None
    if end_spy is not None and ar_spy is not None:
        proj_spy_c, proj_spy_u, proj_spy_l = _project_series(
            ar_spy, 
            av_spy, 
            end_spy)

    # Global y scale (historical + projection together)
    all_vals = list(hist_user)
    if hist_opt:  all_vals += hist_opt
    if hist_spy:  all_vals += hist_spy
    all_vals += proj_user_c + proj_user_u
    if proj_opt_c: all_vals += proj_opt_c + proj_opt_u
    if proj_spy_c: all_vals += proj_spy_c + proj_spy_u

    y_max = max(all_vals)
    y_min = min(0, min(all_vals))
    y_rng = y_max - y_min if y_max != y_min else 1

    def to_px(val):
        return bottom - ((val - y_min) / y_rng) * height

    def to_x_hist(i, n):
        return left + (i / (n - 1)) * hist_w

    def to_x_proj(i, n):
        return divX + (i / (n - 1)) * proj_w

    # Draw axes
    drawLine(left, bottom, left + hist_w + proj_w, bottom, 
             fill=rgb(180,180,180))
    drawLine(left, bottom, left, bottom - height,  fill=rgb(180,180,180))

    # Y-axis tick labels (4 ticks)
    for tick in range(5):
        val = y_min + tick * y_rng / 4
        py  = to_px(val)
        drawLine(left - 4, py, left + hist_w + proj_w, py,
                 fill=rgb(230, 230, 230), lineWidth=0.5)
        drawLabel(_fmt_dollars(val), left - 6, py, size=8,
                  fill='gray', align='right')

    # Divider line between history and projection
    drawLine(divX, bottom - height - 4, divX, bottom + 14,
             fill=rgb(160, 160, 160), lineWidth=1, dashes=True)
    drawLabel("Today", divX, bottom + 22, size=8, fill='gray')
    drawLabel("← Historical", left + hist_w//2, bottom + 22, size=8, 
              fill=rgb(120,120,140))
    drawLabel("Projection →",  divX + proj_w//2, bottom + 22, size=8, 
              fill=rgb(120,120,140))
    
    c_user = rgb(70,  130, 210)
    c_opt  = rgb(60,  180, 110)
    c_spy  = rgb(220,  80,  90)

    # Helper: draw mountain/area chart for one historical series
    def draw_hist(series, color, lw=2):
        n = len(series)

        # Area fill: vertical lines from baseline to value (mountain effect)
        # Draw every 3rd column to keep shape count low
        for i in range(0, n, 3):
            xi = to_x_hist(i, n)
            yi = to_px(series[i])
            yb = to_px(y_min)

            # lighten the fill colour
            fc = (
                rgb(
                    min(255, color.red + 100),
                     min(255, color.green + 100),
                     min(255, color.blue + 100)
                     ) 
                     if hasattr(color, 'red') 
                     else 'lightGray'
            )
            
            drawLine(xi, yi, xi, yb, fill=fc, lineWidth=3, opacity=55)
        
        # Solid outline on top
        for i in range(n - 1):
            drawLine(to_x_hist(i, n),   to_px(series[i]),
                     to_x_hist(i+1, n), to_px(series[i+1]),
                     fill=color, lineWidth=lw)

    # Helper: draw projection band + centre line 
    def draw_proj(centre, upper, lower, color):
        if centre is None:
            return
        n = len(centre)

        # Band: light fill via overlapping lines
        bandColor = rgb(
            min(255, color.red   + 80),
            min(255, color.green + 80),
            min(255, color.blue  + 80)
        ) if hasattr(color, 'red') else 'lightGray'
        
        for i in range(n - 1):
             for s in range(7):
                frac = s / 6
                y_a  = to_px(upper[i]   * (1-frac) + lower[i]   * frac)
                y_b  = to_px(upper[i+1] * (1-frac) + lower[i+1] * frac)
                drawLine(to_x_proj(i, n), y_a, to_x_proj(i+1, n), y_b,
                         fill=bandColor, lineWidth=0.5, opacity=40)
        
        # Central line dashed
        for i in range(n - 1):
            if i % 5 < 3:
                drawLine(to_x_proj(i, n),   to_px(centre[i]),
                         to_x_proj(i+1, n), to_px(centre[i+1]),
                         fill=color, lineWidth=2)

    # Draw historical
    if hist_spy:   draw_hist(hist_spy,  c_spy,  lw=1.2)
    if hist_opt:   draw_hist(hist_opt,  c_opt,  lw=1.5)
    draw_hist(hist_user, c_user, lw=2)

    # Draw projections
    draw_proj(proj_spy_c, proj_spy_u, proj_spy_l, c_spy)
    draw_proj(proj_opt_c, proj_opt_u, proj_opt_l, c_opt)
    draw_proj(proj_user_c, proj_user_u, proj_user_l, c_user)

    # Legend: inside chart, bottom-right corner
    entries = [("Your Portfolio",   c_user),
               ("Optimal (Sharpe)", c_opt),
               ("SPY Benchmark",    c_spy)]
    lgX, lgY, lgW, lgRowH = left + 6, bottom - height + 54, 116, 18
    lgH = len(entries) * lgRowH + 8
    
    drawRect(lgX, lgY, lgW, lgH,
             fill=rgb(245, 247, 255), border=rgb(190, 200, 230), 
             borderWidth=0.8)
    
    for i, (lbl, col) in enumerate(entries):
        ly = lgY + 4 + i * lgRowH + lgRowH // 2
        drawRect(lgX + 6, ly - 5, 9, 9, fill=col, border=None)
        drawLabel(lbl, lgX + 18, ly, size=8, fill=rgb(50,60,90), align='left')

    # Projection summary box
    boxX, boxY, boxW, boxH = left, bottom - height - 58, hist_w + proj_w, 52
    drawRect(boxX, boxY, boxW, boxH, fill=rgb(240, 244, 255),
             border=rgb(180, 198, 235))
    
    drawLabel(f"5-Year Projection  (starting ${pythonRound(investment, 0):,})",
              boxX + boxW//2, boxY + 12, size=10, bold=True, 
              fill=rgb(30, 50, 110))

    col_x = [boxX + 90, boxX + 230, boxX + 370]
    hdrs  = ["Your Portfolio", "Optimal", "SPY"]
    cols  = [c_user, c_opt, c_spy]

    ends  = [proj_user_c[-1],
             proj_opt_c[-1] if proj_opt_c else None,
             proj_spy_c[-1] if proj_spy_c else None]
    
    for cx, hdr, col, val in zip(col_x, hdrs, cols, ends):
        drawLabel(hdr, cx, boxY + 26, size=9, bold=True, fill=col)
        
        if val is not None:
            drawLabel(_fmt_dollars(val), cx, boxY + 40, size=10, bold=True,
                      fill=rgb(25, 90, 40))
        else:
            drawLabel("N/A", cx, boxY + 40, size=9, fill='gray')

def drawAnnualScreen(app):
    m = app.model
    drawScreenHeader("Annual Returns (%)")
    drawBackButton()

    if m.returns_df is None or m.portfolio_returns is None:
        drawLabel("Run Optimize first", 300, 200, fill='gray')
        return
 
    # Build a weighted portfolio return series aligned to the backtest index
    user_cum = getattr(m, 'user_cumulative', None)
    if user_cum is None or len(user_cum) < 2:
        drawLabel("Not enough data", 300, 200, fill='gray')
        return
 
    # Use the aligned returns_df index (same as backtest) to get 
    # yearly boundaries
    idx = m.returns_df.index
    
    # Filter to user-selected date range
    start = pd.Timestamp(m.start_date)
    end   = pd.Timestamp(m.end_date) if m.end_date else idx[-1]
    mask  = (idx >= start) & (idx <= end)
    idx   = idx[mask]
    port_ret = m.portfolio_returns[:len(idx)]
 
    if len(port_ret) < 2:
        drawLabel("Not enough data in selected period", 300, 200, fill='gray')
        return
 
    # Compute annual % change in cumulative returns
    # Group daily returns by year, compound them: (prod(1+r) - 1) * 100
    years_in_range = sorted(set(idx.year))
    annual_pct = []

    for yr in years_in_range:
        yr_mask = idx.year == yr
        yr_rets  = port_ret[yr_mask[:len(port_ret)]]
        if len(yr_rets) == 0:
            continue

        compound = float(np.prod(1 + yr_rets) - 1) * 100   # as percentage
        annual_pct.append((yr, compound))
 
    if not annual_pct:
        drawLabel("No annual data available", 300, 200, fill='gray')
        return
 
    # Chart geometry 
    n      = len(annual_pct)
    left   = 60
    baseline = 360          # y-position of the zero line
    maxBarH  = 160          # max bar height in px (for most extreme value)
    totalW   = 480
    barW     = min(55, (totalW - (n - 1) * 10) // n)
    gap      = (totalW - n * barW) // max(n - 1, 1)
 
    max_abs = max(abs(v) for _, v in annual_pct) or 1
 
    # Axes
    drawLine(left, baseline, left + totalW, baseline, fill=rgb(150, 160, 190))
    drawLine(left, baseline - maxBarH - 20, left, baseline + maxBarH + 20,
             fill=rgb(150, 160, 190))
 
    # Y-axis labels
    for pct in [max_abs, max_abs / 2, 0, -max_abs / 2, -max_abs]:
        py = baseline - (pct / max_abs) * maxBarH
        
        drawLine(left - 4, py, left + totalW, py,
                 fill=rgb(215, 220, 235), lineWidth=0.5)
        
        drawLabel(f"{pythonRound(pct, 1)}%", left - 6, py,
                  size=8, fill=rgb(100,110,140), align='right')
 
    # Bars
    for i, (yr, pct) in enumerate(annual_pct):
        bx = left + i * (barW + gap)
        h  = (pct / max_abs) * maxBarH
 
        if pct >= 0:
            drawRect(bx, baseline - h, barW, h,
                     fill=rgb(60, 170, 90), border=rgb(30, 120, 55))
            drawLabel(f"+{pythonRound(pct, 1)}%",
                      bx + barW / 2, baseline - h - 10,
                      size=9, bold=True, fill=rgb(25, 100, 45))
        else:
            drawRect(bx, baseline, barW, -h,
                     fill=rgb(210, 60, 60), border=rgb(150, 25, 25))
            drawLabel(f"{pythonRound(pct, 1)}%",
                      bx + barW / 2, baseline - h + 14,
                      size=9, bold=True, fill=rgb(150, 25, 25))
        drawLabel(str(yr), bx + barW / 2, baseline + 14, size=10, bold=True,
                  fill=rgb(40, 50, 90))
    drawLabel(
        f"Portfolio annual return  |  {m.start_date[:4]} – "
        f"{(m.end_date or str(idx[-1].year))[:4]}",
        left + totalW / 2, 65, size=9, fill=rgb(100, 110, 140))
 
def getColor(value):
    clamped = max(-1, min(1, value))
    # t goes 0 (value=-1) to 1 (value=1)
    t = (clamped + 1) / 2
    # Light green (240,255,240) to dark green (20,110,40)
    r = int(240 - 220 * t)
    g = int(255 - 145 * t)
    b = int(240 - 200 * t)
    return rgb(r, g, b)

def drawHeatmap(app):
    corr = app.model.corr_matrix
    drawScreenHeader("Correlation Heatmap")
    drawBackButton()

    # Require computed correlation matrix
    if corr is None:
        drawLabel("Run Optimize first", 300, 200, fill='gray')
        return
    
    # Instructions box
    corr_lines = [
        (
            "Correlation matrix shows how returns of assets move "
            "relative to each other.", 
            False
        ),
        (
            "Value > 0: both assets move together. "
            "Value < 0: they move in opposite directions.", 
            False
        ),
        (
            "Assets with low or negative correlation are used for " 
            "diversification,", 
            False
        ),
        (
            "which reduces overall portfolio risk.", 
            False
        ),
    ]
    drawInfoBox(corr_lines, x=10, y=62, w=580, center=True, bullets=True)

    # Grid setup
    n = len(corr)
    cell = 50
    startX = 300 - n * cell // 2
    startY = 185
    
    # Draw heatmap cells
    for i in range(n):
        for j in range(n):
            color = getColor(corr[i][j])
            drawRect(startX + j * cell, startY + i * cell, cell, cell, 
                     fill=color)
            
            # Display correlation value inside each cell
            drawLabel(f"{pythonRound(corr[i][j], 2)}",
                      startX + j * cell + cell // 2,
                      startY + i * cell + cell // 2,
                      fill='white', size=10)
    
    # Axis labels (tickers)
    for i, ticker in enumerate(app.model.tickers):
        drawLabel(ticker, startX + i * cell + cell // 2, startY - 12,
                  size=10, bold=True, fill=rgb(30,45,100))
        drawLabel(ticker, startX - 12, startY + i * cell + cell // 2,
                  size=10, bold=True, align='right', fill=rgb(30,45,100))
    
    # Vertical gradient legend
    lgX   = startX + n * cell + 20
    lgY   = startY
    lgH   = n * cell
    barW  = 14
    
    # Draw gradient bar
    for s in range(lgH):
        t   = 1 - s / lgH          # top = 1(dark), bottom = 0(light)
        val = t * 2 - 1              # maps to [-1, 1]
        col = getColor(val)
        
        drawRect(lgX, lgY + s, barW, 1, fill=col, border=None)
    
    drawRect(lgX, lgY, barW, lgH, fill=None, border=rgb(140,165,140))
    
    # Tick labels
    for val, label in [(1, "1.0"), (0, "0.0"), (-1, "-1.0")]:
        ty = lgY + (1 - (val + 1) / 2) * lgH
        drawLine(lgX + barW, ty, lgX + barW + 4, ty, fill=rgb(80,110,80))
        drawLabel(label, lgX + barW + 6, ty, size=8, fill=rgb(40,70,40), 
                  align='left')
    
    drawLabel("Corr.", lgX + barW // 2, lgY - 10, size=8, fill=rgb(40,70,40))

def drawEfficientFrontier(app):
    m = app.model

    # Title bar
    drawScreenHeader("Efficient Frontier")
    drawBackButton()

    # Instruction box: offset from title bar, extra gap before chart
    ef_lines = [
        (
            "To build a portfolio we need: expected return E(r), " 
            "standard deviation σ, and correlation corr(X,Y).", 
            False
        ),
        (
            "The Efficient Frontier generates thousands of portfolios " 
            "with different weight combinations.", 
            False
        ),
        (
            "Each portfolio's return and risk (volatility) is computed " 
            "and plotted as a dot.", 
            False
        ),
        (
            "The yellow line marks the best portfolios at every " 
            "risk level.", 
            False
        ),
        (
            "The optimal portfolio has the highest return per unit " 
            "of risk (Max Sharpe Ratio).", 
            False
        ),
    ]
    drawInfoBox(ef_lines, x=10, y=54, w=580, center=True, bullets=True)
  
    # Require computed frontier
    if not m.efficient_frontier:
        drawLabel("Run Optimize first", 300, 420, fill='gray')
        return
    
    # Chart geometry
    left, bottom = 72, 510
    width, height = 445, 330

    ef_vols = [x[0] for x in m.efficient_frontier]
    ef_rets = [x[1] for x in m.efficient_frontier]
    all_pts = m.all_portfolios

    # Axis bounds, 10% pad from each side
    all_vols = [v for v, _ in all_pts] + ef_vols
    all_rets = [r for r, _ in all_pts] + ef_rets
    v_min = min(all_vols); v_max = max(all_vols)
    r_min = min(all_rets); r_max = max(all_rets)
    v_rng = (v_max - v_min) or 0.01
    r_rng = (r_max - r_min) or 0.01

    # Coordinate transforms
    def px(vol):
        # Clamp to [left, left+width] so no dot escapes the plot area
        raw = left + ((vol - v_min) / v_rng) * width
        return max(left, min(left + width, raw))
    def py(ret):
        # Clamp to [bottom-height, bottom]
        raw = bottom - ((ret - r_min) / r_rng) * height
        return max(bottom - height, min(bottom, raw))
    
    # Axes
    drawLine(left, bottom, left + width, bottom, fill=rgb(120, 130, 160), 
             lineWidth=1.5)
    drawLine(left, bottom, left, bottom - height, fill=rgb(120, 130, 160), 
             lineWidth=1.5)

    # X ticks (volatility)
    for i in range(6):
        v = v_min + i * v_rng / 5
        xp = px(v)
        # Snap to nearest 20px column so it aligns with background grid
        xp_snap = pythonRound(xp / 20) * 20
        drawLine(xp_snap, bottom, xp_snap, bottom - height,
                 fill=rgb(215, 220, 235), lineWidth=1.2)
        drawLine(xp_snap, bottom, xp_snap, bottom + 4, fill=rgb(120,130,160))
        drawLabel(f"{pythonRound(v, 2)}", xp_snap, bottom + 13,
                  size=8, fill=rgb(80,90,120))

    # Y ticks (return)
    for i in range(6):
        r = r_min + i * r_rng / 5
        yp = py(r)

        # Snap to nearest 20px row
        yp_snap = pythonRound(yp / 20) * 20
        drawLine(left, yp_snap, left + width, yp_snap,
                 fill=rgb(215, 220, 235), lineWidth=1.2)
        drawLine(left - 4, yp_snap, left, yp_snap, fill=rgb(120,130,160))
        drawLabel(f"{pythonRound(r, 2)}", left - 6, yp_snap,
                  size=8, fill=rgb(80,90,120), align='right')
    
    # Axis labels
    drawLabel("Volatility (Std. Deviation)",
              left + width / 2, bottom + 26, size=10, fill=rgb(80,90,120))
    drawLabel("Expected Return", left - 44, bottom - height / 2,
              size=10, fill=rgb(80,90,120), rotateAngle=-90)
    
    # Scatter cloud with all simulated portfolios
    # Color depends on Sharpe ratio: low = blue, high = orange
    min_sharpe = min(r/v for v, r in all_pts if v > 0)
    max_sharpe = max(r/v for v, r in all_pts if v > 0)
    sharpe_rng = max_sharpe - min_sharpe or 1

    for vol, ret in all_pts:
        sharpe_norm = ((ret/vol) - min_sharpe) / sharpe_rng if vol > 0 else 0

        # Gradient where blue is low Sharpe and orange is high Sharpe
        if sharpe_norm < 0.5:
            t = sharpe_norm * 2
            col = rgb(int(80 + 120*t), int(120 + 80*t), int(200 - 100*t))
        else:
            t = (sharpe_norm - 0.5) * 2
            col = rgb(int(200 + 30*t), int(200 - 120*t), int(100 - 80*t))
        drawCircle(px(vol), py(ret), 2, fill=col, opacity=60, border=None)

    # Create efficient frontier line
    for i in range(len(ef_vols) - 1):
        drawLine(px(ef_vols[i]), py(ef_rets[i]),
                 px(ef_vols[i+1]), py(ef_rets[i+1]),
                 fill=rgb(255, 210, 0), lineWidth=2.5)
    
    # User portfolio
    u_pt = getattr(m, 'user_frontier_point', None)
    if u_pt:
        ux, uy = px(u_pt[0]), py(u_pt[1])
        drawCircle(ux, uy, 9, fill=rgb(70, 130, 210), border='white', 
                   borderWidth=2)

    # Optimal (max Sharpe) portfolio
    o_pt = getattr(m, 'optimal_frontier_point', None)
    if o_pt:
        ox, oy = px(o_pt[0]), py(o_pt[1])
        # White halo so star is visible against any background colour
        drawCircle(ox, oy, 13, fill='white', border=None, opacity=80)
        drawStar(ox, oy, 13, 5,
                 fill=rgb(255, 210, 0), border=rgb(160, 90, 0), 
                 borderWidth=1.5)

    # Legend
    lgX, lgY = left + 8, bottom - height + 8
    drawRect(lgX, lgY, 132, 64,
             fill=rgb(245, 247, 255), border=rgb(180, 198, 235), 
             borderWidth=0.8)
    drawLine(lgX + 8, lgY + 12, lgX + 22, lgY + 12,
             fill=rgb(255, 210, 0), lineWidth=2.5)
    drawLabel("Efficient Frontier", lgX + 26, lgY + 12,
              size=8, fill=rgb(60,70,100), align='left')
    drawCircle(lgX + 15, lgY + 28, 5,
               fill=rgb(70, 130, 210), border='white')
    drawLabel("Your Portfolio", lgX + 26, lgY + 28,
              size=8, fill=rgb(60,70,100), align='left')
    drawStar(lgX + 15, lgY + 46, 7, 5, fill=rgb(255, 210, 0))
    drawLabel("Max Sharpe ★",  lgX + 26, lgY + 46,
              size=8, fill=rgb(60,70,100), align='left')

def drawAnnualTable(app):
    # Get returns data from a model
    df = app.model.returns_df
    drawScreenHeader("Annual Returns Table")
    drawBackButton()

    if df is None:
        drawLabel("Run Optimize first", 300, 200, fill='gray')
        return
    
    df = df.copy()
    df['year'] = df.index.year
    
    # Aggregate returns by year
    annual = df.groupby('year').sum()

    drawLabel("Year", 150, 62, bold=True, size=11, fill='white')
    drawLabel("Avg Return", 350, 62, bold=True, size=11, fill='white')
    
    # Draw each year's row
    for i, year in enumerate(annual.index):
        y = 82 + i * 32
        rowFill = rgb(248, 250, 255) if i % 2 == 0 else rgb(252, 253, 255)
        drawRect(80, y - 12, 440, 28, fill=rowFill, border=rgb(210, 220, 238))
        drawLabel(str(year), 150, y + 2)
        val = annual.iloc[i].mean()
        color = rgb(30, 120, 50) if val >= 0 else rgb(180, 35, 35)
        drawLabel(f"{pythonRound(val * 100, 2)}%", 350, y + 2, fill=color)

# Mouse & keyboard
def onMousePress(app, x, y):
    # Always check boxes first: set active and return
    for box in [app.start_year_box, app.end_year_box, app.amount_box, 
                app.csv_path_box]:
        if inside(box, x, y):
            app.active_box = box
            return

    for row in app.asset_rows:
        for key in row:
            if inside(row[key], x, y):
                app.active_box = row[key]
                return

    # Input screen specific buttons
    if app.screen == 'input':
        # Optimize button
        if 350 <= x <= 500 and 505 <= y <= 543:
            app.active_box = None
            if processInput(app):
                app.screen = 'weights'
            return

        # CSV upload button: reads file from the path typed in csv_path_box
        if 355 <= x <= 535 and 172 <= y <= 198:
            path = app.csv_path_box['text'].strip()
            if not path:
                app.status_message = (
                    'Enter a file path in the box below the button'
                )
                return

            tickers, result = load_portfolio_from_csv(path)

            if tickers is None:
                app.status_message = f'Error: {result}'
                app.csv_loaded = False
                return

            # Clear rows then populate from CSV
            for row in app.asset_rows:
                row['ticker']['text'] = ''
                row['weight']['text'] = ''

            weights_loaded = result
            for i, (t, w) in enumerate(zip(tickers, weights_loaded)):
                app.asset_rows[i]['ticker']['text'] = t
                app.asset_rows[i]['weight']['text'] = str(
                    pythonRound(w * 100, 1)
                )
            app.status_message = f'Loaded {len(tickers)} tickers from CSV'
            app.csv_loaded = True
            return

        # Use Historical toggle
        if 160 <= x <= 212 and 185 <= y <= 207:
            app.use_historical = not app.use_historical
            return

        # Benchmark toggle
        if 160 <= x <= 212 and 215 <= y <= 237:
            app.use_benchmark = not app.use_benchmark
            return

    # Navigation bar (only on result screens)
    if app.screen != 'input':
        for i in range(len(app.result_screens)):
            x1 = 20 + i * 82
            x2 = x1 + 74
            if x1 <= x <= x2 and 560 <= y <= 590:
                app.screen = app.result_screens[i]
                app.active_box = None
                return

        # Back to input button
        if 12 <= x <= 84 and 10 <= y <= 36:
            app.screen = 'input'
            app.active_box = None
            return

    # Clicked on empty space: deactivate box
    app.active_box = None

def onKeyPress(app, key):
    if app.active_box is None:
        return

    if key == 'backspace':
        if app.active_box['text']:
            app.active_box['text'] = app.active_box['text'][:-1]
    elif key == 'escape':
        app.active_box = None
    elif key == 'enter':
        if processInput(app):
            app.screen = 'weights'
        app.active_box = None
    elif app.active_box is app.csv_path_box:
        # CSV path: accept all printable characters exactly as typed
        if key == 'space':
            app.active_box['text'] += ' '
        elif len(key) == 1:
            app.active_box['text'] += key   # no uppercase, no filtering
    elif key == 'space':
        pass  # no spaces in tickers/numbers
    elif len(key) == 1:
        if key.isalpha():
            app.active_box['text'] += key.upper()
        elif key in '0123456789.-':
            app.active_box['text'] += key

# Draw 

def redrawAll(app):
    drawBackground()   # graph-paper texture on every screen

    if app.screen == 'input':
        drawInputScreen(app)
    else:
        if app.screen == 'weights':
            drawWeightsScreen(app)
        elif app.screen == 'summary':
            drawSummaryScreen(app)
        elif app.screen == 'growth':
            drawGrowthScreen(app)
        elif app.screen == 'annual':
            drawAnnualScreen(app)
        elif app.screen == 'correlation':
            drawHeatmap(app)
        elif app.screen == 'frontier':
            drawEfficientFrontier(app)
        elif app.screen == 'table':
            drawAnnualTable(app)

        drawNavigation(app)

runApp(width=600, height=600)