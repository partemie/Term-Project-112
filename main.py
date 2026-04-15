# Multi-asset portfolio optimization simulator

from cmu_graphics import *
import yfinance as yf
import pandas as pd
import numpy as np

def load_market_data(tickers, start='2020-01-01', end=None):
    data = yf.download(tickers, start=start, end=end)['Close']
    if isinstance(data, pd.Series):
        data = data.to_frame()
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

        lines = [l.strip() for l in raw.splitlines() if l.strip()]
        if len(lines) < 2:
            return None, "CSV is empty or has no data rows"

        # Parse header — find Ticker and Weight column indices
        header = [h.strip().lower() for h in lines[0].split(',')]
        if 'ticker' not in header or 'weight' not in header:
            return None, "CSV must have 'Ticker' and 'Weight' columns"

        ti = header.index('ticker')
        wi = header.index('weight')

        tickers, weights = [], []
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
                continue  # skip malformed rows

        if len(tickers) == 0:
            return None, "No valid rows found in CSV"
        if len(tickers) > 5:
            return None, f"CSV has {len(tickers)} tickers — maximum is 5"

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

    def calculate_portfolio(self):
        if len(self.tickers) == 0:
            return
        self.price_data = load_market_data(self.tickers, start=self.start_date, end=self.end_date)
        self.returns_df = np.log(self.price_data / self.price_data.shift(1)).dropna()
        self.corr_matrix = self.returns_df.corr().values
        weights_array = np.array(self.weights)
        if len(weights_array) != self.returns_df.shape[1]:
            return
        portfolio_returns = self.returns_df.values @ weights_array
        if len(portfolio_returns) == 0:
            return
        self.expected_return = np.mean(portfolio_returns)
        self.variance = np.var(portfolio_returns)
        self.volatility = np.sqrt(self.variance)
        value = self.investment
        self.cumulative_returns = []
        for r in portfolio_returns:
            value *= (1 + r)
            self.cumulative_returns.append(value)
        self.generate_efficient_frontier()
        self.run_backtest()

    def generate_efficient_frontier(self, num_portfolios=8000):
        if self.returns_df is None:
            return
        returns = self.returns_df.values
        mu = np.mean(returns, axis=0) * 252
        n = len(mu)
        if returns.shape[1] == 1:
            sigma = np.array([[np.var(returns[:, 0]) * 252]])
        else:
            sigma = np.cov(returns.T) * 252
        # Adaptive per-asset cap: only enforce when n>=4, and generous enough
        max_weight = max(1.0 / n + 0.15, 0.6)
        results = []
        for _ in range(num_portfolios):
            weights = np.random.random(n)
            weights = weights / np.sum(weights)
            if n >= 4 and np.any(weights > max_weight):
                continue
            port_return = np.dot(mu, weights)
            port_vol = np.sqrt(np.dot(weights.T, np.dot(sigma, weights)))
            if port_vol <= 0:
                continue
            results.append((port_vol, port_return, weights))
        if not results:
            self.efficient_frontier = []
            self.optimal_weights = None
            return
        results = sorted(results, key=lambda x: x[0])
        efficient = []
        max_return = -np.inf
        for vol, ret, w in results:
            if ret > max_return:
                efficient.append((vol, ret, w))
                max_return = ret
        self.efficient_frontier = efficient
        best = max(results, key=lambda x: x[1] / x[0])
        self.optimal_weights = best[2]

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
        benchmark = yf.download("SPY", start=self.start_date, end=self.end_date)['Close']
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

    # Compute performance of the portfolio with optimized weights if not None
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
    return {
        'ticker': createBox('', 100, y, 80, 30),
        'weight': createBox('', 200, y, 80, 30),
        'min': createBox('0', 300, y, 60, 30),
        'max': createBox('100', 380, y, 60, 30)
    }

def createBox(text, x, y, w, h):
    return {'text': text, 'x': x, 'y': y, 'w': w, 'h': h}

def onAppStart(app):
    app.setMaxShapeCount(10000)
    app.model = PortfolioAppModel()

    # FIX: screens list must NOT include 'input' — nav tabs map to result screens only
    app.screen = 'input'
    app.result_screens = ['weights', 'summary', 'growth',
                          'annual', 'correlation',
                          'frontier', 'table']

    app.graph_type = 'line'
    app.active_box = None
    app.status_message = ''  # for user feedback

    app.start_year_box = createBox('2020-01-01', 190, 120, 115, 30)
    app.end_year_box   = createBox('2025-01-01', 190, 160, 115, 30)
    app.amount_box = createBox('10000', 420, 120, 100, 30)

    app.use_historical = True
    app.use_benchmark = True
    app.ticker_names = {}   # populated lazily after optimise runs
    app.csv_loaded = False
    app.csv_path_box = createBox('', 380, 215, 200, 26)

    app.asset_rows = [createRow(300 + i * 40) for i in range(5)]

def inside(box, x, y):
    return (box['x'] <= x <= box['x'] + box['w'] and
            box['y'] <= y <= box['y'] + box['h'])

def processInput(app):
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
    if len(tickers) == 0:
        app.status_message = 'Enter at least one ticker and weight'
        return False
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
    # FIX: nav labels now correctly map to result_screens (no 'input' in list)
    labels = ["Weights", "Summary", "Growth",
              "Annual", "Correlation",
              "Frontier", "Table"]
    for i, label in enumerate(labels):
        x = 20 + i * 82
        isActive = app.screen == app.result_screens[i]
        fillColor = 'steelBlue' if isActive else 'lightGray'
        textColor = 'white' if isActive else 'black'
        drawRect(x, 560, 74, 30, fill=fillColor, border='gray')
        drawLabel(label, x + 37, 575, fill=textColor, size=11)

# Input screen

def drawBox(box, isActive=False):
    # FIX: always pass isActive correctly; active box gets red border
    borderColor = 'red' if isActive else 'black'
    borderWidth = 2 if isActive else 1
    drawRect(box['x'], box['y'], box['w'], box['h'],
             border=borderColor, borderWidth=borderWidth, fill='white')
    drawLabel(box['text'], box['x'] + box['w'] / 2, box['y'] + box['h'] / 2, size=12)

def sumSafeWeights(app):
    total = 0
    for row in app.asset_rows:
        try:
            total += float(row['weight']['text'])
        except:
            pass
    return total

def drawInputScreen(app):
    drawLabel("Portfolio Optimization Simulator", 300, 30, size=18, bold=True)

    # Left panel — dates
    drawLabel("Date Range", 150, 80, bold=True)
    drawLabel("Start Date:", 100, 128, size=11)
    drawLabel("(YYYY or YYYY-MM-DD)", 100, 141, size=9, fill='gray')
    drawBox(app.start_year_box, app.active_box is app.start_year_box)
    drawLabel("End Date:", 100, 168, size=11)
    drawLabel("(YYYY or YYYY-MM-DD)", 100, 181, size=9, fill='gray')
    drawBox(app.end_year_box, app.active_box is app.end_year_box)

    # Toggles
    drawLabel("Use Historical", 100, 207)
    drawRect(200, 193, 60, 28, fill='lightGreen' if app.use_historical else 'lightGray', border='gray')
    drawLabel("Yes" if app.use_historical else "No", 230, 207, size=12)

    drawLabel("Benchmark (SPY)", 100, 247)
    drawRect(200, 233, 60, 28, fill='lightGreen' if app.use_benchmark else 'lightGray', border='gray')
    drawLabel("Yes" if app.use_benchmark else "No", 230, 247, size=12)

    # Right panel — investment amount
    drawLabel("Investment", 450, 80, bold=True)
    drawLabel("Amount ($):", 380, 128)
    # FIX: drawBox called once with correct isActive check
    drawBox(app.amount_box, app.active_box is app.amount_box)

    # CSV upload section
    csvLoaded = getattr(app, 'csv_loaded', False)
    btnFill   = rgb(180, 230, 180) if csvLoaded else rgb(190, 220, 245)
    btnBorder = rgb(80, 160, 80)   if csvLoaded else 'gray'
    drawRect(380, 163, 140, 28, fill=btnFill, border=btnBorder)
    btnLabel = "CSV Loaded" if csvLoaded else "Upload CSV"
    drawLabel(btnLabel, 450, 177, size=11, bold=csvLoaded,
              fill=rgb(30, 100, 30) if csvLoaded else rgb(30, 60, 120))
    # Path input box
    drawLabel("CSV path:", 380, 209, size=9, fill='gray', align='left')
    drawBox(app.csv_path_box, app.active_box is app.csv_path_box)
    drawLabel("e.g. /Users/name/portfolio.csv", 480, 248, size=8, fill=rgb(180,180,180))

    # Asset table
    drawLabel("Portfolio Assets", 250, 268, bold=True)
    drawLabel("Ticker", 140, 286, size=11, bold=True)
    drawLabel("Weight %", 240, 286, size=11, bold=True)
    drawLabel("Min %", 330, 286, size=11, bold=True)
    drawLabel("Max %", 410, 286, size=11, bold=True)

    for row in app.asset_rows:
        for key in row:
            drawBox(row[key], app.active_box is row[key])

    # Totals and button
    total = sumSafeWeights(app)
    color = 'green' if abs(total - 100) < 0.1 else 'red'
    drawLabel(f"Total: {pythonRound(total, 1)}%  (should sum to 100)", 250, 522, fill=color, size=12)

    # Status message
    if app.status_message:
        drawLabel(app.status_message, 300, 542, fill='darkBlue', size=11)

    # Optimize button
    drawRect(350, 505, 150, 38, fill='lightGreen', border='darkGreen')
    drawLabel("Optimize Portfolio", 425, 524, bold=True)

# Result screens 

import math

# Palette — 5 distinct colours used consistently across both donuts
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
        info = yf.Ticker(ticker).info
        return info.get('shortName') or info.get('longName') or ticker
    except:
        return ticker

def ensureTickerNames(app):
    """Populate app.ticker_names once after optimisation runs."""
    if not hasattr(app, 'ticker_names') or app.ticker_names is None:
        app.ticker_names = {}
    m = app.model
    for t in m.tickers:
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
    start = 90.0          # start from top (12 o'clock)
    for i, w in enumerate(weights):
        angle = 360.0 * w / total
        color = DONUT_COLORS[i % len(DONUT_COLORS)]
        drawArc(cx, cy, r * 2, r * 2, start, angle,
                fill=color, border='white', borderWidth=1.5)
        start += angle
    # punch the hole
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
    drawRect(x, y, tableW, rowH, fill=rgb(240, 243, 248), border=rgb(200, 210, 225))
    cx_off = 0
    for h, w in zip(headers, colW):
        drawLabel(h, x + cx_off + w // 2, y + rowH // 2,
                  size=10, bold=True, fill=rgb(60, 80, 110))
        cx_off += w

    # Data rows — one rect per row (not per cell) to minimise shape count
    for i, (ticker, weight) in enumerate(zip(tickers, weights)):
        ry = y + rowH * (i + 1)
        rowFill = rgb(252, 253, 255) if i % 2 == 0 else 'white'
        drawRect(x, ry, tableW, rowH, fill=rowFill, border=rgb(220, 228, 238))

        swatchColor = DONUT_COLORS[i % len(DONUT_COLORS)]
        drawRect(x + 4, ry + 7, 8, 8, fill=swatchColor, border=None)
        drawLabel(ticker, x + 16, ry + rowH // 2,
                  size=10, bold=True, fill=rgb(30, 50, 90), align='left')

        name = names.get(ticker, ticker)
        if len(name) > 14:
            name = name[:13] + '…'
        drawLabel(name, x + colW[0] + 4, ry + rowH // 2,
                  size=9, fill=rgb(60, 70, 90), align='left')

        pct = f"{pythonRound(weight * 100, 1)}%"
        drawLabel(pct, x + colW[0] + colW[1] + colW[2] // 2, ry + rowH // 2,
                  size=10, bold=True, fill=rgb(40, 100, 60))

    return y + rowH * (len(tickers) + 1)

# Section panel (title bar + table + donut)
def drawPortfolioPanel(title, subtitle, tickers, weights, names,
                       panelY, panelH, appWidth=600):
    """Draw one half-page panel with title, table on the left, donut on right."""
    # Panel background
    drawRect(10, panelY, appWidth - 20, panelH,
             fill=rgb(248, 250, 254), border=rgb(210, 220, 235), borderWidth=1)

    # Title bar
    titleBarH = 28
    drawRect(10, panelY, appWidth - 20, titleBarH,
             fill=rgb(44, 82, 160), border=None)
    drawLabel(title, appWidth // 2, panelY + titleBarH // 2,
              size=12, bold=True, fill='white')

    contentY = panelY + titleBarH + 8

    # Left: table
    tableX = 18
    tableY = contentY + 2
    drawAllocationTable(tickers, weights, names, tableX, tableY)

    # Right: donut
    donutCX = 460
    donutCY = panelY + panelH // 2 + 4
    drawDonut(donutCX, donutCY, weights, r=62, hole=30)

    # Subtitle (Sharpe / user label) below donut
    drawLabel(subtitle, donutCX, donutCY + 72, size=9,
              fill=rgb(100, 110, 130), italic=True)

    # Percentage labels around donut for each slice
    start = 90.0
    for i, w in enumerate(weights):
        angle = 360.0 * w / sum(weights)
        mid_angle = start + angle / 2
        rad = math.radians(mid_angle)
        lx = donutCX + 80 * math.cos(rad)
        ly = donutCY - 80 * math.sin(rad)
        pct = f"{pythonRound(w * 100, 0)}%"
        drawLabel(pct, lx, ly, size=9, bold=True,
                  fill=DONUT_COLORS[i % len(DONUT_COLORS)])
        start += angle

# Main weights screen
def drawWeightsScreen(app):
    m = app.model

    # Page title
    drawRect(0, 0, 600, 46, fill=rgb(25, 55, 120), border=None)
    drawLabel("Portfolio Optimization Results", 300, 23,
              size=15, bold=True, fill='white')

    # Ensure company names are fetched
    ensureTickerNames(app)
    names = app.ticker_names

    panelGap = 8
    panel1Y = 50
    panel1H = 220

    # ── Panel 1 : user's portfolio ──
    drawPortfolioPanel(
        title="Provided Portfolio",
        subtitle="Your allocation",
        tickers=m.tickers,
        weights=m.weights,
        names=names,
        panelY=panel1Y,
        panelH=panel1H
    )

    # ── Panel 2 : optimal portfolio ──
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
    ann_return = np.mean(returns) * 252
    vol = np.std(returns) * np.sqrt(252)
    sharpe = ann_return / vol if vol > 0 else 0
    cumulative = np.cumprod(1 + returns)
    return [cumulative[0], cumulative[-1], ann_return, vol, sharpe]

def drawSummaryScreen(app):
    m = app.model
    drawLabel("Performance Summary", 300, 30, bold=True, size=16)
    if m.portfolio_returns is None:
        drawLabel("Run Optimize first", 300, 200, fill='gray')
        return
    p = computeMetrics(m.portfolio_returns)
    rows = ["Start Value", "End Value", "Ann. Return", "Volatility", "Sharpe"]
    drawLabel("Metric", 150, 90, bold=True)
    drawLabel("Your Portfolio", 300, 90, bold=True)
    if m.optimal_returns is not None:
        o = computeMetrics(m.optimal_returns)
        drawLabel("Optimal Portfolio", 450, 90, bold=True)
    for i, label in enumerate(rows):
        y = 130 + i * 45
        drawRect(60, y - 15, 480, 38, fill='snow', border='lightGray')
        drawLabel(label, 150, y + 4)
        drawLabel(f"{pythonRound(p[i], 3)}", 300, y + 4)
        if m.optimal_returns is not None:
            drawLabel(f"{pythonRound(o[i], 3)}", 450, y + 4)

def drawMultiLine(data, labels, colors, left=60, bottom=450,
                  width=460, height=220, dashed=False, lineWidth=1.5):
    """Generic multi-series line chart. Shared y-scale, independent x-scale."""
    valid = [(d, c, lbl) for d, c, lbl in zip(data, colors, labels)
             if d is not None and len(d) > 1]
    if not valid:
        return

    max_val = max(max(d) for d, _, _ in valid)
    min_val = min(min(d) for d, _, _ in valid)
    rng     = max_val - min_val if max_val != min_val else 1

    for series, color, _ in valid:
        n     = len(series)
        stepX = width / (n - 1)
        for i in range(n - 1):
            x1 = left + i       * stepX
            y1 = bottom - ((series[i]   - min_val) / rng) * height
            x2 = left + (i + 1) * stepX
            y2 = bottom - ((series[i+1] - min_val) / rng) * height
            if dashed and i % 6 < 3:       # crude dashed effect
                continue
            drawLine(x1, y1, x2, y2, fill=color, lineWidth=lineWidth)

    return min_val, max_val          # caller may need scale info


def _fmt_dollars(v):
    """Format a dollar value compactly: $1.23M, $456K, $1,234."""
    if v >= 1_000_000:
        return f"${pythonRound(v/1_000_000, 2)}M"
    if v >= 1_000:
        return f"${pythonRound(v/1_000, 1)}K"
    return f"${pythonRound(v, 0)}"


def _project_series(ann_return, ann_vol, start_value, years=5, steps_per_year=12):
    """
    Monte-Carlo-free deterministic projection:
      central path  = start * (1 + ann_return)^t
      upper band    = start * (1 + ann_return + ann_vol)^t
      lower band    = start * (1 + ann_return - ann_vol)^t   (floored at 0)
    Returns (central, upper, lower) as lists of length years*steps_per_year + 1.
    """
    n      = years * steps_per_year
    r_mo   = ann_return / steps_per_year
    v_mo   = ann_vol    / steps_per_year
    centre, upper, lower = [start_value], [start_value], [start_value]
    for _ in range(n):
        centre.append(centre[-1] * (1 + r_mo))
        upper.append( upper[-1]  * (1 + r_mo + v_mo))
        lower.append( max(lower[-1] * (1 + r_mo - v_mo), 0))
    return centre, upper, lower


def drawGrowthScreen(app):
    m = app.model
    investment = m.investment

    # title
    drawRect(0, 0, 600, 46, fill=rgb(25, 55, 120), border=None)
    drawLabel("Portfolio Growth & 5-Year Projection", 300, 23,
              size=14, bold=True, fill='white')

    if not m.cumulative_returns:
        drawLabel("Run Optimize first", 300, 300, fill='gray')
        return

    # chart geometry
    left, bottom = 58, 460
    hist_w  = 240          # historical portion width
    proj_w  = 170          # projection portion width
    height  = 340          # total chart height
    divX    = left + hist_w   # x where history ends / projection begins

    # build historical series (in $)
    def scale(cum):
        """Convert a cumulative-return series (starting at 1) to dollar values."""
        if cum is None or len(cum) < 2:
            return None
        base = cum[0]
        return [investment * v / base for v in cum]

    user_cum = getattr(m, 'user_cumulative', None)
    hist_user = scale(user_cum if user_cum is not None else
                      (m.cumulative_returns if m.cumulative_returns else None))
    hist_opt  = scale(m.optimal_cumulative)
    hist_spy  = scale(m.benchmark_cumulative)

    if hist_user is None:
        drawLabel("Not enough data", 300, 300, fill='gray')
        return

    # compute annualised stats for projection
    def ann_stats(returns_arr):
        if returns_arr is None or len(returns_arr) < 20:
            return None, None
        r = np.mean(returns_arr)  * 252
        v = np.std(returns_arr)   * np.sqrt(252)
        return float(r), float(v)

    ar_user, av_user = ann_stats(m.portfolio_returns)
    ar_opt,  av_opt  = ann_stats(m.optimal_returns)
    ar_spy,  av_spy  = ann_stats(
        (m.benchmark_cumulative[1:] / m.benchmark_cumulative[:-1] - 1)
        if m.benchmark_cumulative is not None and len(m.benchmark_cumulative) > 1
        else None
    )

    # End values of historical series → projection start values
    end_user = hist_user[-1]
    end_opt  = hist_opt[-1]  if hist_opt  else None
    end_spy  = hist_spy[-1]  if hist_spy  else None

    # build projections
    proj_user_c, proj_user_u, proj_user_l = (
        _project_series(ar_user, av_user, end_user) if ar_user is not None
        else ([end_user], [end_user], [end_user])
    )
    proj_opt_c = proj_opt_u = proj_opt_l = None
    if end_opt is not None and ar_opt is not None:
        proj_opt_c, proj_opt_u, proj_opt_l = _project_series(ar_opt, av_opt, end_opt)

    proj_spy_c = proj_spy_u = proj_spy_l = None
    if end_spy is not None and ar_spy is not None:
        proj_spy_c, proj_spy_u, proj_spy_l = _project_series(ar_spy, av_spy, end_spy)

    # global y scale (historical + projection together)
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

    # draw axes
    drawLine(left, bottom, left + hist_w + proj_w, bottom, fill=rgb(180,180,180))
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
    drawLabel("← Historical", left + hist_w//2, bottom + 22, size=8, fill=rgb(120,120,140))
    drawLabel("Projection →",  divX + proj_w//2, bottom + 22, size=8, fill=rgb(120,120,140))

    # helper: draw mountain/area chart for one historical series
    def draw_hist(series, color, lw=2):
        n = len(series)
        # area fill: vertical lines from baseline to value (mountain effect)
        # draw every 3rd column to keep shape count low
        for i in range(0, n, 3):
            xi = to_x_hist(i, n)
            yi = to_px(series[i])
            yb = to_px(y_min)
            # lighten the fill colour
            fc = rgb(min(255, color.red + 100),
                     min(255, color.green + 100),
                     min(255, color.blue + 100)) if hasattr(color, 'red') else 'lightGray'
            drawLine(xi, yi, xi, yb, fill=fc, lineWidth=3, opacity=55)
        # solid outline on top
        for i in range(n - 1):
            drawLine(to_x_hist(i, n),   to_px(series[i]),
                     to_x_hist(i+1, n), to_px(series[i+1]),
                     fill=color, lineWidth=lw)

    # helper: draw projection band + centre line 
    def draw_proj(centre, upper, lower, color):
        if centre is None:
            return
        n = len(centre)
        # band (light fill via overlapping lines — CMU Graphics has no polygon fill)
        bandColor = rgb(
            min(255, color.red   + 80),
            min(255, color.green + 80),
            min(255, color.blue  + 80)
        ) if hasattr(color, 'red') else 'lightGray'
        for i in range(n - 1):
            # draw thin lines between upper and lower to fake band fill
            steps = 6
            for s in range(steps + 1):
                frac = s / steps
                y_a  = to_px(upper[i]   * (1-frac) + lower[i]   * frac)
                y_b  = to_px(upper[i+1] * (1-frac) + lower[i+1] * frac)
                drawLine(to_x_proj(i, n), y_a, to_x_proj(i+1, n), y_b,
                         fill=bandColor, lineWidth=0.5, opacity=40)
        # centre line dashed
        for i in range(n - 1):
            if i % 5 < 3:
                drawLine(to_x_proj(i, n),   to_px(centre[i]),
                         to_x_proj(i+1, n), to_px(centre[i+1]),
                         fill=color, lineWidth=2)

    # colours
    c_user = rgb(70,  130, 210)
    c_opt  = rgb(60,  180, 110)
    c_spy  = rgb(220,  80,  90)

    # draw historical
    if hist_spy:   draw_hist(hist_spy,  c_spy,  lw=1.2)
    if hist_opt:   draw_hist(hist_opt,  c_opt,  lw=1.5)
    draw_hist(hist_user, c_user, lw=2)

    # draw projections
    draw_proj(proj_spy_c, proj_spy_u, proj_spy_l, c_spy)
    draw_proj(proj_opt_c, proj_opt_u, proj_opt_l, c_opt)
    draw_proj(proj_user_c, proj_user_u, proj_user_l, c_user)

    # legend — inside chart, bottom-right corner
    legendX = divX + proj_w - 4   # right edge of projection zone
    legendY = bottom - 14
    entries = [("Your Portfolio",   c_user),
               ("Optimal (Sharpe)", c_opt),
               ("SPY Benchmark",    c_spy)]
    drawRect(legendX - 102, legendY - len(entries)*18 + 4, 102, len(entries)*18 + 4,
             fill=rgb(250,250,252), border=rgb(210,215,225), borderWidth=0.5)
    for i, (lbl, col) in enumerate(entries):
        ly = legendY - (len(entries) - 1 - i) * 18
        drawRect(legendX - 96, ly - 5, 9, 9, fill=col, border=None)
        drawLabel(lbl, legendX - 84, ly, size=8, fill='dimGray', align='left')

    # projection summary box
    boxX, boxY, boxW, boxH = left, bottom - height - 58, hist_w + proj_w, 52
    drawRect(boxX, boxY, boxW, boxH, fill=rgb(245, 248, 255),
             border=rgb(200, 215, 240))
    drawLabel(f"5-Year Projection  (starting ${pythonRound(investment, 0):,})",
              boxX + boxW//2, boxY + 12, size=10, bold=True, fill=rgb(40, 60, 120))

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
                      fill=rgb(30, 80, 40))
        else:
            drawLabel("N/A", cx, boxY + 40, size=9, fill='gray')

def drawAnnualScreen(app):
    df = app.model.returns_df
    drawLabel("Annual Returns", 300, 30, bold=True, size=16)
    if df is None:
        drawLabel("Run Optimize first", 300, 200, fill='gray')
        return
    df = df.copy()
    df['year'] = df.index.year
    annual = df.groupby('year').sum()
    values = annual.mean(axis=1).values
    years = list(annual.index)
    if len(values) == 0:
        return
    max_v = max(abs(v) for v in values) or 1
    barW = min(40, 400 // len(values))
    for i, v in enumerate(values):
        h = (v / max_v) * 150
        x = 80 + i * (barW + 10)
        color = 'green' if v >= 0 else 'red'
        if h >= 0:
            drawRect(x, 350 - h, barW, h, fill=color)
        else:
            drawRect(x, 350, barW, -h, fill=color)
        drawLabel(str(years[i]), x + barW / 2, 370, size=9, rotateAngle=45)
    drawLine(80, 350, 80 + len(values) * (barW + 10), 350, fill='gray')

def getColor(value):
    clamped = max(-1, min(1, value))
    red = int(255 * (1 - clamped) / 2)
    blue = int(255 * (1 + clamped) / 2)
    return rgb(red, 0, blue)

def drawHeatmap(app):
    corr = app.model.corr_matrix
    drawLabel("Correlation Heatmap", 300, 30, bold=True, size=16)
    if corr is None:
        drawLabel("Run Optimize first", 300, 200, fill='gray')
        return
    n = len(corr)
    cell = 50
    startX = 300 - n * cell // 2
    startY = 120
    for i in range(n):
        for j in range(n):
            color = getColor(corr[i][j])
            drawRect(startX + j * cell, startY + i * cell, cell, cell, fill=color)
            drawLabel(f"{pythonRound(corr[i][j], 2)}",
                      startX + j * cell + cell // 2,
                      startY + i * cell + cell // 2,
                      fill='white', size=10)
    for i, ticker in enumerate(app.model.tickers):
        drawLabel(ticker, startX + i * cell + cell // 2, startY - 12, size=10, bold=True)
        drawLabel(ticker, startX - 12, startY + i * cell + cell // 2, size=10, bold=True, align='right')

def drawEfficientFrontier(app):
    data = app.model.efficient_frontier
    drawLabel("Efficient Frontier", 300, 30, bold=True, size=16)
    if not data:
        drawLabel("Run Optimize first", 300, 200, fill='gray')
        return
    vols = [x[0] for x in data]
    rets = [x[1] for x in data]
    left, bottom = 60, 450
    width, height = 480, 220
    drawLine(left, bottom, left + width, bottom, fill='gray')
    drawLine(left, bottom, left, bottom - height, fill='gray')
    drawLabel("Risk (Volatility)", 300, bottom + 20)
    drawLabel("Return", left - 30, bottom - height // 2, rotateAngle=90)
    max_vol = max(vols)
    max_ret = max(rets)
    min_ret = min(rets)
    rng = max_ret - min_ret or 1
    for i in range(len(vols)):
        x = left + (vols[i] / max_vol) * width
        y = bottom - ((rets[i] - min_ret) / rng) * height
        drawCircle(x, y, 3, fill='steelBlue', opacity=70)

def drawAnnualTable(app):
    df = app.model.returns_df
    drawLabel("Annual Returns Table", 300, 30, bold=True, size=16)
    if df is None:
        drawLabel("Run Optimize first", 300, 200, fill='gray')
        return
    df = df.copy()
    df['year'] = df.index.year
    annual = df.groupby('year').sum()
    drawLabel("Year", 150, 80, bold=True)
    drawLabel("Avg Return", 350, 80, bold=True)
    for i, year in enumerate(annual.index):
        y = 110 + i * 32
        fill = 'snow' if i % 2 == 0 else 'white'
        drawRect(80, y - 12, 440, 28, fill=fill, border='lightGray')
        drawLabel(str(year), 150, y + 2)
        val = annual.iloc[i].mean()
        color = 'green' if val >= 0 else 'red'
        drawLabel(f"{pythonRound(val * 100, 2)}%", 350, y + 2, fill=color)

# Mouse & keyboard

def onMousePress(app, x, y):
    # Always check boxes first — set active and return
    for box in [app.start_year_box, app.end_year_box, app.amount_box, app.csv_path_box]:
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

        # CSV upload button — reads file from the path typed in csv_path_box
        if 380 <= x <= 520 and 163 <= y <= 191:
            path = app.csv_path_box['text'].strip()
            if not path:
                app.status_message = 'Enter a file path in the box below the button'
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

            for i, (t, w) in enumerate(zip(tickers, result)):
                app.asset_rows[i]['ticker']['text'] = t
                app.asset_rows[i]['weight']['text'] = str(pythonRound(w * 100, 1))

            app.status_message = f'Loaded {len(tickers)} tickers from CSV'
            app.csv_loaded = True
            return

        # Use Historical toggle
        if 200 <= x <= 260 and 193 <= y <= 221:
            app.use_historical = not app.use_historical
            return

        # Benchmark toggle
        if 200 <= x <= 260 and 233 <= y <= 261:
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
        if 20 <= x <= 100 and 20 <= y <= 50:
            app.screen = 'input'
            app.active_box = None
            return

    # Clicked on empty space — deactivate box
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
    elif key == 'space':
        pass  # no spaces in tickers or numbers
    elif len(key) == 1:
        # Accept digits, dot, minus, and letters (for tickers)
        if key.isalpha():
            app.active_box['text'] += key.upper()
        elif key in '0123456789.-':
            app.active_box['text'] += key

# Draw 

def redrawAll(app):
    drawRect(0, 0, 600, 600, fill='white')

    if app.screen == 'input':
        drawInputScreen(app)
    else:
        # Back button
        drawRect(20, 20, 80, 28, fill='lightGray', border='gray')
        drawLabel("← Back", 60, 34, size=11)

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
