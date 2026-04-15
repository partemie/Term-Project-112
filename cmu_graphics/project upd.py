from cmu_graphics import *
import random

# 12-month example data (prices in USD)
example_data = {
    'FB': [200, 210, 220, 230, 225, 235, 240, 245, 250, 260, 270, 280],
    'AAPL': [150, 152, 155, 158, 160, 162, 165, 167, 170, 172, 175, 180],
    'AMZN': [3000, 3050, 3100, 3120, 3150, 3200, 3250, 3300, 3350, 3400, 3450, 3500],
    'NFLX': [500, 510, 520, 530, 525, 535, 540, 545, 550, 560, 570, 580],
    'GOOG': [1200, 1210, 1220, 1230, 1240, 1250, 1260, 1270, 1280, 1290, 1300, 1310]
}

class PortfolioAppModel:
    def __init__(self):
        self.screen = 'input'
        self.tickers = ['FB', 'AAPL', 'AMZN']  # default
        self.weights = [0.33, 0.33, 0.34]     # default
        self.investment = 10000                # default
        self.duration = 1                       # years (MVP = multiply monthly data by duration)
        self.cumulative_returns = []
        self.expected_return = 0
        self.volatility = 0
        self.variance = 0
        self.graph_type = 'line'  # line, pie, bar

    # Calculate percent changes
    def pct_change(self, prices):
        returns = []
        for i in range(1, len(prices)):
            r = (prices[i] - prices[i-1]) / prices[i-1]
            returns.append(r)
        return returns

    # Calculate portfolio metrics
    def calculate_portfolio(self):
        n_months = len(example_data[self.tickers[0]])
        # Calculate monthly returns for each asset
        asset_returns = []
        for t in self.tickers:
            asset_returns.append(self.pct_change(example_data[t]))

        # Portfolio monthly return
        portfolio_returns = []
        for i in range(n_months - 1):
            r = 0
            for j in range(len(self.tickers)):
                r += asset_returns[j][i] * self.weights[j]
            portfolio_returns.append(r)

        # Expected monthly return
        self.expected_return = sum(portfolio_returns) / len(portfolio_returns)
        # Variance & volatility
        self.variance = sum((r - self.expected_return) ** 2 for r in portfolio_returns) / len(portfolio_returns)
        self.volatility = self.variance ** 0.5

        # Cumulative portfolio value
        value = self.investment
        self.cumulative_returns = []
        for r in portfolio_returns:
            value *= (1 + r)
            self.cumulative_returns.append(value)

app = PortfolioAppModel()

# Text input boxes
ticker_boxes = []
weight_boxes = []
investment_box = None
duration_box = None

def onAppStart(app):
    global ticker_boxes, weight_boxes, investment_box, duration_box
    app.screen = 'input'
    # Create input boxes
    ticker_boxes = [TextInput('', 200, 100 + i*40, width=100, height=30) for i in range(5)]
    weight_boxes = [TextInput('', 350, 100 + i*40, width=50, height=30) for i in range(5)]
    investment_box = TextInput(str(app.investment), 200, 320, width=100, height=30)
    duration_box = TextInput(str(app.duration), 350, 320, width=50, height=30)

def onMousePress(app, x, y):
    # Input
    if app.screen == 'input':
        # Next button
        if 150 <= x <= 250 and 370 <= y <= 410:
            # Read inputs
            tickers = [b.text.strip() for b in ticker_boxes if b.text.strip() != '']
            weights = []
            for w in weight_boxes[:len(tickers)]:
                try:
                    weights.append(float(w.text)/100)
                except:
                    weights.append(0)
            total_weight = sum(weights)
            if total_weight != 1:
                # Normalize
                weights = [w/total_weight for w in weights]
            try:
                investment = float(investment_box.text)
            except:
                investment = 10000
            try:
                duration = int(duration_box.text)
            except:
                duration = 1

            # Update model
            app.tickers = tickers
            app.weights = weights
            app.investment = investment
            app.duration = duration
            app.calculate_portfolio()
            app.screen = 'results'

   # results
    elif app.screen == 'results':
        # Back button
        if 20 <= x <= 100 and 20 <= y <= 50:
            app.screen = 'input'
        # Graph buttons
        elif 150 <= x <= 250 and 500 <= y <= 530:
            app.graph_type = 'line'
        elif 270 <= x <= 370 and 500 <= y <= 530:
            app.graph_type = 'pie'
        elif 390 <= x <= 490 and 500 <= y <= 530:
            app.graph_type = 'bar'

def redrawAll(app):
    if app.screen == 'input':
        drawInputScreen(app)
    elif app.screen == 'results':
        drawResultsScreen(app)

def drawInputScreen(app):
    drawLabel("Portfolio MVP Simulator", 200, 50, size=18, bold=True)
    drawLabel("Enter up to 5 tickers and weights (sum=100%)", 200, 80)
    # Draw labels
    for i in range(5):
        drawLabel(f"Ticker {i+1}:", 150, 115 + i*40)
        drawLabel("Weight %:", 300, 115 + i*40)
    # Draw input boxes
    for b in ticker_boxes + weight_boxes + [investment_box, duration_box]:
        b.draw()
    drawLabel("Investment ($):", 150, 335)
    drawLabel("Duration (years):", 350, 335)
    # Next button
    drawRect(150, 370, 100, 40, fill='lightGreen')
    drawLabel("Show Results", 200, 390)

def drawResultsScreen(app):
    drawLabel("Portfolio Results", 200, 30, size=18, bold=True)
    # Back button
    drawRect(20, 20, 80, 30, fill='lightGray')
    drawLabel("Back", 60, 35)
    # Portfolio summary
    drawLabel(f"Expected Monthly Return: {round(app.expected_return*100,2)}%", 200, 70)
    drawLabel(f"Volatility: {round(app.volatility*100,2)}%", 200, 100)
    drawLabel(f"Variance: {round(app.variance*100,2)}%", 200, 130)
    drawLabel(f"Total Investment: ${app.investment}", 200, 160)

    # Graph type buttons
    drawRect(150, 500, 100, 30, fill='lightBlue')
    drawLabel("Line Graph", 200, 515)
    drawRect(270, 500, 100, 30, fill='lightBlue')
    drawLabel("Pie Chart", 320, 515)
    drawRect(390, 500, 100, 30, fill='lightBlue')
    drawLabel("Bar Chart", 440, 515)

    # Draw graph
    if app.graph_type == 'line':
        drawLineGraph(app)
    elif app.graph_type == 'pie':
        drawPieChart(app)
    elif app.graph_type == 'bar':
        drawBarChart(app)


def drawLineGraph(app):
    path = app.cumulative_returns
    if len(path) < 2:
        return
    left, bottom = 50, 500
    width, height = 500, 200
    drawLine(left, bottom, left + width, bottom)   # x-axis
    drawLine(left, bottom, left, bottom - height)  # y-axis
    max_val = max(path)
    scaleY = height / max_val
    stepX = width / (len(path) - 1)
    for i in range(len(path)-1):
        x1 = left + i*stepX
        y1 = bottom - path[i]*scaleY
        x2 = left + (i+1)*stepX
        y2 = bottom - path[i+1]*scaleY
        drawLine(x1, y1, x2, y2, lineWidth=2, fill='red')

def drawPieChart(app):
    total = sum(app.weights)
    start_angle = 0
    radius = 100
    centerX, centerY = 300, 400
    colors = ['red','green','blue','yellow','purple']
    for i, w in enumerate(app.weights):
        angle = 360 * w / total
        drawPieSlice(centerX, centerY, radius, start_angle, start_angle + angle, fill=colors[i%len(colors)])
        start_angle += angle
    drawLabel("Portfolio Allocation", centerX, centerY - radius - 20, size=14)

def drawBarChart(app):
    left, bottom = 100, 450
    width, maxHeight = 50, 150
    colors = ['red','green','blue','yellow','purple']
    for i, w in enumerate(app.weights):
        barHeight = w/max(app.weights) * maxHeight
        drawRect(left + i*(width+20), bottom - barHeight, width, barHeight, fill=colors[i%len(colors)])
        drawLabel(app.tickers[i], left + i*(width+20) + width/2, bottom + 10, align='center')

runApp(width=600, height=600)