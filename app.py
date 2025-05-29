import os
import logging
import json
from datetime import datetime, timedelta
from flask import Flask, request, abort, send_from_directory
import yfinance as yf
import mplfinance as mpf
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import re
import threading

from linebot.v3 import WebhookHandler
from linebot.v3.exceptions import InvalidSignatureError
from linebot.v3.messaging import (
    Configuration, ApiClient, MessagingApi,
    ReplyMessageRequest, PushMessageRequest,
    TextMessage, ImageMessage, FlexMessage, 
    FlexContainer, QuickReply, QuickReplyItem, 
    MessageAction
)
from linebot.v3.webhooks import MessageEvent, TextMessageContent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv("id.env")
CHANNEL_SECRET = os.getenv("CHANNEL_SECRET")
CHANNEL_ACCESS_TOKEN = os.getenv("LINE_ACCESS_TOKEN")

# Provide fallback values for testing
if not CHANNEL_SECRET:
    CHANNEL_SECRET = "your_channel_secret_here"
    logger.warning("CHANNEL_SECRET not found in environment, using fallback")

if not CHANNEL_ACCESS_TOKEN:
    CHANNEL_ACCESS_TOKEN = "your_channel_access_token_here"
    logger.warning("CHANNEL_ACCESS_TOKEN not found in environment, using fallback")

app = Flask(__name__)
configuration = Configuration(access_token=CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(CHANNEL_SECRET)

# User state tracking
user_states = {}

# User watchlist storage
USER_WATCHLISTS_FILE = "user_watchlists.json"

# Constants
STATIC_DIR = "./static"
os.makedirs(STATIC_DIR, exist_ok=True)

# Stock market information
MARKETS = {
    "TW": {"suffix": ".TW", "name": "台灣"},
    "US": {"suffix": "", "name": "美國"},
    "HK": {"suffix": ".HK", "name": "香港"}
}

def load_user_watchlists():
    """Load user watchlists from file."""
    try:
        if os.path.exists(USER_WATCHLISTS_FILE):
            with open(USER_WATCHLISTS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        logger.error(f"Error loading watchlists: {e}")
    return {}

def save_user_watchlists(watchlists):
    """Save user watchlists to file."""
    try:
        with open(USER_WATCHLISTS_FILE, 'w', encoding='utf-8') as f:
            json.dump(watchlists, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Error saving watchlists: {e}")

def get_user_watchlist(user_id):
    """Get user's watchlist."""
    watchlists = load_user_watchlists()
    return watchlists.get(user_id, [])

def add_to_watchlist(user_id, symbol, name=None):
    """Add stock to user's watchlist."""
    watchlists = load_user_watchlists()
    if user_id not in watchlists:
        watchlists[user_id] = []
    
    # Check if stock already exists
    for stock in watchlists[user_id]:
        if stock['symbol'] == symbol:
            return False, "股票已存在於自選清單中"
    
    # Add new stock
    stock_info = {
        'symbol': symbol,
        'name': name or symbol,
        'added_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    watchlists[user_id].append(stock_info)
    save_user_watchlists(watchlists)
    return True, "已成功添加到自選清單"

def remove_from_watchlist(user_id, symbol):
    """Remove stock from user's watchlist."""
    watchlists = load_user_watchlists()
    if user_id not in watchlists:
        return False, "您的自選清單為空"
    
    original_length = len(watchlists[user_id])
    watchlists[user_id] = [stock for stock in watchlists[user_id] if stock['symbol'] != symbol]
    
    if len(watchlists[user_id]) < original_length:
        save_user_watchlists(watchlists)
        return True, "已從自選清單中移除"
    return False, "股票不在自選清單中"

def get_stock_info(symbol):
    """Get basic stock information."""
    try:
        ticker = yf.Ticker(symbol)
        
        info = ticker.info
        return {
            'name': info.get('longName', info.get('shortName', symbol)),
            'current_price': info.get('currentPrice', 'N/A'),
            'previous_close': info.get('previousClose', 'N/A'),
            'market_cap': info.get('marketCap', 'N/A'),
            'pe_ratio': info.get('trailingPE', 'N/A')
        }
    except Exception as e:
        logger.error(f"Error getting stock info for {symbol}: {e}")
        return {'name': symbol, 'current_price': 'N/A', 'previous_close': 'N/A', 'market_cap': 'N/A', 'pe_ratio': 'N/A'}

def create_watchlist_message(user_id):
    """Create watchlist display message."""
    watchlist = get_user_watchlist(user_id)
    if not watchlist:
        return "📋 您的自選清單為空\n\n使用方式：\n• 輸入「加入自選 股票代碼」添加股票\n• 例如：加入自選 2330"
    
    message = "📋 您的自選清單：\n\n"
    for i, stock in enumerate(watchlist, 1):
        symbol = stock['symbol']
        name = stock['name']
        
        # Get current price info
        try:
            if symbol.isdigit() and len(symbol) <= 6:
                full_symbol = symbol + ".TW"
            else:
                full_symbol = symbol
                
            ticker = yf.Ticker(full_symbol)
            hist = ticker.history(period='2d')
            if not hist.empty:
                current_price = float(hist['Close'].iloc[-1])
                if len(hist) > 1:
                    prev_price = float(hist['Close'].iloc[-2])
                    change = current_price - prev_price
                    change_percent = (change / prev_price) * 100
                    
                    # Fixed change_str formatting
                    if change > 0:
                        change_str = f"📈 +{change:.2f} (+{change_percent:.2f}%)"
                    elif change < 0:
                        change_str = f"📉 {change:.2f} ({change_percent:.2f}%)"
                    else:
                        change_str = "➖ 0.00 (0.00%)"
                    
                    message += f"{i}. {name} ({symbol})\n"
                    message += f"   💰 {current_price:.2f}  {change_str}\n\n"
                else:
                    message += f"{i}. {name} ({symbol})\n"
                    message += f"   💰 {current_price:.2f}\n\n"
            else:
                message += f"{i}. {name} ({symbol})\n"
                message += f"   💰 無法取得價格\n\n"
        except Exception as e:
            message += f"{i}. {name} ({symbol})\n"
            message += f"   💰 無法取得價格\n\n"
    
    message += "📝 管理自選：\n"
    message += "• 「移除自選 股票代碼」移除股票\n"
    message += "• 「清空自選」清空所有股票\n"
    message += "• 「查詢自選 股票代碼」查看K線圖"
    
    return message

def validate_date(date_str):
    """Validate if the string is a valid date in YYYY-MM-DD format."""
    try:
        datetime.strptime(date_str, '%Y-%m-%d')
        return True
    except ValueError:
        return False

def validate_stock_code(code, market=None):
    """Validate stock code format."""
    if market == "TW":
        # Taiwan stock codes are typically 4-6 digits
        return bool(re.match(r'^\d{4,6}$', code))
    elif market == "US":
        # US stock symbols are typically letters, may include dots or hyphens
        return bool(re.match(r'^[A-Za-z\.\-]+$', code))
    elif market == "HK":
        # Hong Kong stock codes are typically 4-5 digits
        return bool(re.match(r'^\d{4,5}$', code))
    else:
        # General validation
        return bool(re.match(r'^[A-Za-z0-9\.\-]+$', code))

def generate_stock_chart(symbol, start_date, end_date, user_id):
    """Generate a stock chart and return the file path."""
    try:
        logger.info(f"Downloading data for {symbol} from {start_date} to {end_date}")
        
        # Add the explicit ticker and data fetching lines
        aapl = yf.Ticker(symbol)
        stock_data = aapl.history(start=start_date, end=end_date)
        
        # Use the explicitly fetched data
        df = stock_data
        
        # Check if data is empty
        if df.empty:
            logger.warning(f"No data found for {symbol}")
            return None, "查無資料，請確認股票代碼與日期範圍是否正確。"
        
        # Ensure data integrity
        df = df.apply(pd.to_numeric, errors='coerce')
        
        # Calculate additional indicators
        if len(df) >= 5:
            df['MA5'] = df['Close'].rolling(window=5).mean()
        if len(df) >= 20:
            df['MA20'] = df['Close'].rolling(window=20).mean()
        if len(df) >= 60:
            df['MA60'] = df['Close'].rolling(window=60).mean()
        
        # Calculate MACD
        if len(df) >= 26:
            exp1 = df['Close'].ewm(span=12, adjust=False).mean()
            exp2 = df['Close'].ewm(span=26, adjust=False).mean()
            df['MACD'] = exp1 - exp2
            df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
            df['Histogram'] = df['MACD'] - df['Signal']
        
        # Calculate RSI
        if len(df) >= 14:
            delta = df['Close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            rs = avg_gain / avg_loss
            df['RSI'] = 100 - (100 / (1 + rs))
        
        # Create subplots for additional indicators
        apds = []
        if 'MA5' in df.columns:
            apds.append(mpf.make_addplot(df['MA5'], color='blue', width=0.7))
        if 'MA20' in df.columns:
            apds.append(mpf.make_addplot(df['MA20'], color='orange', width=0.7))
        if 'MA60' in df.columns:
            apds.append(mpf.make_addplot(df['MA60'], color='red', width=0.7))
        
        # Prepare panel configuration
        panel_ratios = [6, 2]  # Main chart + Volume
        panel_count = 1  # Start with 1 additional panel (volume)
        
        if 'RSI' in df.columns:
            apds.append(mpf.make_addplot(df['RSI'], panel=1 + panel_count, color='purple', ylabel='RSI'))
            panel_count += 1
            panel_ratios.append(2)
        
        if 'MACD' in df.columns:
            panel_idx = 1 + panel_count
            apds.append(mpf.make_addplot(df['MACD'], panel=panel_idx, color='green', ylabel='MACD'))
            apds.append(mpf.make_addplot(df['Signal'], panel=panel_idx, color='red'))
            apds.append(mpf.make_addplot(df['Histogram'], panel=panel_idx, type='bar', color='dimgray'))
            panel_count += 1
            panel_ratios.append(2)
        
        # Ensure directory exists
        save_path = f"{STATIC_DIR}/{user_id}_chart.png"
        
        # Extract values safely
        try:
            last_close = float(df['Close'].iloc[-1]) if not df.empty else None
        except (TypeError, ValueError):
            last_close = None
            
        try:
            previous_close = float(df['Close'].iloc[-2]) if len(df) > 1 else None
        except (TypeError, ValueError):
            previous_close = None
            
        change = last_close - previous_close if previous_close is not None and last_close is not None else None
        change_percent = (change / previous_close * 100) if previous_close is not None and change is not None else None
        
        # Build title with improved formatting for zero changes
        title = f"{symbol} K線圖"
        if last_close is not None:
            title += f"\n收盤: {last_close:.2f}"
            
        # Improved change display logic
        if change is not None and change_percent is not None:
            if abs(change) < 0.01:  # Consider very small changes as flat
                title += " | 持平"
            elif change > 0:
                title += f" | 漲跌: +{change:.2f} (+{change_percent:.2f}%)"
            else:  # change < 0
                title += f" | 漲跌: {change:.2f} ({change_percent:.2f}%)"
        
        # Create the plot
        mpf.plot(
            df,
            type='candle',
            style='charles',
            title=title,
            ylabel='價格',
            ylabel_lower='成交量',
            volume=True,
            figsize=(12, 10),
            panel_ratios=tuple(panel_ratios),
            addplot=apds if apds else None,
            savefig=save_path
        )
        
        # Generate summary
        summary = {
            "symbol": symbol,
            "last_close": last_close if last_close is not None else "N/A",
            "change": change if change is not None else "N/A",
            "change_percent": change_percent if change_percent is not None else "N/A",
            "period": f"{start_date} 至 {end_date}"
        }
        
        try:
            summary["volume"] = float(df['Volume'].iloc[-1]) if not df.empty and 'Volume' in df.columns else "N/A"
        except (TypeError, ValueError):
            summary["volume"] = "N/A"
            
        try:
            summary["high"] = float(df['High'].max()) if not df.empty and 'High' in df.columns else "N/A"
        except (TypeError, ValueError):
            summary["high"] = "N/A"
            
        try:
            summary["low"] = float(df['Low'].min()) if not df.empty and 'Low' in df.columns else "N/A"
        except (TypeError, ValueError):
            summary["low"] = "N/A"
            
        try:
            summary["avg"] = float(df['Close'].mean()) if not df.empty and 'Close' in df.columns else "N/A"
        except (TypeError, ValueError):
            summary["avg"] = "N/A"
        
        return save_path, summary
        
    except Exception as e:
        logger.error(f"Error generating chart: {str(e)}")
        return None, f"生成圖表時發生錯誤: {str(e)}"

def create_summary_message(summary):
    """Create a text summary of the stock data."""
    try:
        # Handle various data types safely
        if isinstance(summary['change'], (int, float)):
            change_str = f"{summary['change']:.2f}"
        else:
            change_str = str(summary['change'])
            
        if isinstance(summary['change_percent'], (int, float)):
            change_percent_str = f"{summary['change_percent']:.2f}%"
        else:
            change_percent_str = str(summary['change_percent'])
            
        # Format numbers safely
        try:
            close_str = f"{float(summary['last_close']):.2f}" if summary['last_close'] != "N/A" else "N/A"
        except (ValueError, TypeError):
            close_str = str(summary['last_close'])
            
        try:
            volume_str = f"{float(summary['volume']):,}" if summary['volume'] != "N/A" else "N/A"
        except (ValueError, TypeError):
            volume_str = str(summary['volume'])
            
        try:
            high_str = f"{float(summary['high']):.2f}" if summary['high'] != "N/A" else "N/A"
        except (ValueError, TypeError):
            high_str = str(summary['high'])
            
        try:
            low_str = f"{float(summary['low']):.2f}" if summary['low'] != "N/A" else "N/A"
        except (ValueError, TypeError):
            low_str = str(summary['low'])
            
        try:
            avg_str = f"{float(summary['avg']):.2f}" if summary['avg'] != "N/A" else "N/A"
        except (ValueError, TypeError):
            avg_str = str(summary['avg'])
        
        # Build message
        message = (
            f"📊 {summary['symbol']} 股票摘要\n\n"
            f"📈 收盤價: {close_str}\n"
            f"📉 漲跌: {change_str} ({change_percent_str})\n"
            f"💹 成交量: {volume_str}\n"
            f"⏰ 資料期間: {summary['period']}\n"
            f"🔺 最高價: {high_str}\n"
            f"🔻 最低價: {low_str}\n"
            f"📊 平均價: {avg_str}"
        )
        return message
    except Exception as e:
        logger.error(f"Error creating summary: {str(e)}")
        return "無法生成股票摘要"

def get_default_dates():
    """Return default date range (past 6 months)."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=180)
    return start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')

def create_quick_reply_items():
    """Create quick reply items for common stocks."""
    common_stocks = [
        {"label": "台積電", "text": "2330"},
        {"label": "鴻海", "text": "2317"},
        {"label": "聯發科", "text": "2454"},
        {"label": "蘋果", "text": "AAPL"},
        {"label": "特斯拉", "text": "TSLA"},
        {"label": "美光", "text": "MU"},
        {"label": "輝達", "text": "NVDA"},
        {"label": "預設日期", "text": "預設日期"},
        {"label": "說明", "text": "說明"}
    ]
    
    items = []
    for stock in common_stocks:
        items.append(
            QuickReplyItem(
                action=MessageAction(
                    label=stock["label"],
                    text=stock["text"]
                )
            )
        )
    return items

def create_watchlist_quick_reply_items(user_id):
    """Create quick reply items for user's watchlist."""
    watchlist = get_user_watchlist(user_id)
    items = []
    
    for stock in watchlist[:8]:  # Limit to 8 items
        symbol = stock['symbol']
        name = stock['name']
        display_name = name[:8] if len(name) > 8 else name
        items.append(
            QuickReplyItem(
                action=MessageAction(
                    label=display_name,
                    text=symbol
                )
            )
        )
    
    return items

def process_chart_generation_async(user_id, symbol, start_date, end_date, base_url):
    """Process chart generation asynchronously."""
    try:
        with ApiClient(configuration) as api_client:
            line_bot_api = MessagingApi(api_client)
            
            # Add Taiwan suffix if needed
            if symbol.isdigit() and len(symbol) <= 6:
                full_symbol = symbol + ".TW"
            else:
                full_symbol = symbol
                
            chart_path, result = generate_stock_chart(
                full_symbol,
                start_date,
                end_date,
                user_id
            )
            
            if chart_path:
                # Create image URL using passed base_url
                image_url = base_url + chart_path.replace("./", "")
                if image_url.startswith("http://"):
                    image_url = image_url.replace("http://", "https://")
                
                # Send summary message
                summary_text = create_summary_message(result)
                
                line_bot_api.push_message(
                    PushMessageRequest(
                        to=user_id,
                        messages=[TextMessage(text=summary_text)]
                    )
                )
                
                # Send chart image
                line_bot_api.push_message(
                    PushMessageRequest(
                        to=user_id,
                        messages=[
                            ImageMessage(
                                original_content_url=image_url,
                                preview_image_url=image_url
                            )
                        ]
                    )
                )
            else:
                line_bot_api.push_message(
                    PushMessageRequest(
                        to=user_id,
                        messages=[TextMessage(text=str(result))]
                    )
                )
                
    except Exception as e:
        logger.error(f"Error in async chart generation: {e}")
        try:
            with ApiClient(configuration) as api_client:
                line_bot_api = MessagingApi(api_client)
                line_bot_api.push_message(
                    PushMessageRequest(
                        to=user_id,
                        messages=[TextMessage(text="處理過程中發生錯誤，請稍後再試。")]
                    )
                )
        except Exception as inner_e:
            logger.error(f"Error sending error message: {inner_e}")

def process_chart_generation(user_id, symbol, start_date, end_date, reply_token=None):
    """Process chart generation and send response."""
    try:
        with ApiClient(configuration) as api_client:
            line_bot_api = MessagingApi(api_client)
            
            # Capture base URL while in request context
            base_url = request.url_root
            
            # Send processing message
            if reply_token:
                line_bot_api.reply_message(
                    ReplyMessageRequest(
                        reply_token=reply_token,
                        messages=[TextMessage(text="資料處理中，請稍後...")]
                    )
                )
            else:
                line_bot_api.push_message(
                    PushMessageRequest(
                        to=user_id,
                        messages=[TextMessage(text="資料處理中，請稍後...")]
                    )
                )
            
            # Start async processing with base_url
            thread = threading.Thread(
                target=process_chart_generation_async,
                args=(user_id, symbol, start_date, end_date, base_url)
            )
            thread.daemon = True
            thread.start()
            
    except Exception as e:
        logger.error(f"Error starting chart generation process: {e}")
        try:
            with ApiClient(configuration) as api_client:
                line_bot_api = MessagingApi(api_client)
                if reply_token:
                    line_bot_api.reply_message(
                        ReplyMessageRequest(
                            reply_token=reply_token,
                            messages=[TextMessage(text="處理過程中發生錯誤，請稍後再試。")]
                        )
                    )
                else:
                    line_bot_api.push_message(
                        PushMessageRequest(
                            to=user_id,
                            messages=[TextMessage(text="處理過程中發生錯誤，請稍後再試。")]
                        )
                    )
        except Exception as inner_e:
            logger.error(f"Error sending error message: {inner_e}")
@app.route("/")
def home():
    """Home page endpoint."""
    return "LINE Stock Chart Bot is running!"

@app.route("/static/<path:filename>")
def serve_static(filename):
    """Serve static files."""
    return send_from_directory(STATIC_DIR, filename)

@app.route("/callback", methods=['POST'])
def callback():
    """Handle LINE webhook callback."""
    signature = request.headers.get('X-Line-Signature', '')
    body = request.get_data(as_text=True)
    
    logger.info(f"Request body: {body}")
    
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        logger.error("Invalid signature")
        abort(400)
    return 'OK'

@handler.add(MessageEvent, message=TextMessageContent)
def handle_message(event):
    """Handle incoming messages."""
    user_id = event.source.user_id
    text = event.message.text.strip()
    
    with ApiClient(configuration) as api_client:
        line_bot_api = MessagingApi(api_client)
        
        if user_id not in user_states:
            user_states[user_id] = {"step": -1}
        
        state = user_states[user_id]
        
        # Help command
        if text == "說明":
            reply = (
                "📈 股票K線圖查詢機器人 📉\n\n"
                "📊 基本查詢:\n"
                "• 「股票資訊」- 開始查詢股票\n"
                "• 「快速查詢 股票代碼 開始日期 結束日期」\n\n"
                "⭐ 自選股票:\n"
                "• 「自選清單」- 查看自選股票\n"
                "• 「加入自選 股票代碼」- 添加股票\n"
                "• 「移除自選 股票代碼」- 移除股票\n"
                "• 「查詢自選 股票代碼」- 查看K線圖\n"
                "• 「清空自選」- 清空所有自選股票\n\n"
                "🔧 其他功能:\n"
                "• 「預設日期」- 使用過去6個月\n"
                "• 「取消」- 取消當前操作\n\n"
                "範例: 加入自選 2330"
            )
            
            line_bot_api.reply_message(
                ReplyMessageRequest(
                    reply_token=event.reply_token,
                    messages=[TextMessage(text=reply)]
                )
            )
            return

        # Quick query command
        if text.startswith("快速查詢"):
            parts = text.split()
            if len(parts) >= 4:
                symbol = parts[1].upper()
                start_date = parts[2]
                end_date = parts[3]
                
                # Validate inputs
                if not validate_stock_code(symbol):
                    reply = "股票代碼格式不正確。"
                elif not validate_date(start_date):
                    reply = "開始日期格式不正確，請使用 YYYY-MM-DD 格式。"
                elif not validate_date(end_date):
                    reply = "結束日期格式不正確，請使用 YYYY-MM-DD 格式。"
                else:
                    process_chart_generation(user_id, symbol, start_date, end_date, event.reply_token)
                    return
            else:
                reply = "請輸入正確格式：快速查詢 股票代碼 開始日期 結束日期\n例如：快速查詢 2330 2023-01-01 2023-12-31"
            
            line_bot_api.reply_message(
                ReplyMessageRequest(
                    reply_token=event.reply_token,
                    messages=[TextMessage(text=reply)]
                )
            )
            return

        # Watchlist commands
        if text == "自選清單":
            reply = create_watchlist_message(user_id)
            watchlist_items = create_watchlist_quick_reply_items(user_id)
            
            management_items = [
                QuickReplyItem(action=MessageAction(label="加入自選", text="加入自選 ")),
                QuickReplyItem(action=MessageAction(label="移除自選", text="移除自選 ")),
                QuickReplyItem(action=MessageAction(label="清空自選", text="清空自選"))
            ]
            
            all_items = watchlist_items + management_items
            
            line_bot_api.reply_message(
                ReplyMessageRequest(
                    reply_token=event.reply_token,
                    messages=[
                        TextMessage(
                            text=reply,
                            quick_reply=QuickReply(items=all_items[:13]) if all_items else None
                        )
                    ]
                )
            )
            return

        if text.startswith("加入自選"):
            parts = text.split()
            if len(parts) >= 2:
                symbol = parts[1].upper()
                
                if not validate_stock_code(symbol):
                    reply = "股票代碼格式不正確，請重新輸入。"
                else:
                    # Add Taiwan suffix if needed
                    if symbol.isdigit() and len(symbol) <= 6:
                        full_symbol = symbol + ".TW"
                    else:
                        full_symbol = symbol
                    
                    # Get stock name
                    try:
                        ticker = yf.Ticker(full_symbol)
                        info = ticker.info
                        stock_name = info.get('longName', info.get('shortName', symbol))
                    except:
                        stock_name = symbol
                    
                    success, message = add_to_watchlist(user_id, symbol, stock_name)
                    reply = message
            else:
                reply = "請輸入正確格式：加入自選 股票代碼\n例如：加入自選 2330"
            
            line_bot_api.reply_message(
                ReplyMessageRequest(
                    reply_token=event.reply_token,
                    messages=[TextMessage(text=reply)]
                )
            )
            return

        if text.startswith("移除自選"):
            parts = text.split()
            if len(parts) >= 2:
                symbol = parts[1].upper()
                success, message = remove_from_watchlist(user_id, symbol)
                reply = message
            else:
                reply = "請輸入正確格式：移除自選 股票代碼\n例如：移除自選 2330"
            
            line_bot_api.reply_message(
                ReplyMessageRequest(
                    reply_token=event.reply_token,
                    messages=[TextMessage(text=reply)]
                )
            )
            return

        if text == "清空自選":
            watchlists = load_user_watchlists()
            watchlists[user_id] = []
            save_user_watchlists(watchlists)
            
            reply = "已清空所有自選股票。"
            
            line_bot_api.reply_message(
                ReplyMessageRequest(
                    reply_token=event.reply_token,
                    messages=[TextMessage(text=reply)]
                )
            )
            return

        if text.startswith("查詢自選"):
            parts = text.split()
            if len(parts) >= 2:
                symbol = parts[1].upper()
                
                # Check if symbol is in user's watchlist
                watchlist = get_user_watchlist(user_id)
                found = False
                for stock in watchlist:
                    if stock['symbol'] == symbol:
                        found = True
                        break
                
                if not found:
                    reply = f"股票 {symbol} 不在您的自選清單中。"
                    line_bot_api.reply_message(
                        ReplyMessageRequest(
                            reply_token=event.reply_token,
                            messages=[TextMessage(text=reply)]
                        )
                    )
                    return
                
                # Use default dates
                start_date, end_date = get_default_dates()
                
                # Process chart generation
                process_chart_generation(user_id, symbol, start_date, end_date, event.reply_token)
                return
            else:
                reply = "請輸入正確格式：查詢自選 股票代碼\n例如：查詢自選 2330"
                line_bot_api.reply_message(
                    ReplyMessageRequest(
                        reply_token=event.reply_token,
                        messages=[TextMessage(text=reply)]
                    )
                )
                return

        # Handle direct stock code input
        if text.upper() in ["AAPL", "TSLA", "NVDA", "MU"] or (text.isdigit() and len(text) <= 6):
            symbol = text.upper()
            start_date, end_date = get_default_dates()
            process_chart_generation(user_id, symbol, start_date, end_date, event.reply_token)
            return

        # Stock info command
        if text == "股票資訊":
            reply = "請輸入股票代碼，例如：2330（台積電）或 AAPL（蘋果）"
            quick_reply_items = create_quick_reply_items()
            
            line_bot_api.reply_message(
                ReplyMessageRequest(
                    reply_token=event.reply_token,
                    messages=[
                        TextMessage(
                            text=reply,
                            quick_reply=QuickReply(items=quick_reply_items)
                        )
                    ]
                )
            )
            return

        # Default response
        reply = (
            "歡迎使用股票K線圖查詢機器人！\n\n"
            "請輸入以下指令：\n"
            "• 「說明」- 查看使用說明\n"
            "• 「股票資訊」- 開始查詢\n"
            "• 「自選清單」- 管理自選股票\n"
            "• 直接輸入股票代碼（如：2330、AAPL）"
        )
        
        line_bot_api.reply_message(
            ReplyMessageRequest(
                reply_token=event.reply_token,
                messages=[TextMessage(text=reply)]
            )
        )
        
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()  # 讀取 .env

    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
"""
if __name__ == "__main__":
    # For production, use gunicorn or similar WSGI server
    # For development, use Flask's built-in server
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
"""