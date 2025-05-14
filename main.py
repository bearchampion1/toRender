import os
import logging
from datetime import datetime, timedelta
from flask import Flask, request, abort, send_from_directory
import yfinance as yf
import mplfinance as mpf
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import re

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
load_dotenv("id_code.env")
CHANNEL_SECRET = os.getenv("CHANNEL_SECRET")
CHANNEL_ACCESS_TOKEN = os.getenv("CHANNEL_ACCESS_TOKEN")

if not CHANNEL_SECRET or not CHANNEL_ACCESS_TOKEN:
    logger.error("LINE API credentials not found. Check your environment variables.")
    exit(1)

app = Flask(__name__)
configuration = Configuration(access_token=CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(CHANNEL_SECRET)

# User state tracking
user_states = {}

# Constants
STATIC_DIR = "./static"
os.makedirs(STATIC_DIR, exist_ok=True)

# Stock market information
MARKETS = {
    "TW": {"suffix": ".TW", "name": "台灣"},
    "US": {"suffix": "", "name": "美國"},
    "HK": {"suffix": ".HK", "name": "香港"}
}

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
        aapl = yf.Ticker(symbol)
        df = aapl.history(start=start_date, end=end_date)
        
        # Check if data is emptydf 
    
        if df.empty:
            logger.warning(f"No data found for {symbol}")
            return None, "查無資料，請確認股票代碼與日期範圍是否正確。"
        
        # Ensure data integrity
        df = df.apply(pd.to_numeric, errors='coerce')
        
        # Calculate additional indicators
        df['MA5'] = df['Close'].rolling(window=5).mean()
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['MA60'] = df['Close'].rolling(window=60).mean()
        
        # Calculate MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['Histogram'] = df['MACD'] - df['Signal']
        
        # Calculate RSI
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Create subplots for additional indicators
        apds = [
            mpf.make_addplot(df['MA5'], color='blue', width=0.7),
            mpf.make_addplot(df['MA20'], color='orange', width=0.7),
            mpf.make_addplot(df['MA60'], color='red', width=0.7),
            mpf.make_addplot(df['RSI'], panel=1, color='purple', ylabel='RSI'),
            mpf.make_addplot(df['MACD'], panel=2, color='green', ylabel='MACD'),
            mpf.make_addplot(df['Signal'], panel=2, color='red'),
            mpf.make_addplot(df['Histogram'], panel=2, type='bar', color='dimgray')
        ]
        
        # Ensure directory exists
        save_path = f"{STATIC_DIR}/{user_id}_chart.jpg"
        
        # Extract values as native Python types, not pandas Series
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
        
        # Build title with safe string formatting
        title = f"{symbol} K線圖"
        if last_close is not None:
            title += f"\n收盤: {last_close:.2f}"
            
        if change is not None and change_percent is not None:
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
            panel_ratios=(6, 2, 2),
            addplot=apds,
            savefig=save_path
        )
        
        # Generate summary with safe values
        summary = {
            "symbol": symbol
        }
        
        # Safely extract values
        try:
            summary["last_close"] = float(df['Close'].iloc[-1]) if not df.empty else "N/A"
        except (TypeError, ValueError):
            summary["last_close"] = "N/A"
            
        summary["change"] = change if change is not None else "N/A"
        summary["change_percent"] = change_percent if change_percent is not None else "N/A"
        
        try:
            summary["volume"] = float(df['Volume'].iloc[-1]) if not df.empty and 'Volume' in df.columns else "N/A"
        except (TypeError, ValueError):
            summary["volume"] = "N/A"
            
        summary["period"] = f"{start_date} 至 {end_date}"
        
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
            
        # Format numbers safely with fallbacks
        try:
            close_str = f"{float(summary['last_close']):.2f}"
        except (ValueError, TypeError):
            close_str = str(summary['last_close'])
            
        try:
            volume_str = f"{float(summary['volume']):,}"
        except (ValueError, TypeError):
            volume_str = str(summary['volume'])
            
        try:
            high_str = f"{float(summary['high']):.2f}"
        except (ValueError, TypeError):
            high_str = str(summary['high'])
            
        try:
            low_str = f"{float(summary['low']):.2f}"
        except (ValueError, TypeError):
            low_str = str(summary['low'])
            
        try:
            avg_str = f"{float(summary['avg']):.2f}"
        except (ValueError, TypeError):
            avg_str = str(summary['avg'])
        
        # Build message with safe strings
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

@app.route("/static/k_line_chart.jpg")
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
                "使用方式:\n"
                "1. 輸入「股票資訊」開始查詢\n"
                "2. 輸入股票代碼 (台股代碼如「2330」)\n"
                "3. 輸入開始日期 (格式: YYYY-MM-DD)\n"
                "4. 輸入結束日期 (格式: YYYY-MM-DD)\n\n"
                "✨ 特殊命令 ✨\n"
                "- 「預設日期」: 使用過去6個月\n"
                "- 「取消」: 隨時取消當前操作\n"
                "- 「快速查詢」: 一次性查詢格式\n   (例: 快速查詢 2330 2023-01-01 2023-06-30)"
            )
            
            line_bot_api.reply_message(
                ReplyMessageRequest(
                    reply_token=event.reply_token,
                    messages=[TextMessage(text=reply)]
                )
            )
            return

        # Cancel command
        if text == "取消":
            user_states[user_id] = {"step": -1}
            reply = "已取消當前操作。輸入「股票資訊」開始新的查詢。"
            
            line_bot_api.reply_message(
                ReplyMessageRequest(
                    reply_token=event.reply_token,
                    messages=[TextMessage(text=reply)]
                )
            )
            return
            
        # Default dates command
        if text == "預設日期" and state["step"] in [1, 2]:
            start_date, end_date = get_default_dates()
            
            if state["step"] == 1:
                state["start"] = start_date
                state["step"] = 2
                reply = f"已設定開始日期為 {start_date}。請輸入結束日期 (格式: YYYY-MM-DD)，或再次輸入「預設日期」使用今天作為結束日期。"
            else:  # step 2
                state["start"] = state.get("start", start_date)
                state["end"] = end_date
                
                # Process the request with the symbol and dates
                line_bot_api.push_message(
                    PushMessageRequest(
                        to=user_id,
                        messages=[TextMessage(text="資料處理中，請稍後...")]
                    )
                )
                
                symbol = state['symbol']
                if symbol.isdigit() and len(symbol) <= 6:
                    display_symbol = symbol
                    symbol += ".TW"  # Add Taiwan stock suffix
                else:
                    display_symbol = symbol
                
                chart_path, result = generate_stock_chart(
                    symbol,
                    state['start'],
                    state['end'],
                    user_id
                )
                
                if chart_path:
                    # Create image URL
                    image_url = request.url_root + chart_path.replace("./", "")
                    if not image_url.startswith("https"):
                        image_url = image_url.replace("http", "https")
                    
                    # Create and send summary message
                    summary_text = create_summary_message(result)
                    
                    line_bot_api.push_message(
                        PushMessageRequest(
                            to=user_id,
                            messages=[TextMessage(text=summary_text)]
                        )
                    )
                    
                    # Send the chart image
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
                    
                    # Reset user state
                    user_states[user_id] = {"step": -1}
                else:
                    # Send error message
                    line_bot_api.push_message(
                        PushMessageRequest(
                            to=user_id,
                            messages=[TextMessage(text=str(result))]
                        )
                    )
                    user_states[user_id] = {"step": -1}
                return
            
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
            if len(parts) == 4:
                _, symbol, start_date, end_date = parts
                
                if not validate_stock_code(symbol):
                    reply = "股票代碼格式不正確，請重新輸入。"
                elif not validate_date(start_date) or not validate_date(end_date):
                    reply = "日期格式不正確，請使用 YYYY-MM-DD 格式。"
                else:
                    line_bot_api.push_message(
                        PushMessageRequest(
                            to=user_id,
                            messages=[TextMessage(text="資料處理中，請稍後...")]
                        )
                    )
                    
                    if symbol.isdigit() and len(symbol) <= 6:
                        display_symbol = symbol
                        symbol += ".TW"  # Add Taiwan stock suffix
                    else:
                        display_symbol = symbol
                    
                    chart_path, result = generate_stock_chart(
                        symbol,
                        start_date,
                        end_date,
                        user_id
                    )
                    
                    if chart_path:
                        # Create image URL
                        image_url = request.url_root + chart_path.replace("./", "")
                        if not image_url.startswith("https"):
                            image_url = image_url.replace("http", "https")
                        
                        # Create and send summary message
                        summary_text = create_summary_message(result)
                        
                        line_bot_api.push_message(
                            PushMessageRequest(
                                to=user_id,
                                messages=[TextMessage(text=summary_text)]
                            )
                        )
                        
                        # Send the chart image
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
                        
                        return
                    else:
                        reply = str(result)
            else:
                reply = "快速查詢格式不正確。請使用: 快速查詢 股票代碼 開始日期 結束日期"
                
            line_bot_api.reply_message(
                ReplyMessageRequest(
                    reply_token=event.reply_token,
                    messages=[TextMessage(text=reply)]
                )
            )
            return
        
        # Main conversation flow
        if text == "股票資訊":
            state["step"] = 0
            quick_reply = QuickReply(items=create_quick_reply_items())
            
            reply = "請輸入股票代碼 (如台股2330或美股AAPL)"
            
            line_bot_api.reply_message(
                ReplyMessageRequest(
                    reply_token=event.reply_token,
                    messages=[
                        TextMessage(
                            text=reply,
                            quick_reply=quick_reply
                        )
                    ]
                )
            )
            return

        elif state["step"] == 0:
            # Validate stock code
            if not validate_stock_code(text):
                reply = "股票代碼格式不正確，請重新輸入有效的股票代碼。"
                line_bot_api.reply_message(
                    ReplyMessageRequest(
                        reply_token=event.reply_token,
                        messages=[TextMessage(text=reply)]
                    )
                )
                return
                
            state["symbol"] = text.upper()
            state["step"] = 1
            
            # Create quick reply for date selection
            default_start, default_end = get_default_dates()
            reply = f"請輸入起始日期（格式：YYYY-MM-DD）或輸入「預設日期」使用 {default_start}"
            
            line_bot_api.reply_message(
                ReplyMessageRequest(
                    reply_token=event.reply_token,
                    messages=[
                        TextMessage(
                            text=reply,
                            quick_reply=QuickReply(
                                items=[
                                    QuickReplyItem(
                                        action=MessageAction(
                                            label="預設日期",
                                            text="預設日期"
                                        )
                                    ),
                                    QuickReplyItem(
                                        action=MessageAction(
                                            label="取消",
                                            text="取消"
                                        )
                                    )
                                ]
                            )
                        )
                    ]
                )
            )
            return

        elif state["step"] == 1:
            # Validate date format
            if not validate_date(text):
                reply = "日期格式不正確，請使用 YYYY-MM-DD 格式或輸入「預設日期」。"
                line_bot_api.reply_message(
                    ReplyMessageRequest(
                        reply_token=event.reply_token,
                        messages=[TextMessage(text=reply)]
                    )
                )
                return
                
            state["start"] = text
            state["step"] = 2
            
            # Create quick reply for end date
            reply = "請輸入結束日期（格式：YYYY-MM-DD）或輸入「預設日期」使用今天"
            
            line_bot_api.reply_message(
                ReplyMessageRequest(
                    reply_token=event.reply_token,
                    messages=[
                        TextMessage(
                            text=reply,
                            quick_reply=QuickReply(
                                items=[
                                    QuickReplyItem(
                                        action=MessageAction(
                                            label="預設日期",
                                            text="預設日期"
                                        )
                                    ),
                                    QuickReplyItem(
                                        action=MessageAction(
                                            label="取消",
                                            text="取消"
                                        )
                                    )
                                ]
                            )
                        )
                    ]
                )
            )
            return

        elif state["step"] == 2:
            # Validate date format
            if not validate_date(text):
                reply = "日期格式不正確，請使用 YYYY-MM-DD 格式或輸入「預設日期」。"
                line_bot_api.reply_message(
                    ReplyMessageRequest(
                        reply_token=event.reply_token,
                        messages=[TextMessage(text=reply)]
                    )
                )
                return
                
            state["end"] = text

            # Processing message
            line_bot_api.push_message(
                PushMessageRequest(
                    to=user_id,
                    messages=[TextMessage(text="資料處理中，請稍後...")]
                )
            )

            symbol = state['symbol']
            # Handle Taiwan stock codes
            if symbol.isdigit() and len(symbol) <= 6:
                display_symbol = symbol
                symbol += ".TW"  # Add Taiwan stock suffix
            else:
                display_symbol = symbol

            try:
                chart_path, result = generate_stock_chart(
                    symbol,
                    state['start'],
                    state['end'],
                    user_id
                )
                
                if chart_path:
                    # Create image URL
                    image_url = request.url_root + chart_path.replace("./", "")
                    if not image_url.startswith("https"):
                        image_url = image_url.replace("http", "https")
                    
                    # Create and send summary message
                    summary_text = create_summary_message(result)
                    
                    line_bot_api.push_message(
                        PushMessageRequest(
                            to=user_id,
                            messages=[TextMessage(text=summary_text)]
                        )
                    )
                    
                    # Send the chart image
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
                    # Send error message
                    line_bot_api.push_message(
                        PushMessageRequest(
                            to=user_id,
                            messages=[TextMessage(text=str(result))]
                        )
                    )
            except Exception as e:
                logger.error(f"Error processing request: {str(e)}")
                line_bot_api.push_message(
                    PushMessageRequest(
                        to=user_id,
                        messages=[TextMessage(text=f"發生錯誤：{str(e)}")]
                    )
                )
            
            # Reset user state
            user_states[user_id] = {"step": -1}
            return

        else:
            # Default reply for unknown state or command
            quick_reply = QuickReply(
                items=[
                    QuickReplyItem(
                        action=MessageAction(
                            label="股票資訊",
                            text="股票資訊"
                        )
                    ),
                    QuickReplyItem(
                        action=MessageAction(
                            label="說明",
                            text="說明"
                        )
                    )
                ]
            )
            
            reply = "請輸入「股票資訊」開始查詢，或輸入「說明」查看使用方式。"
            
            line_bot_api.reply_message(
                ReplyMessageRequest(
                    reply_token=event.reply_token,
                    messages=[
                        TextMessage(
                            text=reply,
                            quick_reply=quick_reply
                        )
                    ]
                )
            )
            return
"""
if __name__ == "__main__":
    # For production, use gunicorn or similar WSGI server
    # For development, use Flask's built-in server
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
"""