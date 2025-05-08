import os
from time import sleep
import pandas as pd
import matplotlib.pyplot as plt
import mpl_finance as mpf
from flask import Flask, request, abort
import crawler_module as m
from dotenv import load_dotenv
from linebot.v3 import WebhookHandler
from linebot.v3.exceptions import InvalidSignatureError
from linebot.v3.messaging import (
    Configuration, ApiClient, MessagingApi,
    ReplyMessageRequest, PushMessageRequest,
    TextMessage, ImageMessage
)
from linebot.v3.webhooks import MessageEvent, TextMessageContent
from datetime import datetime
import gc
import logging

# 配置日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 加載環境變數
load_dotenv("id.env")

app = Flask(__name__)

# LINE 配置
line_access_token = os.environ.get("LINE_ACCESS_TOKEN")
line_channel_secret = os.environ.get("LINE_CHANNEL_SECRET")

configuration = Configuration(access_token=line_access_token)
whandler = WebhookHandler(line_channel_secret)

# 使用者狀態管理
user_states = {}

def validate_date(date_str):
    """驗證日期格式和合理性"""
    try:
        date = datetime.strptime(date_str, "%Y%m%d").date()
        if date > datetime.now().date():
            return False, "日期不能是未來日期"
        return True, ""
    except ValueError:
        return False, "日期格式錯誤，請使用 YYYYMMDD 格式"

@app.route("/", methods=["GET"])
def index():
    return "The server is running!"

@app.route("/callback", methods=['POST'])
def callback():
    signature = request.headers['X-Line-Signature']
    body = request.get_data(as_text=True)
    try:
        whandler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
    return 'OK'

def process_stock_data(user_id, symbol, start_date, end_date):
    """處理股票數據的獨立函數"""
    with ApiClient(configuration) as api_client:
        line_bot_api = MessagingApi(api_client)
        
        try:
            # 通知開始處理
            line_bot_api.push_message(
                PushMessageRequest(
                    to=user_id,
                    messages=[TextMessage(text="資料處理中，請稍候...")]
                )
            )

            stock_symbol, dates = m.get_data()
            all_list = []
            df_columns = None

            for date in dates:
                try:
                    df = m.crawl_data(date, stock_symbol)
                    if df is not None:
                        all_list.append(df[0])
                        if df_columns is None:
                            df_columns = df[1]
                except Exception as e:
                    logger.error(f"Error processing date {date}: {str(e)}")
                    continue

            if not all_list:
                return False, "查無資料，請確認日期或代碼。"

            # 清理記憶體
            gc.collect()

            # 處理數據
            all_df = pd.DataFrame(all_list, columns=df_columns)
            # ... (保持原有的數據處理邏輯)

            # 繪製圖表
            fig, (ax, ax2) = plt.subplots(2, 1, sharex=True, figsize=(24, 15), dpi=100)
            # ... (保持原有的繪圖邏輯)

            # 儲存圖片
            os.makedirs("./static", exist_ok=True)
            save_path = "./static/k_line_chart.jpg"
            plt.savefig(save_path, bbox_inches='tight')
            plt.close(fig)  # 明確關閉圖表釋放記憶體

            return True, save_path

        except Exception as e:
            logger.error(f"處理股票數據時發生錯誤: {str(e)}")
            return False, f"處理數據時發生錯誤: {str(e)}"

@whandler.add(MessageEvent, message=TextMessageContent)
def handle_message(event):
    user_id = event.source.user_id
    text = event.message.text.strip()

    with ApiClient(configuration) as api_client:
        line_bot_api = MessagingApi(api_client)

        if user_id not in user_states:
            user_states[user_id] = {"step": -1}

        state = user_states[user_id]

        if text == "股票資訊":
            state["step"] = 0
            reply = "請輸入股票代碼"

        elif state["step"] == 0:
            state["symbol"] = text
            state["step"] = 1
            reply = "請輸入起始日期（格式：YYYYMMDD）"

        elif state["step"] == 1:
            is_valid, msg = validate_date(text)
            if is_valid:
                state["start"] = text
                state["step"] = 2
                reply = "請輸入結束日期（格式：YYYYMMDD）"
            else:
                reply = msg

        elif state["step"] == 2:
            is_valid, msg = validate_date(text)
            if is_valid:
                state["end"] = text
                
                # 儲存查詢資訊
                with open('stock.txt', 'w') as f:
                    f.write(f"{state['symbol']},{state['start']},{state['end']}")

                # 在背景處理數據
                success, result = process_stock_data(
                    user_id, state["symbol"], state["start"], state["end"]
                )

                if success:
                    image_url = request.url_root + "static/k_line_chart.jpg"
                    image_url = image_url.replace("http", "https")
                    image_message = ImageMessage(
                        original_content_url=image_url,
                        preview_image_url=image_url
                    )
                    line_bot_api.push_message(
                        PushMessageRequest(
                            to=user_id,
                            messages=[image_message]
                        )
                    )
                else:
                    line_bot_api.push_message(
                        PushMessageRequest(
                            to=user_id,
                            messages=[TextMessage(text=result)]
                        )
                    )

                # 重置狀態
                user_states[user_id] = {"step": -1}
                return
            else:
                reply = msg

        else:
            reply = "請輸入「股票資訊」來查詢 K 線圖"

        line_bot_api.reply_message(
            ReplyMessageRequest(
                reply_token=event.reply_token,
                messages=[TextMessage(text=reply)]
            )
        )