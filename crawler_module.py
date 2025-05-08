import requests
from io import StringIO
import pandas as pd
import datetime
import logging
from typing import Tuple, List, Optional

# 配置日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_setting() -> List[str]:
    """讀取股票設定檔
    
    Returns:
        List[str]: [股票代號, 起始日期, 結束日期]
                  若讀取失敗則返回空列表
    """
    try:
        with open('stock.txt', 'r', encoding='utf-8') as f:
            content = f.read().strip()
            logger.info(f'讀取設定檔內容: {content}')
            
            if not content:
                logger.error('stock.txt 是空的')
                return []
                
            parts = content.split(',')
            if len(parts) != 3:
                logger.error('stock.txt 格式錯誤，應為「代號,起始日期,結束日期」')
                return []
                
            return parts
            
    except FileNotFoundError:
        logger.error('stock.txt 檔案不存在')
    except Exception as e:
        logger.error(f'讀取 stock.txt 時發生錯誤: {str(e)}')
    
    return []

def get_data() -> Tuple[str, List[str]]:
    """獲取股票代號和日期範圍
    
    Returns:
        Tuple[str, List[str]]: (股票代號, 日期列表)
    """
    data = get_setting()
    if not data or len(data) != 3:
        logger.error('無效的設定資料')
        return None, []
    
    symbol = data[0].strip()
    start_date_str = data[1].strip()
    end_date_str = data[2].strip()
    
    try:
        start_date = datetime.datetime.strptime(start_date_str, '%Y%m%d')
        end_date = datetime.datetime.strptime(end_date_str, '%Y%m%d')
        
        if start_date > end_date:
            logger.error('起始日期不能晚於結束日期')
            return symbol, []
            
        if start_date > datetime.datetime.now():
            logger.error('起始日期不能是未來日期')
            return symbol, []
            
        dates = []
        for daynumber in range((end_date - start_date).days + 1):
            date = start_date + datetime.timedelta(days=daynumber)
            if date.weekday() < 5:  # 0-4 是週一到週五
                dates.append(date.strftime('%Y%m%d'))
                
        logger.info(f'獲取 {symbol} 從 {start_date_str} 到 {end_date_str} 的 {len(dates)} 個交易日')
        return symbol, dates
        
    except ValueError as e:
        logger.error(f'日期格式錯誤: {str(e)}')
        return symbol, []

def crawl_data(date: str, symbol: str) -> Optional[Tuple[list, list]]:
    """爬取指定日期的股票資料
    
    Args:
        date (str): 日期 (YYYYMMDD)
        symbol (str): 股票代號
        
    Returns:
        Optional[Tuple[list, list]]: (資料行, 欄位名稱) 或 None
    """
    try:
        # 請求參數驗證
        if not date or not symbol:
            logger.error('日期或股票代號為空')
            return None
            
        # 設置請求參數
        url = 'https://www.twse.com.tw/exchangeReport/MI_INDEX'
        params = {
            'response': 'csv',
            'date': date,
            'type': 'ALL'
        }
        
        logger.info(f'請求 {url} 日期: {date}')
        
        # 發送請求
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()  # 檢查 HTTP 錯誤
        
        # 內容驗證
        content_type = response.headers.get('Content-Type', '')
        if 'text/csv' not in content_type:
            logger.error(f'無效的內容類型: {content_type}')
            return None
            
        if len(response.text) < 100:
            logger.error('回應內容過短')
            return None
            
        # 資料清洗
        lines = [line for line in response.text.split('\n') 
                if len(line.split('",')) == 17 
                and not line.startswith('=')]
                
        if not lines:
            logger.error(f'{date}: 無有效的 CSV 資料行')
            return None
            
        # 解析 CSV
        try:
            df = pd.read_csv(StringIO("\n".join(lines)), header=0)
        except pd.errors.EmptyDataError:
            logger.error(f'{date}: CSV 解析失敗 - 空資料')
            return None
        except Exception as e:
            logger.error(f'{date}: CSV 解析失敗 - {str(e)}')
            return None
            
        # 資料清理
        df = df.drop(columns=['Unnamed: 16'], errors='ignore')
        
        # 過濾目標股票
        filter_df = df[df["證券代號"] == symbol]
        if filter_df.empty:
            logger.warning(f'{date}: 找不到股票代號 {symbol}')
            return None
            
        # 添加日期欄位
        filter_df = filter_df.copy()
        filter_df.insert(0, "日期", date)
        
        # 返回第一行資料和欄位名稱
        return list(filter_df.iloc[0]), list(filter_df.columns)
        
    except requests.exceptions.RequestException as e:
        logger.error(f'{date}: 請求失敗 - {str(e)}')
    except Exception as e:
        logger.error(f'{date}: 處理失敗 - {str(e)}')
        
    return None