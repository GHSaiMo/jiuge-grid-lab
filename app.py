import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import platform
import os
import hashlib
from datetime import datetime, date, timedelta
import random

# 导入数据获取模块
try:
    import akshare as ak
    AKSHARE_AVAILABLE = True
except ImportError:
    AKSHARE_AVAILABLE = False

try:
    import tushare as ts
    TUSHARE_AVAILABLE = True
except ImportError:
    TUSHARE_AVAILABLE = False

st.set_page_config(page_title="基金 / 股票动态网格回测系统", layout="wide")

# 检查数据源可用性
data_sources_status = []
if AKSHARE_AVAILABLE:
    data_sources_status.append("✅ AkShare")
else:
    data_sources_status.append("❌ AkShare")

if TUSHARE_AVAILABLE:
    data_sources_status.append("✅ Tushare")
else:
    data_sources_status.append("❌ Tushare")

if not AKSHARE_AVAILABLE and not TUSHARE_AVAILABLE:
    st.error("未找到数据源模块，请安装：pip install akshare tushare")
    st.stop()

# ==========================================
# 名称映射函数
# ==========================================
@st.cache_data
def load_name_mapping():
    """
    加载name.csv文件，建立代码到名称的映射
    """
    try:
        # 尝试多个可能的路径
        possible_paths = ["策略回测云端/name.csv", "name.csv", "./name.csv"]
        name_df = None
        
        for path in possible_paths:
            try:
                if os.path.exists(path):
                    # 确保Code列被读取为字符串，保留前导零
                    name_df = pd.read_csv(path, dtype={'Code': str})
                    break
            except:
                continue
        
        if name_df is None:
            return {}
            
        # 创建代码到名称的映射字典
        name_mapping = dict(zip(name_df['Code'].astype(str), name_df['Name']))
        return name_mapping
    except Exception as e:
        st.warning(f"无法加载name.csv文件: {str(e)}")
        return {}

def get_display_name(code):
    """
    根据代码获取显示名称，格式：代码
    """
    name_mapping = load_name_mapping()
    name = name_mapping.get(code, "未知")
    return f"{name} ({code})"

# ==========================================
# 股票代码检测和数据源选择
# ==========================================

# Tushare Token - 从 secrets 中读取
try:
    TUSHARE_TOKEN = st.secrets["tushare"]["token"]
except KeyError:
    st.error("❌ 未找到 Tushare Token，请在 Streamlit Cloud 中配置 secrets")
    TUSHARE_TOKEN = None

def is_stock_code(code):
    """
    检测6位数代码是否为A股股票代码
    
    Args:
        code (str): 6位数代码
    
    Returns:
        bool: True表示是股票代码，False表示可能是基金等其他代码
    """
    if not code or len(code) != 6 or not code.isdigit():
        return False
    
    # A股股票代码规则
    stock_prefixes = [
        '000',  # 深圳主板
        '002',  # 深圳中小板
        '300',  # 创业板
        '600',  # 上海主板
        '601',  # 上海主板
        '603',  # 上海主板
        '605',  # 上海主板
        '688',  # 科创板
    ]
    
    return any(code.startswith(prefix) for prefix in stock_prefixes)

def format_stock_code(code):
    """
    将6位数代码转换为Tushare标准格式
    
    Args:
        code (str): 6位数股票代码
    
    Returns:
        str: Tushare标准格式代码
    """
    if not code or len(code) != 6 or not code.isdigit():
        raise ValueError(f"代码格式错误: {code}，请输入6位数字")
    
    # 深圳交易所
    if code.startswith(('000', '002', '300')):
        return f"{code}.SZ"
    # 上海交易所
    elif code.startswith(('600', '601', '603', '605', '688')):
        return f"{code}.SH"
    else:
        raise ValueError(f"不支持的股票代码前缀: {code[:3]}")

@st.cache_data(ttl=3600*12)  # 缓存有效期12小时
def fetch_tushare_stock_data(symbol):
    """
    使用Tushare获取股票历史数据（近10年）
    
    Args:
        symbol (str): 6位数股票代码
    
    Returns:
        tuple: (df, status_info)
    """
    if not TUSHARE_AVAILABLE:
        return None, {"error": "Tushare模块未安装"}
    
    if not TUSHARE_TOKEN:
        return None, {"error": "Tushare Token未配置"}
    
    try:
        # 设置token
        ts.set_token(TUSHARE_TOKEN)
        pro = ts.pro_api()
        
        # 转换代码格式
        ts_code = format_stock_code(symbol)
        
        # 计算近10年的日期范围
        end_date = datetime.now().strftime('%Y%m%d')
        start_date = (datetime.now() - timedelta(days=365*10)).strftime('%Y%m%d')
        
        status_messages = [f"🔄 正在使用Tushare获取股票数据 {ts_code}..."]
        
        # 获取股票日线数据
        df = pro.daily(
            ts_code=ts_code,
            start_date=start_date,
            end_date=end_date
        )
        
        if df is None or df.empty:
            # 如果是.SZ失败，尝试.SH
            if ts_code.endswith('.SZ'):
                ts_code_sh = f"{symbol}.SH"
                status_messages.append(f"🔄 尝试上海交易所代码: {ts_code_sh}")
                df = pro.daily(
                    ts_code=ts_code_sh,
                    start_date=start_date,
                    end_date=end_date
                )
                ts_code = ts_code_sh
        
        if df is None or df.empty:
            raise ValueError(f"未找到股票代码 {symbol} 的数据")
        
        # 数据清洗，转换为通用格式
        df = df.copy()
        df = df.sort_values('trade_date').reset_index(drop=True)
        df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
        
        # 重命名列为通用格式
        df = df.rename(columns={
            'trade_date': 'Date',
            'open': 'Open',
            'close': 'Close', 
            'high': 'High',
            'low': 'Low'
        })
        
        # 只保留需要的列
        required_cols = ['Date', 'Open', 'High', 'Low', 'Close']
        df = df[required_cols].dropna()
        
        status_messages.append(f"✅ Tushare获取成功：{len(df)} 条股票数据")
        
        status_info = {
            "success": True,
            "data_source": f"Tushare Pro ({ts_code})",
            "data_count": len(df),
            "messages": status_messages,
            "ts_code": ts_code
        }
        
        return df, status_info
        
    except Exception as e:
        error_msg = f"Tushare获取失败: {str(e)}"
        return None, {"error": error_msg, "messages": status_messages + [f"❌ {error_msg}"]}

# ==========================================
# 数据获取函数 (带缓存)
# ==========================================

@st.cache_data(ttl=3600*12)  # 缓存有效期12小时，避免频繁请求
def fetch_online_data(symbol):
    """
    智能获取基金/股票历史数据
    - 先检测是否为股票代码，如果是则优先使用Tushare获取近10年数据
    - 如果不是股票或Tushare失败，则使用AkShare获取基金数据
    
    Args:
        symbol (str): 6位数字的基金/股票代码，如 '510880', '000001'
    
    Returns:
        tuple: (df, status_info) - 数据和状态信息
    """
    # 验证输入
    if not symbol or not symbol.isdigit() or len(symbol) != 6:
        return None, {"error": "请输入6位数字的基金/股票代码，如：510880"}
    
    df = None
    status_messages = []
    
    # 第一步：检测是否为股票代码
    if is_stock_code(symbol):
        status_messages.append(f"🔍 检测到股票代码: {symbol}")
        
        # 优先使用Tushare获取股票数据
        if TUSHARE_AVAILABLE:
            df, tushare_status = fetch_tushare_stock_data(symbol)
            if df is not None:
                # Tushare成功
                tushare_status["messages"] = status_messages + tushare_status["messages"]
                return df, tushare_status
            else:
                # Tushare失败，记录错误信息
                status_messages.extend(tushare_status.get("messages", []))
                status_messages.append("⚠️ Tushare获取失败，尝试AkShare...")
        else:
            status_messages.append("❌ Tushare模块未安装，尝试AkShare...")
    else:
        status_messages.append(f"🔍 检测到基金/ETF代码: {symbol}")
    
    # 第二步：使用AkShare获取数据（基金或股票备用）
    if AKSHARE_AVAILABLE:
        try:
            status_messages.append("🔄 正在使用AkShare获取数据...")
            # fund_etf_hist_em 接口来自东方财富，包含开高低收
            df = ak.fund_etf_hist_em(symbol=symbol, adjust="qfq")  # qfq=前复权
            
            if df is None or df.empty:
                raise ValueError(f"AkShare未找到代码 {symbol} 的数据")
            
            # 数据清洗，适配策略框架
            df = df.rename(columns={
                "日期": "Date",
                "开盘": "Open", 
                "收盘": "Close",
                "最高": "High",
                "最低": "Low"
            })
            
            status_messages.append("✅ AkShare获取成功")
            data_source = "AkShare"
            
        except Exception as e:
            status_messages.append(f"⚠️ AkShare获取失败: {str(e)}")
            df = None
    else:
        status_messages.append("❌ AkShare模块未安装")
    
    # 如果所有数据源都失败
    if df is None:
        error_msg = f"❌ 数据获取失败，代码: {symbol}"
        available_sources = []
        if TUSHARE_AVAILABLE:
            available_sources.append("Tushare")
        if AKSHARE_AVAILABLE:
            available_sources.append("AkShare")
        
        if not available_sources:
            error_msg += "\n建议：安装数据源模块 pip install akshare tushare"
        else:
            error_msg += f"\n已尝试数据源: {', '.join(available_sources)}"
            error_msg += "\n建议：检查网络连接或确认代码是否正确"
        
        return None, {"error": error_msg, "messages": status_messages}
    
    try:
        # 统一数据处理
        # 检查必要列是否存在
        required_cols = ['Date', 'Open', 'High', 'Low', 'Close']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"数据缺少必要列: {missing_cols}")
        
        # 转换日期格式
        df['Date'] = pd.to_datetime(df['Date'])
        
        # 只要需要的列
        df = df[required_cols]
        
        # 删除空值行
        df = df.dropna()
        
        # 排序
        df = df.sort_values('Date').reset_index(drop=True)
        
        # 数据清理：只保留最近10年的数据
        ten_years_ago = datetime.now() - timedelta(days=365*10)
        df = df[df['Date'] >= ten_years_ago].reset_index(drop=True)
        
        # 验证数据质量
        warning_msg = None
        if len(df) < 100:
            warning_msg = f"数据量较少，仅有 {len(df)} 条记录，可能影响回测效果"
        
        # 构建状态信息
        status_info = {
            "success": True,
            "data_source": data_source,
            "data_count": len(df),
            "messages": status_messages,
            "warning": warning_msg
        }
        
        return df, status_info
        
    except Exception as e:
        return None, {"error": f"数据处理失败: {str(e)}", "messages": status_messages}



# ==========================================
# lines: 需要插入的空行数
# ==========================================

def sidebar_space(lines=1):
    """
    lines: 需要插入的空行数
    """
    for _ in range(lines):
        st.sidebar.write("") 

# ==========================================
# 1. 核心账户类 (支持融资与理财)
# ==========================================

def calculate_metrics(daily_values):
    """计算最大回撤和夏普比率"""
    if len(daily_values) < 2: return 0, 0
    cum_max = daily_values.cummax()
    drawdown = (daily_values - cum_max) / cum_max
    max_dd = drawdown.min()
    returns = daily_values.pct_change().dropna()
    if returns.std() == 0:
        sharpe = 0
    else:
        sharpe = (returns.mean() * 252 - 0.02) / (returns.std() * np.sqrt(252))
    return max_dd, sharpe

class BacktestAccount:
    def __init__(self, initial_capital, fee_rate, min_fee, margin_rate=0.05, deposit_rate=0.015, max_position_ratio=2.0):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions = 0
        self.fee_rate = fee_rate
        self.min_fee = min_fee
        
        # 利率参数 (年化)
        self.margin_rate = margin_rate   # 融资利率 (Cash < 0)
        self.deposit_rate = deposit_rate # 存款利率 (Cash > 0)
        
        # 仓位控制参数
        self.max_position_ratio = max_position_ratio  # 最大仓位比例
        
        self.history = [] 
        self.trade_records = []

    def _calc_fee(self, amount):
        return max(self.min_fee, amount * self.fee_rate)

    def get_net_asset(self, price):
        return self.cash + (self.positions * price)

    def buy(self, date, price, volume):
        if volume <= 0: return False
        
        # 杠杆检查：如果买入后持仓市值 > 净资产 * max_position_ratio，则禁止交易
        current_net_asset = self.get_net_asset(price)
        # 预估买入后的持仓市值
        post_trade_pos_val = (self.positions + volume) * price
        
        if current_net_asset > 0 and (post_trade_pos_val / current_net_asset) > self.max_position_ratio:
            # st.toast(f"{date.date()} 触发杠杆风控：禁止开仓超过{self.max_position_ratio*100:.0f}%", icon="⚠️")
            return False

        amount = price * volume
        fee = self._calc_fee(amount)
        cost = amount + fee
        
        # 允许现金扣减为负数 (融资)
        self.cash -= cost
        self.positions += volume
        
        self._log_trade(date, '买入', price, volume, fee)
        return True

    def sell(self, date, price, volume):
        if volume <= 0: return False
        if self.positions >= volume:
            amount = price * volume
            fee = self._calc_fee(amount)
            revenue = amount - fee
            
            self.cash += revenue
            self.positions -= volume
            
            self._log_trade(date, '卖出', price, volume, fee)
            return True
        return False
    
    def _log_trade(self, date, action, price, vol, fee):
        self.trade_records.append({
            '日期': date.strftime('%Y-%m-%d'),
            '操作': action,
            '价格': price,
            '数量': vol,
            '手续费': round(fee, 2),
            '持仓市值': round(self.positions * price, 2),
            '现金/负债': round(self.cash, 2)
        })

    def settle_daily_interest(self):
        """每日收盘后结算利息"""
        if self.cash > 0:
            # 获得存款利息
            interest = self.cash * (self.deposit_rate / 365)
            self.cash += interest
        elif self.cash < 0:
            # 支付融资利息 (负数 += 负数，债务增加)
            interest_cost = self.cash * (self.margin_rate / 365)
            self.cash += interest_cost

    def record_daily(self, date, price):
        # 1. 先结算当天的利息
        self.settle_daily_interest()
        
        # 2. 记录净值
        total_val = self.get_net_asset(price)
        self.history.append({
            'date': date,
            'strategy_value': total_val,
            'price': price,
            'cash': self.cash,
            'pos_val': self.positions * price
        })

def calc_benchmark(df, capital):
    """
    基准：100% 满仓持有，不择时，不融资。
    """
    first_price = df['Close'].iloc[0]
    bench_shares = int(capital / first_price / 100) * 100
    # 剩余的一点零钱
    residual_cash = capital - (bench_shares * first_price)
    
    # 基准也简单算一点点零钱的利息，或者忽略。这里为了纯粹对比Beta，只算股价变动
    bench_series = df['Close'] * bench_shares + residual_cash
    return bench_series

def plot_results(res_df, bench_series, strategy_name, symbol_code=None):
    # 设置中文字体
    import matplotlib.font_manager as fm
    
    # 获取字体路径
    font_path = os.path.join(os.path.dirname(__file__), "SourceHanSansSC-Regular.otf")
    
    # 添加字体到管理器
    fm.fontManager.addfont(font_path)
    
    # 设置全局字体为该字体的名称
    prop = fm.FontProperties(fname=font_path)
    plt.rcParams['font.family'] = prop.get_name()  # 自动获取该字体的内部名称
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    
    strat_dd, strat_sharpe = calculate_metrics(res_df['strategy_value'])
    bench_dd, bench_sharpe = calculate_metrics(bench_series)
    final_val = res_df['strategy_value'].iloc[-1]
    
    init_cap = bench_series.iloc[0]
    strat_ret = (final_val - init_cap) / init_cap * 100
    bench_ret = (bench_series.iloc[-1] - init_cap) / init_cap * 100
    
    # 计算差值
    ret_diff = strat_ret - bench_ret
    dd_diff = (strat_dd - bench_dd) * 100
    sharpe_diff = strat_sharpe - bench_sharpe
    
    c1, c2, c3, c4 = st.columns(4)
    
    c1.metric("策略收益率", f"{strat_ret:.2f}%", delta=f"{ret_diff:.2f}%", delta_color="inverse")
    c2.metric("最大回撤", f"{strat_dd*100:.2f}%", delta=f"{dd_diff:.2f}%", delta_color="inverse")
    c3.metric("夏普比率", f"{strat_sharpe:.2f}", delta=f"{sharpe_diff:.2f}", delta_color="inverse")
    
    bench_final_val = bench_series.iloc[-1]
    asset_diff = final_val - bench_final_val
    c4.metric("策略最终资产", f"{final_val:,.0f}", delta=f"{asset_diff:,.0f}", delta_color="inverse")

    # 图表1: 净值对比
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [1, 1]})
    ax1.plot(res_df['date'], res_df['strategy_value'], label='策略净值', color='#d62728', linewidth=1.5)
    ax1.plot(res_df['date'], bench_series, label='基准(满仓持有)', color='gray', linestyle='--', alpha=0.6)
  
    # 在图表上方添加基金/股票名称的总表头
    if symbol_code:
        display_name = get_display_name(symbol_code)
        st.markdown(f"<h3 style='text-align: center;'>{display_name} {strategy_name}策略回测结果</h3>", unsafe_allow_html=True)

    # 设置图表标题
    chart_title = "净值对比"
    ax1.set_title(chart_title)
    ax1.legend()
    ax1.grid(True, alpha=0.2)
    ax1.set_xlabel("日期")
    ax1.set_ylabel("净值")

    # 图表2: 仓位/杠杆监控
    # 计算实际杠杆 = 持仓市值 / 净资产价值
    leverage_ratio = res_df['pos_val'] / res_df['strategy_value']
    ax2.plot(res_df['date'], leverage_ratio * 100, label='实际仓位%', color='#1f77b4', linewidth=1)
    ax2.axhline(y=100, color='gray', linestyle=':', alpha=0.5, label='100%本金线')
    
    # 融资区域: 当仓位 > 100% 时，在100%线和仓位线之间填充红色
    ax2.fill_between(res_df['date'], 100, leverage_ratio * 100, where=(leverage_ratio>1), color='red', alpha=0.1, label='融资区域')
    
    # 现金管理区域: 当仓位 < 100% 时，在仓位线和100%线之间填充绿色
    ax2.fill_between(res_df['date'], leverage_ratio * 100, 100, where=(leverage_ratio<=1), color='green', alpha=0.1, label='现金管理区域')
    
    ax2.set_title("历史仓位变化(%)")
    ax2.set_xlabel("日期")
    ax2.set_ylabel("仓位比例")
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.2)
    
    plt.tight_layout()
    st.pyplot(fig)

# ==========================================
# 2. 策略逻辑 (适配 200% 仓位与融资)
# ==========================================

def run_ma_strategy(df, capital, fee, min_fee, m_rate, d_rate, initial_pct, ma_period, deviation, trade_pct, max_pos_ratio):
    acc = BacktestAccount(capital, fee, min_fee, m_rate, d_rate, max_pos_ratio)
    for i in range(len(df)):
        price = df['Close'].iloc[i]; date = df['Date'].iloc[i]
        
        if i == 0:
            shares = int((capital * initial_pct) / price / 100) * 100
            acc.buy(date, price, shares); acc.record_daily(date, price); continue
            
        if i >= ma_period:
            ma = df['Close'].iloc[i-ma_period:i].mean()
            ratio = price / ma
            
            # 计算交易股数：基于当前净资产的百分比
            net_asset = acc.get_net_asset(price)
            trade_amount = net_asset * trade_pct
            trade_shares = int(trade_amount / price / 100) * 100
            
            if ratio < (1 - deviation): acc.buy(date, price, trade_shares)
            elif ratio > (1 + deviation): acc.sell(date, price, trade_shares)
        acc.record_daily(date, price)
    return acc

def run_bollinger_strategy(df, capital, fee, min_fee, m_rate, d_rate, initial_pct, window, k_dev, trade_pct, max_pos_ratio):
    """
    布林通道策略：
    - 超过上轨卖出
    - 跌破下轨买入
    - 使用单笔交易百分比
    """
    acc = BacktestAccount(capital, fee, min_fee, m_rate, d_rate, max_pos_ratio)
    
    for i in range(len(df)):
        price = df['Close'].iloc[i]
        date = df['Date'].iloc[i]
        
        # 初始建仓
        if i == 0:
            shares = int((capital * initial_pct) / price / 100) * 100
            acc.buy(date, price, shares)
            acc.record_daily(date, price)
            continue
        
        # 等待足够的数据计算布林带
        if i >= window:
            # 计算布林带
            hist = df['Close'].iloc[i-window:i]
            ma = hist.mean()
            std = hist.std()
            upper_band = ma + k_dev * std
            lower_band = ma - k_dev * std
            
            # 计算交易股数：基于当前净资产的百分比
            net_asset = acc.get_net_asset(price)
            trade_amount = net_asset * trade_pct
            trade_shares = int(trade_amount / price / 100) * 100
            
            # 交易逻辑
            if price > upper_band:  # 超过上轨，卖出
                acc.sell(date, price, trade_shares)
            elif price < lower_band:  # 跌破下轨，买入
                acc.buy(date, price, trade_shares)
        
        acc.record_daily(date, price)
    
    return acc

# ==========================================
# 3. 自动更新辅助函数
# ==========================================

def get_param_hash(params):
    """生成参数的哈希值，用于检测参数变化"""
    param_str = str(sorted(params.items()))
    return hashlib.md5(param_str.encode()).hexdigest()

def auto_run_strategy(strategy_func, params, strategy_name, tab_key):
    """自动运行策略并缓存结果"""
    param_hash = get_param_hash(params)
    cache_key = f"{tab_key}_result"
    hash_key = f"{tab_key}_hash"
    
    # 检查参数是否变化
    if hash_key not in st.session_state or st.session_state[hash_key] != param_hash:
        # 参数变化，重新计算
        with st.spinner(f"正在计算{strategy_name}策略..."):
            result = strategy_func(**params)
            st.session_state[cache_key] = result
            st.session_state[hash_key] = param_hash
    
    return st.session_state.get(cache_key)

# ==========================================
# 4. Streamlit UI
# ==========================================

st.sidebar.header("📊 数据获取")

# 检查数据源可用性并显示状态
if not AKSHARE_AVAILABLE and not TUSHARE_AVAILABLE:
    st.error("❌ 未找到数据源模块")
    st.info("请安装数据源模块：pip install akshare tushare")
    st.stop()

# AkShare数据获取 - 使用两列布局
col1, col2 = st.sidebar.columns([1, 1])
with col1:
    symbol = st.text_input("输入ETF/股票代码", value="510880", help="例如: 510880", label_visibility="collapsed", placeholder="ETF/股票代码")
with col2:
    get_data_btn = st.button("获取数据", type="secondary", use_container_width=True)

if get_data_btn:
    if symbol:
        # 检查是否是默认的510880代码
        if symbol == "510880":
            local_csv_path = "策略回测云端/510880.csv"
            if os.path.exists(local_csv_path):
                st.sidebar.info("💡 检测到本地已有510880数据，如需更新请继续")
        
        with st.spinner(f"正在从网络下载 {symbol} 历史数据..."):
            online_df, status_info = fetch_online_data(symbol)
            if online_df is not None and not online_df.empty:
                raw_df = online_df
                st.sidebar.success(f"✅ 网络获取成功：{len(raw_df)} 条数据")
                
                # 1. 保存数据到 session_state
                st.session_state['akshare_data'] = raw_df
                st.session_state['akshare_symbol'] = symbol
                st.session_state['data_status'] = status_info
                
                # =========================================================
                # [关键修复]：重置日期范围
                # 获取新数据的起止时间
                min_d, max_d = raw_df['Date'].iloc[0].date(), raw_df['Date'].iloc[-1].date()
                # 强制将 date_range 更新为新数据的全范围，防止越界报错
                st.session_state['date_range'] = (min_d, max_d)
                # =========================================================

                # 如果获取的是510880数据，询问是否保存到本地
                if symbol == "510880":
                    st.sidebar.info("💾 已更新510880数据，可考虑保存到本地CSV文件")
                
                # 强制刷新页面以应用新的日期范围
                st.rerun() 
                
            else:
                st.sidebar.error("❌ 网络获取失败，请检查代码是否正确")
                if 'error' in status_info:
                    st.session_state['data_status'] = status_info
    else:
        st.sidebar.warning("⚠️ 请输入代码")

# 添加读取本地CSV文件的函数
@st.cache_data
def load_local_csv():
    """
    读取本地510880.csv文件作为默认数据，尝试多个可能的路径
    """
    try:
        # 尝试多个可能的路径
        possible_paths = ["策略回测云端/510880.csv", "510880.csv", "./510880.csv"]
        df = None
        used_path = None
        
        for path in possible_paths:
            try:
                if os.path.exists(path):
                    df = pd.read_csv(path)
                    used_path = path
                    break
            except:
                continue
        
        if df is None:
            return None, {"error": "未找到510880.csv文件，已尝试路径: " + ", ".join(possible_paths)}
            
        # 确保日期列格式正确
        df['Date'] = pd.to_datetime(df['Date'])
        # 检查必要的列是否存在
        required_cols = ['Date', 'Open', 'High', 'Low', 'Close']
        if all(col in df.columns for col in required_cols):
            df = df[required_cols].dropna().sort_values('Date').reset_index(drop=True)
            return df, {
                "success": True,
                "data_source": "本地CSV文件",
                "data_count": len(df),
                "messages": [f"✅ 成功读取本地510880.csv文件"],
                "file_path": used_path
            }
        else:
            return None, {"error": f"CSV文件缺少必要列: {required_cols}"}
    except Exception as e:
        return None, {"error": f"读取CSV文件失败: {str(e)}"}

# 初始化数据
raw_df = pd.DataFrame()

# 如果session_state中有数据，使用它
if 'akshare_data' in st.session_state:
    raw_df = st.session_state['akshare_data']
else:
    # 默认先尝试读取本地510880.csv文件
    with st.spinner("正在加载默认数据..."):
        raw_df, status_info = load_local_csv()
        
        if raw_df is not None and not raw_df.empty:
            # 成功读取本地文件
            st.session_state['akshare_data'] = raw_df
            st.session_state['akshare_symbol'] = "510880"
            st.session_state['data_status'] = status_info
        else:
            # 本地文件读取失败，尝试网络获取
            st.info("本地文件不存在，正在从网络获取510880数据...")
            raw_df, status_info = fetch_online_data("510880")
            if raw_df is not None and not raw_df.empty:
                st.session_state['akshare_data'] = raw_df
                st.session_state['akshare_symbol'] = "510880"
                st.session_state['data_status'] = status_info
            else:
                st.error("无法获取默认数据，请手动输入代码")
                if 'error' in status_info:
                    st.session_state['data_status'] = status_info
                st.stop()

# 显示数据获取状态信息（在侧边栏）
if 'data_status' in st.session_state:
    status_info = st.session_state['data_status']
    
    if 'success' in status_info and status_info['success']:
        # 成功获取数据的情况
        current_symbol = st.session_state.get('akshare_symbol', 'Unknown')
        data_source = status_info['data_source']
        
        # 获取显示名称
        display_name = get_display_name(current_symbol)
        
        # 根据数据来源设置不同的图标
        if "本地CSV" in data_source:
            icon = "💾"
            source_color = "🟢"
        elif "Tushare" in data_source:
            icon = "🌐"
            source_color = "🔵"
        else:
            icon = "🌐"
            source_color = "🟠"
        
        latest_status = f"{icon} 数据来源: {data_source} | 数据量: {status_info['data_count']} 条"
        
        with st.sidebar.expander(f"{source_color} {display_name}", expanded=False):
            st.success(latest_status)
            
            # 显示获取过程的详细信息
            if 'messages' in status_info:
                for msg in status_info['messages']:
                    if "🔄" in msg:
                        st.info(msg)
                    elif "✅" in msg:
                        st.success(msg)
                    elif "⚠️" in msg:
                        st.warning(msg)
            
            # 显示警告信息（如果有）
            if 'warning' in status_info and status_info['warning']:
                st.warning(status_info['warning'])
                
            # 如果是本地数据，提供刷新选项
            if "本地CSV" in data_source:
                st.info("💡 如需获取最新数据，请点击上方'获取数据'按钮")
                
    elif 'error' in status_info:
        # 获取数据失败的情况
        with st.sidebar.expander("❌ 数据获取失败", expanded=True):
            st.error(status_info['error'])
            
            # 显示尝试过程
            if 'messages' in status_info:
                for msg in status_info['messages']:
                    if "🔄" in msg:
                        st.info(msg)
                    elif "⚠️" in msg:
                        st.warning(msg)
else:
    # 如果没有状态信息，显示简单的当前数据信息
    current_symbol = st.session_state.get('akshare_symbol', 'Unknown')
    display_name = get_display_name(current_symbol) if current_symbol != 'Unknown' else current_symbol
    st.sidebar.info(f"📊 当前数据: {display_name}")

# 确保有数据才继续
if raw_df.empty:
    st.info("请输入代码并点击获取数据")
    st.stop()

st.sidebar.markdown(
    """<div style="height: 20px;"></div>""", 
    unsafe_allow_html=True
)

raw_df.columns = [c.capitalize() for c in raw_df.columns]
raw_df['Date'] = pd.to_datetime(raw_df['Date'])
raw_df = raw_df.sort_values('Date').reset_index(drop=True)

min_date, max_date = raw_df['Date'].iloc[0].date(), raw_df['Date'].iloc[-1].date()

st.sidebar.header("📅 回测区间")

# 初始化session_state中的日期范围
if 'date_range' not in st.session_state:
    st.session_state['date_range'] = (min_date, max_date)

date_range = st.sidebar.date_input("选择时间", value=st.session_state['date_range'], min_value=min_date, max_value=max_date)

# 快速时间区间按钮
st.sidebar.markdown("**快速选择：**")
col1, col2, col3 = st.sidebar.columns(3)

with col1:
    if st.button("近5年", key="5y"):
        five_years_ago = max_date - timedelta(days=365*5)
        start_date_5y = max(min_date, five_years_ago)
        st.session_state['date_range'] = (start_date_5y, max_date)
        st.rerun()

with col2:
    if st.button("近3年", key="3y"):
        three_years_ago = max_date - timedelta(days=365*3)
        start_date_3y = max(min_date, three_years_ago)
        st.session_state['date_range'] = (start_date_3y, max_date)
        st.rerun()

with col3:
    if st.button("近2年", key="2y"):
        two_years_ago = max_date - timedelta(days=365*2)
        start_date_2y = max(min_date, two_years_ago)
        st.session_state['date_range'] = (start_date_2y, max_date)
        st.rerun()

# 第二排按钮
col4, col5 = st.sidebar.columns(2)

with col4:
    if st.button("近1年", key="1y"):
        one_year_ago = max_date - timedelta(days=365)
        start_date_1y = max(min_date, one_year_ago)
        st.session_state['date_range'] = (start_date_1y, max_date)
        st.rerun()

with col5:
    if st.button("随机1-3年", key="random", help="用于测试策略有效性，防止过拟合"):
        # 计算数据总时长（年）
        total_days = (max_date - min_date).days
        total_years = total_days / 365.25
        
        # 根据数据长度确定随机测试的年数
        if total_years >= 3:
            # 数据长度>=3年，随机选择1-3年
            test_years = random.uniform(1, 3)
        elif total_years >= 1:
            # 数据长度1-3年，随机选择1年到数据长度
            test_years = random.uniform(1, total_years)
        else:
            # 数据长度<1年，使用全部数据
            test_years = total_years
        
        # 计算测试天数
        test_days = int(test_years * 365.25)
        
        # 随机选择起始点
        max_start_days = total_days - test_days
        if max_start_days > 0:
            random_start_days = random.randint(0, max_start_days)
            random_start_date = min_date + timedelta(days=random_start_days)
            random_end_date = random_start_date + timedelta(days=test_days)
        else:
            # 如果测试期间>=数据长度，使用全部数据
            random_start_date = min_date
            random_end_date = max_date
        
        st.session_state['date_range'] = (random_start_date, random_end_date)
        
        # 显示随机选择的信息
        st.sidebar.info(f"🎲 随机选择: {test_years:.1f}年 ({random_start_date} 至 {random_end_date})")
        st.rerun()

if len(date_range) != 2:
    st.stop()
start_date, end_date = date_range

# 更新session_state中的日期范围
st.session_state['date_range'] = date_range

mask = (raw_df['Date'] >= pd.to_datetime(start_date)) & (raw_df['Date'] <= pd.to_datetime(end_date))
df = raw_df.loc[mask].reset_index(drop=True)

if df.empty or len(df) < 50:
    st.error("数据不足")
    st.stop()
    
sidebar_space(2) # 插入两个空行
st.sidebar.header("🎯 仓位设置")
max_position_pct = st.sidebar.slider("最大仓位 (%)", 100, 200, 150, 5, help="允许的最大仓位百分比，超过100%表示使用融资") / 100

sidebar_space(2) # 插入两个空行
st.sidebar.header("⚙️ 资金设置")
init_capital = st.sidebar.number_input("初始资金", value=300000, min_value=10000, step=10000)
fee_rate = st.sidebar.number_input("费率", 0.00025, format="%.5f")
min_fee = st.sidebar.number_input("最低手续费", 5.0)

sidebar_space(2) # 插入两个空行
st.sidebar.header("💸 利率设置")
margin_rate = st.sidebar.slider("融资年化利率 (负债)", 0.0, 8.0, 4.0, 0.1, help="当现金为负时，需要支付的年化利息") / 100
deposit_rate = st.sidebar.slider("存款年化利率 (现金)", 0.0, 4.0, 1.5, 0.1, help="当现金为正时，获得的年化理财收益") / 100

sidebar_space(2) # 插入两个空行
st.sidebar.header("👨‍💻 关于作者")
st.sidebar.subheader("九哥")
st.sidebar.markdown(
    """<small>版本：v1.0.0 &nbsp;&nbsp;|&nbsp;&nbsp; 更新时间：2025-12-13</small>""", 
    unsafe_allow_html=True
)

sidebar_space(2) # 插入两个空行
st.sidebar.markdown("### 💝 支持作者")
st.sidebar.markdown(
    """
    <a href="https://github.com/GHSaiMo/jiuge-grid-lab" target="_blank" style="
        display: inline-block;
        padding: 8px 12px;
        background-color: #28a745;
        color: white;
        text-decoration: none;
        border-radius: 6px;
        font-size: 12px;
        text-align: center;
        width: 100%;
        box-sizing: border-box;
        margin-top: 10px;
        font-weight: bold;
    ">⭐ 给项目点个 Star 支持！</a>
    """,
    unsafe_allow_html=True
)
st.sidebar.markdown(
    """
    <a href="https://tushare.pro/register?reg=923874" target="_blank" style="
        display: inline-block;
        padding: 10px 15px;
        background-color: #ff6b6b;
        color: white;
        text-decoration: none;
        border-radius: 8px;
        font-size: 13px;
        text-align: center;
        width: 100%;
        box-sizing: border-box;
        margin-bottom: 10px;
        font-weight: bold;
    ">🎁 注册 Tushare 数据源</a>
    """,
    unsafe_allow_html=True
)

st.title("基金 / 股票动态网格回测系统")

st.caption("""⚠️ **免责声明**：本系统仅用于历史数据回测与策略验证，不构成任何投资建议。股市有风险，投资需谨慎。""")

tab1, tab2 = st.tabs(["📉 MA回归", "🌊 布林通道"])

# --- Tab 1: MA回归 ---
with tab1:
    st.markdown("#### 📉 MA 回归策略")
    c1,c2,c3,c4 = st.columns(4)
    p1_base = c1.slider("初始仓位", 0.0, max_position_pct, 1.0, 0.1, key="t1_b",help="1.0代表100%仓位")
    p1_ma = c2.number_input("MA周期", value=20, min_value=5, max_value=250, key="t1_p",help="均线的周期")
    p1_dev = c3.slider("偏离%", 1.0, 10.0, 2.5, key="t1_d", help="偏离均线的幅度")/100  # 默认2.5%
    p1_pct = c4.slider("单笔交易%", 1.0, 30.0, 15.0, 0.5, key="t1_pct", help="每次交易占当前净资产的百分比")/100
    
    # 自动运行策略
    ma_params = {
        'df': df, 'capital': init_capital, 'fee': fee_rate, 'min_fee': min_fee,
        'm_rate': margin_rate, 'd_rate': deposit_rate, 'initial_pct': p1_base,
        'ma_period': p1_ma, 'deviation': p1_dev, 'trade_pct': p1_pct, 'max_pos_ratio': max_position_pct
    }
    
    acc = auto_run_strategy(run_ma_strategy, ma_params, "MA回归", "ma")
    if acc:
        bench = calc_benchmark(df, init_capital)
        current_symbol = st.session_state.get('akshare_symbol', None)
        plot_results(pd.DataFrame(acc.history), bench, "MA回归", current_symbol)
        with st.expander("查看详细交易单"):
            st.dataframe(pd.DataFrame(acc.trade_records))

# --- Tab 2: 布林通道 ---
with tab2:
    st.markdown("#### 🌊 布林通道策略")
    c1,c2,c3,c4 = st.columns(4)
    p2_base = c1.slider("初始仓位", 0.0, max_position_pct, 1.0, 0.1, key="t2_b",help="1.0代表100%仓位")
    p2_win = c2.number_input("布林周期", value=26, min_value=5, max_value=250, key="t2_w",help="布林线的周期")  # 默认26，范围10-120
    p2_k = c3.slider("标准差倍数", 0.5, 3.0, 1.5, key="t2_k", help="布林线通道的标准差倍数")  # 默认2
    p2_pct = c4.slider("单笔交易%", 1.0, 30.0, 15.0, 0.5, key="t2_pct", help="每次交易占当前净资产的百分比")/100
    
    # 自动运行策略
    bollinger_params = {
        'df': df, 'capital': init_capital, 'fee': fee_rate, 'min_fee': min_fee,
        'm_rate': margin_rate, 'd_rate': deposit_rate, 'initial_pct': p2_base,
        'window': p2_win, 'k_dev': p2_k, 'trade_pct': p2_pct, 'max_pos_ratio': max_position_pct
    }
    
    acc = auto_run_strategy(run_bollinger_strategy, bollinger_params, "布林通道", "bollinger")
    if acc:
        bench = calc_benchmark(df, init_capital)
        current_symbol = st.session_state.get('akshare_symbol', None)
        plot_results(pd.DataFrame(acc.history), bench, "布林通道", current_symbol)
        with st.expander("查看详细交易单"):
            st.dataframe(pd.DataFrame(acc.trade_records))