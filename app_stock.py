import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
import re
import os
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="我的股市助手", page_icon="📈", layout="wide")

st.title("📈 我的股市助手")
st.caption("自选股票 · 多指标评分 · AI舆情分析 · 人话解读")


# ============================================================
# 环境变量读取（部署时用）
# ============================================================
def get_config(key, default=""):
    """优先从环境变量读，读不到再用默认值"""
    return os.environ.get(key, default)


# ============================================================
# 数据源 1：股价（akshare）
# ============================================================
@st.cache_data(ttl=3600)
def get_stock_data_akshare(code, days=500):
    try:
        import akshare as ak
        
        if code.startswith('6'):
            symbol = f"sh{code}"
        else:
            symbol = f"sz{code}"
        
        start_date = (datetime.now() - timedelta(days=days)).strftime("%Y%m%d")
        
        df = ak.stock_zh_a_daily(symbol=symbol, start_date=start_date, adjust="qfq")
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
        info_dict = {}
        try:
            info = ak.stock_individual_info_em(symbol=code)
            info_dict = dict(zip(info['item'], info['value']))
        except:
            pass
        
        return {"success": True, "df": df, "info": info_dict, "source": "akshare"}
    except Exception as e:
        return {"success": False, "error": str(e), "source": "akshare"}


@st.cache_data(ttl=3600)
def get_stock_data_sina(code):
    try:
        import akshare as ak
        
        if code.startswith('6'):
            symbol = f"sh{code}"
        else:
            symbol = f"sz{code}"
        
        df = ak.stock_zh_a_daily(symbol=symbol, adjust="qfq")
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        df = df.tail(500)
        
        return {"success": True, "df": df, "info": {}, "source": "sina"}
    except Exception as e:
        return {"success": False, "error": str(e), "source": "sina"}


def get_stock_data(code, days=500):
    result = get_stock_data_akshare(code, days)
    if result["success"]:
        return result
    result2 = get_stock_data_sina(code)
    if result2["success"]:
        return result2
    return {"success": False, "error": "所有数据源均失败", "source": "none"}


# ============================================================
# 新闻源
# ============================================================
@st.cache_data(ttl=1800)
def get_news_eastmoney(keyword):
    try:
        import requests
        import json
        
        url = "https://search-api-web.eastmoney.com/search/jsonp"
        param_json = (
            '{"uid":"","keyword":"' + keyword + '",'
            '"type":["cmsArticleWebOld"],'
            '"client":"web","clientType":"web","clientVersion":"curr",'
            '"param":{"cmsArticleWebOld":{'
            '"searchScope":"default","sort":"default",'
            '"pageIndex":1,"pageSize":10,'
            '"preTag":"<em>","postTag":"</em>"}}}'
        )
        params = {"cb": "jQuery", "param": param_json, "_": "1700000000000"}
        headers = {"User-Agent": "Mozilla/5.0"}
        
        response = requests.get(url, params=params, headers=headers, timeout=10)
        text = response.text
        json_str = text[text.index('(')+1:text.rindex(')')]
        data = json.loads(json_str)
        
        articles = data.get('result', {}).get('cmsArticleWebOld', [])
        if not articles:
            return []
        
        return [{
            '标题': a.get('title', '').replace('<em>', '').replace('</em>', ''),
            '摘要': a.get('content', '').replace('<em>', '').replace('</em>', '')[:300],
            '链接': a.get('url', ''),
            '时间': a.get('date', ''),
            '来源': '东方财富'
        } for a in articles]
    except:
        return []


@st.cache_data(ttl=1800)
def get_news_sina(keyword):
    try:
        import requests
        url = "https://feed.mix.sina.com.cn/api/roll/get"
        params = {"pageid": "153", "lid": "2516", "k": keyword, "num": "10", "page": "1"}
        headers = {"User-Agent": "Mozilla/5.0"}
        
        response = requests.get(url, params=params, headers=headers, timeout=10)
        data = response.json()
        
        items = data.get('result', {}).get('data', [])
        if not items:
            return []
        
        return [{
            '标题': item.get('title', ''),
            '摘要': item.get('intro', '')[:300],
            '链接': item.get('url', ''),
            '时间': item.get('ctime', ''),
            '来源': '新浪财经'
        } for item in items]
    except:
        return []


def get_news_multi(keyword, max_items=8):
    all_news = []
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(get_news_eastmoney, keyword),
            executor.submit(get_news_sina, keyword)
        ]
        for f in futures:
            try:
                result = f.result(timeout=15)
                all_news.extend(result)
            except:
                pass
    
    if not all_news:
        return None
    
    seen = set()
    unique = []
    for n in all_news:
        if n['标题'] and n['标题'] not in seen:
            seen.add(n['标题'])
            unique.append(n)
    
    return pd.DataFrame(unique[:max_items])


# ============================================================
# AI 批量分析新闻
# ============================================================
@st.cache_data(ttl=3600, show_spinner=False)
def analyze_news_batch_ai(titles_summaries, api_key, endpoint_id):
    if not titles_summaries:
        return []
    
    try:
        from volcenginesdkarkruntime import Ark
        
        client = Ark(
            base_url="https://ark.cn-beijing.volces.com/api/v3",
            api_key=api_key,
        )
        
        news_text = ""
        for i, (title, summary) in enumerate(titles_summaries, 1):
            news_text += f"{i}. 标题：{title}\n摘要：{summary[:100]}\n\n"
        
        prompt = f"""请分析以下{len(titles_summaries)}条A股财经新闻，逐条判断利好/利空/中性。

{news_text}

请严格按以下格式回答，每行一条，不要有多余内容：
1. 利好 | 简短理由
2. 利空 | 简短理由
3. 中性 | 简短理由
...

要求：
- 每条只输出一行
- 序号从1开始，与上面新闻对应
- 理由不超过15字
- 只输出"利好"、"利空"或"中性"，不要其他词"""
        
        response = client.chat.completions.create(
            model=endpoint_id,
            messages=[
                {"role": "system", "content": "你是资深A股分析师，严格按用户要求的格式输出。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=800,
        )
        
        result = response.choices[0].message.content.strip()
        lines = result.split('\n')
        parsed = {}
        for line in lines:
            match = re.match(r'^\s*(\d+)[.、]\s*(利好|利空|中性)\s*[|｜]\s*(.+)$', line.strip())
            if match:
                idx = int(match.group(1))
                sentiment = match.group(2)
                reason = match.group(3).strip()
                parsed[idx] = (sentiment, reason)
        
        results = []
        for i in range(1, len(titles_summaries) + 1):
            if i in parsed:
                sentiment, reason = parsed[i]
                emoji = "🟢" if sentiment == "利好" else ("🔴" if sentiment == "利空" else "⚪")
                results.append(f"{emoji} {sentiment} | {reason}")
            else:
                title, summary = titles_summaries[i-1]
                results.append(analyze_news_keyword(title, summary))
        
        return results
    except Exception as e:
        return [analyze_news_keyword(t, s) for t, s in titles_summaries]


def analyze_news_keyword(title, summary):
    text = f"{title} {summary}"
    positive_words = ['涨', '利好', '增长', '突破', '买入', '上调', '涨停', '新高', '盈利', '超预期', '大涨', '飙升']
    negative_words = ['跌', '利空', '下滑', '跌破', '卖出', '下调', '跌停', '新低', '亏损', '不及预期', '大跌', '暴跌']
    
    pos = sum(1 for w in positive_words if w in text)
    neg = sum(1 for w in negative_words if w in text)
    
    if pos > neg:
        return "🟢 利好 | 关键词匹配"
    elif neg > pos:
        return "🔴 利空 | 关键词匹配"
    else:
        return "⚪ 中性 | 关键词匹配"


# ============================================================
# AI 生成综合建议
# ============================================================
@st.cache_data(ttl=3600, show_spinner=False)
def generate_advice_ai(score, trend, risk, rsi, news_summary, api_key, endpoint_id):
    try:
        from volcenginesdkarkruntime import Ark
        
        client = Ark(
            base_url="https://ark.cn-beijing.volces.com/api/v3",
            api_key=api_key,
        )
        
        prompt = f"""基于以下信息，用一句话（不超过30字）给出投资建议：

- 综合评分：{score}/100
- 趋势：{trend}
- 风险：{risk}
- RSI：{rsi:.1f}
- 新闻情绪：{news_summary}

只输出一句话建议，不要解释，不要分点。"""
        
        response = client.chat.completions.create(
            model=endpoint_id,
            messages=[
                {"role": "system", "content": "你是资深A股分析师，回答简洁专业。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=80,
        )
        
        return response.choices[0].message.content.strip()
    except:
        return generate_advice_rule(score, trend, risk)


def generate_advice_rule(score, trend, risk):
    if score >= 70:
        return "可以考虑关注，仓位建议不超过总资产的30%"
    elif score >= 50:
        return "建议观望，等待更明确的信号"
    else:
        return "建议谨慎，短期风险较大"


# ============================================================
# 技术指标
# ============================================================
def calc_indicators(df):
    close = df['close']
    
    ma5 = close.rolling(5).mean().iloc[-1]
    ma20 = close.rolling(20).mean().iloc[-1]
    ma60 = close.rolling(60).mean().iloc[-1]
    current = close.iloc[-1]
    
    if current > ma5 > ma20 > ma60:
        trend, trend_score = "强势上涨", 90
    elif current > ma20 > ma60:
        trend, trend_score = "温和上涨", 70
    elif current < ma5 < ma20 < ma60:
        trend, trend_score = "强势下跌", 10
    else:
        trend, trend_score = "震荡整理", 50
    
    returns = close.pct_change().dropna()
    volatility = returns.tail(60).std() * np.sqrt(252)
    
    if volatility < 0.20:
        risk, risk_score = "低", 80
    elif volatility < 0.35:
        risk, risk_score = "中等", 60
    else:
        risk, risk_score = "高", 30
    
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs)).iloc[-1]
    
    if rsi > 70:
        rsi_signal = "超买"
    elif rsi < 30:
        rsi_signal = "超卖"
    else:
        rsi_signal = "正常"
    
    change_1d = (close.iloc[-1] / close.iloc[-2] - 1) * 100 if len(close) >= 2 else 0
    change_3d = (close.iloc[-1] / close.iloc[-4] - 1) * 100 if len(close) >= 4 else 0
    change_5d = (close.iloc[-1] / close.iloc[-6] - 1) * 100 if len(close) >= 6 else 0
    change_20d = (close.iloc[-1] / close.iloc[-21] - 1) * 100 if len(close) >= 21 else 0
    
    return {
        "current": float(current),
        "trend": trend,
        "trend_score": trend_score,
        "volatility": float(volatility),
        "risk": risk,
        "risk_score": risk_score,
        "rsi": float(rsi),
        "rsi_signal": rsi_signal,
        "change_1d": float(change_1d),
        "change_3d": float(change_3d),
        "change_5d": float(change_5d),
        "change_20d": float(change_20d),
        "close_series": close
    }


def calc_total_score(ind):
    rsi = ind["rsi"]
    if 40 <= rsi <= 60:
        rsi_score = 80
    elif 30 <= rsi <= 70:
        rsi_score = 60
    else:
        rsi_score = 30
    
    total = ind["trend_score"] * 0.4 + ind["risk_score"] * 0.3 + rsi_score * 0.3
    return int(total)


# ============================================================
# 核心：分析所有股票
# ============================================================
def analyze_all_stocks(watchlist):
    results = []
    progress = st.progress(0, text="正在分析股票...")
    
    for i, code in enumerate(watchlist):
        progress.progress((i+1) / len(watchlist), text=f"正在分析 {code}...")
        
        data = get_stock_data(code)
        if data["success"]:
            ind = calc_indicators(data["df"])
            score = calc_total_score(ind)
            
            results.append({
                "code": code,
                "name": data["info"].get("股票简称", code),
                "price": ind["current"],
                "change_1d": ind["change_1d"],
                "change_3d": ind["change_3d"],
                "change_5d": ind["change_5d"],
                "change_20d": ind["change_20d"],
                "trend": ind["trend"],
                "risk": ind["risk"],
                "rsi": ind["rsi"],
                "score": score,
                "indicators": ind,
                "source": data.get("source", "unknown")
            })
        else:
            results.append({
                "code": code,
                "name": f"{code} (数据获取失败)",
                "price": 0,
                "change_1d": 0,
                "change_3d": 0,
                "change_5d": 0,
                "change_20d": 0,
                "trend": "未知",
                "risk": "未知",
                "rsi": 0,
                "score": 0,
                "indicators": None,
                "source": "none",
                "error": data.get("error", "未知错误")
            })
    
    progress.empty()
    return results


# ============================================================
# 侧边栏
# ============================================================
with st.sidebar:
    st.header("📌 我的股票池")
    
    if "watchlist" not in st.session_state:
        st.session_state.watchlist = ["600519", "000858", "300750"]
    
    new_code = st.text_input("添加股票代码", placeholder="如：600519")
    if st.button("➕ 添加") and new_code:
        new_code = new_code.strip()
        # 校验：必须是6位数字
        if not (len(new_code) == 6 and new_code.isdigit()):
            st.error("❌ 股票代码必须是6位数字（如：600519）")
        elif new_code in st.session_state.watchlist:
            st.warning("⚠️ 该股票已在自选中")
        else:
            st.session_state.watchlist.append(new_code)
            if "stock_results" in st.session_state:
                del st.session_state.stock_results
            st.rerun()
    
    st.divider()
    st.markdown("**当前自选（点 ❌ 删除）：**")
    for code in st.session_state.watchlist:
        col1, col2 = st.columns([4, 1])
        col1.markdown(f"- `{code}`")
        if col2.button("❌", key=f"del_{code}"):
            st.session_state.watchlist.remove(code)
            if "stock_results" in st.session_state:
                del st.session_state.stock_results
            st.rerun()
    
    st.divider()
    st.subheader("🤖 AI 设置")
    
    # 优先从环境变量读，读不到才让用户填
    default_key = get_config("ARK_API_KEY", "")
    default_endpoint = get_config("ENDPOINT_ID", "")
    
    if default_key and default_endpoint:
        st.success("✅ 已自动加载 API 配置")
        api_key = default_key
        endpoint_id = default_endpoint
        use_ai = st.checkbox("使用 AI 分析", value=True)
    else:
        api_key = st.text_input("火山引擎 API Key", type="password")
        endpoint_id = st.text_input("接入点 ID", placeholder="ep-xxxxxxxx")
        use_ai = st.checkbox("使用 AI 分析", value=False,
                             help="批量分析新闻 + AI生成建议")


# ============================================================
# 主区域
# ============================================================
st.subheader("📊 股票池行情")

if st.button("🔄 刷新所有股票", type="primary"):
    if "stock_results" in st.session_state:
        del st.session_state.stock_results
    st.rerun()

if "stock_results" not in st.session_state:
    st.session_state.stock_results = analyze_all_stocks(st.session_state.watchlist)


# ============================================================
# 显示结果
# ============================================================
if "stock_results" in st.session_state:
    results = st.session_state.stock_results
    
    for r in results:
        if r.get("error") or r["price"] == 0:
            st.error(f"❌ {r['code']} 数据获取失败：{r.get('error', '未知错误')}")
            st.divider()
            continue
        
        if r["score"] >= 70:
            score_icon, signal = "🟢", "看多"
        elif r["score"] >= 50:
            score_icon, signal = "🟡", "中性"
        else:
            score_icon, signal = "🔴", "看空"
        
        st.markdown(f"### {r['name']} ({r['code']})  {score_icon} {r['score']}分 · {signal}")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.markdown(f"**现价**")
        col1.markdown(f"### ¥{r['price']:.2f}")
        col2.markdown(f"**近1日**")
        col2.markdown(f"### {'🟢' if r['change_1d'] > 0 else '🔴'} {r['change_1d']:+.2f}%")
        col3.markdown(f"**近3日**")
        col3.markdown(f"### {'🟢' if r['change_3d'] > 0 else '🔴'} {r['change_3d']:+.2f}%")
        col4.markdown(f"**近5日**")
        col4.markdown(f"### {'🟢' if r['change_5d'] > 0 else '🔴'} {r['change_5d']:+.2f}%")
        col5.markdown(f"**近20日**")
        col5.markdown(f"### {'🟢' if r['change_20d'] > 0 else '🔴'} {r['change_20d']:+.2f}%")
        
        with st.expander(f"📖 {r['name']} 详细分析"):
            ind = r["indicators"]
            
            st.markdown("**📈 近60日走势**")
            close_recent = ind["close_series"].tail(60)
            ma20_recent = ind["close_series"].rolling(20).mean().tail(60)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=close_recent.index, y=close_recent.values,
                mode='lines', name='收盘价', line=dict(color='blue', width=2)
            ))
            fig.add_trace(go.Scatter(
                x=ma20_recent.index, y=ma20_recent.values,
                mode='lines', name='MA20', line=dict(color='orange', width=1)
            ))
            fig.update_layout(height=300, margin=dict(l=0, r=0, t=0, b=0),
                              hovermode='x unified')
            st.plotly_chart(fig, use_container_width=True)
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("趋势", ind["trend"])
            c2.metric("风险", ind["risk"])
            c3.metric("RSI", f"{ind['rsi']:.1f} ({ind['rsi_signal']})")
            c4.metric("波动率", f"{ind['volatility']*100:.1f}%")
            
            st.markdown("**💡 人话解读**")
            if ind["trend_score"] >= 70:
                st.success(f"📈 趋势：{ind['trend']}，近期走势良好")
            elif ind["trend_score"] >= 50:
                st.warning(f"➡️ 趋势：{ind['trend']}，方向不明朗")
            else:
                st.error(f"📉 趋势：{ind['trend']}，短期承压")
            
            if ind["risk_score"] >= 70:
                st.success(f"🛡️ 风险：{ind['risk']}，波动较小")
            elif ind["risk_score"] >= 50:
                st.warning(f"⚠️ 风险：{ind['risk']}，注意波动")
            else:
                st.error(f"🔥 风险：{ind['risk']}，波动剧烈")
            
            if ind["rsi_signal"] == "超卖":
                st.info(f"💡 RSI：{ind['rsi']:.1f}，超卖，可能有反弹机会")
            elif ind["rsi_signal"] == "超买":
                st.info(f"💡 RSI：{ind['rsi']:.1f}，超买，注意回调风险")
            else:
                st.info(f"💡 RSI：{ind['rsi']:.1f}，正常区间")
            
            # 新闻
            st.markdown("**📰 相关新闻（多源）**")
            search_name = r['name'] if r['name'] != r['code'] else r['code']
            news = get_news_multi(search_name, max_items=8)
            
            news_summary = "无"
            sentiments = []
            
            if news is not None and len(news) > 0:
                pairs = list(zip(news['标题'].tolist(), news['摘要'].tolist()))
                
                if use_ai and api_key and endpoint_id:
                    sentiments = analyze_news_batch_ai(pairs, api_key, endpoint_id)
                else:
                    sentiments = [analyze_news_keyword(t, s) for t, s in pairs]
                
                pos_count = sum(1 for s in sentiments if "🟢" in s)
                neg_count = sum(1 for s in sentiments if "🔴" in s)
                total = len(sentiments)
                
                news_summary = f"{pos_count}条利好、{neg_count}条利空，共{total}条"
                
                col_n1, col_n2, col_n3 = st.columns(3)
                col_n1.metric("利好新闻", pos_count)
                col_n2.metric("利空新闻", neg_count)
                col_n3.metric("情绪倾向",
                              "偏多" if pos_count > neg_count else ("偏空" if neg_count > pos_count else "中性"))
                
                for (_, row), sentiment in zip(news.iterrows(), sentiments):
                    title = row['标题']
                    url = row['链接']
                    source = row['来源']
                    if url:
                        st.markdown(f"- {sentiment}｜[{title}]({url}) `{source}`")
                    else:
                        st.markdown(f"- {sentiment}｜{title} `{source}`")
            else:
                st.caption(f"暂无「{search_name}」相关新闻")
            
            # AI 生成建议
            st.markdown("**🎯 综合建议**")
            if use_ai and api_key and endpoint_id:
                advice = generate_advice_ai(
                    r["score"], ind["trend"], ind["risk"],
                    ind["rsi"], news_summary,
                    api_key, endpoint_id
                )
            else:
                advice = generate_advice_rule(r["score"], ind["trend"], ind["risk"])
            
            if r["score"] >= 70:
                st.success(f"💡 {advice}")
            elif r["score"] >= 50:
                st.warning(f"💡 {advice}")
            else:
                st.error(f"💡 {advice}")
        
        st.divider()


st.caption("""
⚠️ **免责声明**：本工具基于历史数据和技术指标分析，
AI分析仅供参考，不构成投资建议。股市有风险，投资需谨慎。
""")