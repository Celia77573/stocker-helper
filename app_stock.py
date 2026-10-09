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
# 配置读取（优先环境变量，其次Secrets）
# ============================================================
def get_config(key, default=""):
    val = os.environ.get(key, "")
    if val:
        return val
    try:
        return st.secrets.get(key, default)
    except:
        return default


# ============================================================
# 股价数据
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
# 新闻
# ============================================================
@st.cache_data(ttl=1800)
def get_news_eastmoney(keyword):
    try:
        import requests, json
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
        futures = [executor.submit(get_news_eastmoney, keyword),
                   executor.submit(get_news_sina, keyword)]
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
            temperature=0.1, max_tokens=800,
        )
        result = response.choices[0].message.content.strip()
        lines = result.split('\n')
        parsed = {}
        for line in lines:
            match = re.match(r'^\s*(\d+)[.、]\s*(利好|利空|中性)\s*[|｜]\s*(.+)$', line.strip())
            if match:
                idx = int(match.group(1))
                parsed[idx] = (match.group(2), match.group(3).strip())
        results = []
        for i in range(1, len(titles_summaries) + 1):
            if i in parsed:
                sentiment, reason = parsed[i]
                emoji = "🟢" if sentiment == "利好" else ("🔴" if sentiment == "利空" else "⚪")
                results.append(f"{emoji} {sentiment} | {reason}")
            else:
                t, s = titles_summaries[i-1]
                results.append(analyze_news_keyword(t, s))
        return results
    except:
        return [analyze_news_keyword(t, s) for t, s in titles_summaries]


def analyze_news_keyword(title, summary):
    text = f"{title} {summary}"
    pos_words = ['涨', '利好', '增长', '突破', '买入', '上调', '涨停', '新高', '盈利', '超预期', '大涨', '飙升']
    neg_words = ['跌', '利空', '下滑', '跌破', '卖出', '下调', '跌停', '新低', '亏损', '不及预期', '大跌', '暴跌']
    pos = sum(1 for w in pos_words if w in text)
    neg = sum(1 for w in neg_words if w in text)
    if pos > neg:
        return "🟢 利好 | 关键词匹配"
    elif neg > pos:
        return "🔴 利空 | 关键词匹配"
    else:
        return "⚪ 中性 | 关键词匹配"


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
        "ma5": float(ma5), "ma20": float(ma20), "ma60": float(ma60),
        "trend": trend, "trend_score": trend_score,
        "volatility": float(volatility), "risk": risk, "risk_score": risk_score,
        "rsi": float(rsi), "rsi_signal": rsi_signal,
        "change_1d": float(change_1d), "change_3d": float(change_3d),
        "change_5d": float(change_5d), "change_20d": float(change_20d),
        "close_series": close
    }


def calc_tech_score(ind):
    """纯技术面评分（0-100）"""
    rsi = ind["rsi"]
    if 40 <= rsi <= 60:
        rsi_score = 80
    elif 30 <= rsi <= 70:
        rsi_score = 60
    else:
        rsi_score = 30
    return int(ind["trend_score"] * 0.4 + ind["risk_score"] * 0.3 + rsi_score * 0.3)


def calc_news_score(sentiments):
    """新闻情绪评分（0-100）"""
    if not sentiments:
        return 50
    total = len(sentiments)
    pos = sum(1 for s in sentiments if "🟢" in s)
    neg = sum(1 for s in sentiments if "🔴" in s)
    # 利好比例映射到0-100
    return int(50 + (pos - neg) / total * 50)


def calc_total_score(tech_score, news_score):
    """综合评分：技术85% + 新闻15%"""
    return int(tech_score * 0.85 + news_score * 0.15)


# ============================================================
# 生成综合建议（具体版）
# ============================================================
def generate_detailed_advice(r, ind, sentiments, news_summary):
    """生成详细的综合建议（规则版，稳定输出）"""
    tech_score = r["tech_score"]
    news_score = r["news_score"]
    total = r["score"]
    
    # 判断技术面和消息面方向
    tech_direction = "偏多" if tech_score >= 60 else ("偏空" if tech_score <= 40 else "中性")
    news_direction = "偏多" if news_score >= 60 else ("偏空" if news_score <= 40 else "中性")
    
    # 分歧判断
    has_divergence = (
        (tech_score >= 60 and news_score <= 40) or
        (tech_score <= 40 and news_score >= 60)
    )
    
    lines = []
    
    # 分歧提示
    if has_divergence:
        lines.append(f"⚠️ **技术面与消息面存在分歧**")
        lines.append(f"- 📉 技术面：{ind['trend']}，RSI {ind['rsi']:.1f}（{ind['rsi_signal']}）")
        lines.append(f"- 📰 消息面：{news_summary}")
        lines.append("")
        lines.append("**可能原因**：市场对消息的反应滞后，或存在其他利空因素未反映在新闻中。")
    else:
        lines.append(f"✅ **技术面与消息面方向一致**")
        lines.append(f"- 📊 技术面：{ind['trend']}，RSI {ind['rsi']:.1f}（{ind['rsi_signal']}）")
        lines.append(f"- 📰 消息面：{news_summary}")
    
    lines.append("")
    lines.append("**🎯 操作建议**")
    
    # 根据技术面+消息面给出操作建议
    if tech_direction == "偏多" and news_direction == "偏多":
        lines.append("- 可考虑建仓，仓位建议不超过总资产的30%")
        lines.append("- 回踩MA20可加仓")
        lines.append(f"- 目标位：前期高点附近")
        lines.append("- 止损位：跌破MA20")
        
    elif tech_direction == "偏空" and news_direction == "偏空":
        lines.append("- 建议观望或减持，避免抄底")
        lines.append("- 等待技术面企稳（如站上MA20）再考虑")
        lines.append("- 如已持仓，设置止损位")
        
    elif tech_direction == "偏多" and news_direction == "偏空":
        lines.append("- 技术面尚可，但消息面不利，谨慎持有")
        lines.append("- 仓位建议不超过10%")
        lines.append("- 关注消息面是否有进一步恶化")
        lines.append("- 如跌破MA20，果断减仓")
        
    elif tech_direction == "偏空" and news_direction == "偏多":
        lines.append("- 技术面偏弱，但消息面利好，存在反转可能")
        lines.append("- 暂时观望，等待K线企稳信号")
        lines.append("- 如果放量突破MA20，可小仓位试仓（不超过10%）")
        lines.append("- 如果继续跌破前期低点，果断放弃")
        
    else:
        lines.append("- 多空信号不明确，建议观望")
        lines.append("- 等待更明确的信号后再操作")
    
    lines.append("")
    lines.append("**🛡️ 风险提示**")
    
    if ind["rsi"] > 70:
        lines.append("- ⚠️ RSI超买，短期可能回调")
    elif ind["rsi"] < 30:
        lines.append("- 💡 RSI超卖，可能有反弹机会")
    
    if ind["volatility"] > 0.35:
        lines.append("- ⚠️ 波动率较高，注意仓位控制")
    
    if has_divergence:
        lines.append("- ⚠️ 技术面与消息面分歧，需谨慎判断")
    
    if tech_direction == "偏空":
        lines.append("- 📉 短期趋势向下，不宜重仓")
    
    return "\n".join(lines)


# ============================================================
# 分析所有股票
# ============================================================
def analyze_all_stocks(watchlist, api_key, endpoint_id, use_ai):
    results = []
    progress = st.progress(0, text="正在分析股票...")
    
    for i, code in enumerate(watchlist):
        progress.progress((i+1) / len(watchlist), text=f"正在分析 {code}...")
        
        data = get_stock_data(code)
        if data["success"]:
            ind = calc_indicators(data["df"])
            tech_score = calc_tech_score(ind)
            
            # 抓新闻
            name = data["info"].get("股票简称", code)
            search_name = name if name != code else code
            news = get_news_multi(search_name, max_items=8)
            
            sentiments = []
            news_summary = "无新闻"
            if news is not None and len(news) > 0:
                pairs = list(zip(news['标题'].tolist(), news['摘要'].tolist()))
                if use_ai and api_key and endpoint_id:
                    sentiments = analyze_news_batch_ai(pairs, api_key, endpoint_id)
                else:
                    sentiments = [analyze_news_keyword(t, s) for t, s in pairs]
                
                pos = sum(1 for s in sentiments if "🟢" in s)
                neg = sum(1 for s in sentiments if "🔴" in s)
                news_summary = f"{pos}条利好、{neg}条利空，共{len(sentiments)}条"
            
            news_score = calc_news_score(sentiments)
            total_score = calc_total_score(tech_score, news_score)
            
            results.append({
                "code": code,
                "name": name,
                "price": ind["current"],
                "change_1d": ind["change_1d"],
                "change_3d": ind["change_3d"],
                "change_5d": ind["change_5d"],
                "change_20d": ind["change_20d"],
                "trend": ind["trend"],
                "risk": ind["risk"],
                "rsi": ind["rsi"],
                "tech_score": tech_score,
                "news_score": news_score,
                "score": total_score,
                "indicators": ind,
                "news": news,
                "sentiments": sentiments,
                "news_summary": news_summary,
                "source": data.get("source", "unknown")
            })
        else:
            results.append({
                "code": code, "name": f"{code} (数据失败)",
                "price": 0, "change_1d": 0, "change_3d": 0,
                "change_5d": 0, "change_20d": 0,
                "trend": "未知", "risk": "未知", "rsi": 0,
                "tech_score": 0, "news_score": 0, "score": 0,
                "indicators": None, "error": data.get("error", "未知"),
                "source": "none"
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
        if not (len(new_code) == 6 and new_code.isdigit()):
            st.error("❌ 股票代码必须是6位数字")
        elif new_code in st.session_state.watchlist:
            st.warning("⚠️ 已在自选中")
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
        use_ai = st.checkbox("使用 AI 分析", value=False)


# ============================================================
# 主区域
# ============================================================
st.subheader("📊 股票池行情")

if st.button("🔄 刷新所有股票", type="primary"):
    if "stock_results" in st.session_state:
        del st.session_state.stock_results
    st.rerun()

if "stock_results" not in st.session_state:
    st.session_state.stock_results = analyze_all_stocks(
        st.session_state.watchlist, api_key, endpoint_id, use_ai
    )


# ============================================================
# 显示结果
# ============================================================
if "stock_results" in st.session_state:
    for r in st.session_state.stock_results:
        if r.get("error") or r["price"] == 0:
            st.error(f"❌ {r['code']} 数据获取失败：{r.get('error', '未知')}")
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
            
            # K线
            st.markdown("**📈 近60日走势**")
            close_recent = ind["close_series"].tail(60)
            ma20_recent = ind["close_series"].rolling(20).mean().tail(60)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=close_recent.index, y=close_recent.values,
                                     mode='lines', name='收盘价', line=dict(color='blue', width=2)))
            fig.add_trace(go.Scatter(x=ma20_recent.index, y=ma20_recent.values,
                                     mode='lines', name='MA20', line=dict(color='orange', width=1)))
            fig.update_layout(height=300, margin=dict(l=0, r=0, t=0, b=0), hovermode='x unified')
            st.plotly_chart(fig, use_container_width=True)
            
            # 指标
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("趋势", ind["trend"])
            c2.metric("风险", ind["risk"])
            c3.metric("RSI", f"{ind['rsi']:.1f} ({ind['rsi_signal']})")
            c4.metric("波动率", f"{ind['volatility']*100:.1f}%")
            
            # 评分拆解
            st.markdown("**📊 评分拆解**")
            col_s1, col_s2, col_s3 = st.columns(3)
            col_s1.metric("技术面", f"{r['tech_score']}分", help="85%权重")
            col_s2.metric("消息面", f"{r['news_score']}分", help="15%权重")
            col_s3.metric("综合", f"{r['score']}分")
            
            # 人话解读
            st.markdown("**💡 人话解读**")
            if ind["trend_score"] >= 70:
                st.success(f"📈 趋势：{ind['trend']}")
            elif ind["trend_score"] >= 50:
                st.warning(f"➡️ 趋势：{ind['trend']}")
            else:
                st.error(f"📉 趋势：{ind['trend']}")
            
            # 新闻
            st.markdown("**📰 相关新闻（多源）**")
            news = r.get("news")
            sentiments = r.get("sentiments", [])
            
            if news is not None and len(news) > 0 and sentiments:
                pos = sum(1 for s in sentiments if "🟢" in s)
                neg = sum(1 for s in sentiments if "🔴" in s)
                
                col_n1, col_n2, col_n3 = st.columns(3)
                col_n1.metric("利好新闻", pos)
                col_n2.metric("利空新闻", neg)
                col_n3.metric("情绪倾向",
                              "偏多" if pos > neg else ("偏空" if neg > pos else "中性"))
                
                for (_, row), sentiment in zip(news.iterrows(), sentiments):
                    if row['链接']:
                        st.markdown(f"- {sentiment}｜[{row['标题']}]({row['链接']}) `{row['来源']}`")
                    else:
                        st.markdown(f"- {sentiment}｜{row['标题']} `{row['来源']}`")
            else:
                st.caption("暂无相关新闻")
            
            # 综合建议（详细版）
            st.markdown("**🎯 综合建议**")
            advice = generate_detailed_advice(r, ind, sentiments, r.get("news_summary", "无"))
            st.markdown(advice)
        
        st.divider()


st.caption("""
⚠️ **免责声明**：本工具基于历史数据和技术指标分析，
AI分析仅供参考，不构成投资建议。股市有风险，投资需谨慎。
""")
