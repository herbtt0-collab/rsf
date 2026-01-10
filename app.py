# =========================================================
# 📊 RSF Survival Prediction System - Professional Edition
# 专业版生存风险预测系统（适合论文发表）
# 时间单位：年
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os

# =========================================================
# 🎨 页面配置
# =========================================================
st.set_page_config(
    page_title="Survival Prediction System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# 🎨 专业版 CSS 样式（适合发表）
# =========================================================
st.markdown("""
<style>
    /* 整体背景 */
    .stApp {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
    }
    
    /* 侧边栏 */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e3a5f 0%, #2d5a87 100%);
    }
    
    [data-testid="stSidebar"] .stMarkdown,
    [data-testid="stSidebar"] label {
        color: #ffffff !important;
    }
    
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stNumberInput label {
        color: #e0e7ff !important;
        font-weight: 500;
    }
    
    /* 主标题 */
    .main-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #1e3a5f 0%, #3b82f6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.3rem;
        letter-spacing: -0.5px;
    }
    
    .sub-title {
        font-size: 1.15rem;
        color: #64748b;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    
    /* 结果卡片容器 */
    .results-container {
        display: flex;
        justify-content: center;
        gap: 1.2rem;
        flex-wrap: wrap;
        margin: 1.5rem 0;
    }
    
    /* 生存率卡片 - 专业风格 */
    .survival-card {
        background: white;
        border-radius: 16px;
        padding: 1.8rem 1.5rem;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        border: 1px solid #e2e8f0;
        min-width: 160px;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .survival-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 30px rgba(0,0,0,0.12);
    }
    
    .card-year {
        font-size: 0.9rem;
        color: #64748b;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.5rem;
    }
    
    .card-value {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0.3rem 0;
    }
    
    .card-label {
        font-size: 0.85rem;
        color: #94a3b8;
        font-weight: 500;
    }
    
    /* 颜色主题 */
    .year-1 { color: #10b981; border-top: 4px solid #10b981; }
    .year-2 { color: #3b82f6; border-top: 4px solid #3b82f6; }
    .year-3 { color: #8b5cf6; border-top: 4px solid #8b5cf6; }
    .year-4 { color: #f59e0b; border-top: 4px solid #f59e0b; }
    .risk-card { color: #1e3a5f; border-top: 4px solid #1e3a5f; }
    
    /* 分隔线 */
    .section-divider {
        height: 2px;
        background: linear-gradient(90deg, transparent, #cbd5e1, transparent);
        margin: 2.5rem 0;
    }
    
    /* 按钮样式 */
    .stButton > button {
        background: linear-gradient(135deg, #1e3a5f 0%, #3b82f6 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.85rem 2rem;
        font-size: 1.05rem;
        font-weight: 600;
        width: 100%;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(30, 58, 95, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 25px rgba(30, 58, 95, 0.4);
    }
    
    /* 表格样式 */
    .dataframe {
        border-radius: 12px;
        overflow: hidden;
    }
    
    /* 页脚 */
    .footer {
        text-align: center;
        color: #94a3b8;
        padding: 2rem;
        font-size: 0.9rem;
        border-top: 1px solid #e2e8f0;
        margin-top: 2rem;
    }
    
    /* 信息提示框 */
    .info-box {
        background: #f0f9ff;
        border: 1px solid #bae6fd;
        border-radius: 12px;
        padding: 1.2rem;
        margin: 1rem 0;
        color: #0c4a6e;
    }
    
    /* 隐藏 Streamlit 默认元素 */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# =========================================================
# 📋 特征标签映射
# =========================================================
FEATURE_LABEL_MAP = {
    "post_dm_acarbose_Yes": "α-glucosidase inhibitors",
    "post_htn_raas_Yes": "RAAS inhibitors",
    "post_dm_metformin_Yes": "Metformin",
    "入院年龄": "Age",
    "尿素氮": "Blood urea nitrogen",
    "肌酸激酶": "Creatine Kinase",
    "渗透压": "Serum Osmolality",
    "葡萄糖": "Glucose",
    "CCI_score": "CCI score",
    "纤维蛋白原": "Fibrinogen"
}

LABEL_FEATURE_MAP = {v: k for k, v in FEATURE_LABEL_MAP.items()}

# =========================================================
# 📊 特征配置
# =========================================================
FEATURE_CONFIG = {
    "Age": {
        "type": "number",
        "min": 18.0,
        "max": 100.0,
        "default": 65.0,
        "step": 1.0,
        "unit": "years",
        "description": "Patient age at admission"
    },
    "Blood urea nitrogen": {
        "type": "number",
        "min": 0.0,
        "max": 50.0,
        "default": 6.0,
        "step": 0.1,
        "unit": "mmol/L",
        "description": "Blood urea nitrogen level"
    },
    "Creatine Kinase": {
        "type": "number",
        "min": 0.0,
        "max": 5000.0,
        "default": 100.0,
        "step": 1.0,
        "unit": "U/L",
        "description": "Creatine kinase level"
    },
    "Serum Osmolality": {
        "type": "number",
        "min": 250.0,
        "max": 350.0,
        "default": 290.0,
        "step": 1.0,
        "unit": "mOsm/kg",
        "description": "Serum osmolality"
    },
    "Glucose": {
        "type": "number",
        "min": 2.0,
        "max": 40.0,
        "default": 6.0,
        "step": 0.1,
        "unit": "mmol/L",
        "description": "Blood glucose level"
    },
    "CCI score": {
        "type": "number",
        "min": 0.0,
        "max": 20.0,
        "default": 2.0,
        "step": 1.0,
        "unit": "",
        "description": "Charlson Comorbidity Index"
    },
    "Fibrinogen": {
        "type": "number",
        "min": 0.0,
        "max": 10.0,
        "default": 3.0,
        "step": 0.1,
        "unit": "g/L",
        "description": "Fibrinogen level"
    },
    "α-glucosidase inhibitors": {
        "type": "select",
        "options": ["No", "Yes"],
        "default": "No",
        "description": "α-glucosidase inhibitors use"
    },
    "RAAS inhibitors": {
        "type": "select",
        "options": ["No", "Yes"],
        "default": "No",
        "description": "RAAS inhibitors use"
    },
    "Metformin": {
        "type": "select",
        "options": ["No", "Yes"],
        "default": "No",
        "description": "Metformin use"
    }
}

# =========================================================
# 🔧 模型加载
# =========================================================
@st.cache_resource
def load_model():
    """加载 RSF 模型"""
    possible_paths = [
        "rsf_model.joblib",
        "rsf_model_compressed.joblib",
        r"C:\Users\Serendipity\Desktop\cjj\rsf_model.joblib",
        r"C:\Users\Serendipity\Desktop\cjj\rsf_model_compressed.joblib",
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            try:
                return joblib.load(path)
            except Exception as e:
                st.error(f"Model loading error: {e}")
                return None
    return None

# =========================================================
# 🔧 特征列表加载
# =========================================================
@st.cache_data
def load_feature_list():
    """加载特征列表"""
    possible_paths = [
        "selected_features.txt",
        r"C:\Users\Serendipity\Desktop\cjj\selected_features.txt",
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    features = [line.strip() for line in f if line.strip()]
                if features:
                    return features
            except:
                pass
    
    # 默认特征列表
    return [
        "入院年龄", "尿素氮", "肌酸激酶", "渗透压", "葡萄糖",
        "CCI_score", "纤维蛋白原", "post_dm_acarbose_Yes",
        "post_htn_raas_Yes", "post_dm_metformin_Yes"
    ]

# =========================================================
# 🔧 获取生存概率（正确的阶梯函数插值）
# =========================================================
def get_survival_probability(surv_func, target_time):
    """从生存函数获取指定时间点的生存概率"""
    times = surv_func.x
    probs = surv_func.y
    
    if target_time <= times[0]:
        return 1.0
    if target_time >= times[-1]:
        return probs[-1]
    
    # 阶梯函数：取左边界值
    idx = np.searchsorted(times, target_time, side='right') - 1
    return probs[max(0, idx)]

# =========================================================
# 🔧 预测函数
# =========================================================
def predict_survival(model, input_data):
    """RSF 模型预测"""
    risk_score = model.predict(input_data)
    surv_funcs = model.predict_survival_function(input_data)
    return risk_score[0], surv_funcs[0]

# =========================================================
# 🎨 绘制专业生存曲线（适合发表）
# =========================================================
def plot_survival_curve_professional(surv_func):
    """绘制适合论文发表的生存曲线"""
    
    # 设置专业绘图风格
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.linewidth'] = 1.2
    
    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=150)
    
    # 白色背景
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    
    time_points = surv_func.x
    surv_probs = surv_func.y
    
    # 主曲线 - 使用深蓝色，更粗的线条
    ax.step(time_points, surv_probs, where='post', 
            color='#1e3a5f', linewidth=2.5, label='Survival Probability')
    
    # 淡色填充
    ax.fill_between(time_points, surv_probs, step='post', 
                    color='#3b82f6', alpha=0.15)
    
    # 标记 1、2、3、4 年的点
    colors = ['#10b981', '#3b82f6', '#8b5cf6', '#f59e0b']
    years = [1, 2, 3, 4]
    labels = ['1-Year', '2-Year', '3-Year', '4-Year']
    
    for i, (year, color, label) in enumerate(zip(years, colors, labels)):
        if year <= time_points[-1]:
            prob = get_survival_probability(surv_func, year)
            
            # 绘制点
            ax.scatter([year], [prob], color=color, s=120, zorder=5, 
                      edgecolors='white', linewidths=2)
            
            # 绘制虚线到坐标轴
            ax.plot([year, year], [0, prob], color=color, linestyle='--', 
                   linewidth=1, alpha=0.6)
            ax.plot([0, year], [prob, prob], color=color, linestyle='--', 
                   linewidth=1, alpha=0.6)
            
            # 标注文字
            offset_y = 0.06 if i % 2 == 0 else -0.08
            va = 'bottom' if i % 2 == 0 else 'top'
            ax.annotate(f'{label}: {prob:.1%}', 
                       xy=(year, prob),
                       xytext=(year + 0.15, prob + offset_y),
                       fontsize=11,
                       fontweight='bold',
                       color=color,
                       va=va,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                edgecolor=color, alpha=0.9))
    
    # 设置标题和标签
    ax.set_title('Predicted Survival Curve', fontsize=16, fontweight='bold', 
                 color='#1e293b', pad=20)
    ax.set_xlabel('Time (Years)', fontsize=13, fontweight='600', color='#374151')
    ax.set_ylabel('Survival Probability', fontsize=13, fontweight='600', color='#374151')
    
    # 坐标轴范围
    ax.set_xlim(0, min(5, max(time_points) * 1.05))
    ax.set_ylim(0, 1.02)
    
    # 刻度设置
    ax.set_xticks([0, 1, 2, 3, 4, 5])
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0%', '20%', '40%', '60%', '80%', '100%'])
    
    # 网格
    ax.grid(True, linestyle='-', alpha=0.2, color='#94a3b8')
    ax.set_axisbelow(True)
    
    # 边框
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_color('#cbd5e1')
        ax.spines[spine].set_linewidth(1.2)
    
    # 刻度颜色
    ax.tick_params(colors='#4b5563', labelsize=11)
    
    plt.tight_layout()
    return fig

# =========================================================
# 🏠 主函数
# =========================================================
def main():
    # 标题
    st.markdown('<h1 class="main-title">🏥 Intelligent Platform for Predicting the Risk of Coronary Heart Disease in CKM Syndrome</h1>',
                unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Random Survival Forest Model-Based Clinical Decision Support Tool</p>', 
                unsafe_allow_html=True)
    
    # 加载模型
    model = load_model()
    feature_list = load_feature_list()
    demo_mode = model is None
    
    if demo_mode:
        st.warning("⚠️ **Demo Mode**: Model file not found. Please ensure `rsf_model.joblib` is in the app directory.")
    
    # ----------------------
    # 侧边栏输入
    # ----------------------
    with st.sidebar:
        st.markdown("## 📋 Patient Parameters")
        st.markdown("---")
        
        user_inputs = {}
        
        for feature_name in feature_list:
            display_name = FEATURE_LABEL_MAP.get(feature_name, feature_name)
            config = FEATURE_CONFIG.get(display_name, {})
            
            is_binary = feature_name.endswith("_Yes") or config.get("type") == "select"
            
            if is_binary:
                selection = st.selectbox(
                    label=display_name,
                    options=config.get("options", ["No", "Yes"]),
                    index=0,
                    help=config.get("description", "")
                )
                user_inputs[feature_name] = 1.0 if selection == "Yes" else 0.0
            else:
                unit = config.get("unit", "")
                label = f"{display_name}" + (f" ({unit})" if unit else "")
                
                user_inputs[feature_name] = st.number_input(
                    label=label,
                    min_value=config.get("min", 0.0),
                    max_value=config.get("max", 1000.0),
                    value=config.get("default", 0.0),
                    step=config.get("step", 0.1),
                    help=config.get("description", "")
                )
        
        st.markdown("---")
        predict_button = st.button("🔮 Calculate Survival Probability", use_container_width=True)
    
    # ----------------------
    # 主内容区
    # ----------------------
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    if predict_button:
        input_df = pd.DataFrame([user_inputs])
        input_df = input_df[feature_list]
        
        with st.spinner('Calculating...'):
            if demo_mode:
                # 演示模式
                risk_score = np.random.uniform(20, 80)
                times = np.linspace(0, 5, 100)
                base_rate = 0.15 + (risk_score / 100) * 0.3
                surv_probs = np.exp(-base_rate * times)
                
                class MockSurvFunc:
                    def __init__(self, x, y):
                        self.x = x
                        self.y = y
                
                surv_func = MockSurvFunc(times, surv_probs)
            else:
                risk_score, surv_func = predict_survival(model, input_df)
        
        # 计算 1-4 年生存率
        surv_1y = get_survival_probability(surv_func, 1)
        surv_2y = get_survival_probability(surv_func, 2)
        surv_3y = get_survival_probability(surv_func, 3)
        surv_4y = get_survival_probability(surv_func, 4)
        
        # ----------------------
        # 显示结果卡片
        # ----------------------
        st.markdown("### 📊 Prediction Results")
        
        # 使用5列布局
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.markdown(f"""
            <div class="survival-card risk-card">
                <div class="card-year">Risk Score</div>
                <div class="card-value">{risk_score:.1f}</div>
                <div class="card-label">Relative Risk</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="survival-card year-1">
                <div class="card-year">1-Year</div>
                <div class="card-value">{surv_1y:.1%}</div>
                <div class="card-label">Survival Rate</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="survival-card year-2">
                <div class="card-year">2-Year</div>
                <div class="card-value">{surv_2y:.1%}</div>
                <div class="card-label">Survival Rate</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="survival-card year-3">
                <div class="card-year">3-Year</div>
                <div class="card-value">{surv_3y:.1%}</div>
                <div class="card-label">Survival Rate</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col5:
            st.markdown(f"""
            <div class="survival-card year-4">
                <div class="card-year">4-Year</div>
                <div class="card-value">{surv_4y:.1%}</div>
                <div class="card-label">Survival Rate</div>
            </div>
            """, unsafe_allow_html=True)
        
        # ----------------------
        # 生存曲线
        # ----------------------
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.markdown("### 📈 Survival Curve")
        
        fig = plot_survival_curve_professional(surv_func)
        st.pyplot(fig)
        plt.close()
        
        # ----------------------
        # 预测摘要表格
        # ----------------------
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.markdown("### 📋 Prediction Summary")
        
        col_table1, col_table2 = st.columns(2)
        
        with col_table1:
            st.markdown("**Survival Probabilities**")
            surv_df = pd.DataFrame({
                "Time Point": ["1-Year", "2-Year", "3-Year", "4-Year"],
                "Survival Probability": [f"{surv_1y:.1%}", f"{surv_2y:.1%}", 
                                         f"{surv_3y:.1%}", f"{surv_4y:.1%}"]
            })
            st.dataframe(surv_df, use_container_width=True, hide_index=True)
        
        with col_table2:
            st.markdown("**Input Parameters**")
            input_summary = []
            for feature_name, value in user_inputs.items():
                display_name = FEATURE_LABEL_MAP.get(feature_name, feature_name)
                config = FEATURE_CONFIG.get(display_name, {})
                
                if feature_name.endswith("_Yes") or config.get("type") == "select":
                    display_value = "Yes" if value == 1 else "No"
                else:
                    unit = config.get("unit", "")
                    display_value = f"{value:.2f}" + (f" {unit}" if unit else "")
                
                input_summary.append({"Parameter": display_name, "Value": display_value})
            
            st.dataframe(pd.DataFrame(input_summary), use_container_width=True, hide_index=True)
    
    else:
        # 未点击按钮时的提示
        st.markdown("""
        <div class="info-box">
            <h4 style="margin-top: 0; color: #0c4a6e;">👈 Please enter patient parameters in the sidebar</h4>
            <p style="margin-bottom: 0;">
                Input all required clinical parameters and click <strong>"Calculate Survival Probability"</strong> 
                to obtain the 1-4 year survival prediction results.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # 显示模型说明
        st.markdown("### 📌 Model Features")
        features_display = [FEATURE_LABEL_MAP.get(f, f) for f in feature_list]
        
        cols = st.columns(2)
        mid = len(features_display) // 2 + len(features_display) % 2
        
        with cols[0]:
            for f in features_display[:mid]:
                st.markdown(f"• {f}")
        with cols[1]:
            for f in features_display[mid:]:
                st.markdown(f"• {f}")
    
    # 页脚
    st.markdown("""
    <div class="footer">
        <p>⚕️ This tool is for research and clinical reference only. 
        Please consult healthcare professionals for medical decisions.</p>
        <p>© 2025 Survival Risk Prediction System | Powered by Random Survival Forest</p>
    </div>
    """, unsafe_allow_html=True)

# =========================================================
# 🚀 运行
# =========================================================
if __name__ == "__main__":
    main()
