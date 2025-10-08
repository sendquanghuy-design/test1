import streamlit as st
import pandas as pd
from google import genai
from google.genai.errors import APIError
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# --- Cấu hình Trang Streamlit ---
st.set_page_config(
    page_title="App Phân Tích Báo Cáo Tài Chính",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "Ứng dụng Phân Tích Tài Chính với AI"
    }
)

# --- Custom CSS cho giao diện đẹp ---
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary-color: #1f77b4;
        --success-color: #2ecc71;
        --danger-color: #e74c3c;
        --warning-color: #f39c12;
    }
    
    /* Card styling */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        color: white;
        margin: 10px 0;
    }
    
    .metric-card h3 {
        margin: 0;
        font-size: 1.2em;
        font-weight: 600;
    }
    
    .metric-card p {
        margin: 10px 0 0 0;
        font-size: 2em;
        font-weight: bold;
    }
    
    /* Chat styling */
    .stChatMessage {
        border-radius: 15px;
        padding: 15px;
        margin: 10px 0;
    }
    
    /* Suggest button styling */
    .suggest-btn {
        background-color: #f0f2f6;
        border: 1px solid #e0e0e0;
        border-radius: 20px;
        padding: 8px 16px;
        margin: 5px;
        display: inline-block;
        cursor: pointer;
        transition: all 0.3s;
    }
    
    .suggest-btn:hover {
        background-color: #667eea;
        color: white;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    /* Table styling */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
    }
    
    /* Animation */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .fade-in {
        animation: fadeIn 0.5s ease-in;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 30px;
        border-radius: 15px;
        color: white;
        margin-bottom: 30px;
        text-align: center;
    }
    
    /* Success message */
    .success-toast {
        background-color: #2ecc71;
        color: white;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# --- Khởi tạo session state ---
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

if "financial_data_context" not in st.session_state:
    st.session_state.financial_data_context = None

if "df_processed" not in st.session_state:
    st.session_state.df_processed = None

if "file_history" not in st.session_state:
    st.session_state.file_history = []

if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

# --- Hàm tính toán chính ---
@st.cache_data
def process_financial_data(df):
    """Thực hiện các phép tính Tăng trưởng và Tỷ trọng."""
    numeric_cols = ['Năm trước', 'Năm sau']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    df['Tốc độ tăng trưởng (%)'] = (
        (df['Năm sau'] - df['Năm trước']) / df['Năm trước'].replace(0, 1e-9)
    ) * 100

    tong_tai_san_row = df[df['Chỉ tiêu'].str.contains('TỔNG CỘNG TÀI SẢN', case=False, na=False)]
    
    if tong_tai_san_row.empty:
        raise ValueError("Không tìm thấy chỉ tiêu 'TỔNG CỘNG TÀI SẢN'.")

    tong_tai_san_N_1 = tong_tai_san_row['Năm trước'].iloc[0]
    tong_tai_san_N = tong_tai_san_row['Năm sau'].iloc[0]

    divisor_N_1 = tong_tai_san_N_1 if tong_tai_san_N_1 != 0 else 1e-9
    divisor_N = tong_tai_san_N if tong_tai_san_N != 0 else 1e-9

    df['Tỷ trọng Năm trước (%)'] = (df['Năm trước'] / divisor_N_1) * 100
    df['Tỷ trọng Năm sau (%)'] = (df['Năm sau'] / divisor_N) * 100
    
    return df

# --- Hàm tạo biểu đồ ---
def create_comparison_chart(df):
    """Tạo biểu đồ so sánh Năm trước vs Năm sau"""
    top_items = df.nlargest(10, 'Năm sau')
    
    fig = go.Figure(data=[
        go.Bar(name='Năm trước', x=top_items['Chỉ tiêu'], y=top_items['Năm trước'], 
               marker_color='#3498db'),
        go.Bar(name='Năm sau', x=top_items['Chỉ tiêu'], y=top_items['Năm sau'],
               marker_color='#2ecc71')
    ])
    
    fig.update_layout(
        barmode='group',
        title='Top 10 Chỉ tiêu Tài sản',
        xaxis_title='Chỉ tiêu',
        yaxis_title='Giá trị (VNĐ)',
        height=400,
        template='plotly_white'
    )
    
    return fig

def create_growth_chart(df):
    """Tạo biểu đồ tốc độ tăng trưởng"""
    top_growth = df.nlargest(10, 'Tốc độ tăng trưởng (%)')
    
    colors = ['#2ecc71' if x > 0 else '#e74c3c' for x in top_growth['Tốc độ tăng trưởng (%)']]
    
    fig = go.Figure(data=[
        go.Bar(x=top_growth['Chỉ tiêu'], y=top_growth['Tốc độ tăng trưởng (%)'],
               marker_color=colors)
    ])
    
    fig.update_layout(
        title='Top 10 Tốc độ Tăng trưởng',
        xaxis_title='Chỉ tiêu',
        yaxis_title='Tốc độ tăng trưởng (%)',
        height=400,
        template='plotly_white'
    )
    
    return fig

def create_pie_chart(df):
    """Tạo biểu đồ tròn cơ cấu tài sản"""
    top_5 = df.nlargest(5, 'Năm sau')
    
    fig = px.pie(top_5, values='Năm sau', names='Chỉ tiêu',
                 title='Cơ cấu Top 5 Tài sản (Năm sau)',
                 color_discrete_sequence=px.colors.qualitative.Set3)
    
    fig.update_layout(height=400)
    
    return fig

# --- Hàm gọi API Gemini ---
def get_ai_analysis(data_for_ai, api_key):
    """Gửi dữ liệu phân tích đến Gemini API và nhận nhận xét."""
    try:
        client = genai.Client(api_key=api_key)
        model_name = 'gemini-2.5-flash' 

        prompt = f"""
        Bạn là một chuyên gia phân tích tài chính chuyên nghiệp. Dựa trên các chỉ số tài chính sau, hãy đưa ra một nhận xét khách quan, ngắn gọn (khoảng 3-4 đoạn) về tình hình tài chính của doanh nghiệp. Đánh giá tập trung vào tốc độ tăng trưởng, thay đổi cơ cấu tài sản và khả năng thanh toán hiện hành.
        
        Dữ liệu thô và chỉ số:
        {data_for_ai}
        """

        response = client.models.generate_content(
            model=model_name,
            contents=prompt
        )
        return response.text

    except APIError as e:
        return f"Lỗi gọi Gemini API: Vui lòng kiểm tra Khóa API hoặc giới hạn sử dụng. Chi tiết lỗi: {e}"
    except KeyError:
        return "Lỗi: Không tìm thấy Khóa API 'GEMINI_API_KEY'. Vui lòng kiểm tra cấu hình Secrets trên Streamlit Cloud."
    except Exception as e:
        return f"Đã xảy ra lỗi không xác định: {e}"

def chat_with_gemini(user_message, context, api_key):
    """Gửi câu hỏi của người dùng kèm context dữ liệu tài chính đến Gemini."""
    try:
        client = genai.Client(api_key=api_key)
        model_name = 'gemini-2.5-flash'
        
        prompt = f"""
        Bạn là trợ lý tài chính thông minh. Người dùng đang phân tích báo cáo tài chính của một doanh nghiệp.
        
        Dữ liệu tài chính hiện tại:
        {context if context else "Chưa có dữ liệu tài chính được tải lên."}
        
        Câu hỏi của người dùng: {user_message}
        
        Hãy trả lời câu hỏi dựa trên dữ liệu tài chính đã cung cấp (nếu có). Nếu câu hỏi không liên quan đến dữ liệu, hãy trả lời một cách hữu ích và chuyên nghiệp.
        """
        
        response = client.models.generate_content(
            model=model_name,
            contents=prompt
        )
        return response.text
        
    except Exception as e:
        return f"Lỗi khi chat với Gemini: {e}"

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("### ⚙️ Cài đặt Ứng dụng")
    
    # Theme toggle
    theme_col1, theme_col2 = st.columns(2)
    with theme_col1:
        if st.button("🌙 Dark Mode"):
            st.session_state.dark_mode = True
    with theme_col2:
        if st.button("☀️ Light Mode"):
            st.session_state.dark_mode = False
    
    st.markdown("---")
    
    # Navigation
    page = st.radio("📍 Điều hướng", ["📊 Dashboard", "💬 Chat AI", "📈 Biểu đồ"])
    
    st.markdown("---")
    
    # File history
    st.markdown("### 📁 Lịch sử File")
    if st.session_state.file_history:
        for i, file_info in enumerate(st.session_state.file_history[-5:]):
            st.text(f"{i+1}. {file_info['name']}")
            st.caption(f"   {file_info['time']}")
    else:
        st.info("Chưa có file nào được tải")
    
    st.markdown("---")
    
    # Chat controls
    if page == "💬 Chat AI":
        st.markdown("### 💬 Điều khiển Chat")
        if st.button("🗑️ Xóa lịch sử chat", use_container_width=True):
            st.session_state.chat_messages = []
            st.rerun()
        
        if st.button("📥 Export Chat", use_container_width=True):
            if st.session_state.chat_messages:
                chat_text = "\n\n".join([f"{msg['role'].upper()}: {msg['content']}" 
                                        for msg in st.session_state.chat_messages])
                st.download_button(
                    label="Tải xuống Chat",
                    data=chat_text,
                    file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
    
    st.markdown("---")
    st.markdown("### 💡 Thông tin")
    st.info("Phiên bản: 2.0\n\nPowered by Gemini AI")

# --- MAIN CONTENT ---

# Header
st.markdown("""
<div class="main-header fade-in">
    <h1>📊 Ứng Dụng Phân Tích Báo Cáo Tài Chính</h1>
    <p>Phân tích thông minh với sức mạnh AI</p>
</div>
""", unsafe_allow_html=True)

# --- PAGE: DASHBOARD ---
if page == "📊 Dashboard":
    
    # File uploader
    uploaded_file = st.file_uploader(
        "📤 Tải file Excel Báo cáo Tài chính",
        type=['xlsx', 'xls'],
        help="File Excel cần có 3 cột: Chỉ tiêu | Năm trước | Năm sau"
    )

    if uploaded_file is not None:
        try:
            # Save to history
            if not any(f['name'] == uploaded_file.name for f in st.session_state.file_history):
                st.session_state.file_history.append({
                    'name': uploaded_file.name,
                    'time': datetime.now().strftime('%d/%m/%Y %H:%M')
                })
            
            df_raw = pd.read_excel(uploaded_file)
            df_raw.columns = ['Chỉ tiêu', 'Năm trước', 'Năm sau']
            
            df_processed = process_financial_data(df_raw.copy())
            st.session_state.df_processed = df_processed
            st.session_state.financial_data_context = df_processed.to_markdown(index=False)

            # Success message
            st.success("✅ File đã được tải và xử lý thành công!")
            
            # --- KPI CARDS ---
            st.markdown("### 🎯 Chỉ số Tổng quan")
            
            try:
                tong_tai_san = df_processed[df_processed['Chỉ tiêu'].str.contains('TỔNG CỘNG TÀI SẢN', case=False, na=False)]
                tong_ts_n = tong_tai_san['Năm sau'].iloc[0]
                tong_ts_n_1 = tong_tai_san['Năm trước'].iloc[0]
                tang_truong_tong = ((tong_ts_n - tong_ts_n_1) / tong_ts_n_1) * 100
                
                # KPI Cards
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        label="💰 Tổng Tài sản (Năm sau)",
                        value=f"{tong_ts_n:,.0f}",
                        delta=f"{tang_truong_tong:.2f}%"
                    )
                
                with col2:
                    try:
                        tsnh = df_processed[df_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)]
                        tsnh_growth = tsnh['Tốc độ tăng trưởng (%)'].iloc[0]
                        st.metric(
                            label="📈 TSNH Tăng trưởng",
                            value=f"{tsnh_growth:.2f}%",
                            delta=f"{tsnh_growth:.2f}%"
                        )
                    except:
                        st.metric(label="📈 TSNH Tăng trưởng", value="N/A")
                
                with col3:
                    try:
                        tsnh_n = df_processed[df_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)]['Năm sau'].iloc[0]
                        nnh_n = df_processed[df_processed['Chỉ tiêu'].str.contains('NỢ NGẮN HẠN', case=False, na=False)]['Năm sau'].iloc[0]
                        thanh_toan = tsnh_n / nnh_n
                        st.metric(
                            label="💧 Thanh toán Hiện hành",
                            value=f"{thanh_toan:.2f}",
                            delta="Tốt" if thanh_toan > 1 else "Cần cải thiện"
                        )
                    except:
                        st.metric(label="💧 Thanh toán Hiện hành", value="N/A")
                
                with col4:
                    avg_growth = df_processed['Tốc độ tăng trưởng (%)'].mean()
                    st.metric(
                        label="📊 Tăng trưởng TB",
                        value=f"{avg_growth:.2f}%",
                        delta=f"{avg_growth:.2f}%"
                    )
                
            except Exception as e:
                st.warning(f"Không thể tính một số chỉ số: {e}")
            
            st.markdown("---")
            
            # --- DATA TABLE với Conditional Formatting ---
            st.markdown("### 📋 Bảng Phân tích Chi tiết")
            
            def highlight_growth(val):
                """Tô màu cho tốc độ tăng trưởng"""
                try:
                    if val > 0:
                        return 'background-color: #d4edda; color: #155724'
                    elif val < 0:
                        return 'background-color: #f8d7da; color: #721c24'
                    else:
                        return ''
                except:
                    return ''
            
            styled_df = df_processed.style.format({
                'Năm trước': '{:,.0f}',
                'Năm sau': '{:,.0f}',
                'Tốc độ tăng trưởng (%)': '{:.2f}%',
                'Tỷ trọng Năm trước (%)': '{:.2f}%',
                'Tỷ trọng Năm sau (%)': '{:.2f}%'
            }).applymap(highlight_growth, subset=['Tốc độ tăng trưởng (%)'])
            
            st.dataframe(styled_df, use_container_width=True, height=400)
            
            st.markdown("---")
            
            # --- AI ANALYSIS ---
            st.markdown("### 🤖 Phân tích AI Tự động")
            
            col_btn1, col_btn2 = st.columns([1, 3])
            with col_btn1:
                analyze_btn = st.button("🚀 Yêu cầu AI Phân tích", use_container_width=True)
            
            if analyze_btn:
                api_key = st.secrets.get("GEMINI_API_KEY")
                
                if api_key:
                    with st.spinner('🔄 Đang phân tích dữ liệu...'):
                        try:
                            tsnh_n = df_processed[df_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)]['Năm sau'].iloc[0]
                            tsnh_n_1 = df_processed[df_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)]['Năm trước'].iloc[0]
                            nnh_n = df_processed[df_processed['Chỉ tiêu'].str.contains('NỢ NGẮN HẠN', case=False, na=False)]['Năm sau'].iloc[0]
                            nnh_n_1 = df_processed[df_processed['Chỉ tiêu'].str.contains('NỢ NGẮN HẠN', case=False, na=False)]['Năm trước'].iloc[0]
                            tt_n = tsnh_n / nnh_n
                            tt_n_1 = tsnh_n_1 / nnh_n_1
                            
                            data_for_ai = pd.DataFrame({
                                'Chỉ tiêu': [
                                    'Toàn bộ Bảng phân tích',
                                    'Tăng trưởng TSNH (%)',
                                    'Thanh toán hiện hành (N-1)',
                                    'Thanh toán hiện hành (N)'
                                ],
                                'Giá trị': [
                                    df_processed.to_markdown(index=False),
                                    f"{df_processed[df_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)]['Tốc độ tăng trưởng (%)'].iloc[0]:.2f}%",
                                    f"{tt_n_1:.2f}",
                                    f"{tt_n:.2f}"
                                ]
                            }).to_markdown(index=False)
                        except:
                            data_for_ai = df_processed.to_markdown(index=False)
                        
                        ai_result = get_ai_analysis(data_for_ai, api_key)
                        st.markdown("**📊 Kết quả Phân tích:**")
                        st.info(ai_result)
                else:
                    st.error("❌ Không tìm thấy API Key. Vui lòng cấu hình 'GEMINI_API_KEY' trong Secrets.")

        except ValueError as ve:
            st.error(f"❌ Lỗi cấu trúc dữ liệu: {ve}")
        except Exception as e:
            st.error(f"❌ Có lỗi xảy ra: {e}")

    else:
        st.info("👆 Vui lòng tải lên file Excel để bắt đầu phân tích")

# --- PAGE: CHARTS ---
elif page == "📈 Biểu đồ":
    if st.session_state.df_processed is not None:
        df = st.session_state.df_processed
        
        st.markdown("### 📊 Trực quan hóa Dữ liệu")
        
        tab1, tab2, tab3 = st.tabs(["📊 So sánh", "📈 Tăng trưởng", "🍰 Cơ cấu"])
        
        with tab1:
            st.plotly_chart(create_comparison_chart(df), use_container_width=True)
        
        with tab2:
            st.plotly_chart(create_growth_chart(df), use_container_width=True)
        
        with tab3:
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(create_pie_chart(df), use_container_width=True)
            with col2:
                st.markdown("#### 📌 Nhận xét:")
                st.write("Biểu đồ tròn thể hiện cơ cấu của 5 khoản mục tài sản lớn nhất.")
                st.write("Các màu sắc khác nhau giúp dễ dàng phân biệt các khoản mục.")
    else:
        st.warning("⚠️ Vui lòng tải file dữ liệu ở trang Dashboard trước")

# --- PAGE: CHAT ---
elif page == "💬 Chat AI":
    st.markdown("### 💬 Trò chuyện với Trợ lý Tài chính AI")
    
    # Suggest questions
    st.markdown("#### 💡 Câu hỏi gợi ý (Click để sử dụng):")
    
    suggest_questions = [
        "Phân tích chi tiết chỉ số thanh toán hiện hành",
        "So sánh tốc độ tăng trưởng các khoản mục chính",
        "Đánh giá rủi ro tài chính của doanh nghiệp",
        "Đưa ra khuyến nghị cải thiện tình hình tài chính",
        "Giải thích ý nghĩa của các chỉ số đã tính"
    ]
    
    cols = st.columns(3)
    for i, question in enumerate(suggest_questions):
        with cols[i % 3]:
            if st.button(f"💬 {question[:30]}...", key=f"suggest_{i}", use_container_width=True):
                st.session_state.suggested_question = question
    
    st.markdown("---")
    
    # Chat container
    chat_container = st.container()
    
    with chat_container:
        for message in st.session_state.chat_messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("💭 Nhập câu hỏi của bạn..."):
        handle_chat_input(prompt)
    
    # Handle suggested question
    if 'suggested_question' in st.session_state:
        handle_chat_input(st.session_state.suggested_question)
        del st.session_state.suggested_question

def handle_chat_input(prompt):
    """Xử lý input chat"""
    api_key = st.secrets.get("GEMINI_API_KEY")
    
    if not api_key:
        st.error("❌ Vui lòng cấu hình GEMINI_API_KEY trong Streamlit Secrets!")
        return
    
    # Add user message
    st.session_state.chat_messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get AI response
    with st.chat_message("assistant"):
        with st.spinner("🤔 Đang suy nghĩ..."):
            response = chat_with_gemini(
                prompt,
                st.session_state.financial_data_context,
                api_key
            )
            st.markdown(response)
            st.session_state.chat_messages.append({"role": "assistant", "content": response})

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>💼 Ứng dụng Phân tích Tài chính Chuyên nghiệp</p>
    <p>Powered by <strong>Google Gemini AI</strong> & <strong>Streamlit</strong></p>
</div>
""", unsafe_allow_html=True)
