import streamlit as st
import pymysql
from model import GPT2LMHeadModel
from transformers import BertTokenizer
import argparse
import os
import torch
import time
from generate_title import predict_one_sample
from textrank4zh import  TextRank4Sentence


st.set_page_config(
    page_title="智能创作平台",
    page_icon=":computer:",
    initial_sidebar_state="auto",
    layout="wide"
)
con = pymysql.connect(host="rm-bp1an500l6pntj3f5vo.mysql.rds.aliyuncs.com", user="woshilihang", password="123456abc*", db="zhinengchuangzuo", port=3306, charset="utf8")
#con = pymysql.connect(host="localhost", user="root", password="root", database="python", charset="utf8")

c = con.cursor()

def create_usertable():
    c.execute('CREATE TABLE IF NOT EXISTS userstable(username TEXT, password TEXT)')

def add_userdata(username, password):

    if c.execute('SELECT username FROM userstable WHERE username = %s',(username)):
        st.warning("用户名已存在，请更换一个新的用户名。")
    else:
        c.execute('INSERT INTO userstable(username,password) VALUES(%s,%s)',(username,password))
        con.commit()
        st.success("恭喜，您已成功注册。")
        st.info("请在左侧选择“登录”选项进行登录。")

def login_user(username,password):
    if c.execute('SELECT username FROM userstable WHERE username = %s',(username)):
        c.execute('SELECT * FROM userstable WHERE username = %s AND password = %s',(username,password))
        data=c.fetchall()
        return data
    else:
        st.warning("用户名不存在，请先选择注册按钮完成注册。")

def view_all_users():
    c.execute('SELECT * FROM userstable')
    data = c.fetchall()
    return data

@st.cache(allow_output_mutation=True)
def get_model(device, vocab_path, model_path):
    tokenizer = BertTokenizer.from_pretrained(vocab_path, do_lower_case=True)
    model = GPT2LMHeadModel.from_pretrained(model_path)
    model.to(device)
    model.eval()
    return tokenizer, model

device_ids = 0
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICE"] = str(device_ids)
device = torch.device("cuda" if torch.cuda.is_available() and int(device_ids) >= 0 else "cpu")
tokenizer, model = get_model(device, "vocab/vocab.txt", "output_dir/checkpoint-139805")

def writer():
    st.markdown(
        """
        ## 功能1：输入文章生成标题
        """
    )
    st.sidebar.subheader("标题生成配置参数")
    # batch_size = st.sidebar.slider("batch_size", min_value=0, max_value=10, value=3)
    generate_max_len = st.sidebar.number_input("generate_max_len", min_value=0, max_value=64, value=16, step=1)
    # repetition_penalty = st.sidebar.number_input("repetition_penalty", min_value=0.0, max_value=10.0, value=1.2,
    #                                              step=0.1)
    # top_k = st.sidebar.slider("top_k", min_value=0, max_value=10, value=3, step=1)
    # top_p = st.sidebar.number_input("top_p", min_value=0.0, max_value=1.0, value=0.95, step=0.01)

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=1, type=int, help='生成标题的个数')
    parser.add_argument('--generate_max_len', default=generate_max_len, type=int, help='生成标题的最大长度')
    parser.add_argument('--repetition_penalty', default=1.2, type=float, help='重复处罚率')
    parser.add_argument('--top_k', default=3, type=float, help='解码时保留概率最高的多少个标记')
    parser.add_argument('--top_p', default=1, type=float, help='解码时保留概率累加大于多少的标记')
    parser.add_argument('--max_len', type=int, default=512, help='输入模型的最大长度，要比config中n_ctx小')
    args = parser.parse_args()

    content = st.text_area("输入文章正文", max_chars=10000)
    if st.button("一键生成标题"):
        start_message = st.empty()
        start_message.write("正在生成，请等待...")
        if(len(content)>200):
            content=content[0:200]
        start_time = time.time()
        titles = predict_one_sample(model, tokenizer, device, args, content)
        end_time = time.time()
        start_message.write("生成完成，耗时{}s".format(end_time - start_time))
        for i, title in enumerate(titles):
            st.text_input("生成的标题为", title)
    st.markdown(
        """
        ## 功能2：输入文章生成摘要
        """
    )

    content1 = st.text_area("输入文章正文", max_chars=9999)
    if st.button("一键生成摘要"):
        start_message = st.empty()
        start_message.write("正在生成，请等待...")
        if (len(content1) > 250):
            content1 = content1[0:250]
        start_time = time.time()
        # titles1 = predict_one_sample(model, tokenizer, device, args1, content1)
        tr4s = TextRank4Sentence()
        # 英文单词小写，进行词性过滤并剔除停用词
        tr4s.analyze(text=content1, lower=True, source='all_filters')
        end_time = time.time()
        start_message.write("生成完成，耗时{}s".format(end_time - start_time))
        for item in tr4s.get_key_sentences(num=1):
            st.text_input("生成的摘要为",item.sentence)


def main():
    menu = ["首页","登录","注册", "注销"]

    if 'count' not in st.session_state:
        st.session_state.count = 0

    choice = st.sidebar.selectbox("选项菜单",menu)
    st.sidebar.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 250px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 250px;
        margin-left: -250px;
    }
    </style>
    """,
    unsafe_allow_html=True,)

    if choice =="首页":
        st.header("首页")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(
                """
            ### 项目简介 
            本项目来自:[2022年中国大学生软件设计大赛 A9赛题—智能创作平台](http://www.cnsoftbei.com/plus/view.php?aid=729)
            
            项目概述:基于Longformer和Steamlit的智能创作平台
            """
            )
            # st.image()
        with c2:
            st.markdown(
                """
            ### 团队简介 

            团队名称:我爱NLP

            学校:[东北林业大学](https://www.nefu.edu.cn/)

            项目成员:李航 季子扬 
            """
            )
            # st.image()

    elif choice =="登录":
        placeholder = st.sidebar.empty()
        with placeholder.container():
            st.subheader("登录区域")
            username = st.text_input("用户名")
            password = st.text_input("密码",type = "password")
            denglu = st.checkbox('登录')
        if denglu:
                logged_user = login_user(username,password)
                if logged_user:
                    st.session_state.count += 1
                    if st.session_state.count >= 1:
                        placeholder.empty()
                        st.sidebar.success("您已登录成功，您的用户名是 {}".format(username))
                        writer()

                else:
                    st.sidebar.warning("用户名或者密码不正确，请检查后重试。")

    elif choice =="注册":
        st.subheader("注册")
        new_user = st.sidebar.text_input("用户名")
        new_password = st.sidebar.text_input("密码",type = "password")

        if st.sidebar.button("注册"):
            create_usertable()
            add_userdata(new_user,new_password)

    elif choice =="注销":
        st.session_state.count = 0
        if st.session_state.count == 0:
            st.info("您已成功注销，如果需要，请选择左侧的登录按钮继续登录。")



if __name__=="__main__":
    main()