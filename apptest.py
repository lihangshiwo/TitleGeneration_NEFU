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
    page_title="���ܴ���ƽ̨",
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
        st.warning("�û����Ѵ��ڣ������һ���µ��û�����")
    else:
        c.execute('INSERT INTO userstable(username,password) VALUES(%s,%s)',(username,password))
        con.commit()
        st.success("��ϲ�����ѳɹ�ע�ᡣ")
        st.info("�������ѡ�񡰵�¼��ѡ����е�¼��")

def login_user(username,password):
    if c.execute('SELECT username FROM userstable WHERE username = %s',(username)):
        c.execute('SELECT * FROM userstable WHERE username = %s AND password = %s',(username,password))
        data=c.fetchall()
        return data
    else:
        st.warning("�û��������ڣ�����ѡ��ע�ᰴť���ע�ᡣ")

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
        ## ����1�������������ɱ���
        """
    )
    st.sidebar.subheader("�����������ò���")
    # batch_size = st.sidebar.slider("batch_size", min_value=0, max_value=10, value=3)
    generate_max_len = st.sidebar.number_input("generate_max_len", min_value=0, max_value=64, value=16, step=1)
    # repetition_penalty = st.sidebar.number_input("repetition_penalty", min_value=0.0, max_value=10.0, value=1.2,
    #                                              step=0.1)
    # top_k = st.sidebar.slider("top_k", min_value=0, max_value=10, value=3, step=1)
    # top_p = st.sidebar.number_input("top_p", min_value=0.0, max_value=1.0, value=0.95, step=0.01)

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=1, type=int, help='���ɱ���ĸ���')
    parser.add_argument('--generate_max_len', default=generate_max_len, type=int, help='���ɱ������󳤶�')
    parser.add_argument('--repetition_penalty', default=1.2, type=float, help='�ظ�������')
    parser.add_argument('--top_k', default=3, type=float, help='����ʱ����������ߵĶ��ٸ����')
    parser.add_argument('--top_p', default=1, type=float, help='����ʱ���������ۼӴ��ڶ��ٵı��')
    parser.add_argument('--max_len', type=int, default=512, help='����ģ�͵���󳤶ȣ�Ҫ��config��n_ctxС')
    args = parser.parse_args()

    content = st.text_area("������������", max_chars=10000)
    if st.button("һ�����ɱ���"):
        start_message = st.empty()
        start_message.write("�������ɣ���ȴ�...")
        if(len(content)>200):
            content=content[0:200]
        start_time = time.time()
        titles = predict_one_sample(model, tokenizer, device, args, content)
        end_time = time.time()
        start_message.write("������ɣ���ʱ{}s".format(end_time - start_time))
        for i, title in enumerate(titles):
            st.text_input("���ɵı���Ϊ", title)
    st.markdown(
        """
        ## ����2��������������ժҪ
        """
    )

    content1 = st.text_area("������������", max_chars=9999)
    if st.button("һ������ժҪ"):
        start_message = st.empty()
        start_message.write("�������ɣ���ȴ�...")
        if (len(content1) > 250):
            content1 = content1[0:250]
        start_time = time.time()
        # titles1 = predict_one_sample(model, tokenizer, device, args1, content1)
        tr4s = TextRank4Sentence()
        # Ӣ�ĵ���Сд�����д��Թ��˲��޳�ͣ�ô�
        tr4s.analyze(text=content1, lower=True, source='all_filters')
        end_time = time.time()
        start_message.write("������ɣ���ʱ{}s".format(end_time - start_time))
        for item in tr4s.get_key_sentences(num=1):
            st.text_input("���ɵ�ժҪΪ",item.sentence)


def main():
    menu = ["��ҳ","��¼","ע��", "ע��"]

    if 'count' not in st.session_state:
        st.session_state.count = 0

    choice = st.sidebar.selectbox("ѡ��˵�",menu)
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

    if choice =="��ҳ":
        st.header("��ҳ")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(
                """
            ### ��Ŀ��� 
            ����Ŀ����:[2022���й���ѧ�������ƴ��� A9���⡪���ܴ���ƽ̨](http://www.cnsoftbei.com/plus/view.php?aid=729)
            
            ��Ŀ����:����Longformer��Steamlit�����ܴ���ƽ̨
            """
            )
            # st.image()
        with c2:
            st.markdown(
                """
            ### �ŶӼ�� 

            �Ŷ�����:�Ұ�NLP

            ѧУ:[������ҵ��ѧ](https://www.nefu.edu.cn/)

            ��Ŀ��Ա:� ������ 
            """
            )
            # st.image()

    elif choice =="��¼":
        placeholder = st.sidebar.empty()
        with placeholder.container():
            st.subheader("��¼����")
            username = st.text_input("�û���")
            password = st.text_input("����",type = "password")
            denglu = st.checkbox('��¼')
        if denglu:
                logged_user = login_user(username,password)
                if logged_user:
                    st.session_state.count += 1
                    if st.session_state.count >= 1:
                        placeholder.empty()
                        st.sidebar.success("���ѵ�¼�ɹ��������û����� {}".format(username))
                        writer()

                else:
                    st.sidebar.warning("�û����������벻��ȷ����������ԡ�")

    elif choice =="ע��":
        st.subheader("ע��")
        new_user = st.sidebar.text_input("�û���")
        new_password = st.sidebar.text_input("����",type = "password")

        if st.sidebar.button("ע��"):
            create_usertable()
            add_userdata(new_user,new_password)

    elif choice =="ע��":
        st.session_state.count = 0
        if st.session_state.count == 0:
            st.info("���ѳɹ�ע���������Ҫ����ѡ�����ĵ�¼��ť������¼��")



if __name__=="__main__":
    main()