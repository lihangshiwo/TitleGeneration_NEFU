"""
    文件说明:streamlit实现web部署
    
"""
import streamlit as st
from model import GPT2LMHeadModel
from transformers import BertTokenizer
import argparse
import os
import torch
import time
from generate_title import predict_one_sample

st.set_page_config(
    page_title="智能创作平台",
    page_icon=":rainbow:",
    initial_sidebar_state="auto",
    layout="wide"
)

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
    batch_size = st.sidebar.slider("batch_size", min_value=0, max_value=10, value=3)
    generate_max_len = st.sidebar.number_input("generate_max_len", min_value=0, max_value=64, value=32, step=1)
    repetition_penalty = st.sidebar.number_input("repetition_penalty", min_value=0.0, max_value=10.0, value=1.2,
                                                 step=0.1)
    top_k = st.sidebar.slider("top_k", min_value=0, max_value=10, value=3, step=1)
    top_p = st.sidebar.number_input("top_p", min_value=0.0, max_value=1.0, value=0.95, step=0.01)

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=batch_size, type=int, help='生成标题的个数')
    parser.add_argument('--generate_max_len', default=generate_max_len, type=int, help='生成标题的最大长度')
    parser.add_argument('--repetition_penalty', default=repetition_penalty, type=float, help='重复处罚率')
    parser.add_argument('--top_k', default=top_k, type=float, help='解码时保留概率最高的多少个标记')
    parser.add_argument('--top_p', default=top_p, type=float, help='解码时保留概率累加大于多少的标记')
    parser.add_argument('--max_len', type=int, default=512, help='输入模型的最大长度，要比config中n_ctx小')
    args = parser.parse_args()

    content = st.text_area("输入文章正文", max_chars=512)
    if st.button("一键生成标题"):
        start_message = st.empty()
        start_message.write("正在生成，请等待...")
        start_time = time.time()
        titles = predict_one_sample(model, tokenizer, device, args, content)
        end_time = time.time()
        start_message.write("生成完成，耗时{}s".format(end_time - start_time))
        for i, title in enumerate(titles):
            st.text_input("第{}个结果".format(i + 1), title)
    # else:
    #     st.stop()
    st.markdown(
        """
        ## 功能2：输入文章生成摘要
        """
    )

    content1 = st.text_area("输入文章正文", max_chars=511)
    if st.button("一键生成摘要"):
        start_message = st.empty()
        start_message.write("正在生成，请等待...")
        start_time = time.time()
        titles1 = predict_one_sample(model, tokenizer, device, args, content1)
        end_time = time.time()
        start_message.write("生成完成，耗时{}s".format(end_time - start_time))
        for j, title in enumerate(titles1):
            st.text_input("第{}个结果".format(j + 1), title)
    st.markdown(
        """
    ### About Us
    本项目来自:[2022年中国大学生软件设计大赛 A9赛题—智能创作平台](http://www.cnsoftbei.com/plus/view.php?aid=729)

    团队名称:我爱NLP

    学校:[东北林业大学](https://www.nefu.edu.cn/)

    项目成员:李航 季子扬 孙泽宏 李姗珊
    """

    )


if __name__ == '__main__':
    writer()
