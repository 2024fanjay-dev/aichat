import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM

st.set_page_config(
    page_title="甜甜陪伴",
    page_icon="🍬",
    layout="centered"
)

# 顶部风格
st.title("🍬 甜甜陪伴")
st.markdown("#### 温柔又甜的小陪伴，一直都在～")

# 加载轻量模型
@st.cache_resource
def load_model():
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    )
    return tokenizer, model

tokenizer, model = load_model()

# 超级甜美人设
system_prompt = """
你是一个温柔、甜美、软萌、会安慰人的陪伴AI。
说话轻声细语，温暖治愈，像甜甜的小太阳。
会认真倾听，会记住你说的话，不抬杠、不敷衍。
语气软一点，短一点，贴心一点。
"""

# 聊天记忆
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": system_prompt}
    ]

# 显示聊天记录
for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user", avatar="👤"):
            st.markdown(msg["content"])
    elif msg["role"] == "assistant":
        with st.chat_message("assistant", avatar="🍬"):
            st.markdown(msg["content"])

# 输入框
prompt = st.chat_input("和我说说话吧～")

if prompt:
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 生成回答
    inputs = tokenizer.apply_chat_template(
        st.session_state.messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    )

    outputs = model.generate(
        inputs,
        max_new_tokens=256,
        temperature=0.6,
        top_p=0.95,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

    response = tokenizer.decode(
        outputs[0][len(inputs[0]):],
        skip_special_tokens=True
    )

    with st.chat_message("assistant", avatar="🍬"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
