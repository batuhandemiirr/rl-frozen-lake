import streamlit as st
import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
import numpy as np
import pandas as pd
import time
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import matplotlib.pyplot as plt

# --- 1. SİNİR AĞI MODELİ (DQN İÇİN) ---
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

def state_to_tensor(state, state_size):
    tensor = torch.zeros(state_size)
    tensor[state] = 1.0
    return tensor

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="RL Frozen Lake Projesi", layout="wide", page_icon="🧊")
st.title("🧊 Pekiştirmeli Öğrenme (RL) Eğitim Platformu")
st.write("Klasik Q-Learning, Deep Q-Learning ve Manuel Kontrol modlarını tek bir ekranda deneyimleyin.")

# --- KENAR ÇUBUĞU (ANA MENÜ VE ORTAK AYARLAR) ---
st.sidebar.title("⚙️ Kontrol Paneli")
secili_mod = st.sidebar.radio("Çalışma Modunu Seçin:", ["Klasik Q-Learning", "Deep Q-Learning (Çift Ağ)", "Single Step (Manuel Mod)"])

st.sidebar.markdown("---")
st.sidebar.subheader("🌍 Çevre (Environment) Ayarları")
is_slippery = st.sidebar.checkbox("Slippery Mode (Kaygan Zemin)", value=False)

if st.sidebar.button("🎲 Yeni Rastgele Harita Üret"):
    st.session_state.random_map = generate_random_map(size=4, p=0.8)
    st.session_state.env_reset_needed = True
    st.rerun()

if 'random_map' not in st.session_state:
    st.session_state.random_map = generate_random_map(size=4, p=0.8)

actions_str = ["Sol (0)", "Aşağı (1)", "Sağ (2)", "Yukarı (3)"]

st.markdown("---")

# =====================================================================
# MOD 1: KLASİK Q-LEARNING
# =====================================================================
if secili_mod == "Klasik Q-Learning":
    st.header("🧠 Mod 1: Tablosal (Klasik) Q-Learning")
    
    col_ayarlar, _ = st.columns([1, 2])
    with col_ayarlar:
        episodes = st.number_input("Bölüm Sayısı", 100, 5000, 2000, 100)
        alpha = st.slider("Öğrenme Oranı (Alpha)", 0.01, 1.0, 0.8)
        gamma = st.slider("İndirim Faktörü (Gamma)", 0.1, 1.0, 0.95)

    if st.button("🚀 Q-Learning Eğitimini Başlat"):
        env = gym.make("FrozenLake-v1", desc=st.session_state.random_map, is_slippery=is_slippery, render_mode="rgb_array")
        state_size = env.observation_space.n
        action_size = env.action_space.n
        q_table = np.zeros((state_size, action_size))
        
        epsilon, min_epsilon, epsilon_decay = 1.0, 0.01, 0.995
        
        status_text = st.empty()
        status_text.info("Eğitim yapılıyor...")
        progress_bar = st.progress(0)
        
        for episode in range(episodes):
            state, _ = env.reset()
            terminated = truncated = False
            while not (terminated or truncated):
                if random.uniform(0, 1) < epsilon:
                    action = env.action_space.sample()
                else:
                    action = np.argmax(q_table[state, :])
                new_state, reward, terminated, truncated, _ = env.step(action)
                if terminated and reward == 0: reward = -1 
                q_table[state, action] = q_table[state, action] + alpha * (reward + gamma * np.max(q_table[new_state, :]) - q_table[state, action])
                state = new_state
            epsilon = max(min_epsilon, epsilon * epsilon_decay)
            if episode % (episodes // 100) == 0: progress_bar.progress(episode / episodes)
                
        progress_bar.progress(1.0)
        status_text.success("Eğitim Tamamlandı!")
        
        col_tablo, col_izle = st.columns(2)
        with col_tablo:
            st.subheader("📊 Q-Tablosu")
            st.dataframe(pd.DataFrame(q_table, columns=actions_str).style.background_gradient(cmap='RdYlGn', axis=1))

        with col_izle:
            st.subheader("🤖 Ajanı İzle")
            image_placeholder = st.empty()
            while True:
                state, _ = env.reset()
                term = trunc = False
                while not (term or trunc):
                    action = np.argmax(q_table[state, :])
                    state, r, term, trunc, _ = env.step(action)
                    frame = env.render()
                    image_placeholder.image(frame, channels="RGB", width=400)
                    time.sleep(0.3)
                time.sleep(1)
        env.close()

# =====================================================================
# MOD 2: DEEP Q-LEARNING
# =====================================================================
elif secili_mod == "Deep Q-Learning (Çift Ağ)":
    st.header("🧠 Mod 2: Deep Q-Learning (DQN)")
    
    col_ayarlar, _ = st.columns([1, 2])
    with col_ayarlar:
        episodes_dqn = st.number_input("Bölüm Sayısı", 100, 3000, 1000, 100)
        lr = st.slider("Öğrenme Oranı", 0.0001, 0.01, 0.001, 0.0001)

    if st.button("🚀 DQN Eğitimi Başlat"):
        env = gym.make("FrozenLake-v1", desc=st.session_state.random_map, is_slippery=is_slippery, render_mode="rgb_array")
        state_size = env.observation_space.n
        action_size = env.action_space.n

        policy_net = DQN(state_size, action_size)
        target_net = DQN(state_size, action_size)
        target_net.load_state_dict(policy_net.state_dict())
        optimizer = optim.Adam(policy_net.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        memory = deque(maxlen=2000)

        epsilon, min_epsilon, epsilon_decay = 1.0, 0.01, 0.995
        progress_bar = st.progress(0)
        
        for episode in range(episodes_dqn):
            state, _ = env.reset()
            term = trunc = False
            while not (term or trunc):
                state_t = state_to_tensor(state, state_size)
                if random.uniform(0, 1) < epsilon:
                    action = env.action_space.sample()
                else:
                    with torch.no_grad(): action = torch.argmax(policy_net(state_t)).item()
                new_state, reward, term, trunc, _ = env.step(action)
                if term and reward == 0: reward = -1
                memory.append((state, action, reward, new_state, term or trunc))
                state = new_state
                if len(memory) >= 32:
                    batch = random.sample(memory, 32)
                    b_s = torch.stack([state_to_tensor(s[0], state_size) for s in batch])
                    b_a = torch.tensor([s[1] for s in batch]).unsqueeze(1)
                    b_r = torch.tensor([s[2] for s in batch], dtype=torch.float32)
                    b_ns = torch.stack([state_to_tensor(s[3], state_size) for s in batch])
                    b_d = torch.tensor([s[4] for s in batch], dtype=torch.float32)
                    
                    current_q = policy_net(b_s).gather(1, b_a).squeeze()
                    max_next_q = target_net(b_ns).max(1)[0].detach()
                    target_q = b_r + (0.95 * max_next_q * (1 - b_d))
                    loss = loss_fn(current_q, target_q)
                    optimizer.zero_grad(); loss.backward(); optimizer.step()

            epsilon = max(min_epsilon, epsilon * epsilon_decay)
            if episode % 10 == 0: target_net.load_state_dict(policy_net.state_dict())
            if episode % (episodes_dqn // 100) == 0: progress_bar.progress(episode / episodes_dqn)
                
        st.success("DQN Eğitimi Tamamlandı!")
        image_placeholder = st.empty()
        while True:
            state, _ = env.reset()
            term = trunc = False
            while not (term or trunc):
                with torch.no_grad(): action = torch.argmax(policy_net(state_to_tensor(state, state_size))).item()
                state, r, term, trunc, _ = env.step(action)
                frame = env.render()
                image_placeholder.image(frame, channels="RGB", use_container_width=True)
                time.sleep(0.3)
            time.sleep(1)

# =====================================================================
# MOD 3: MANUEL KONTROL
# =====================================================================
elif secili_mod == "Single Step (Manuel Mod)":
    st.header("🕹️ Mod 3: Manuel Kontrol")
    if 'manuel_env' not in st.session_state or st.session_state.get('env_reset_needed', False):
        st.session_state.manuel_env = gym.make("FrozenLake-v1", desc=st.session_state.random_map, is_slippery=is_slippery, render_mode="rgb_array")
        st.session_state.m_state, _ = st.session_state.manuel_env.reset()
        st.session_state.m_game_over = False
        st.session_state.env_reset_needed = False
        
    if st.button("🔄 Yeniden Başlat"):
        st.session_state.m_state, _ = st.session_state.manuel_env.reset()
        st.session_state.m_game_over = False
        st.rerun()

    frame = st.session_state.manuel_env.render()
    st.image(frame, channels="RGB", use_container_width=True)
    
    if not st.session_state.m_game_over:
        c1, c2, c3, c4 = st.columns(4)
        move = None
        if c1.button("⬅️ Sol (0)"): move = 0
        if c2.button("⬇️ Aşağı (1)"): move = 1
        if c3.button("➡️ Sağ (2)"): move = 2
        if c4.button("⬆️ Yukarı (3)"): move = 3
        
        if move is not None:
            st.session_state.m_state, r, term, trunc, _ = st.session_state.manuel_env.step(move)
            if term or trunc: st.session_state.m_game_over = True
            st.rerun()
    else:
        st.warning("Oyun bitti! Yeniden başlat butonuna basın.")
