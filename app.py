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
        self.fc3 = nn.Linear(32, action_size) [cite: 41]

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x) [cite: 42]

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
# Hocanın o "100 puanlık" notu için Kaygan Zemin ayarı
is_slippery = st.sidebar.checkbox("Slippery Mode (Kaygan Zemin)", value=False, help="Açıkken ajan attığı adımda %100 başarılı olamaz, kayabilir. Bu ajanı daha güvenli (safest) uzun yolları seçmeye zorlar.")

if st.sidebar.button("🎲 Yeni Rastgele Harita Üret"):
    st.session_state.random_map = generate_random_map(size=4, p=0.8) [cite: 68]
    st.session_state.env_reset_needed = True
    st.rerun()

# Harita hafızada yoksa oluştur
if 'random_map' not in st.session_state:
    st.session_state.random_map = generate_random_map(size=4, p=0.8) [cite: 67]

# Yön isimleri
actions_str = ["Sol (0)", "Aşağı (1)", "Sağ (2)", "Yukarı (3)"] [cite: 11]

st.markdown("---")

# =====================================================================
# MOD 1: KLASİK Q-LEARNING
# =====================================================================
if secili_mod == "Klasik Q-Learning":
    st.header("🧠 Mod 1: Tablosal (Klasik) Q-Learning")
    st.write("Ajan, ortamı keşfederek değerleri bir Q-Tablosuna (Matris) kaydeder.")
    
    col_ayarlar, _ = st.columns([1, 2])
    with col_ayarlar:
        episodes = st.number_input("Bölüm Sayısı", 100, 5000, 2000, 100) [cite: 68]
        alpha = st.slider("Öğrenme Oranı (Alpha)", 0.01, 1.0, 0.8) [cite: 68]
        gamma = st.slider("İndirim Faktörü (Gamma)", 0.1, 1.0, 0.95, help="Gelecekteki ödüllerin önemi. Slippery Mode açıkken bu değeri yüksek tutmak ajanın uzağı görmesini sağlar.") [cite: 68]

    if st.button("🚀 Q-Learning Eğitimini Başlat"):
        env = gym.make("FrozenLake-v1", desc=st.session_state.random_map, is_slippery=is_slippery, render_mode="rgb_array")
        state_size = env.observation_space.n
        action_size = env.action_space.n
        q_table = np.zeros((state_size, action_size)) [cite: 69]
        
        epsilon, min_epsilon, epsilon_decay = 1.0, 0.01, 0.995 [cite: 69]
        
        status_text = st.empty()
        status_text.info("Eğitim yapılıyor, lütfen bekleyin...")
        progress_bar = st.progress(0)
        
        for episode in range(episodes):
            state, _ = env.reset()
            terminated = truncated = False
            
            while not (terminated or truncated):
                if random.uniform(0, 1) < epsilon:
                    action = env.action_space.sample()
                else:
                    action = np.argmax(q_table[state, :]) [cite: 70, 71]
                    
                new_state, reward, terminated, truncated, _ = env.step(action)
                
                if terminated and reward == 0:
                    reward = -1  [cite: 5, 71]
                    
                q_table[state, action] = q_table[state, action] + alpha * (reward + gamma * np.max(q_table[new_state, :]) - q_table[state, action]) [cite: 72]
                state = new_state
                
            epsilon = max(min_epsilon, epsilon * epsilon_decay) [cite: 73]
            if episode % (episodes // 100) == 0: progress_bar.progress(episode / episodes)
                
        progress_bar.progress(1.0)
        status_text.success("Eğitim Tamamlandı!")
        
        col_tablo, col_izle = st.columns(2)
        with col_tablo:
            st.subheader("📊 Öğrenilen Q-Tablosu")
            df_q = pd.DataFrame(q_table, columns=actions_str) [cite: 74]
            df_q.index.name = "Durum (State)"
            st.dataframe(df_q.style.background_gradient(cmap='RdYlGn', axis=1))

        with col_izle:
            st.subheader("🤖 Ajanın Hedefe Gidişi")
            image_placeholder = st.empty()
            st.info("Canlı izleme döngüsü çalışıyor... Kapatmak için sağ üstteki Stop'a basın.")
            
            # Sonsuz izleme döngüsü
            while True:
                state, _ = env.reset()
                term = trunc = False
                while not (term or trunc):
                    action = np.argmax(q_table[state, :]) [cite: 75]
                    state, r, term, trunc, _ = env.step(action)
                    frame = env.render()
                    image_placeholder.image(frame, channels="RGB", width=400, caption=f"Anlık Durum: {state}")
                    time.sleep(0.4)
                time.sleep(1.5)
        env.close()

# =====================================================================
# MOD 2: DEEP Q-LEARNING (ÇİFT AĞ)
# =====================================================================
elif secili_mod == "Deep Q-Learning (Çift Ağ)":
    st.header("🧠 Mod 2: Profesyonel DQN (Çift Ağlı)")
    st.write("Ajan, Hareketli Hedef (Moving Target) problemini çözmek için **Policy Network (Ana Ağ)** ve **Target Network (Hedef Ağ)** kullanır.") [cite: 41]
    
    col_ayarlar, _ = st.columns([1, 2])
    with col_ayarlar:
        episodes_dqn = st.number_input("Bölüm Sayısı", 100, 3000, 1000, 100) [cite: 42]
        lr = st.slider("Öğrenme Oranı (Learning Rate)", 0.0001, 0.01, 0.001, 0.0001) [cite: 42]
        gamma_dqn = st.slider("İndirim Faktörü (Gamma)", 0.1, 1.0, 0.95) [cite: 42]
        target_update_freq = st.slider("Hedef Ağ Güncelleme Sıklığı", 1, 50, 10) [cite: 42]

    if st.button("🚀 Çift Ağlı DQN Eğitimi Başlat"):
        env = gym.make("FrozenLake-v1", desc=st.session_state.random_map, is_slippery=is_slippery, render_mode="rgb_array")
        state_size = env.observation_space.n
        action_size = env.action_space.n

        policy_net = DQN(state_size, action_size) [cite: 43]
        target_net = DQN(state_size, action_size) [cite: 43]
        target_net.load_state_dict(policy_net.state_dict()) [cite: 43]
        target_net.eval() [cite: 43]

        optimizer = optim.Adam(policy_net.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        memory = deque(maxlen=2000)
        batch_size = 32

        epsilon, min_epsilon, epsilon_decay = 1.0, 0.01, 0.995
        loss_history, reward_history = [], [] [cite: 44]
        
        status_text = st.empty()
        status_text.info("DQN Eğitimi başlıyor. Target Network sayesinde stabil öğrenme gerçekleşecek...") [cite: 44]
        progress_bar = st.progress(0)
        
        for episode in range(episodes_dqn): [cite: 45]
            state, _ = env.reset()
            term = trunc = False
            total_reward = 0
            
            while not (term or trunc):
                state_tensor = state_to_tensor(state, state_size) [cite: 45]
                
                if random.uniform(0, 1) < epsilon:
                    action = env.action_space.sample()
                else:
                    with torch.no_grad():
                        action = torch.argmax(policy_net(state_tensor)).item() [cite: 47]
                        
                new_state, reward, term, trunc, _ = env.step(action)
                total_reward += reward
                if term and reward == 0: reward = -1 [cite: 48]
                    
                memory.append((state, action, reward, new_state, term or trunc)) [cite: 48]
                state = new_state
                
                if len(memory) >= batch_size: [cite: 49]
                    batch = random.sample(memory, batch_size)
                    b_states = torch.stack([state_to_tensor(s[0], state_size) for s in batch]) [cite: 49]
                    b_actions = torch.tensor([s[1] for s in batch], dtype=torch.int64).unsqueeze(1) [cite: 49]
                    b_rewards = torch.tensor([s[2] for s in batch], dtype=torch.float32) [cite: 50]
                    b_next_states = torch.stack([state_to_tensor(s[3], state_size) for s in batch]) [cite: 50]
                    b_dones = torch.tensor([s[4] for s in batch], dtype=torch.float32) [cite: 50]
                    
                    current_q = policy_net(b_states).gather(1, b_actions).squeeze() [cite: 51]
                    with torch.no_grad():
                        max_next_q = target_net(b_next_states).max(1)[0] [cite: 51]
                        target_q = b_rewards + (gamma_dqn * max_next_q * (1 - b_dones)) [cite: 52]
                        
                    loss = loss_fn(current_q, target_q)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    loss_history.append(loss.item()) [cite: 53]

            reward_history.append(total_reward)
            epsilon = max(min_epsilon, epsilon * epsilon_decay) [cite: 53, 54]
            if episode % target_update_freq == 0: target_net.load_state_dict(policy_net.state_dict()) [cite: 54]
            if episode % (episodes_dqn // 100) == 0: progress_bar.progress(episode / episodes_dqn) [cite: 54, 55]
                
        progress_bar.progress(1.0)
        status_text.success("Eğitim Tamamlandı!") [cite: 55]
        
        st.markdown("---")
        st.subheader("🎬 1. Ajanın Hedefe Gidişi (Canlı İzleme)") [cite: 56]
        col_img1, col_img2, col_img3 = st.columns([1, 2, 1]) [cite: 56]
        with col_img2: image_placeholder = st.empty()
            
        col1, col2 = st.columns(2) [cite: 57]
        with col1:
            st.subheader("📉 2. Eğitim Kaybı (Loss)") [cite: 57]
            if loss_history:
                fig1, ax1 = plt.subplots() [cite: 58]
                ax1.plot(loss_history, color='purple') [cite: 58]
                st.pyplot(fig1)

        with col2:
            st.subheader("📊 3. Sanal Q-Tablosu (Ana Ağ)") [cite: 59]
            q_table_approx = np.zeros((state_size, action_size))
            with torch.no_grad():
                for s in range(state_size):
                    q_table_approx[s] = policy_net(state_to_tensor(s, state_size)).numpy() [cite: 60]
            df_q = pd.DataFrame(q_table_approx, columns=actions_str)
            st.dataframe(df_q.style.background_gradient(cmap='RdYlGn', axis=1))

        st.subheader("📈 4. Bölüm Başına Ödül (Hareketli Ortalama)") [cite: 61]
        if reward_history:
            smoothed = pd.Series(reward_history).rolling(window=50, min_periods=1).mean() [cite: 61]
            fig2, ax2 = plt.subplots(figsize=(10, 3))
            ax2.plot(smoothed, color='green') [cite: 61]
            st.pyplot(fig2)

        while True:
            state, _ = env.reset()
            term = trunc = False
            while not (term or trunc):
                with torch.no_grad():
                    action = torch.argmax(policy_net(state_to_tensor(state, state_size))).item() [cite: 63]
                state, r, term, trunc, _ = env.step(action)
                frame = env.render()
                image_placeholder.image(frame, channels="RGB", use_container_width=True) [cite: 64]
                time.sleep(0.4)
            time.sleep(1.5) [cite: 66]
        env.close()

# =====================================================================
# MOD 3: SİNGLE STEP (MANUEL KONTROL)
# =====================================================================
elif secili_mod == "Single Step (Manuel Mod)":
    st.header("🕹️ Mod 3: Single Step Mode (Manuel Kontrol)")
    st.write("Ajanın yerine geçin! RL etkileşim döngüsünü (Action -> Environment -> Reward -> New State) canlı olarak deneyimleyin.")
    
    # Hafıza kontrolü
    if 'manuel_env' not in st.session_state or st.session_state.get('env_reset_needed', False):
        st.session_state.manuel_env = gym.make("FrozenLake-v1", desc=st.session_state.random_map, is_slippery=is_slippery, render_mode="rgb_array")
        st.session_state.m_state, _ = st.session_state.manuel_env.reset()
        st.session_state.m_game_over = False
        st.session_state.m_reward = 0
        st.session_state.m_step_count = 0
        st.session_state.m_info = {}
        st.session_state.env_reset_needed = False
        
    if st.button("🔄 Oyunu Baştan Başlat"):
        st.session_state.m_state, _ = st.session_state.manuel_env.reset()
        st.session_state.m_game_over = False
        st.session_state.m_reward = 0
        st.session_state.m_step_count = 0
        st.session_state.m_info = {}
        st.rerun()

    st.markdown("---")
    col_game, col_info = st.columns([1, 1])

    with col_game:
        frame = st.session_state.manuel_env.render()
        st.image(frame, channels="RGB", use_container_width=True)
        
        if not st.session_state.m_game_over:
            st.markdown("### Yön Kontrolleri")
            c1, c2, c3 = st.columns([1, 1, 1])
            with c2: up = st.button("⬆️ Yukarı (3)")
            c4, c5, c6 = st.columns([1, 1, 1])
            with c4: left = st.button("⬅️ Sol (0)")
            with c5: down = st.button("⬇️ Aşağı (1)")
            with c6: right = st.button("➡️ Sağ (2)")

            action = None
            if left: action = 0
            elif down: action = 1
            elif right: action = 2
            elif up: action = 3

            if action is not None:
                new_state, reward, term, trunc, info = st.session_state.manuel_env.step(action)
                st.session_state.m_state = new_state
                st.session_state.m_reward = reward
                st.session_state.m_step_count += 1
                st.session_state.m_info = info
                
                if term or trunc:
                    st.session_state.m_game_over = True
                    if reward == 1: st.balloons()
                st.rerun()
        else:
            if st.session_state.m_reward == 1:
                st.success("🏆 TEBRİKLER! Hedefe ulaştınız.")
            else:
                st.error("💥 EYVAH! Deliğe düştünüz.")

    with col_info:
        st.subheader("📊 Ortam Değişkenleri (Environment Dynamics)")
        st.write("Gymnasium ortamının ajana döndürdüğü anlık veriler:")
        
        satir = st.session_state.m_state // 4
        sutun = st.session_state.m_state % 4
        
        st.info(f"**📍 Mevcut Durum (State):** {st.session_state.m_state} \n\n *(Matris Koordinatı: Satır {satir}, Sütun {sutun})*")
        st.warning(f"**💰 Anlık Ödül (Reward):** {st.session_state.m_reward}")
        st.success(f"**🛑 Oyun Bitti mi? (Terminated):** {st.session_state.m_game_over}")
        
        prob = st.session_state.m_info.get('prob', 'Bilinmiyor')
        st.text(f"🎲 Eylem Başarı Olasılığı (prob): {prob} \n(Slippery Mode: {'AÇIK' if is_slippery else 'KAPALI'})")
        st.metric(label="Geçen Toplam Adım", value=st.session_state.m_step_count)
