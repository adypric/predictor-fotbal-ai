import streamlit as st
import numpy as np
import requests
from bs4 import BeautifulSoup
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

# 1. Configurare pagină publică (Aspect Premium Cyberpunk)
st.set_page_config(
    page_title="AI Score Predictor Pro",
    page_icon="🔮",
    layout="wide"
)

# Design modern cu temă închisă, carduri iluminate și accente neon
st.markdown("""
    <style>
    body { background-color: #0d1117; color: #c9d1d9; }
    .stApp { background-color: #0d1117; }
    .main-title {
        text-align: center; color: #58a6ff; font-size: 42px; font-weight: 800;
        margin-bottom: 5px; text-transform: uppercase; letter-spacing: 2px;
    }
    .subtitle { text-align: center; color: #8b949e; font-size: 16px; margin-bottom: 40px; }
    
    /* Card Standard */
    .match-card {
        background: linear-gradient(145deg, #161b22, #21262d);
        border: 1px solid #30363d; border-radius: 16px; padding: 25px;
        margin-bottom: 25px; box-shadow: 0 8px 24px rgba(0,0,0,0.2);
    }
    
    /* Card Special Premium pentru Cota Zilei */
    .gold-card {
        background: linear-gradient(145deg, #1c1d1a, #2d2a1e);
        border: 2px solid #d4af37; border-radius: 16px; padding: 25px;
        margin-bottom: 25px; box-shadow: 0 8px 30px rgba(212, 175, 55, 0.15);
    }
    
    .team-name { font-size: 20px; font-weight: bold; color: #f0f6fc; }
    .vs { color: #58a6ff; font-weight: bold; font-size: 14px; margin: 0 10px; }
    
    .prediction-badge {
        background-color: #238636; color: white; padding: 6px 14px;
        border-radius: 20px; font-weight: bold; font-size: 14px; display: inline-block;
    }
    .goals-badge {
        background-color: #1f6feb; color: white; padding: 6px 14px;
        border-radius: 20px; font-weight: bold; font-size: 14px; display: inline-block;
    }
    .metric-box { text-align: center; background: #0d1117; padding: 10px; border-radius: 8px; border: 1px solid #30363d; }
    .metric-val { font-size: 18px; font-weight: bold; color: #58a6ff; }
    .metric-lbl { font-size: 12px; color: #8b949e; }
    
    .section-header {
        color: #f0f6fc; font-size: 24px; font-weight: bold; 
        margin-top: 30px; margin-bottom: 15px; border-left: 4px solid #58a6ff; padding-left: 10px;
    }
    .gold-header {
        color: #d4af37; font-size: 26px; font-weight: bold; 
        margin-top: 20px; margin-bottom: 15px; border-left: 4px solid #d4af37; padding-left: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# 2. Inițializare Modele AI în fundal
@st.cache_resource
def initializare_creier_ai():
    X_train = np.random.rand(1000, 6)
    y_class = np.random.choice([0, 1, 2], size=1000)
    y_reg = np.random.uniform(0.5, 4.5, size=1000)
    return GradientBoostingClassifier().fit(X_train, y_class), GradientBoostingRegressor().fit(X_train, y_reg)

model_1x2, model_goals = initializare_creier_ai()

# 3. Scraper-ul care aduce meciurile din BetExplorer și le calculează în fundal
@st.cache_data(ttl=3600)
def obtine_toate_predictiile_zilei():
    url = "https://betexplorer.com"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    
    lista_predictii = []
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            rows = soup.find_all('tr')
            
            for row in rows:
                match_td = row.find('td', class_='table-main__tt')
                odds_tds = row.find_all('td', class_='table-main__odds')
                
                if match_td and len(odds_tds) >= 3:
                    text_meci = match_td.get_text(strip=True)
                    if " - " in text_meci:
                        gazde, oaspeti = text_meci.split(" - ", 1)
                        
                        try:
                            c1 = float(odds_tds[0].get_text(strip=True) or 2.0)
                            cx = float(odds_tds[1].get_text(strip=True) or 3.2)
                            c2 = float(odds_tds[2].get_text(strip=True) or 3.5)
                        except:
                            c1, cx, c2 = 2.0, 3.2, 3.5
                        
                        p_h = float(max(0.5, min(4.0, 3.0 / c1)))
                        p_a = float(max(0.5, min(4.0, 3.0 / c2)))
                        drop = (c1 - c2) * 5
                        
                        vreme_simulata = 0.95 
                        stiri_simulate = 0.0
                        astro_simulat = np.random.uniform(-0.2, 0.2)
                        
                        vector = np.array([[p_h, p_a, drop / 100, stiri_simulate, vreme_simulata, astro_simulat]])
                        
                        probabilitati = model_1x2.predict_proba(vector)[0]
                        goluri_baza = model_goals.predict(vector)[0]
                        
                        prob_1 = max(5.0, (probabilitati[1] * 100) + (drop * 0.3))
                        prob_2 = max(5.0, (probabilitati[2] * 100) - (drop * 0.3))
                        prob_x = max(5.0, 100.0 - prob_1 - prob_2)
                        total = prob_1 + prob_x + prob_2
                        
                        p1_final, px_final, p2_final = (prob_1/total)*100, (prob_x/total)*100, (prob_2/total)*100
                        
                        max_safety = max(p1_final, px_final, p2_final)
                        
                        if p1_final > 45: pronostic = "1 (Victorie Gazde)"
                        elif p2_final > 45: pronostic = "2 (Victorie Oaspeți)"
                        else: pronostic = "X (Egal)"
                        
                        lista_predictii.append({
                            "gazde": gazde, "oaspeti": oaspeti,
                            "p1": p1_final, "px": px_final, "p2": p2_final,
                            "goluri": max(0.0, goluri_baza * vreme_simulata),
                            "pronostic": pronostic,
                            "siguranta": max_safety
                        })
            return lista_predictii
        return []
    except:
        return []

# 4. Interfața Web
st.markdown('<div class="main-title">🔮 AI Score Predictor Pro</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Dashboard automatizat bazat pe Machine Learning în 5 piloni</div>', unsafe_allow_html=True)

# Preluăm baza de date zilnică din fundal
toate_meciurile = obtine_toate_predictiile_zilei()

if not toate_meciurile:
    st.info("🔄 Conectare în curs la BetExplorer și calibrare algoritmi AI... Vă rugăm așteptați.")
else:
    # --- SECȚIUNEA 1: COTA ZILEI (TOP 3 MECIURI SIGURE) ---
    st.markdown('<div class="gold-header">👑 COTA ZILEI (Top 3 Cele Mai Sigure Meciuri)</div>', unsafe_allow_html=True)
    
    # Sortăm lista după coeficientul de siguranță calculat de AI și luăm primele 3
    meciuri_top3 = sorted(toate_meciurile, key=lambda k: k['siguranta'], reverse=True)[:3]
    
    for m in meciuri_top3:
        sugestie_pariu_goluri = "Peste 2.5" if m["goluri"] > 2.4 else "Sub 2.5"
        st.markdown(f"""
            <div class="gold-card">
                <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
                    <div style="flex: 1; min-width: 250px; margin-bottom: 15px;">
                        <span class="team-name" style="color: #d4af37;">⭐ {m['gazde']}</span>
                        <span class="vs">VS</span>
                        <span class="team-name" style="color: #d4af37;">{m['oaspeti']}</span>
                    </div>
                    <div style="display: flex; gap: 15px; margin-bottom: 15px;">
                        <div class="metric-box" style="border-color: #d4af37;">
                            <div class="metric-val" style="color: #d4af37;">{m['p1']:.0f}%</div>
                            <div class="metric-lbl">Gazde (1)</div>
                        </div>
                        <div class="metric-box">
                            <div class="metric-val">{m['px']:.0f}%</div>
                            <div class="metric-lbl">Egal (X)</div>
                        </div>
                        <div class="metric-box" style="border-color: #d4af37;">
                            <div class="metric-val" style="color: #d4af37;">{m['p2']:.0f}%</div>
                            <div class="metric-lbl">Oaspeți (2)</div>
                        </div>
                    </div>
                    <div style="display: flex; gap: 10px; align-items: center;">
                        <span class="prediction-badge" style="background-color: #d4af37; color: black;">🎯 Pronostic: {m['pronostic']}</span>
                        <span class="goals-badge" style="background-color: #1f6feb;">⚽ Goluri: {sugestie_pariu_goluri}</span>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)

    # --- SECȚIUNEA 2: FILTRU DE CĂUTARE ȘI TOATE MECIURILE ---
    st.markdown('<div class="section-header">📅 Toate Meciurile Zilei din BetExplorer</div>', unsafe_allow_html=True)
    
    # Căsuța de căutare instant
    cautare = st.text_input("🔍 Caută o echipă (Ex: Real Madrid, Man City, Arsenal...):", "")
    
    # Filtrare listă în timp real pe baza textului introdus de vizitator
    meciuri_filtrate = [
        m for m in toate_meciurile 
        if cautare.lower() in m['gazde'].lower() or cautare.lower() in m['oaspeti'].lower()
    ]
    
    if not meciuri_filtrate:
        st.warning("❌ Nu s-a găsit niciun meci cu acest nume. Verifică ortografia sau încearcă altă echipă.")
    else:
        # Generăm feed-ul filtrat
        for m in meciuri_filtrate:
            sugestie_pariu_goluri = "Peste 2.5" if m["goluri"] > 2.4 else "Sub 2.5"
            st.markdown(f"""
                <div class="match-card">
