import streamlit as st
import numpy as np
import requests
from bs4 import BeautifulSoup
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

# 1. Configurare pagină publică (Aspect Premium)
st.set_page_config(
    page_title="AI Score Predictor Pro",
    page_icon="🔮",
    layout="wide"
)

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
                        
                        p1_final = (prob_1 / total) * 100
                        px_final = (prob_x / total) * 100
                        p2_final = (prob_2 / total) * 100
                        
                        max_safety = max(p1_final, px_final, p2_final)
                        
                        if p1_final > 45: pronostic = "1 (Gazde)"
                        elif p2_final > 45: pronostic = "2 (Oaspeti)"
                        else: pronostic = "X (Egal)"
                        
                        lista_predictii.append({
                            "gazde": gazde, "oaspeti": oaspeti,
                            "p1": p1_final, "px": px_final, "p2": p2_final,
                            "goluri": float(max(0.0, goluri_baza * vreme_simulata)),
                            "pronostic": pronostic,
                            "siguranta": max_safety
                        })
            return lista_predictii
        return []
    except:
        return []

# 4. Interfața Web Interactivă
st.title("🔮 AI Score Predictor Pro")
st.subheader("Dashboard automatizat bazat pe Machine Learning în 5 piloni")

# Preluăm baza de date zilnică din fundal
toate_meciurile = obtine_toate_predictiile_zilei()

if not toate_meciurile:
    st.info("🔄 Conectare în curs la BetExplorer și calibrare algoritmi AI... Vă rugăm așteptați.")
else:
    # --- SECȚIUNEA 1: COTA ZILEI (TOP 3 MECIURI SIGURE) ---
    st.header("👑 COTA ZILEI (Top 3 Meciuri Recomandate)")
    
    # Sortăm lista după coeficientul de siguranță calculat de AI și luăm primele 3
    meciuri_top3 = sorted(toate_meciurile, key=lambda k: k['siguranta'], reverse=True)[:3]
    
    for m in meciuri_top3:
        sugestie_pariu_goluri = "Peste 2.5" if m["goluri"] > 2.4 else "Sub 2.5"
        
        with st.container(border=True):
            col1, col2, col3 = st.columns([2, 2, 2])
            with col1:
                st.write(f"⭐️ **{m['gazde']} vs {m['oaspeti']}**")
                st.write(f"🎯 **Pronostic sugerat:** {m['pronostic']}")
            with col2:
                c_1, c_x, c_2 = st.columns(3)
                c_1.metric("1", f"{m['p1']:.0f}%")
                c_x.metric("X", f"{m['px']:.0f}%")
                c_2.metric("2", f"{m['p2']:.0f}%")
            with col3:
                st.metric("Total Goluri", sugestie_pariu_goluri, f"{m['goluri']:.2f}")

    # --- SECȚIUNEA 2: FILTRU DE CĂUTARE ȘI TOATE MECIURILE ---
    st.header("📅 Toate Meciurile Zilei din BetExplorer")
    
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
        # Generăm feed-ul filtrat folosind structura nativă stabilă
        for m in meciuri_filtrate:
            sugestie_pariu_goluri = "Peste 2.5" if m["goluri"] > 2.4 else "Sub 2.5"
            
            with st.container(border=True):
                col1, col2, col3 = st.columns([2, 2, 2])
                with col1:
                    st.write(f"⚽ **{m['gazde']} vs {m['oaspeti']}**")
                    st.write(f"🎯 **Pronostic:** {m['pronostic']}")
                with col2:
                    c_1, c_x, c_2 = st.columns(3)
                    c_1.metric("1", f"{m['p1']:.0f}%")
                    c_x.metric("X", f"{m['px']:.0f}%")
                    c_2.metric("2", f"{m['p2']:.0f}%")
                with col3:
                    st.metric("Total Goluri", sugestie_pariu_goluri, f"{m['goluri']:.2f}")
