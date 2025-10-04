# 🌦️ Codeius — NASA Space Apps Challenge 2025

**Challenge:** Will It Rain On My Parade?  
**Team:** Mohamed Ashraf, Fatma Osama, Khaled Nageh, Ahmed Bakr, Hassan Ashraf

---

## 🚀 About
Codeius is an AI-powered web application that forecasts rainfall risk for parades and outdoor events.  
It combines **NASA POWER data**, **Open-Meteo forecasts**, and **NASA GIBS IMERG** satellite overlays.  

- Provides **Rain Risk Score** (Green/Yellow/Red).  
- Uses **Ensemble AI (RandomForest + GB + Logistic calibration)**.  
- Identifies **safe windows** for outdoor events.  
- Offers **interactive UI/UX** with explainability & export options.  

---

## 📂 Project Structure
- `app.py` — Streamlit web app  
- `train_pro.py` — Model training (NASA POWER data)  
- `models/risk_model.pkl` — Trained ensemble AI model  
- `requirements.txt` — Dependencies  
- `slides/Codeius_Presentation.pptx` — Project demo slides  

---

## 🖥️ Run Locally
```bash
pip install -r requirements.txt
python train_pro.py
streamlit run app.py
