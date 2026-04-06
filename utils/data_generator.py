import numpy as np
import pandas as pd
import os

# ========== HEALTH DATA (with coverage) ==========
def generate_health_data(n=100000, seed=42):
    np.random.seed(seed)
    age = np.random.randint(18, 70, n)
    gender = np.random.choice(['male', 'female'], n)
    bmi = np.round(np.random.normal(28, 6, n), 1)
    children = np.random.randint(0, 5, n)
    smoker = np.random.choice(['yes', 'no'], n, p=[0.25, 0.75])
    region = np.random.choice(['northeast', 'northwest', 'southeast', 'southwest'], n)
    
    # Coverage options (sum insured in lakhs)
    coverage_options = [5, 10, 25, 50, 100]  # in lakhs ₹
    coverage = np.random.choice(coverage_options, n)  # in lakhs
    
    # Premium = base + age*? + bmi*? + smoker*? + coverage*? (coverage has high impact)
    premium = (2000 + 
               age * 250 + 
               bmi * 100 + 
               (smoker == 'yes') * 6500 + 
               children * 350 +
               coverage * 1200 +          # each lakh coverage adds ₹1200 premium
               np.random.normal(0, 800, n))
    premium = np.maximum(premium, 3000)
    return pd.DataFrame({
        'age':age, 'gender':gender, 'bmi':bmi, 'children':children,
        'smoker':smoker, 'region':region, 'coverage_lakhs':coverage, 'premium':premium
    })

# ========== CAR DATA (with IDV) ==========
def generate_car_data(n=100000, seed=42):
    np.random.seed(seed)
    age = np.random.randint(18, 80, n)
    gender = np.random.choice(['male', 'female'], n)
    driving_exp = np.random.randint(0, 70, n)
    driving_exp = np.clip(driving_exp, 0, age-16)
    vehicle_age = np.random.randint(0, 20, n)
    vehicle_type = np.random.choice(['sedan', 'suv', 'truck', 'sports'], n, p=[0.4,0.3,0.2,0.1])
    location = np.random.choice(['urban', 'suburban', 'rural'], n)
    prev_claims = np.random.poisson(0.3, n)
    annual_mileage = np.random.randint(5000, 35000, n)
    
    # IDV (Insured Declared Value) in lakhs
    idv_options = [3, 5, 8, 12, 20]  # lakhs
    idv = np.random.choice(idv_options, n)
    
    premium = (2000 + 
               age * 30 + 
               (driving_exp < 5) * 800 + 
               vehicle_age * 200 + 
               (vehicle_type == 'sports') * 2500 + 
               (location == 'urban') * 600 +
               prev_claims * 1200 + 
               annual_mileage * 0.05 +
               idv * 400 +               # each lakh IDV adds ₹400 premium
               np.random.normal(0, 300, n))
    premium = np.maximum(premium, 2000)
    return pd.DataFrame({
        'age':age, 'gender':gender, 'driving_experience':driving_exp,
        'vehicle_age':vehicle_age, 'vehicle_type':vehicle_type, 'location':location,
        'previous_claims':prev_claims, 'annual_mileage':annual_mileage,
        'idv_lakhs':idv, 'premium':premium
    })

# ========== LIFE DATA (with sum assured) ==========
def generate_life_data(n=100000, seed=42):
    np.random.seed(seed)
    age = np.random.randint(20, 70, n)
    gender = np.random.choice(['male', 'female'], n)
    smoker = np.random.choice(['yes', 'no'], n, p=[0.2, 0.8])
    health = np.random.choice(['poor','average','good','excellent'], n, p=[0.1,0.3,0.4,0.2])
    income = np.random.randint(30000, 250000, n)
    term = np.random.choice([10,20,30], n)
    
    # Sum assured in lakhs
    sa_options = [25, 50, 75, 100, 150, 200]  # lakhs
    sum_assured = np.random.choice(sa_options, n)
    
    health_mult = {'poor':1.8, 'average':1.4, 'good':1.1, 'excellent':0.9}
    mult = np.array([health_mult[h] for h in health])
    
    premium = (1000 + 
               age * 40 + 
               (smoker=='yes') * 1500 + 
               income * 0.0003 +
               sum_assured * 15 +          # each lakh sum assured adds ₹15 premium
               term * 20 +
               np.random.normal(0, 200, n))
    premium = np.maximum(premium, 1500)
    return pd.DataFrame({
        'age':age,'gender':gender,'smoker':smoker,'health_status':health,
        'annual_income':income,'term_length':term,'sum_assured_lakhs':sum_assured,
        'premium':premium
    })

# ========== HOME DATA (with coverage) ==========
def generate_home_data(n=100000, seed=42):
    np.random.seed(seed)
    home_age = np.random.randint(0, 80, n)
    location = np.random.choice(['urban', 'suburban', 'rural'], n)
    sqft = np.random.randint(800, 5000, n)
    construction = np.random.choice(['wood','brick','concrete'], n, p=[0.25,0.55,0.2])
    roof = np.random.choice(['shingle','tile','metal'], n)
    security = np.random.choice(['none','alarm','monitored'], n)
    prev_claims = np.random.poisson(0.15, n)
    
    # Home coverage (structure+content) in lakhs
    coverage_options = [10, 20, 30, 50, 75, 100]  # lakhs
    home_coverage = np.random.choice(coverage_options, n)
    
    premium = (1500 + 
               home_age * 20 + 
               (location=='urban') * 800 + 
               sqft * 0.15 + 
               (construction=='wood') * 600 + 
               (roof=='shingle') * 200 +
               (security=='none') * 500 + 
               prev_claims * 800 +
               home_coverage * 30 +          # each lakh coverage adds ₹30 premium
               np.random.normal(0, 250, n))
    premium = np.maximum(premium, 2000)
    return pd.DataFrame({
        'home_age':home_age,'location':location,'sqft':sqft,
        'construction_type':construction,'roof_type':roof,
        'security_system':security,'previous_claims':prev_claims,
        'coverage_lakhs':home_coverage,'premium':premium
    })

# ========== FRAUD DATA (no change needed) ==========
def generate_fraud_data(n=100000, seed=42):
    np.random.seed(seed)
    policy_type = np.random.choice(['health','car','life','home'], n)
    claim_amount = np.random.uniform(500, 80000, n)
    incident_type = np.random.choice(['theft','collision','fire','natural disaster','other'], n)
    incident_severity = np.random.choice(['minor','moderate','severe'], n, p=[0.4,0.35,0.25])
    witnesses = np.random.randint(0, 6, n)
    police_report = np.random.choice(['yes','no'], n, p=[0.8,0.2])
    
    fraud = (((claim_amount > 50000) & (incident_severity == 'severe') & (witnesses == 0)) |
             ((incident_type == 'theft') & (police_report == 'no')) |
             ((claim_amount > 30000) & (incident_severity == 'moderate') & (police_report == 'no')) |
             ((incident_type == 'fire') & (witnesses == 0)) |
             ((policy_type == 'car') & (incident_type == 'theft') & (claim_amount > 40000)) |
             ((policy_type == 'health') & (incident_type == 'critical illness') & (claim_amount > 35000)) |
             ((incident_type == 'natural disaster') & (police_report == 'no'))).astype(int)
    random_fraud = np.random.random(n) < 0.08
    fraud = np.where(random_fraud, 1, fraud)
    fraud = np.where((claim_amount < 5000) & (fraud == 1), 0, fraud)
    return pd.DataFrame({
        'policy_type':policy_type,'claim_amount':claim_amount,
        'incident_type':incident_type,'incident_severity':incident_severity,
        'witnesses':witnesses,'police_report':police_report,'fraud_reported':fraud
    })

def generate_all_data():
    os.makedirs('data', exist_ok=True)
    generate_health_data().to_csv('data/health_insurance.csv', index=False)
    generate_car_data().to_csv('data/car_insurance.csv', index=False)
    generate_life_data().to_csv('data/life_insurance.csv', index=False)
    generate_home_data().to_csv('data/home_insurance.csv', index=False)
    generate_fraud_data().to_csv('data/fraud_claims.csv', index=False)
    print("✅ Data generated with coverage features (100k rows each).")

if __name__ == "__main__":
    generate_all_data()