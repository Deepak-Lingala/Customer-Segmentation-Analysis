import os
import pandas as pd
import numpy as np

def generate_batch(batch_num, batch_size=50000):
    start_id = batch_num * batch_size + 1
    end_id = start_id + batch_size
    
    # Baseline RFM
    recency = np.random.randint(0, 366, size=batch_size)
    freq_raw = np.random.lognormal(mean=1.5, sigma=1.0, size=batch_size)
    frequency = np.clip(np.round(freq_raw), 1, 100).astype(int)
    monetary = np.round(np.random.lognormal(mean=4.0, sigma=1.2, size=batch_size) * 10, 2)
    
    # 1. Tenure
    tenure_addition = np.random.randint(0, 365*3, size=batch_size)
    tenure_offset = np.where(frequency > 1, np.random.randint(1, 100, size=batch_size), 0)
    tenure = recency + tenure_addition + tenure_offset
    
    # 2. Return Rate
    return_rate = np.round(np.random.beta(a=0.5, b=5.0, size=batch_size), 2)
    
    # 3. Discount Usage
    discount_usage = np.round(np.random.beta(a=0.8, b=0.8, size=batch_size), 2)
    
    # 4. Satisfaction Score (Vectorized rather than looped for extreme speed increase)
    # Give base random choices based on standard distribution, then overwrite specific indices
    base_choices = [1, 2, 3, 4, 5]
    satisfaction = np.random.choice(base_choices, p=[0.05, 0.05, 0.1, 0.4, 0.4], size=batch_size)
    # Mask where return rate > 0.4
    high_return_mask = return_rate > 0.4
    num_high_return = np.sum(high_return_mask)
    if num_high_return > 0:
        satisfaction[high_return_mask] = np.random.choice(base_choices, p=[0.4, 0.3, 0.2, 0.05, 0.05], size=num_high_return)
            
    # 5. Avg Time on Site
    base_time = np.random.lognormal(mean=1.5, sigma=0.5, size=batch_size)
    avg_time_on_site = np.clip(np.round(base_time + (frequency * 0.1), 1), 1.0, 120.0)
    
    # 6. Support Calls (Vectorized Poisson)
    # Start with base lambda 0.5
    lams = np.full(batch_size, 0.5)
    lams[satisfaction <= 2] += 2.0
    lams[return_rate > 0.3] += 1.5
    support_calls = np.random.poisson(lams)

    # Add noise / outliers
    num_outliers = int(batch_size * 0.025)
    outlier_indices = np.random.choice(batch_size, num_outliers, replace=False)
    
    monetary[outlier_indices] = monetary[outlier_indices] * np.random.uniform(5, 50, size=num_outliers)
    frequency[outlier_indices] = np.clip(frequency[outlier_indices] * np.random.uniform(2, 5, size=num_outliers), 1, 300).astype(int)
    
    df = pd.DataFrame({
        'CustomerID': np.arange(start_id, end_id),
        'Recency': recency,
        'Frequency': frequency,
        'Monetary': monetary,
        'Tenure': tenure,
        'Return_Rate': return_rate,
        'Discount_Usage': discount_usage,
        'Satisfaction_Score': satisfaction,
        'Avg_Time_Site_Min': avg_time_on_site,
        'Support_Calls': support_calls
    })
    
    return df

def main():
    os.makedirs('data', exist_ok=True)
    file_path = 'data/customer_data.csv'
    
    print("Generating advanced dataset via vectorized operations...")
    total_batches = 10
    batch_size = 50000
    
    first_batch = True
    for i in range(total_batches):
        df_batch = generate_batch(i, batch_size)
        mode = 'w' if first_batch else 'a'
        header = first_batch
        df_batch.to_csv(file_path, mode=mode, header=header, index=False)
        first_batch = False
        print(f"Batch {i+1}/10 generated and saved.")
        
    print(f"Data generation complete! Saved to {file_path}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR inline: {e}")
