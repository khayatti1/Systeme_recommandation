import os
import pandas as pd

def extract_data(filepath):
    data = []
    current = {}

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            
            if not line:  # Si ligne vide, on passe à l'entrée suivante
                if current:  # Si current n'est pas vide
                    data.append(current)
                    current = {}
                continue
                
            if line.startswith("product/productId:"):
                current["product_id"] = line.split("product/productId:")[1].strip()
            elif line.startswith("product/title:"):
                current["title"] = line.split("product/title:")[1].strip()
            elif line.startswith("product/price:"):
                price = line.split("product/price:")[1].strip()
                current["price"] = float(price) if price != "unknown" else None
            elif line.startswith("review/userId:"):
                current["user_id"] = line.split("review/userId:")[1].strip()
            elif line.startswith("review/profileName:"):
                current["user_name"] = line.split("review/profileName:")[1].strip().replace('"', '')
            elif line.startswith("review/helpfulness:"):
                current["helpfulness"] = line.split("review/helpfulness:")[1].strip()
            elif line.startswith("review/score:"):
                current["rating"] = float(line.split("review/score:")[1].strip())
            elif line.startswith("review/time:"):
                current["review_time"] = int(line.split("review/time:")[1].strip())
            elif line.startswith("review/summary:"):
                current["review_summary"] = line.split("review/summary:")[1].strip()
            elif line.startswith("review/text:"):
                current["review_text"] = line.split("review/text:")[1].strip()
    
        # Ajouter le dernier enregistrement s'il existe
        if current:
            data.append(current)

    df = pd.DataFrame(data)
    return df

if __name__ == "__main__":
    input_path = "data/raw/Industrial_&_Scientific.txt"
    output_path = "data/processed/industrial_cleaned.csv"
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df = extract_data(input_path)
    df.to_csv(output_path, index=False)
    print(f"✅ Données brutes sauvegardées dans : {output_path}")
    print(f"Nombre total d'enregistrements : {len(df)}")
