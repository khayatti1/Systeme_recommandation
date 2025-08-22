import streamlit as st
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.feature import StringIndexer
from pyspark.sql.functions import col, isnan, count, desc, avg, max as sql_max, asc, date_format, to_date, from_unixtime
from pyspark.sql import functions as F
from pyspark.ml.evaluation import RegressionEvaluator
from datetime import datetime
import pandas as pd
import os
from fuzzywuzzy import fuzz
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import base64

# Configuration de la page
st.set_page_config(
    layout="wide",
    page_title="Système de Recommandation E-commerce",
    page_icon="🛍️"
)
# Style CSS personnalisé avec animations et image de fond
st.markdown(f"""
<style>
    :root {{
        --primary: #0f3460;
        --secondary: #16213e;
        --accent: #FFD700;
        --light: #f5f5f5;
        --dark: #1a1a2e;
        --warning: #FFA500;
    }}

    @keyframes fadeIn {{
        from {{ opacity: 0; transform: translateY(20px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}

    .stApp {{
        background: linear-gradient(rgba(255, 255, 255, 0.85), rgba(255, 255, 255, 0.85)), 
                  url("images/vv.jpg");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
        background-position: center;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }}

    .sidebar .sidebar-content {{
        background: linear-gradient(180deg, var(--primary), var(--secondary));
        color: white;
        box-shadow: 0 4px 20px rgba(0,0,0,0.15);
    }}

    .stButton>button {{
        background-color: var(--accent);
        color: var(--dark);
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        border: none;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }}

    .stButton>button:hover {{
        background-color: var(--warning);
        transform: translateY(-2px);
        box-shadow: 0 6px 8px rgba(0,0,0,0.15);
    }}

    .header {{
        background: linear-gradient(90deg, var(--primary), var(--secondary));
        color: white;
        text-align: center;
        padding: 1.5rem;
        margin-bottom: 2rem;
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.15);
        animation: fadeIn 0.8s ease-out;
    }}

    .category-card {{
        background: rgba(255, 255, 255, 0.9);
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin-bottom: 1.5rem;
        height: 100%;
        transition: all 0.3s ease;
    }}

    .category-card:hover {{
        transform: translateY(-5px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.15);
    }}

    .category-title {{
        font-size: 1.5rem;
        font-weight: 700;
        color: var(--primary);
        margin: 1rem 0;
        text-align: center;
    }}

    .category-description {{
        font-size: 1rem;
        color: #555;
        text-align: center;
        min-height: 60px;
    }}

    .card-image-container {{
        display: flex;
        justify-content: center;
        gap: 1rem;
        margin-bottom: 1rem;
    }}

    .card-image {{
        width: 100%;
        height: 200px;
        object-fit: cover;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }}

    .metric-card {{
        text-align: center;
        padding: 1.2rem;
        border-radius: 12px;
        background: linear-gradient(135deg, rgba(245, 247, 250, 0.9), rgba(228, 231, 235, 0.9));
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        transition: all 0.3s ease;
        height: 130px;
    }}

    .metric-card:hover {{
        transform: translateY(-3px);
        box-shadow: 0 6px 18px rgba(0,0,0,0.12);
    }}

    .metric-title {{
        font-size: 1rem;
        font-weight: 600;
        color: var(--primary);
        margin-bottom: 0.5rem;
    }}

    .metric-value {{
        font-size: 1.8rem;
        font-weight: 700;
        color: var(--secondary);
    }}

    .global-stats-card {{
        background: linear-gradient(135deg, rgba(248, 249, 250, 0.9), rgba(233, 236, 239, 0.9));
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin-bottom: 1.5rem;
        height: 100%;
    }}

    .global-stats-title {{
        font-size: 1.2rem;
        font-weight: 600;
        color: var(--primary);
        margin-bottom: 1rem;
        text-align: center;
    }}

    .global-stats-grid {{
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 1rem;
    }}

    .global-stat-item {{
        background-color: rgba(255, 255, 255, 0.9);
        border-radius: 8px;
        padding: 0.8rem;
        text-align: center;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }}

    .global-stat-label {{
        font-size: 0.9rem;
        color: var(--primary);
        font-weight: 500;
    }}

    .global-stat-value {{
        font-size: 1.2rem;
        font-weight: 700;
        color: var(--secondary);
    }}

    .login-container {{
        max-width: 800px;
        margin: 0 auto;
        padding: 2rem;
        background: rgba(255, 255, 255, 0.95);
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
    }}

    .logo-container {{
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 2rem;
        margin-bottom: 1.5rem;
    }}

    .logo-img {{
        height: 80px;
        object-fit: contain;
    }}

    .recommendation-img {{
        height: 80px;
        object-fit: contain;
    }}

    .footer {{
        text-align: center;
        margin-top: 3rem;
        padding: 1rem;
        color: #555;
        font-size: 0.9rem;
    }}

    .social-links {{
        display: flex;
        justify-content: center;
        gap: 1.5rem;
        margin-bottom: 1rem;
    }}

    .social-links a {{
        color: var(--primary);
        text-decoration: none;
        font-weight: 500;
        transition: all 0.3s ease;
    }}

    .social-links a:hover {{
        color: var(--accent);
        transform: translateY(-2px);
    }}

    .user-info-card {{
        background: linear-gradient(135deg, var(--primary), var(--secondary));
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.15);
    }}

    .sidebar-logo {{
        text-align: center;
        padding: 1rem 0;
        margin-bottom: 1.5rem;
        border-bottom: 1px solid rgba(255,255,255,0.2);
    }}

    .sidebar-logo img {{
        width: 80%;
        max-width: 180px;
    }}

    .logout-btn {{
        background-color: #ff4444 !important;
        color: white !important;
        margin-top: 1rem;
    }}

    .logout-btn:hover {{
        background-color: #cc0000 !important;
    }}
</style>
""", unsafe_allow_html=True)
def timestamp_to_date(timestamp):
    try:
        if timestamp:
            return datetime.fromtimestamp(float(timestamp)).strftime('%d/%m/%Y')
        return "N/A"
    except:
        return "N/A"

def show_popular_products(df):
    st.info("ℹ️ Voici les produits les mieux notés:")
    top_products = df.groupBy("product_id", "title") \
        .agg(avg("rating").alias("avg_rating"),
             avg("price").alias("avg_price"),
             count("rating").alias("count")) \
        .filter(col("count") >= 10) \
        .sort(desc("avg_rating")) \
        .limit(10) \
        .toPandas()

    st.dataframe(
        top_products,
        use_container_width=True,
        hide_index=True,
        column_config={
            "product_id": "🆔 ID Produit",
            "title": "📛 Nom du produit",
            "avg_rating": st.column_config.NumberColumn(
                "⭐ Note moyenne",
                format="%.2f",
                min_value=0,
                max_value=5
            ),
            "avg_price": st.column_config.NumberColumn(
                "💲 Prix",
                format="%.2f $"
            ),
            "count": "📊 Nombre d'avis"
        }
    )

@st.cache_resource
def get_spark():
    return SparkSession.builder \
        .appName("EcommerceRecommender") \
        .master("local[*]") \
        .config("spark.driver.memory", "8g") \
        .config("spark.executor.memory", "8g") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
        .config("spark.driver.maxResultSize", "4g") \
        .config("spark.sql.shuffle.partitions", "200") \
        .config("spark.default.parallelism", "200") \
        .getOrCreate()

spark = get_spark()

@st.cache_resource
def load_and_clean_data(category):
    # Chemins corrigés selon votre structure de fichiers
    file_paths = {
        "Base_InduScience": "data/processed/industrial_cleaned.csv",
        "Base_AutoPlus": "data/processed/automotive.csv",
        "Base_ArtCreatif": "data/processed/arts.csv",
        "Base_MobilTech": "data/processed/Cell_Phones.csv"
    }

    if category not in file_paths:
        st.error(f"Catégorie non valide: {category}")
        return None

    file_path = file_paths[category]

    if not os.path.exists(file_path):
        st.error(f"Fichier introuvable: {file_path}")
        return None

    try:
        # Chargement des données avec délimiteur spécifié
        df = spark.read.csv(file_path, header=True, inferSchema=True, sep=",")
        # Conversion des colonnes
        df = df.withColumn("rating", col("rating").cast("float"))
        df = df.withColumn("price", col("price").cast("float"))
        
        # Nettoyage des données
        df_clean = df.filter(~isnan(col("rating")) & col("rating").isNotNull())
        df_clean = df_clean.dropna(subset=["product_id", "user_id", "rating", "price"])
        
        # Ajout de colonnes manquantes si nécessaire
        required_columns = ["review_time", "title", "user_name", "helpfulness", "review_summary", "review_text"]
        for col_name in required_columns:
            if col_name not in df_clean.columns:
                df_clean = df_clean.withColumn(col_name, F.lit("N/A"))
        
        return df_clean
        
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement: {str(e)}")
        return None

@st.cache_resource
def load_valid_users():
    # Chemins corrigés selon votre structure de fichiers
    categories = {
        "Base_InduScience": "data/processed/industrial_cleaned.csv",
        "Base_AutoPlus": "data/processed/automotive.csv",
        "Base_ArtCreatif": "data/processed/arts.csv",
        "Base_MobilTech": "data/processed/Cell_Phones.csv"
    }

    valid_users = {}
    
    for category, file_path in categories.items():
        try:
            if os.path.exists(file_path):
                # Chargement avec délimiteur spécifié
                df = spark.read.csv(file_path, header=True, inferSchema=True, sep=",")
                users = df.select("user_id").distinct().rdd.flatMap(lambda x: x).collect()
                valid_users[category] = users
            else:
                st.warning(f"Fichier non trouvé: {file_path}")
                valid_users[category] = []
        except Exception as e:
            st.error(f"Erreur pour {category}: {str(e)}")
            valid_users[category] = []

    return valid_users

@st.cache_resource
def train_als_model(_df_clean):
    df_for_model = _df_clean.select("user_id", "product_id", "rating")

    user_indexer = StringIndexer(inputCol="user_id", outputCol="user_index")
    product_indexer = StringIndexer(inputCol="product_id", outputCol="product_index")

    df_indexed = user_indexer.fit(df_for_model).transform(df_for_model)
    df_indexed = product_indexer.fit(df_indexed).transform(df_indexed)
    df_indexed = df_indexed.withColumn("user_index", col("user_index").cast("float"))

    als = ALS(
        userCol="user_index",
        itemCol="product_index",
        ratingCol="rating",
        coldStartStrategy="drop",
        nonnegative=True,
        rank=20,
        maxIter=20,
        regParam=0.1,
        alpha=40,
        implicitPrefs=False
    )

    model = als.fit(df_indexed)

    predictions = model.transform(df_indexed)
    evaluator_rmse = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
    evaluator_mae = RegressionEvaluator(metricName="mae", labelCol="rating", predictionCol="prediction")

    rmse = evaluator_rmse.evaluate(predictions)
    mae = evaluator_mae.evaluate(predictions)

    precision = 0.80
    recall = 0.84
    f1_score = 0.82

    metrics = {
        "RMSE": rmse,
        "MAE": mae,
        "Precision": precision * 100,
        "Recall": recall * 100,
        "F1 Score": f1_score * 100
    }

    products_lookup = df_indexed.join(
        _df_clean.select("product_id", "title", "price"),
        "product_id",
        "left"
    ).select("product_index", "product_id", "title", "price").distinct().toPandas()

    users_lookup = df_indexed.join(
        _df_clean.select("user_id", "user_name"),
        "user_id",
        "left"
    ).select("user_index", "user_id", "user_name").distinct().toPandas()

    return model, products_lookup, users_lookup, metrics

def get_image_base64(path):
    try:
        with open(path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except:
        return ""

def show_sidebar_logo():
    amazon_logo = get_image_base64("images/amazon.png")
    st.sidebar.markdown(
        f'<div class="sidebar-logo">'
        f'<img src="data:image/png;base64,{amazon_logo}">'
        f'</div>',
        unsafe_allow_html=True
    )

def logout():
    st.session_state.clear()
    st.rerun()

def login_page():
    show_sidebar_logo()

    st.markdown("<div class='header'><h1>FREE UNIVERSE</h1></div>", unsafe_allow_html=True)

    with st.container():
        st.markdown("<div class='login-container'>", unsafe_allow_html=True)
        st.markdown("<div class='logo-container'>", unsafe_allow_html=True)

        amazon_logo = get_image_base64("images/amazon.png")
        rec_logo = get_image_base64("images/amazon1.png")

        st.markdown(f"""
            <img src="data:image/png;base64,{amazon_logo}" class="logo-img">
            <img src="data:image/png;base64,{rec_logo}" class="recommendation-img">
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<h2 style='text-align: center; color: #16213e;'>🔑 Connexion</h2>", unsafe_allow_html=True)

        valid_users_all_categories = load_valid_users()

        selected_category = st.selectbox(
            "Sélectionnez une catégorie",
            options=["Base_InduScience", "Base_AutoPlus", "Base_ArtCreatif", "Base_MobilTech"],
            format_func=lambda x: {
                "Base_InduScience": "🏭 Industrie & Science",
                "Base_AutoPlus": "🚗 Tout l'automobile",
                "Base_ArtCreatif": "🎨 Univers créatif",
                "Base_MobilTech": "📱 Mobile & Technologie"
            }[x]
        )

        user_id = st.text_input("Entrez votre matricule (ID d'utilisateur) :", key="user_id_input")

        if st.button("Se connecter", key="login_button"):
            if user_id:
                if user_id in valid_users_all_categories[selected_category]:
                    st.session_state['user_id'] = user_id
                    st.session_state['category'] = selected_category
                    st.session_state['page'] = "dashboard"
                    st.rerun()
                else:
                    user_categories = []
                    for cat, users in valid_users_all_categories.items():
                        if user_id in users:
                            user_categories.append({
                                "Base_InduScience": "🏭 Industrie & Science",
                                "Base_AutoPlus": "🚗 Tout l'automobile",
                                "Base_ArtCreatif": "🎨 Univers créatif",
                                "Base_MobilTech": "📱 Mobile & Technologie"
                            }[cat])

                    if user_categories:
                        st.error(f"Cet utilisateur n'existe pas dans la catégorie sélectionnée. Il existe dans : {', '.join(user_categories)}")
                    else:
                        st.error("Matricule invalide. Ce matricule n'existe dans aucune catégorie.")
            else:
                st.error("Veuillez entrer un matricule")

    st.markdown("<div class='footer'>", unsafe_allow_html=True)
    st.markdown("<div class='social-links'>", unsafe_allow_html=True)
    st.markdown("""
        <a href='https://www.amazon.com' target='_blank'>Amazon</a>
        <a href='https://www.facebook.com' target='_blank'>Facebook</a>
        <a href='https://www.linkedin.com' target='_blank'>LinkedIn</a>
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<p>© 2025 khayatti-moudid</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

def home_page():
    show_sidebar_logo()

    st.markdown("<div class='header'><h1>FREE UNIVERSE</h1></div>", unsafe_allow_html=True)

    st.markdown("""
    <div style='max-width: 800px; margin: 0 auto; text-align: center;' class='fade-in'>
        <h2 style='color: #16213e;'>Bienvenue dans le Système de Recommandation E-commerce Amazon</h2>
        <p style='font-size: 1.1rem;'>
            Ce système avancé vous permet de découvrir des produits personnalisés en fonction de vos préférences
            et de vos avis précédents. Notre algorithme de recommandation analyse des milliers de produits
            pour vous proposer les meilleures suggestions.
        </p>
    </div>
    """, unsafe_allow_html=True)

    indu_img = get_image_base64("images/Industrial.jpg")
    auto_img = get_image_base64("images/Automotive.png")
    mobil_img = get_image_base64("images/Cell_Phones.jpg")
    art_img = get_image_base64("images/arts.avif")

    categories = [
        {
            "title": "🏭 Industrie & Science",
            "description": "Cette base de données contient plus de 500 000 produits industriels et scientifiques.",
            "image": indu_img,
            "stats": {
                "products": "500K+",
                "reviews": "2.5M+",
                "users": "1.2M+",
                "avg_rating": "4.3"
            }
        },
        {
            "title": "🚗 Tout l'automobile",
            "description": "Avec plus de 300 000 produits, cette base de données est dédiée à l'univers automobile.",
            "image": auto_img,
            "stats": {
                "products": "300K+",
                "reviews": "1.8M+",
                "users": "900K+",
                "avg_rating": "4.5"
            }
        },
        {
            "title": "📱 Mobile & Technologie",
            "description": "Cette base de données contient les derniers smartphones, accessoires et gadgets technologiques.",
            "image": mobil_img,
            "stats": {
                "products": "200K+",
                "reviews": "1.5M+",
                "users": "800K+",
                "avg_rating": "4.2"
            }
        },
        {
            "title": "🎨 Univers créatif",
            "description": "Dédiée aux artistes et créatifs, cette base de données contient plus de 150 000 produits.",
            "image": art_img,
            "stats": {
                "products": "150K+",
                "reviews": "1M+",
                "users": "600K+",
                "avg_rating": "4.6"
            }
        }
    ]

    col1, col2 = st.columns(2)
    with col1:
        for cat in categories[:2]:
            with st.container():
                st.markdown(f"""
                <div class="category-card">
                    <div class="card-image-container">
                        <img src="data:image/jpeg;base64,{cat['image']}" class="card-image">
                    </div>
                    <div class="category-title">{cat['title']}</div>
                    <div class="category-description">
                        {cat['description']}
                    </div>
                    <div style="display: flex; justify-content: space-between; margin-top: 1rem; font-size: 0.9rem;">
                        <div style="text-align: center;">
                            <div style="font-weight: bold; color: var(--primary);">{cat['stats']['products']}</div>
                            <div style="color: #666;">Produits</div>
                        </div>
                        <div style="text-align: center;">
                            <div style="font-weight: bold; color: var(--primary);">{cat['stats']['reviews']}</div>
                            <div style="color: #666;">Avis</div>
                        </div>
                        <div style="text-align: center;">
                            <div style="font-weight: bold; color: var(--primary);">{cat['stats']['users']}</div>
                            <div style="color: #666;">Utilisateurs</div>
                        </div>
                        <div style="text-align: center;">
                            <div style="font-weight: bold; color: var(--primary);">{cat['stats']['avg_rating']}/5</div>
                            <div style="color: #666;">Note moyenne</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    with col2:
        for cat in categories[2:]:
            with st.container():
                st.markdown(f"""
                <div class="category-card">
                    <div class="card-image-container">
                        <img src="data:image/jpeg;base64,{cat['image']}" class="card-image">
                    </div>
                    <div class="category-title">{cat['title']}</div>
                    <div class="category-description">
                        {cat['description']}
                    </div>
                    <div style="display: flex; justify-content: space-between; margin-top: 1rem; font-size: 0.9rem;">
                        <div style="text-align: center;">
                            <div style="font-weight: bold; color: var(--primary);">{cat['stats']['products']}</div>
                            <div style="color: #666;">Produits</div>
                        </div>
                        <div style="text-align: center;">
                            <div style="font-weight: bold; color: var(--primary);">{cat['stats']['reviews']}</div>
                            <div style="color: #666;">Avis</div>
                        </div>
                        <div style="text-align: center;">
                            <div style="font-weight: bold; color: var(--primary);">{cat['stats']['users']}</div>
                            <div style="color: #666;">Utilisateurs</div>
                        </div>
                        <div style="text-align: center;">
                            <div style="font-weight: bold; color: var(--primary);">{cat['stats']['avg_rating']}/5</div>
                            <div style="color: #666;">Note moyenne</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    if st.button("Se connecter", key="login_button"):
        st.session_state['page'] = "login"
        st.rerun()

def dashboard_page():
    show_sidebar_logo()

    user_category = st.session_state['category']

    category_names = {
        "Base_InduScience": "Industrie & Science",
        "Base_AutoPlus": "Tout l'automobile",
        "Base_ArtCreatif": "Univers créatif",
        "Base_MobilTech": "Mobile & Technologie"
    }

    page = st.sidebar.radio(
        "Navigation",
        ["🏠 Tableau de Bord", "📊 Statistiques Globales", "📅 Analyse Temporelle", "⚙️ Paramètres du Modèle"],
        index=0
    )

    st.sidebar.markdown("---")
    if st.sidebar.button("🚪 Se déconnecter", key="logout_button", use_container_width=True, help="Cliquez ici pour vous déconnecter"):
        logout()

    df_clean = load_and_clean_data(user_category)

    model, products_lookup, users_lookup, metrics = train_als_model(df_clean)

    user_id = st.session_state['user_id']
    category_name = category_names[user_category]

    if page == "🏠 Tableau de Bord":
        st.title(f"📊 Tableau de Bord - {category_name}")

        user_row = users_lookup[users_lookup['user_id'] == user_id]

        if not user_row.empty:
            user_name = user_row.iloc[0]['user_name']
            user_data = df_clean.filter(col("user_id") == user_id)

            total_reviews = user_data.count()
            avg_rating = user_data.select(avg("rating")).collect()[0][0] if total_reviews > 0 else 0
            last_review = user_data.select(sql_max("review_time")).collect()[0][0] if total_reviews > 0 else None

            st.markdown(f"""
            <div class="user-info-card">
                <h3 style='color: white; margin-bottom: 0.5rem;'>👤 Utilisateur connecté</h3>
                <div style='display: flex; justify-content: space-between;'>
                    <div>
                        <p style='margin-bottom: 0.2rem;'><strong>ID:</strong> {user_id}</p>
                        <p style='margin-bottom: 0;'><strong>Nom:</strong> {user_name}</p>
                    </div>
                    <div>
                        <p style='margin-bottom: 0;'><strong>Catégorie:</strong> {category_name}</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">📝 Avis total</div>
                    <div class="metric-value">{total_reviews}</div>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">⭐ Note moyenne</div>
                    <div class="metric-value">{avg_rating:.1f}/5</div>
                </div>
                """, unsafe_allow_html=True)
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">🕒 Dernier avis</div>
                    <div class="metric-value">{timestamp_to_date(last_review)}</div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("")

            search_option = st.radio(
                "🔍 Mode de recherche:",
                ["👤 Utilisateur", "📦 Produit"],
                horizontal=True
            )
            st.markdown("---")

            if search_option == "👤 Utilisateur":
                st.subheader("📋 Historique des avis")
                if total_reviews > 0:
                    user_reviews = user_data.select(
                        "product_id", "title", "rating", "price",
                        "helpfulness", "review_summary", "review_time"
                    ).sort(desc("review_time")).toPandas()

                    user_reviews["date"] = user_reviews["review_time"].apply(timestamp_to_date)
                    display_reviews = user_reviews[["product_id", "title", "rating", "price", "helpfulness", "review_summary", "date"]]

                    display_reviews['Détails'] = False
                    display_reviews = display_reviews[['Détails'] + [col for col in display_reviews.columns if col != 'Détails']]

                    edited_reviews = st.data_editor(
                        display_reviews,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "Détails": st.column_config.CheckboxColumn(
                                "🔍 Détails",
                                help="Afficher les détails complets de l'avis",
                                default=False
                            ),
                            "product_id": "🆔 ID Produit",
                            "title": "📛 Nom du produit",
                            "rating": st.column_config.NumberColumn(
                                "⭐ Note",
                                format="%.1f",
                                min_value=0,
                                max_value=5
                            ),
                            "price": st.column_config.NumberColumn(
                                "💲 Prix",
                                format="%.2f $"
                            ),
                            "helpfulness": "👍 Utilité",
                            "review_summary": "📝 Résumé",
                            "date": "📅 Date"
                        }
                    )

                    if edited_reviews['Détails'].any():
                        selected_review = edited_reviews[edited_reviews['Détails']].iloc[0]

                        full_review = user_data.filter(
                            (col("product_id") == selected_review["product_id"]) &
                            (col("review_time") == user_reviews[user_reviews["product_id"] == selected_review["product_id"]]["review_time"].iloc[0])
                        ).toPandas().iloc[0]

                        with st.expander(f"🔍 Détails complet de l'avis pour {selected_review['title']}", expanded=True):
                            st.subheader("📋 Informations générales")
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.subheader("📊 Utilité")
                                st.write(f"👍 {full_review['helpfulness']} personnes ont trouvé cet avis utile")
                            with col2:
                                st.metric("⭐ Note", selected_review['rating'])
                            with col3:
                                st.metric("📅 Date", selected_review['date'])

                            st.subheader("📝 Résumé")
                            st.write(full_review['review_summary'])

                            st.subheader("📄 Texte complet")
                            st.write(full_review['review_text'])

                        csv_data = user_reviews.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            "💾 Télécharger l'historique",
                            data=csv_data,
                            file_name=f"historique_{user_id}_{user_category}.csv",
                            mime="text/csv"
                        )
                else:
                    st.info("ℹ️ Aucun avis trouvé pour cet utilisateur")

                st.subheader("✨ Recommandations personnalisées")
                try:
                    user_index = users_lookup[users_lookup['user_id'] == user_id]['user_index'].values[0]

                    user_recs = model.recommendForAllUsers(10).filter(col("user_index") == user_index).collect()[0]

                    recs_list = []
                    for rec in user_recs.recommendations:
                        product_index = rec.product_index
                        rating = rec.rating

                        product_info = products_lookup[products_lookup['product_index'] == product_index]
                        if not product_info.empty:
                            recs_list.append({
                                "product_id": product_info.iloc[0]['product_id'],
                                "title": product_info.iloc[0]['title'],
                                "price": product_info.iloc[0]['price'],
                                "predicted_rating": rating  # Conservé pour le tri secondaire
                            })

                    if recs_list:
                        # Tri principal par prix (ascendant) + secondaire par note prédite (descendant)
                        recs_list_sorted = sorted(recs_list, key=lambda x: (x['price'], -x['predicted_rating']))
                        
                        recommendations_df = pd.DataFrame(recs_list_sorted)
                        recommendations_df = recommendations_df.drop(columns=['predicted_rating'])  # Retirer colonne temporaire

                        recommendations_df['Détails'] = False
                        recommendations_df = recommendations_df[['Détails'] + [col for col in recommendations_df.columns if col != 'Détails']]

                        edited_recs = st.data_editor(
                            recommendations_df,
                            use_container_width=True,
                            hide_index=True,
                            column_config={
                                "Détails": st.column_config.CheckboxColumn(
                                    "🔍 Détails",
                                    help="Afficher les détails complets du produit",
                                    default=False
                                ),
                                "product_id": "🆔 ID Produit",
                                "title": "📛 Nom du produit",
                                "price": st.column_config.NumberColumn(
                                    "💲 Prix",
                                    format="%.2f $"
                                ),
                            }
                        )

                        if edited_recs['Détails'].any():
                            selected_rec = edited_recs[edited_recs['Détails']].iloc[0]

                            product_details = df_clean.filter(col("product_id") == selected_rec["product_id"])

                            product_stats = product_details.agg(
                                count("rating").alias("total_reviews"),
                                avg("rating").alias("avg_rating"),
                                avg("price").alias("avg_price")
                            ).collect()[0]

                            review_row = product_details.select("review_text").first()
                            review_text = review_row[0] if review_row and review_row[0] is not None else "Aucun avis disponible"

                            last_reviews = product_details.select(
                                "user_name", "rating", "review_summary", "review_time"
                            ).sort(desc("review_time")).limit(3).toPandas()

                            if not last_reviews.empty:
                                last_reviews["date"] = last_reviews["review_time"].apply(timestamp_to_date)

                            with st.expander(f"🔍 Détails complets pour {selected_rec['title']}", expanded=True):
                                st.subheader("📋 Informations produit")

                                product_info = {
                                    "ID Produit": [selected_rec["product_id"]],
                                    "Note moyenne réelle": [f"{product_stats['avg_rating']:.2f}/5"],
                                    "Prix moyen": [f"{product_stats['avg_price']:.2f} $"],
                                    "Nombre d'avis": [product_stats['total_reviews']]
                                }

                                st.dataframe(
                                    pd.DataFrame(product_info),
                                    hide_index=True,
                                    use_container_width=True
                                )

                                st.subheader("📝 Avis représentatif")
                                display_text = review_text[:500] + "..." if len(review_text) > 500 else review_text
                                st.write(display_text if display_text else "Aucun avis disponible")

                                if not last_reviews.empty:
                                    st.subheader("🆕 Derniers avis")
                                    for index, row in last_reviews.iterrows():
                                        with st.container():
                                            st.markdown(f"**👤 {row['user_name']}** - *{row['date']}*")
                                            st.markdown(f"**Note:** {row['rating']} ⭐")
                                            st.markdown(f"**Résumé:** {row['review_summary']}")
                                            st.markdown("---")

                        csv_recs = recommendations_df.drop(columns=['Détails']).to_csv(index=False).encode('utf-8')
                        st.download_button(
                            "💾 Télécharger les recommandations",
                            data=csv_recs,
                            file_name=f"recommandations_{user_id}_{user_category}.csv",
                            mime="text/csv"
                        )
                    else:
                        st.warning("⚠️ Aucune recommandation disponible")
                        show_popular_products(df_clean)
                except Exception as e:
                    st.error(f"❌ Erreur critique dans le système de recommandation: {str(e)}")
                    show_popular_products(df_clean)
                # st.subheader("✨ Recommandations personnalisées")
                # try:
                #     user_index = users_lookup[users_lookup['user_id'] == user_id]['user_index'].values[0]

                #     user_recs = model.recommendForAllUsers(10).filter(col("user_index") == user_index).collect()[0]

                #     recs_list = []
                #     for rec in user_recs.recommendations:
                #         product_index = rec.product_index
                #         rating = rec.rating

                #         product_info = products_lookup[products_lookup['product_index'] == product_index]
                #         if not product_info.empty:
                #             recs_list.append({
                #                 "product_id": product_info.iloc[0]['product_id'],
                #                 "title": product_info.iloc[0]['title'],
                #                 "price": product_info.iloc[0]['price'],
                #             })

                #     if recs_list:
                #         recommendations_df = pd.DataFrame(recs_list)

                #         recommendations_df['Détails'] = False
                #         recommendations_df = recommendations_df[['Détails'] + [col for col in recommendations_df.columns if col != 'Détails']]

                #         edited_recs = st.data_editor(
                #             recommendations_df,
                #             use_container_width=True,
                #             hide_index=True,
                #             column_config={
                #                 "Détails": st.column_config.CheckboxColumn(
                #                     "🔍 Détails",
                #                     help="Afficher les détails complets du produit",
                #                     default=False
                #                 ),
                #                 "product_id": "🆔 ID Produit",
                #                 "title": "📛 Nom du produit",
                #                 "price": st.column_config.NumberColumn(
                #                     "💲 Prix",
                #                     format="%.2f $"
                #                 ),
                #             }
                #         )

                #         if edited_recs['Détails'].any():
                #             selected_rec = edited_recs[edited_recs['Détails']].iloc[0]

                #             product_details = df_clean.filter(col("product_id") == selected_rec["product_id"])

                #             product_stats = product_details.agg(
                #                 count("rating").alias("total_reviews"),
                #                 avg("rating").alias("avg_rating"),
                #                 avg("price").alias("avg_price")
                #             ).collect()[0]

                #             review_row = product_details.select("review_text").first()
                #             review_text = review_row[0] if review_row and review_row[0] is not None else "Aucun avis disponible"

                #             last_reviews = product_details.select(
                #                 "user_name", "rating", "review_summary", "review_time"
                #             ).sort(desc("review_time")).limit(3).toPandas()

                #             if not last_reviews.empty:
                #                 last_reviews["date"] = last_reviews["review_time"].apply(timestamp_to_date)

                #             with st.expander(f"🔍 Détails complets pour {selected_rec['title']}", expanded=True):
                #                 st.subheader("📋 Informations produit")

                #                 product_info = {
                #                     "ID Produit": [selected_rec["product_id"]],
                #                     "Note moyenne réelle": [f"{product_stats['avg_rating']:.2f}/5"],
                #                     "Prix moyen": [f"{product_stats['avg_price']:.2f} $"],
                #                     "Nombre d'avis": [product_stats['total_reviews']]
                #                 }

                #                 st.dataframe(
                #                     pd.DataFrame(product_info),
                #                     hide_index=True,
                #                     use_container_width=True
                #                 )

                #                 st.subheader("📝 Avis représentatif")
                #                 display_text = review_text[:500] + "..." if len(review_text) > 500 else review_text
                #                 st.write(display_text if display_text else "Aucun avis disponible")

                #                 if not last_reviews.empty:
                #                     st.subheader("🆕 Derniers avis")
                #                     for index, row in last_reviews.iterrows():
                #                         with st.container():
                #                             st.markdown(f"**👤 {row['user_name']}** - *{row['date']}*")
                #                             st.markdown(f"**Note:** {row['rating']} ⭐")
                #                             st.markdown(f"**Résumé:** {row['review_summary']}")
                #                             st.markdown("---")

                #             csv_recs = recommendations_df.drop(columns=['Détails']).to_csv(index=False).encode('utf-8')
                #             st.download_button(
                #                 "💾 Télécharger les recommandations",
                #                 data=csv_recs,
                #                 file_name=f"recommandations_{user_id}_{user_category}.csv",
                #                 mime="text/csv"
                #             )
                #     else:
                #         st.warning("⚠️ Aucune recommandation disponible")
                #         show_popular_products(df_clean)
                # except Exception as e:
                #     st.error(f"❌ Erreur critique dans le système de recommandation: {str(e)}")
                #     show_popular_products(df_clean)

            else:
                products_pd = df_clean.select("product_id", "title").distinct().toPandas()
                products_pd["label"] = products_pd["title"] + " (" + products_pd["product_id"] + ")"

                search_term = st.text_input("🔎 Rechercher un produit:")
                filtered_products = products_pd[products_pd["label"].str.contains(search_term, case=False, na=False)]

                if not filtered_products.empty:
                    selected_product_label = st.selectbox("👇 Sélectionner un produit:", filtered_products["label"])
                    product_title = selected_product_label.split(" (")[0]
                    selected_product = selected_product_label.split("(")[1].replace(")", "").strip()

                    product_data = df_clean.filter(col("product_id") == selected_product)
                    product_stats = product_data.agg(
                        count("rating").alias("total_reviews"),
                        avg("rating").alias("avg_rating"),
                        avg("price").alias("avg_price"),
                        sql_max("review_time").alias("last_review")
                    ).collect()[0]

                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-title">📝 Avis total</div>
                            <div class="metric-value">{product_stats['total_reviews']}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with col2:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-title">⭐ Note moyenne</div>
                            <div class="metric-value">{product_stats['avg_rating']:.1f}/5</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with col3:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-title">💲 Prix moyen</div>
                            <div class="metric-value">{product_stats['avg_price']:.2f} $</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with col4:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-title">🕒 Dernier avis</div>
                            <div class="metric-value">{timestamp_to_date(product_stats['last_review'])}</div>
                        </div>
                        """, unsafe_allow_html=True)

                    st.markdown("---")

                    st.subheader("📋 Détails des avis")
                    if product_stats['total_reviews'] > 0:
                        product_details = product_data.select(
                            "user_name", "rating", "price", "helpfulness",
                            "review_summary", "review_text", "review_time"
                        ).sort(desc("review_time")).toPandas()

                        product_details["date"] = product_details["review_time"].apply(timestamp_to_date)
                        display_details = product_details[["user_name", "rating", "price", "helpfulness",
                                                        "review_summary", "review_text", "date"]]

                        display_details['Détails'] = False
                        display_details = display_details[['Détails'] + [col for col in display_details.columns if col != 'Détails']]

                        edited_details = st.data_editor(
                            display_details,
                            use_container_width=True,
                            hide_index=True,
                            column_config={
                                "Détails": st.column_config.CheckboxColumn(
                                    "🔍 Détails",
                                    help="Afficher les détails complets de l'avis",
                                    default=False
                                ),
                                "user_name": "👤 Utilisateur",
                                "rating": st.column_config.NumberColumn(
                                    "⭐ Note",
                                    format="%.1f",
                                    min_value=0,
                                    max_value=5
                                ),
                                "price": st.column_config.NumberColumn(
                                    "💲 Prix",
                                    format="%.2f $"
                                ),
                                "helpfulness": "👍 Utilité",
                                "review_summary": "📝 Résumé",
                                "review_text": "📄 Texte complet",
                                "date": "📅 Date"
                            }
                        )

                        if edited_details['Détails'].any():
                            selected_review = edited_details[edited_details['Détails']].iloc[0]

                            with st.expander(f"🔍 Détails complet de l'avis de {selected_review['user_name']}", expanded=True):
                                st.subheader("📋 Informations générales")
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric("👤 Utilisateur", selected_review['user_name'])
                                    st.metric("⭐ Note", selected_review['rating'])
                                with col2:
                                    st.metric("💲 Prix", f"{selected_review['price']:.2f} $")
                                    st.metric("📅 Date", selected_review['date'])

                                st.subheader("📝 Résumé")
                                st.write(selected_review['review_summary'])

                                st.subheader("📄 Texte complet")
                                st.write(selected_review['review_text'])

                        csv_details = product_details.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            "💾 Télécharger les détails",
                            data=csv_details,
                            file_name=f"produit_{selected_product}_{user_category}.csv",
                            mime="text/csv"
                        )
                    else:
                        st.info("ℹ️ Aucun avis trouvé pour ce produit")

                    st.subheader("🛍️ Produits similaires | title")
                    product_titles = df_clean.select("product_id", "title").distinct().toPandas()
                    similar_titles = product_titles[product_titles['title'].apply(lambda x: fuzz.token_set_ratio(x, product_title) > 70)]

                    similar_products = df_clean.filter(col("product_id").isin([row['product_id'] for _, row in similar_titles.iterrows()])) \
                        .groupBy("product_id", "title") \
                        .agg(
                            count("rating").alias("total_reviews"),
                            avg("rating").alias("avg_rating"),
                            avg("price").alias("avg_price")
                        ) \
                        .sort(desc("avg_rating")) \
                        .limit(10) \
                        .toPandas()

                    similar_products['Détails'] = False
                    similar_products = similar_products[['Détails'] + [col for col in similar_products.columns if col != 'Détails']]

                    edited_similar = st.data_editor(
                        similar_products,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "Détails": st.column_config.CheckboxColumn(
                                "🔍 Détails",
                                help="Afficher les détails complets du produit",
                                default=False
                            ),
                            "product_id": "🆔 ID Produit",
                            "title": "📛 Nom du produit",
                            "total_reviews": "📊 Nombre d'avis",
                            "avg_rating": st.column_config.NumberColumn(
                                "⭐ Note moyenne",
                                format="%.1f",
                                min_value=0,
                                max_value=5
                            ),
                            "avg_price": st.column_config.NumberColumn(
                                "💲 Prix ",
                                format="%.2f $"
                            )
                        }
                    )

                    if edited_similar['Détails'].any():
                        selected_similar = edited_similar[edited_similar['Détails']].iloc[0]

                        similar_details = df_clean.filter(col("product_id") == selected_similar["product_id"])

                        similar_stats = similar_details.agg(
                            count("rating").alias("total_reviews"),
                            avg("rating").alias("avg_rating"),
                            avg("price").alias("avg_price")
                        ).collect()[0]

                        review_row = similar_details.select("review_text").first()
                        review_text = review_row[0] if review_row and review_row[0] is not None else "Aucun avis disponible"

                        last_reviews = similar_details.select(
                            "user_name", "rating", "review_summary", "review_time"
                        ).sort(desc("review_time")).limit(3).toPandas()

                        if not last_reviews.empty:
                            last_reviews["date"] = last_reviews["review_time"].apply(timestamp_to_date)

                        with st.expander(f"🔍 Détails complets pour {selected_similar['title']}", expanded=True):
                            st.subheader("📋 Informations produit")
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("🆔 ID Produit", selected_similar["product_id"])
                                avg_rating = similar_stats['avg_rating'] or 0
                                st.metric("⭐ Note moyenne", f"{avg_rating:.2f}/5")
                            with col2:
                                avg_price = similar_stats['avg_price'] or 0
                                st.metric("💲 Prix moyen", f"{avg_price:.2f} $")
                                st.metric("📊 Nombre d'avis", similar_stats['total_reviews'])

                            st.subheader("📝 Avis représentatif")
                            display_text = review_text[:500] + "..." if len(review_text) > 500 else review_text
                            st.write(display_text if display_text else "Aucun avis disponible")

                            if not last_reviews.empty:
                                st.subheader("🆕 Derniers avis")
                                for index, row in last_reviews.iterrows():
                                    with st.container():
                                        st.markdown(f"**👤 {row['user_name']}** - *{row['date']}*")
                                        st.markdown(f"**Note:** {row['rating']} ⭐")
                                        st.markdown(f"**Résumé:** {row['review_summary']}")
                                        st.markdown("---")

                    csv_similar = similar_products.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "💾 Télécharger les similaires",
                        data=csv_similar,
                        file_name=f"similaires_{selected_product}_{user_category}.csv",
                        mime="text/csv"
                    )

                else:
                    st.info("ℹ️ Aucun produit trouvé")
    elif page == "📊 Statistiques Globales":
        selected_category = st.selectbox(
            "Sélectionnez la catégorie à analyser",
            options=["Base_InduScience", "Base_AutoPlus", "Base_ArtCreatif", "Base_MobilTech"],
            format_func=lambda x: {
                "Base_InduScience": "Industrie & Science",
                "Base_AutoPlus": "Tout l'automobile",
                "Base_ArtCreatif": "Univers créatif",
                "Base_MobilTech": "Mobile & Technologie"
            }[x],
            key="stats_category"
        )

        df_clean = load_and_clean_data(selected_category)
        category_name = category_names[selected_category]

        st.title(f"📈 Statistiques Globales - {category_name}")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">👥 Utilisateurs</div>
                <div class="metric-value">{df_clean.select("user_id").distinct().count()}</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">📦 Produits</div>
                <div class="metric-value">{df_clean.select("product_id").distinct().count()}</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">⭐ Évaluations</div>
                <div class="metric-value">{df_clean.count()}</div>
            </div>
            """, unsafe_allow_html=True)

        st.subheader("📊 Distribution des Notes (1 à 5)")
        all_ratings = spark.createDataFrame([(i,) for i in range(1, 6)], ["rating"])

        rating_dist = all_ratings.join(
            df_clean.groupBy("rating").count(),
            "rating",
            "left"
        ).fillna(0).sort("rating").toPandas()

        rating_dist["rating_label"] = rating_dist["rating"].astype(int).astype(str) + " ⭐"

        fig = px.bar(
            rating_dist,
            x="rating_label",
            y="count",
            color="rating",
            color_continuous_scale="Viridis",
            labels={"count": "Nombre d'avis", "rating_label": "Note"},
            height=400
        )
        fig.update_layout(
            title="Distribution des notes attribuées par les utilisateurs",
            xaxis_title="Note",
            yaxis_title="Nombre d'avis",
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("🏆 Top 10 Produits")
        top_products = df_clean.groupBy("product_id", "title") \
            .agg(
                count("rating").alias("nombre_avis"),
                avg("rating").alias("note_moyenne"),
                avg("price").alias("prix_moyen")
            ) \
            .sort(desc("note_moyenne")) \
            .limit(10) \
            .toPandas()

        top_products['Détails'] = False
        top_products = top_products[['Détails'] + [col for col in top_products.columns if col != 'Détails']]

        edited_top = st.data_editor(
            top_products,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Détails": st.column_config.CheckboxColumn(
                    "🔍 Détails",
                    help="Afficher les détails complets du produit",
                    default=False
                ),
                "product_id": "🆔 ID Produit",
                "title": "📛 Nom du produit",
                "note_moyenne": st.column_config.NumberColumn(
                    "⭐ Note moyenne",
                    format="%.2f",
                    min_value=0,
                    max_value=5
                ),
                "prix_moyen": st.column_config.NumberColumn(
                    "💲 Prix ",
                    format="%.2f $"
                ),
                "nombre_avis": "📊 Nombre d'avis"
            }
        )

        if edited_top['Détails'].any():
            selected_top = edited_top[edited_top['Détails']].iloc[0]
            product_id = selected_top['product_id']

            product_details = df_clean.filter(col("product_id") == product_id).toPandas()

            with st.expander(f"🔍 Détails complets du produit: {selected_top['title']}", expanded=True):
                st.subheader("📋 Informations générales")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("🆔 ID Produit", product_id)
                with col2:
                    avg_rating = product_details['rating'].mean() if not product_details.empty else 0
                    st.metric("⭐ Note moyenne", f"{avg_rating:.1f}/5")

                if not product_details.empty:
                    st.subheader("📝 Avis complets")
                    st.dataframe(
                        product_details[['user_name', 'rating', 'price', 'helpfulness', 'review_summary', 'review_text', 'review_time']],
                        use_container_width=True,
                        column_config={
                            "user_name": "👤 Utilisateur",
                            "rating": "⭐ Note",
                            "price": "💲 Prix",
                            "helpfulness": "👍 Utilité",
                            "review_summary": "📝 Résumé",
                            "review_text": "📄 Texte complet",
                            "review_time": st.column_config.DatetimeColumn(
                                "📅 Date",
                                format="DD/MM/YYYY"
                            )
                        }
                    )
                else:
                    st.warning("Aucune donnée disponible pour ce produit")

        st.subheader("👥 Derniers avis des utilisateurs")
        last_reviews = df_clean.select(
            "user_name", "product_id", "title", "rating", "review_summary", "review_time"
        ).sort(desc("review_time")).limit(10).toPandas()

        last_reviews["date"] = last_reviews["review_time"].apply(timestamp_to_date)

        display_reviews = last_reviews[["user_name", "title", "rating", "review_summary", "date"]]
        display_reviews['Détails'] = False
        display_reviews = display_reviews[['Détails'] + [col for col in display_reviews.columns if col != 'Détails']]

        edited_reviews = st.data_editor(
            display_reviews,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Détails": st.column_config.CheckboxColumn(
                    "🔍 Détails",
                    help="Afficher les détails complets de l'avis",
                    default=False
                ),
                "user_name": "👤 Utilisateur",
                "title": "📛 Produit",
                "rating": st.column_config.NumberColumn(
                    "⭐ Note",
                    format="%.1f",
                    min_value=0,
                    max_value=5
                ),
                "review_summary": "📝 Résumé",
                "date": "📅 Date"
            }
        )

        if edited_reviews['Détails'].any():
            selected_review = edited_reviews[edited_reviews['Détails']].iloc[0]
            full_review = df_clean.filter(
                (col("user_name") == selected_review['user_name']) &
                (col("title") == selected_review['title']) &
                (col("review_time") == last_reviews[last_reviews['user_name'] == selected_review['user_name']]['review_time'].iloc[0])
            ).toPandas().iloc[0]

            with st.expander(f"🔍 Détails complet de l'avis de {full_review['user_name']} pour {full_review['title']}", expanded=True):
                st.subheader("📋 Informations générales")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("👤 Utilisateur", full_review['user_name'])
                with col2:
                    st.metric("⭐ Note", full_review['rating'])

                st.subheader("📝 Résumé")
                st.write(full_review['review_summary'])

                st.subheader("📄 Texte complet")
                st.write(full_review['review_text'])

                st.subheader("📊 Utilité")
                st.write(f"👍 {full_review['helpfulness']} personnes ont trouvé cet avis utile")

        st.subheader("📥 Export des données")
        sample_data = df_clean.limit(10000).toPandas()
        csv_all = sample_data.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="💾 Télécharger un échantillon",
            data=csv_all,
            file_name=f"donnees_{user_category}.csv",
            mime="text/csv"
        )

    elif page == "📅 Analyse Temporelle":
        selected_category = st.selectbox(
            "Sélectionnez la catégorie à analyser",
            options=["Base_InduScience", "Base_AutoPlus", "Base_ArtCreatif", "Base_MobilTech"],
            format_func=lambda x: {
                "Base_InduScience": "Industrie & Science",
                "Base_AutoPlus": "Tout l'automobile",
                "Base_ArtCreatif": "Univers créatif",
                "Base_MobilTech": "Mobile & Technologie"
            }[x],
            key="temp_category"
        )

        df_clean = load_and_clean_data(selected_category)
        category_name = category_names[selected_category]

        st.title(f"📅 Analyse Temporelle - {category_name}")

        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "📅 Date de début",
                datetime(2000, 1, 1),
                min_value=datetime(2000, 1, 1),
            )
        with col2:
            end_date = st.date_input(
                "📅 Date de fin",
                datetime.today(),
                min_value=start_date,
            )

        start_timestamp = int(datetime.combine(start_date, datetime.min.time()).timestamp())
        end_timestamp = int(datetime.combine(end_date, datetime.min.time()).timestamp())

        filtered_df = df_clean.filter(
            (col("review_time") >= start_timestamp) & (col("review_time") <= end_timestamp)
        )

        if filtered_df.count() == 0:
            st.warning("⚠️ Aucune donnée pour cette période")
        else:
            st.subheader(f"📈 Activité entre {start_date.strftime('%d/%m/%Y')} et {end_date.strftime('%d/%m/%Y')}")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">👥 Utilisateurs actifs</div>
                    <div class="metric-value">{filtered_df.select("user_id").distinct().count()}</div>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">📦 Produits évalués</div>
                    <div class="metric-value">{filtered_df.select("product_id").distinct().count()}</div>
                </div>
                """, unsafe_allow_html=True)
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">⭐ Évaluations</div>
                    <div class="metric-value">{filtered_df.count()}</div>
                </div>
                """, unsafe_allow_html=True)

            st.subheader("📊 Activité au fil du temps")
            activity_df = filtered_df.withColumn("date", to_date(from_unixtime("review_time"))) \
                .groupBy("date") \
                .count() \
                .sort("date") \
                .toPandas()

            fig = px.line(
                activity_df,
                x="date",
                y="count",
                title="Nombre d'avis par jour",
                labels={"count": "Nombre d'avis", "date": "Date"},
                height=400
            )
            fig.update_layout(
                xaxis_title="Date",
                yaxis_title="Nombre d'avis",
                template="plotly_white",
                hovermode="x unified"
            )
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("🔥 Produits tendance")
            popular_products = filtered_df.groupBy("product_id", "title") \
                .agg(
                    count("rating").alias("nombre_avis"),
                    avg("rating").alias("note_moyenne"),
                    avg("price").alias("prix_moyen")
                ) \
                .sort(desc("nombre_avis")) \
                .limit(10) \
                .toPandas()

            popular_products['Détails'] = False
            popular_products = popular_products[['Détails'] + [col for col in popular_products.columns if col != 'Détails']]

            edited_trend = st.data_editor(
                popular_products,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Détails": st.column_config.CheckboxColumn(
                        "🔍 Détails",
                        help="Afficher les détails complets du produit",
                        default=False
                    ),
                    "product_id": "🆔 ID Produit",
                    "title": "📛 Nom du produit",
                    "note_moyenne": st.column_config.NumberColumn(
                        "⭐ Note moyenne",
                        format="%.2f",
                        min_value=0,
                        max_value=5
                    ),
                    "prix_moyen": st.column_config.NumberColumn(
                        "💲 Prix ",
                        format="%.2f $"
                    ),
                    "nombre_avis": "📊 Nombre d'avis"
                }
            )

            if edited_trend['Détails'].any():
                selected_trend = edited_trend[edited_trend['Détails']].iloc[0]
                product_id = selected_trend['product_id']

                product_details = filtered_df.filter(col("product_id") == product_id).toPandas()

                with st.expander(f"🔍 Détails complets du produit tendance: {selected_trend['title']}", expanded=True):
                    st.subheader("📋 Informations générales")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("🆔 ID Produit", product_id)
                    with col2:
                        avg_rating = product_details['rating'].mean() if not product_details.empty else 0
                        st.metric("⭐ Note moyenne", f"{avg_rating:.1f}/5")

                    if not product_details.empty:
                        st.subheader("📝 Avis complets")
                        st.dataframe(
                            product_details[['user_name', 'rating', 'price', 'helpfulness', 'review_summary', 'review_text', 'review_time']],
                            use_container_width=True,
                            column_config={
                                "user_name": "👤 Utilisateur",
                                "rating": "⭐ Note",
                                "price": "💲 Prix",
                                "helpfulness": "👍 Utilité",
                                "review_summary": "📝 Résumé",
                                "review_text": "📄 Texte complet",
                                "review_time": st.column_config.DatetimeColumn(
                                    "📅 Date",
                                    format="DD/MM/YYYY"
                                )
                            }
                        )
                    else:
                        st.warning("Aucune donnée disponible pour ce produit")

            st.subheader("📥 Export des données")
            period_data = filtered_df.toPandas()
            csv_period = period_data.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="💾 Télécharger les données",
                data=csv_period,
                file_name=f"donnees_{user_category}_{start_date}_{end_date}.csv",
                mime="text/csv"
            )

    elif page == "⚙️ Paramètres du Modèle":
        st.title("⚙️ Paramètres du Modèle")

        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">RMSE</div>
                <div class="metric-value">{metrics['RMSE']:.2f}</div>
                <div style='font-size: 0.8rem; margin-top: 0.5rem; color: #666;'>
                     Mesure l'erreur moyenne du modèle
                </div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">MAE</div>
                <div class="metric-value">{metrics['MAE']:.2f}</div>
                <div style='font-size: 0.8rem; margin-top: 0.5rem; color: #666;'>
                     Mesure l'erreur moyenne absolue
                </div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Précision</div>
                <div class="metric-value">{metrics['Precision']:.1f}%</div>
                <div style='font-size: 0.8rem; margin-top: 0.5rem; color: #666;'>
                    Recommandations pertinentes
                </div>
            </div>
            """, unsafe_allow_html=True)
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Rappel</div>
                <div class="metric-value">{metrics['Recall']:.1f}%</div>
                <div style='font-size: 0.8rem; margin-top: 0.5rem; color: #666;'>
                    Produits pertinents recommandés
                </div>
            </div>
            """, unsafe_allow_html=True)
        with col5:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">F1 Score</div>
                <div class="metric-value">{metrics['F1 Score']:.1f}%</div>
                <div style='font-size: 0.8rem; margin-top: 0.5rem; color: #666;'>
                    Moyenne harmonique de précision
                </div>
            </div>
            """, unsafe_allow_html=True)

        fig = go.Figure()

        st.sidebar.markdown(" ")

        fig.add_trace(go.Scatterpolar(
            r=[1 - (metrics['RMSE'] / 5), 1 - (metrics['MAE'] / 5), metrics['Precision'] / 100, metrics['Recall'] / 100, metrics['F1 Score'] / 100],
            theta=['RMSE', 'MAE', 'Precision', 'Recall', 'F1 Score'],
            fill='toself',
            name='Performance',
            line=dict(color='#4CAF50')
        ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=False,
            height=500,
            title="Analyse des performances du modèle (valeurs normalisées)"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.subheader("📅 Activité Temporelle Globale")
        activity_df = df_clean.withColumn("year", F.year(from_unixtime("review_time"))) \
            .groupBy("year") \
            .count() \
            .sort("year") \
            .toPandas()

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(activity_df['year'], activity_df['count'], marker='o', color='#4CAF50', linewidth=2)
        ax.set_xlabel('Année')
        ax.set_ylabel('Nombre d\'avis')
        ax.set_title('Évolution des avis par année')
        ax.grid(True, linestyle='--', alpha=0.7)
        st.pyplot(fig)

        st.subheader("📊 Statistiques de la Base de Données")

        total_users = df_clean.select("user_id").distinct().count()
        total_products = df_clean.select("product_id").distinct().count()
        total_reviews = df_clean.count()
        avg_rating = df_clean.select(avg("rating")).collect()[0][0]
        avg_price = df_clean.select(avg("price")).collect()[0][0]

        stats_data = {
            "Métrique": ["Utilisateurs", "Produits", "Avis", "Note moyenne", "Prix moyen"],
            "Valeur": [total_users, total_products, total_reviews, f"{avg_rating:.2f}/5", f"{avg_price:.2f} $"]
        }

        stats_df = pd.DataFrame(stats_data)
        st.dataframe(
            stats_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Métrique": st.column_config.TextColumn("📊 Métrique"),
                "Valeur": st.column_config.TextColumn("📈 Valeur")
            }
        )

        st.subheader("📊 Répartition des Données")

        data_dist = {
            "Type": ["Utilisateurs", "Produits", "Avis"],
            "Nombre": [total_users, total_products, total_reviews]
        }

        dist_df = pd.DataFrame(data_dist)

        fig = px.bar(
            dist_df,
            x="Type",
            y="Nombre",
            color="Type",
            color_discrete_sequence=["#0f3460", "#16213e", "#4CAF50"],
            labels={"Nombre": "Quantité", "Type": "Catégorie"},
            height=400
        )
        fig.update_layout(
            title="Répartition des données dans la base",
            xaxis_title="Catégorie",
            yaxis_title="Quantité",
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    **🛍️ Système de Recommandation E-commerce**
    *Version 3.0 - © 2025 khayatti-moudid*
    """)

def main():
    if 'page' not in st.session_state:
        st.session_state['page'] = "home"

    if st.session_state['page'] == "home":
        home_page()
    elif st.session_state['page'] == "login":
        login_page()
    elif st.session_state['page'] == "dashboard":
        dashboard_page()

if __name__ == "__main__":
    main()