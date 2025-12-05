import pickle
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

st.set_page_config(page_title="Car price demo", layout="wide")
st.title("Предсказание стоимости авто")

bundle = pickle.load(open("car_price_model.pickle", "rb"))
model, scaler, ohe = bundle["model"], bundle["scaler"], bundle["ohe"]
NUM, CAT, ALL = (
    bundle["numeric_features"],
    bundle["categorical_features"],
    bundle["all_features"],
)
NUM_MED = bundle.get("numeric_medians", {})
CAT_LEVELS = bundle.get("categorical_levels", {})
EDA_DF = bundle.get("eda_df", pd.DataFrame())


def clean_numeric(df):
    x = df.copy()
    x["mileage"] = x["mileage"].astype(str).str.replace(r" kmpl| km/kg", "", regex=True)
    x["engine"] = x["engine"].astype(str).str.replace(" CC", "", regex=False)
    x["max_power"] = x["max_power"].astype(str).str.replace(" bhp", "", regex=False)
    for c in ["mileage", "engine", "max_power"]:
        x[c] = x[c].replace(["nan", ""], np.nan).astype(float)
    return x


# claude.ai, использовал аналогичный промпт в ноутбуке для парсинга строки
def parse_torque(df):
    d = df.copy()
    d["torque_value"], d["max_torque_rpm"] = np.nan, np.nan
    for i, t in d["torque"].astype(str).items():
        if t == "nan" or pd.isna(t):
            continue
        m = re.search(r"([\\d.]+)\\s*(?:Nm|nm|NM|kgm)", t, re.IGNORECASE)
        r = re.search(
            r"@?\\s*([\\d,]+)(?:-[\\d,]+)?\\s*(?:rpm|\\(rpm)", t, re.IGNORECASE
        )
        if m:
            v = float(m.group(1))
            if "kgm" in t.lower():
                v *= 9.80665
            d.at[i, "torque_value"] = v
        if r:
            d.at[i, "max_torque_rpm"] = float(r.group(1).replace(",", ""))
    return d.drop(columns=["torque"], errors="ignore")


def add_brand_flags(df):
    # claude.ai: использовал чтобы разбить бренды авто по "премиальности" (аналогично в ноутбуке)
    g = {
        "budget_mass": ["Maruti", "Tata", "Datsun", "Daewoo"],
        "mass_market": [
            "Hyundai",
            "Honda",
            "Ford",
            "Renault",
            "Nissan",
            "Chevrolet",
            "Fiat",
        ],
        "mass_premium": [
            "Toyota",
            "Mahindra",
            "Skoda",
            "Volkswagen",
            "Kia",
            "MG",
            "Jeep",
        ],
        "specialized": ["Force", "Isuzu", "Mitsubishi", "Peugeot", "Ambassador"],
        "premium": ["Land", "Lexus", "Volvo"],
        "luxury": ["Audi", "BMW", "Mercedes-Benz", "Jaguar"],
    }
    x = df.copy()
    if "name" not in x:
        x["name"] = "Unknown"
    x["brand"] = x["name"].str.split().str[0]
    for k, cars in g.items():
        x[f"is_{k}"] = x["brand"].isin(cars).astype(int)
    return x.drop(columns=["brand"])


def preprocess(df):
    x = df.copy()
    if "torque" in x.columns:
        x = parse_torque(x)
    for c in ["year", "km_driven", "seats"]:
        x[c] = pd.to_numeric(x[c], errors="coerce") if c in x else np.nan
    if {"mileage", "engine", "max_power"}.issubset(x.columns):
        x = clean_numeric(x)
    for c in NUM:
        if c not in x:
            x[c] = np.nan
        x[c] = x[c].fillna(NUM_MED.get(c, x[c].median()))
    x["engine"] = x["engine"].astype(int, errors="ignore")
    x["seats"] = x["seats"].astype(int, errors="ignore")
    x = add_brand_flags(x)
    for c, v in CAT_LEVELS.items():
        x[c] = x.get(c, pd.Series([v[0] if v else "Unknown"] * len(x))).fillna(
            v[0] if v else x[c].mode().iloc[0]
        )
    for c in CAT:
        if c not in x:
            x[c] = 0
    return x


def transform(df):
    n = pd.DataFrame(scaler.transform(df[NUM]), columns=NUM, index=df.index)
    c = pd.DataFrame(
        ohe.transform(df[CAT]), columns=ohe.get_feature_names_out(CAT), index=df.index
    )
    return pd.concat([n, c], axis=1).reindex(columns=ALL, fill_value=0)


def predict_df(df):
    return pd.Series(model.predict(transform(preprocess(df))), index=df.index)


eda_tab, predict_tab, weights_tab = st.tabs(["EDA", "Предсказание", "Веса модели"])

with eda_tab:
    st.subheader("EDA по трейну из pickle")
    if EDA_DF.empty:
        st.warning("EDA недоступен: в pickle нет трейна.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots()
            sns.histplot(EDA_DF["selling_price"], bins=40, ax=ax, color="#2b8cbe")
            ax.set_title("Распределение цены")
            st.pyplot(fig)
            fig, ax = plt.subplots()
            sns.boxplot(
                x=EDA_DF["fuel"], y=EDA_DF["selling_price"], ax=ax, palette="Set2"
            )
            ax.set_ylabel("Цена")
            st.pyplot(fig)
        with col2:
            fig, ax = plt.subplots()
            sns.scatterplot(
                x=EDA_DF["year"], y=EDA_DF["selling_price"], hue=EDA_DF["fuel"], ax=ax
            )
            ax.set_title("Цена vs год")
            st.pyplot(fig)
            fig, ax = plt.subplots()
            corr = EDA_DF[NUM + ["selling_price"]].corr()
            sns.heatmap(corr, annot=False, cmap="coolwarm", ax=ax)
            ax.set_title("Корреляции числовых фич")
            st.pyplot(fig)

with predict_tab:
    st.subheader("Загрузить CSV или ввести руками")
    uploaded = st.file_uploader("CSV в исходных колонках датасета", type=["csv"])
    if uploaded:
        df_csv = pd.read_csv(uploaded)
        preds = predict_df(df_csv)
        out = df_csv.copy()
        out["predicted_price"] = preds
        st.success(f"Посчитано {len(out)} строк")
        st.dataframe(out.head())
        st.download_button(
            "Скачать с предсказаниями",
            data=out.to_csv(index=False),
            file_name="predictions.csv",
        )
    st.markdown("### Ручной ввод одного авто")
    with st.form("manual"):
        c1, c2, c3 = st.columns(3)
        name = c1.text_input("Модель", value="Hyundai i20 Sportz")
        # claude.ai: использовал чтобы руками не переписывать и кучу полей
        year = c2.number_input("Год", 1995, 2025, 2015)
        km = c3.number_input("Пробег, км", 0, 400000, 70000)
        fuel = c1.selectbox("Топливо", CAT_LEVELS.get("fuel", ["Petrol"]))
        seller = c2.selectbox("Тип продавца", CAT_LEVELS.get("seller_type", ["Dealer"]))
        transmission = c3.selectbox(
            "Коробка", CAT_LEVELS.get("transmission", ["Manual"])
        )
        owner = c1.selectbox("Владелец", CAT_LEVELS.get("owner", ["First Owner"]))
        seats = c2.selectbox("Сиденья", CAT_LEVELS.get("seats", [5]))
        mileage = c3.number_input("Расход (kmpl)", 5.0, 40.0, 18.0, step=0.1)
        engine = c1.number_input("Объем, CC", 600, 5000, 1200)
        power = c2.number_input("Мощность, bhp", 30.0, 400.0, 80.0, step=1.0)
        torque = c3.text_input("Крутящий момент (строкой)", value="190Nm@ 2000rpm")
        submitted = st.form_submit_button("Посчитать цену")
    if submitted:
        sample = pd.DataFrame(
            [
                {
                    "name": name,
                    "year": year,
                    "km_driven": km,
                    "fuel": fuel,
                    "seller_type": seller,
                    "transmission": transmission,
                    "owner": owner,
                    "mileage": mileage,
                    "engine": engine,
                    "max_power": power,
                    "torque": torque,
                    "seats": seats,
                }
            ]
        )
        st.success(f"Прогноз: **{predict_df(sample).iloc[0]:,.0f}** ₽")

with weights_tab:
    st.subheader("Коэффициенты модели Ridge")
    coefs = pd.DataFrame({"feature": ALL, "coef": model.coef_})
    top = pd.concat([coefs.nlargest(10, "coef"), coefs.nsmallest(10, "coef")])
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(data=top, x="coef", y="feature", palette="coolwarm", ax=ax)
    ax.axvline(0, color="black", lw=1)
    st.pyplot(fig)
    st.dataframe(coefs.sort_values("coef", ascending=False).reset_index(drop=True))

st.caption(
    f"Модель: {bundle.get('model_name', 'Ridge')} | alpha={bundle.get('best_alpha')} | R²={bundle.get('test_r2', 0):.3f} | бизнес-метрика={bundle.get('business_metric', 0)}%"
)
