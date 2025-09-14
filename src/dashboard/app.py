import re

import pandas as pd
import streamlit as st
from services.comparison import show_comparison
from services.evolution import show_evolution
from services.forecast import show_forecast
from services.indicators import show_indicators
from services.ranking import show_ranking
from services.rmvp_analysis import show_rmvp_analysis

st.set_page_config(page_title="Dashboard Dengue", layout="wide")
st.title("Dashboard de Casos de Dengue")


def main_dashboard():
    df_rmvp = pd.read_csv("src/data/rmvp.csv")
    municipality_col = "Município de notificação"
    months = ["Jan", "Fev", "Mar", "Abr", "Mai", "Jun", "Jul", "Ago", "Set", "Out", "Nov", "Dez"]
    years = sorted(df_rmvp["Ano"].unique())

    municipality_pattern = re.compile(r"[A-ZÀ-Ÿ]{2,}.*")
    municipalities_vale = [
        m
        for m in df_rmvp[municipality_col].dropna().unique()
        if isinstance(m, str)
        and municipality_pattern.match(m)
        and m not in ["SÃO PAULO", "SAO PAULO"]
    ]
    municipalities = sorted(municipalities_vale)

    df_rmvp_filtered = df_rmvp[~df_rmvp[municipality_col].isin(["SÃO PAULO", "SAO PAULO"])]
    show_indicators(df_rmvp=df_rmvp_filtered, municipality_col=municipality_col)
    show_rmvp_analysis(df_rmvp=df_rmvp_filtered, municipality_col=municipality_col, months=months)
    show_forecast(df_rmvp=df_rmvp_filtered)
    show_evolution(
        df_rmvp=df_rmvp_filtered, municipality_col=municipality_col, municipalities=municipalities
    )
    show_ranking(df_rmvp=df_rmvp_filtered, municipality_col=municipality_col, years=years)
    show_comparison(
        df_rmvp=df_rmvp_filtered, municipality_col=municipality_col, municipalities=municipalities
    )


if __name__ == "__main__":
    main_dashboard()
