import re
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st
from services.comparison import show_comparison
from services.evolution import show_evolution
from services.forecast import show_forecast
from services.incidence_rmvp import show_incidence_rmvp
from services.indicators import show_indicators
from services.ranking import show_ranking
from services.rmvp_analysis import show_rmvp_analysis

st.set_page_config(page_title="Dashboard Dengue", layout="wide")
st.title("Dashboard de Casos de Dengue")
st.markdown(
    f"<h6>Última atualização: {datetime.now(ZoneInfo('America/Sao_Paulo')).strftime('%d-%b-%Y')}</h6>",
    unsafe_allow_html=True,
)


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

    # 1. Gráfico de previsão (real vs previsto) - DESTAQUE NO TOPO
    show_forecast(df_rmvp=df_rmvp_filtered)

    # 2. Indicadores resumidos (cards)
    show_indicators(df_rmvp=df_rmvp_filtered, municipality_col=municipality_col)

    # 3. Gráfico de incidência/casos (lado a lado)
    col1, col2 = st.columns(2)
    with col1:
        show_incidence_rmvp()
    with col2:
        show_comparison(
            df_rmvp=df_rmvp_filtered,
            municipality_col=municipality_col,
            municipalities=municipalities,
        )

    # 4. Ranking de municípios e evolução temporal (em colunas)
    col3, col4 = st.columns(2)
    with col3:
        show_ranking(df_rmvp=df_rmvp_filtered, municipality_col=municipality_col, years=years)
    with col4:
        show_evolution(
            df_rmvp=df_rmvp_filtered,
            municipality_col=municipality_col,
            municipalities=municipalities,
        )

    # 5. Análise RMVP
    show_rmvp_analysis(df_rmvp=df_rmvp_filtered, municipality_col=municipality_col, months=months)


if __name__ == "__main__":
    main_dashboard()
