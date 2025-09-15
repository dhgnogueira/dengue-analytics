import streamlit as st


def show_indicators(df_rmvp, municipality_col):
    df_validos = df_rmvp[
        (df_rmvp[municipality_col].notnull())
        & (df_rmvp[municipality_col] != "0")
        & (df_rmvp[municipality_col] != "")
    ]

    def format_value(value):
        if value >= 1_000_000:
            return f"{value / 1_000_000:.2f} milhões"
        elif value >= 1_000:
            return f"{value / 1_000:.0f} mil"
        else:
            return f"{value}"

    total_casos = int(df_validos["Total"].sum())
    media_anual = int(df_validos.groupby("Ano")["Total"].sum().mean())
    pico_ano = df_validos.groupby("Ano")["Total"].sum().idxmax()
    pico_valor = int(df_validos.groupby("Ano")["Total"].sum().max())
    muni_pico = df_validos.groupby(municipality_col)["Total"].sum().idxmax()
    muni_pico_valor = int(df_validos.groupby(municipality_col)["Total"].sum().max())
    # Adiciona CSS responsivo para o indicador
    st.markdown(
        """
        <style>
        .indicador-total {
            font-size: 2rem;
            word-break: break-word;
            max-width: 100%;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: normal;
        }
        @media (max-width: 600px) {
            .indicador-total {
                font-size: 1.2rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    col1, col2, col3, col4 = st.columns(4)
    col1.markdown(
        f"""<div class="indicador-total"><b>Total de casos</b><br>{format_value(total_casos)}</div>""",
        unsafe_allow_html=True,
    )
    col2.markdown(
        f"""<div class="indicador-total"><b>Média anual de casos</b><br>{format_value(media_anual)}</div>""",
        unsafe_allow_html=True,
    )
    col3.markdown(
        f"""<div class="indicador-total"><b>Ano com mais casos</b><br>{pico_ano} ({format_value(pico_valor)})</div>""",
        unsafe_allow_html=True,
    )
    col4.markdown(
        f"""<div class="indicador-total"><b>Município com mais casos (total acumulado)</b><br>{muni_pico} ({format_value(muni_pico_valor)})</div>""",
        unsafe_allow_html=True,
    )
