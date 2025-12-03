import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# -----------------------------
# Configurações iniciais da página
# -----------------------------
st.set_page_config(
    page_title="Sistema de Suporte à Previsão de Vendas",
    layout="wide"
)

# -----------------------------
# Estilos (CSS) - fundo e tabelas
# -----------------------------
st.markdown("""
    <style>
        /* Fundo geral em bege rosado */
        .stApp {
            background-color: #f9efeb;
        }

        /* Centralizar título principal */
        .block-container {
            padding-top: 1rem;
            padding-bottom: 2rem;
        }

        /* Tabelas com fundo branco */
        .stDataFrame, .stTable {
            background-color: white !important;
            border-radius: 8px;
            padding: 8px;
        }
    </style>
""", unsafe_allow_html=True)

# -----------------------------
# Sidebar – Upload do arquivo
# -----------------------------
st.sidebar.header("Upload dos dados")
arquivo = st.sidebar.file_uploader(
    "Envie um arquivo CSV consolidado",
    type=["csv"]
)

# -----------------------------
# Logo (se existir no diretório)
# -----------------------------
st.markdown("<div style='height:25px'></div>", unsafe_allow_html=True)
st.image("logo_bistro.png", width=110)
st.title("Mini App de Previsão de Vendas")
st.write("Ferramenta simples para apoiar o planejamento de compras do bistrô, utilizando o histórico de vendas de cada produto.")

# -----------------------------
# Se não houver arquivo, mostra aviso
# -----------------------------
if arquivo is None:
    st.info("Envie um arquivo CSV para continuar.")
else:
    # -----------------------------
    # Leitura e preparação dos dados
    # -----------------------------
    df = pd.read_csv(arquivo)

    # Garante tipo de data
    df["data_ref"] = pd.to_datetime(df["data_ref"])

    st.subheader("Pré-visualização dos dados enviados")

    # Cópia só para exibição, formatando a data_ref como AAAA-MM
    df_preview = df.copy()
    df_preview["data_ref"] = pd.to_datetime(df_preview["data_ref"]).dt.strftime("%Y-%m")
    
    st.dataframe(df_preview)

    st.sidebar.subheader("Configurações da Previsão")

    # Produtos em ordem alfabética
    lista_produtos = sorted(df["produto"].unique())
    produto_selecionado = st.sidebar.selectbox(
        "Selecione o produto",
        lista_produtos
    )

    meses_prever = st.sidebar.slider(
        "Quantos meses deseja prever?",
        min_value=1,
        max_value=12,
        value=3
    )

    st.subheader(f"Produto selecionado: {produto_selecionado}")
    st.write(f"Meses a prever: **{meses_prever}**")

    # -----------------------------
    # Monta série temporal do produto
    # -----------------------------
    df_produto = df[df["produto"] == produto_selecionado].copy()

    serie = (
        df_produto
        .groupby("data_ref", as_index=False)["quantidade"]
        .sum()
        .rename(columns={"data_ref": "ds", "quantidade": "y"})
    )

    # Tabela da série (datas sem horário)
    serie_display = serie.copy()
    serie_display["Período"] = serie_display["ds"].dt.strftime("%Y-%m")
    serie_display["Quantidade vendida"] = serie_display["y"]

    # Remove colunas antigas
    serie_display = serie_display[["Período", "Quantidade vendida"]]
        
    st.subheader("Série temporal do produto selecionado")
    st.dataframe(serie_display)

    # -----------------------------
    # Se houver dados suficientes, permite previsão
    # -----------------------------
    st.subheader("Previsão de vendas")

    if len(serie) < 3:
        st.warning("Histórico muito curto para gerar previsão. É recomendável ter pelo menos 3 a 4 registros mensais.")
    else:
        if st.button("Gerar previsão"):
            # -----------------------------
            # Ajuste do modelo Prophet
            # -----------------------------
            modelo = Prophet()
            modelo.fit(serie)

            # Gera datas futuras
            futuro = modelo.make_future_dataframe(
                periods=meses_prever,
                freq="MS"
            )
            previsao = modelo.predict(futuro)

            # Apenas os meses futuros para a tabela de previsão
            previsao_futuro = previsao.tail(meses_prever).copy()

            # Arredondar valores para inteiros
            previsao_futuro["yhat"] = previsao_futuro["yhat"].round().astype(int)
            previsao_futuro["yhat_lower"] = previsao_futuro["yhat_lower"].round().astype(int)
            previsao_futuro["yhat_upper"] = previsao_futuro["yhat_upper"].round().astype(int)

            # Tabela principal (com limites)
            tabela_prev = pd.DataFrame({
                "Data": previsao_futuro["ds"].dt.strftime("%Y-%m-%d"),
                "Previsão (unidades)": previsao_futuro["yhat"],
                "Limite inferior": previsao_futuro["yhat_lower"],
                "Limite superior": previsao_futuro["yhat_upper"]
            })

            st.write("Tabela de previsão para os próximos meses:")
            st.dataframe(tabela_prev)

            # Tabela resumida para leitura simples
            st.write("A previsão para os próximos meses é:")
            tabela_resumida = tabela_prev[["Data", "Previsão (unidades)"]].copy()
            # Opcional: formato mês/ano
            tabela_resumida["Data"] = pd.to_datetime(tabela_resumida["Data"]).dt.strftime("%m/%Y")
            st.table(tabela_resumida)

            # -----------------------------
            # Botão para download do CSV
            # -----------------------------
            csv_prev = tabela_prev.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Baixar previsão em CSV",
                data=csv_prev,
                file_name=f"previsao_{produto_selecionado}.csv",
                mime="text/csv"
            )

            # -----------------------------
            # Gráfico histórico + previsão
            # -----------------------------
            fig, ax = plt.subplots(figsize=(10, 3))

            # Intervalo de confiança
            ax.fill_between(
                previsao["ds"],
                previsao["yhat_lower"],
                previsao["yhat_upper"],
                alpha=0.2,
                label="Intervalo de confiança"
            )

            # Série observada
            ax.plot(
                serie["ds"],
                serie["y"],
                marker="o",
                linewidth=2,
                label="Vendas observadas"
            )

            # Valores em cima dos pontos observados
            for x, y in zip(serie["ds"], serie["y"]):
                ax.text(
                    x, y + 2,
                    f"{int(y)}",
                    fontsize=8,
                    ha="center",
                    va="bottom"
                )

            # Linha de previsão
            ax.plot(
                previsao["ds"],
                previsao["yhat"],
                linewidth=2,
                label="Previsão Prophet"
            )

            ax.set_title(f"Vendas históricas e previsão - {produto_selecionado}")
            ax.set_xlabel("Mês")
            ax.set_ylabel("Quantidade vendida")
            ax.grid(True)
            plt.xticks(rotation=45)
            ax.legend()

            st.pyplot(fig)
        else:
            st.info("Clique em **Gerar previsão** para calcular os próximos meses.")
