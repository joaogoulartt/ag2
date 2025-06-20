import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)

try:
    import sys
    from pathlib import Path

    # Adicionar o diretório src ao Python path
    src_path = Path(__file__).parent.parent
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    from models.model import CreditRiskModel
    from dotenv import load_dotenv
    from sklearn.metrics import accuracy_score
    from utils.translations import (
        TEST_CASES,
        get_field_display_name,
        get_field_description,
        get_value_description,
        is_numeric_field,
        get_field_options,
    )
except ImportError as e:
    st.error(f"Erro ao importar dependências: {e}")
    st.stop()

st.set_page_config(
    page_title="Análise de Risco de Crédito",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 0.5rem 0;
    }
    
    .success-box {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        border: 1px solid #c3e6cb;
        margin: 1rem 0;
    }
    
    .warning-box {
        background-color: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 5px;
        border: 1px solid #ffeaa7;
        margin: 1rem 0;
    }
    
    .danger-box {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 5px;
        border: 1px solid #f5c6cb;
        margin: 1rem 0;
    }
    
    .stButton > button {
        background-color: #1f77b4;
        color: white;
        border-radius: 10px;
        border: none;
        padding: 0.5rem 2rem;
        font-size: 1rem;
        font-weight: bold;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        background-color: #0d5aa7;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
</style>
""",
    unsafe_allow_html=True,
)


def load_model():
    """Carrega e inicializa o modelo de risco de crédito."""
    if "model" not in st.session_state or "model_trained" not in st.session_state:
        try:
            load_dotenv()

            db_user = os.getenv("DB_USER", "root")
            db_password = os.getenv("DB_PASSWORD", "root")
            db_host = os.getenv("DB_HOST", "localhost")
            db_port = int(os.getenv("DB_PORT", "32768"))
            db_name = os.getenv("DB_NAME", "statlog")

            with st.spinner("🔄 Conectando ao banco de dados e carregando o modelo..."):
                model = CreditRiskModel(db_user, db_password, db_host, db_name, db_port)

                X_train, X_test, y_train, y_test = model.load_and_prepare_data()
                model.train(X_train, y_train)

                y_pred = model.model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)

                st.session_state.model = model
                st.session_state.model_trained = True
                st.session_state.model_accuracy = accuracy
                st.session_state.X_test = X_test
                st.session_state.y_test = y_test
                st.session_state.y_pred = y_pred

            st.success("✅ Modelo carregado e treinado com sucesso!")
            return model

        except Exception as e:
            st.error(f"❌ Erro ao carregar o modelo: {e}")
            st.stop()

    return st.session_state.model


def create_prediction_form(model):
    st.subheader("🔮 Nova Predição de Risco")

    # Adicionar casos de teste pré-definidos
    st.markdown("### 🧪 Casos de Teste Rápidos")
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("👤 Cliente Ideal", use_container_width=True):
            st.session_state.test_case = "Cliente Ideal"

    with col2:
        if st.button("⚠️ Cliente de Risco", use_container_width=True):
            st.session_state.test_case = "Cliente de Risco"

    with col3:
        if st.button("📊 Cliente Médio", use_container_width=True):
            st.session_state.test_case = "Cliente Médio"

    st.markdown("---")

    with st.expander("ℹ️ Informações sobre os campos"):
        st.write(
            """
        **Campos necessários para a análise:**
        - Use os casos de teste para ver exemplos pré-configurados
        - Para campos categóricos, selecione uma das opções disponíveis
        - Para campos numéricos, insira valores apropriados
        - Todos os campos são obrigatórios para a análise
        """
        )

    cols = st.columns(3)
    user_input = []

    # Verificar se há um caso de teste selecionado
    selected_values = {}
    if "test_case" in st.session_state and st.session_state.test_case in TEST_CASES:
        selected_values = TEST_CASES[st.session_state.test_case]["values"]
        st.info(
            f"📋 Usando valores do caso: **{st.session_state.test_case}** - {TEST_CASES[st.session_state.test_case]['description']}"
        )

    # Usar apenas as features originais para entrada do usuário
    original_features = [
        "laufkont",
        "laufzeit",
        "moral",
        "verw",
        "hoehe",
        "sparkont",
        "beszeit",
        "rate",
        "famges",
        "buerge",
        "wohnzeit",
        "verm",
        "alter",
        "weitkred",
        "wohn",
        "bishkred",
        "beruf",
        "pers",
        "telef",
        "gastarb",
    ]

    fields_per_col = len(original_features) // 3 + 1

    for i, col_name in enumerate(original_features):
        col_idx = i // fields_per_col
        display_name = get_field_display_name(col_name)
        description = get_field_description(col_name)

        with cols[col_idx]:
            # Mostrar o nome do campo com tooltip
            st.markdown(f"**{display_name}**")
            if description:
                st.caption(f"💡 {description}")

            if is_numeric_field(col_name):
                # Campo numérico
                default_value = selected_values.get(col_name, 0)
                if col_name == "laufzeit":
                    # Garantir que o valor padrão não seja menor que o mínimo
                    safe_default = (
                        max(6, int(default_value)) if default_value > 0 else 12
                    )
                    value = st.number_input(
                        f"Valor para {display_name}",
                        min_value=6,
                        max_value=72,
                        value=safe_default,
                        key=f"input_{col_name}",
                        help="Duração em meses (6 a 72 meses)",
                    )
                elif col_name == "hoehe":
                    # Garantir que o valor padrão não seja menor que o mínimo
                    safe_default = (
                        max(250, int(default_value)) if default_value > 0 else 1000
                    )
                    value = st.number_input(
                        f"Valor para {display_name}",
                        min_value=250,
                        max_value=20000,
                        value=safe_default,
                        key=f"input_{col_name}",
                        help="Valor em marcos alemães (DM)",
                    )
                elif col_name == "alter":
                    # Garantir que o valor padrão não seja menor que o mínimo
                    safe_default = (
                        max(18, int(default_value)) if default_value > 0 else 25
                    )
                    value = st.number_input(
                        f"Valor para {display_name}",
                        min_value=18,
                        max_value=80,
                        value=safe_default,
                        key=f"input_{col_name}",
                        help="Idade em anos",
                    )
                else:
                    value = st.number_input(
                        f"Valor para {display_name}",
                        value=float(default_value),
                        key=f"input_{col_name}",
                    )
            else:
                # Campo categórico
                options = get_field_options(col_name)
                if options:
                    option_labels = [f"{k}: {v}" for k, v in options.items()]
                    option_values = list(options.keys())

                    default_index = 0
                    if col_name in selected_values:
                        try:
                            default_index = option_values.index(
                                selected_values[col_name]
                            )
                        except ValueError:
                            default_index = 0

                    selected_option = st.selectbox(
                        f"Selecione {display_name}",
                        option_labels,
                        index=default_index,
                        key=f"select_{col_name}",
                    )
                    # Extrair o valor numérico da opção selecionada
                    value = int(selected_option.split(":")[0])
                else:
                    value = st.number_input(
                        f"Valor para {display_name}",
                        value=float(selected_values.get(col_name, 0)),
                        key=f"number_{col_name}",
                    )

            user_input.append(value)

    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("🚀 Analisar Risco de Crédito", type="primary"):
            try:
                with st.spinner("🔮 Analisando perfil do cliente..."):
                    # Criar DataFrame com as features originais
                    original_features = [
                        "laufkont",
                        "laufzeit",
                        "moral",
                        "verw",
                        "hoehe",
                        "sparkont",
                        "beszeit",
                        "rate",
                        "famges",
                        "buerge",
                        "wohnzeit",
                        "verm",
                        "alter",
                        "weitkred",
                        "wohn",
                        "bishkred",
                        "beruf",
                        "pers",
                        "telef",
                        "gastarb",
                    ]
                    input_df = pd.DataFrame([user_input], columns=original_features)

                    # Aplicar feature engineering
                    input_df = model.create_engineered_features(input_df)

                    # Reorganizar colunas para corresponder ao modelo treinado
                    final_input = []
                    for col in model.columns:
                        if col in input_df.columns:
                            final_input.append(input_df[col].iloc[0])
                        else:
                            final_input.append(0)  # Valor padrão para features ausentes

                    # Fazer a predição
                    details = model.get_prediction_details(final_input)
                    result = details["prediction"]
                    prob_good = details["probability_good"]
                    prob_bad = details["probability_bad"]
                    confidence = details["confidence"]
                    risk_factors = details["risk_factors"]

                st.markdown("---")
                st.subheader("📊 Resultado da Análise")

                if result == "BOM":
                    st.markdown(
                        f"""
                        <div class="success-box">
                            <h3>✅ RISCO BAIXO</h3>
                            <p><strong>Recomendação:</strong> APROVAR o crédito</p>
                            <p><strong>Probabilidade de Bom Pagador:</strong> {prob_good:.1%}</p>
                            <p><strong>Confiança da Predição:</strong> {confidence:.1%}</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                elif result == "MODERADO":
                    st.markdown(
                        f"""
                        <div class="warning-box">
                            <h3>⚠️ RISCO MODERADO</h3>
                            <p><strong>Recomendação:</strong> ANALISAR com critério adicional</p>
                            <p><strong>Probabilidade de Bom Pagador:</strong> {prob_good:.1%}</p>
                            <p><strong>Confiança da Predição:</strong> {confidence:.1%}</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"""
                        <div class="danger-box">
                            <h3>❌ RISCO ALTO</h3>
                            <p><strong>Recomendação:</strong> NEGAR o crédito</p>
                            <p><strong>Probabilidade de Mau Pagador:</strong> {prob_bad:.1%}</p>
                            <p><strong>Confiança da Predição:</strong> {confidence:.1%}</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                # Mostrar fatores de risco se existirem
                if risk_factors:
                    st.subheader("⚠️ Fatores de Risco Identificados")
                    for factor in risk_factors:
                        st.warning(f"• {factor}")

                # Gráfico de probabilidades
                col1, col2 = st.columns(2)
                with col1:
                    fig_prob = go.Figure(
                        data=[
                            go.Bar(
                                x=["Bom Pagador", "Mau Pagador"],
                                y=[prob_good, prob_bad],
                                marker_color=["green", "red"],
                            )
                        ]
                    )
                    fig_prob.update_layout(
                        title="Probabilidades de Classificação",
                        yaxis_title="Probabilidade",
                        yaxis_range=[0, 1],
                    )
                    st.plotly_chart(fig_prob, use_container_width=True)

                # Mostrar resumo dos dados inseridos
                st.subheader("📝 Resumo dos Dados Inseridos")
                summary_data = []
                for i, col_name in enumerate(original_features):
                    display_name = get_field_display_name(col_name)
                    value_desc = get_value_description(col_name, user_input[i])
                    summary_data.append(
                        {
                            "Campo": display_name,
                            "Valor": user_input[i],
                            "Descrição": value_desc,
                        }
                    )

                if summary_data:
                    summary_df = pd.DataFrame(summary_data)
                    st.dataframe(summary_df, use_container_width=True)

            except Exception as e:
                st.error(f"❌ Erro na predição: {e}")
                st.error("Tente novamente ou verifique os dados inseridos.")

    # Limpar caso de teste após uso
    if st.button("🔄 Limpar Formulário"):
        for key in st.session_state.keys():
            if key.startswith(("input_", "select_", "number_", "test_case")):
                del st.session_state[key]
        st.rerun()


def show_model_performance():
    """Exibe métricas de performance do modelo."""
    if "model_accuracy" in st.session_state:
        st.subheader("📈 Performance do Modelo")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                label="Acurácia",
                value=f"{st.session_state.model_accuracy:.2%}",
                delta=f"{st.session_state.model_accuracy - 0.5:.2%}",
            )

        with col2:
            from sklearn.metrics import precision_score, recall_score

            precision = precision_score(
                st.session_state.y_test, st.session_state.y_pred, average="weighted"
            )
            st.metric(label="Precisão", value=f"{precision:.2%}")

        with col3:
            recall = recall_score(
                st.session_state.y_test, st.session_state.y_pred, average="weighted"
            )
            st.metric(label="Recall", value=f"{recall:.2%}")

        from sklearn.metrics import confusion_matrix

        cm = confusion_matrix(st.session_state.y_test, st.session_state.y_pred)

        fig = px.imshow(
            cm,
            text_auto=True,
            color_continuous_scale="Blues",
            title="Matriz de Confusão",
        )
        fig.update_xaxes(title="Predição")
        fig.update_yaxes(title="Real")

        st.plotly_chart(fig, use_container_width=True)


def show_data_overview():
    """Exibe overview dos dados."""
    if "model" in st.session_state:
        st.subheader("📊 Visão Geral dos Dados")

        try:
            model = st.session_state.model
            query = "SELECT * FROM germancredit LIMIT 1000"
            df = pd.read_sql(query, con=model.engine)

            col1, col2 = st.columns(2)

            with col1:
                st.write("**Estatísticas dos Dados:**")
                st.write(f"- Total de registros: {len(df)}")
                st.write(f"- Número de colunas: {len(df.columns)}")
                st.write(f"- Dados faltantes: {df.isnull().sum().sum()}")

            with col2:
                if "kredit" in df.columns:
                    target_dist = df["kredit"].value_counts()
                    fig = px.pie(
                        values=target_dist.values,
                        names=["Risco Alto", "Risco Baixo"],
                        title="Distribuição dos Riscos",
                    )
                    st.plotly_chart(fig, use_container_width=True)

            st.write("**Amostra dos Dados:**")
            st.dataframe(df.head(10), use_container_width=True)

        except Exception as e:
            st.error(f"Erro ao carregar dados: {e}")


def main():
    """Função principal da aplicação."""

    st.markdown(
        '<h1 class="main-header">💳 Análise de Risco de Crédito</h1>',
        unsafe_allow_html=True,
    )
    st.markdown("---")

    st.sidebar.title("🧭 Navegação")
    page = st.sidebar.selectbox(
        "Escolha uma página:",
        ["🏠 Dashboard", "🔮 Nova Predição", "📊 Dados", "⚙️ Configurações"],
    )

    model = load_model()

    if page == "🏠 Dashboard":
        st.header("📈 Dashboard Principal")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(
                """
            <div class="metric-card">
                <h3>🎯 Modelo Ativo</h3>
                <p>Decision Tree Classifier</p>
                <p>Status: ✅ Treinado e Pronto</p>
            </div>
            """,
                unsafe_allow_html=True,
            )

        with col2:
            st.markdown(
                f"""
            <div class="metric-card">
                <h3>📊 Última Atualização</h3>
                <p>{datetime.now().strftime('%d/%m/%Y %H:%M')}</p>
                <p>Conexão: ✅ Ativa</p>
            </div>
            """,
                unsafe_allow_html=True,
            )

        show_model_performance()

    elif page == "🔮 Nova Predição":
        create_prediction_form(model)

    elif page == "📊 Dados":
        show_data_overview()

    elif page == "⚙️ Configurações":
        st.header("⚙️ Configurações do Sistema")

        st.subheader("🔧 Informações do Ambiente")

        config_info = {
            "Banco de Dados": os.getenv("DB_NAME", "N/A"),
            "Host": os.getenv("DB_HOST", "N/A"),
            "Porta": os.getenv("DB_PORT", "N/A"),
            "Usuário": os.getenv("DB_USER", "N/A"),
        }

        for key, value in config_info.items():
            st.write(f"**{key}:** {value}")

        st.subheader("🎛️ Opções")

        if st.button("🔄 Recarregar Modelo"):
            for key in ["model", "model_trained", "model_accuracy"]:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()

        if st.button("🧹 Limpar Cache"):
            st.cache_data.clear()
            st.success("Cache limpo com sucesso!")

    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666;'>"
        "💡 Sistema de Análise de Risco de Crédito - Powered by Streamlit & Machine Learning"
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
