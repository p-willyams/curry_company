import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="Curry Company - Marketplace",
    page_icon="🍛",
    layout="wide",
    initial_sidebar_state="expanded"
)

def carregar_logo(path='imagens/logo.png'):
    """
    Carrega a imagem do logo da empresa.
    """
    return Image.open(path)

def main():
    logo = carregar_logo('imagens/logo.png')
    col1, col2 = st.columns([1, 4])
    with col1:
        st.image(logo, width=120)
    with col2:
        st.markdown("<h1 style='margin-bottom:0;'>Curry Company</h1>", unsafe_allow_html=True)
        st.markdown("## Marketplace de Entregas")
        st.markdown(
            "Bem-vindo ao painel de controle da Curry Company! "
            "Aqui você pode navegar pelas diferentes visões do nosso marketplace e analisar os principais indicadores de desempenho."
        )
    
    st.markdown("---")
    st.markdown("### Como navegar pelo site:")
    st.markdown(
        """
        Utilize o **menu lateral** à esquerda para acessar as diferentes visões disponíveis:

        - **Visão Empresa**: Métricas e análises gerais do marketplace.
        - **Visão Entregadores**: Desempenho e ranking dos entregadores.
        - **Visão Restaurantes**: Indicadores e análises dos restaurantes parceiros.

        Basta clicar na visão desejada no menu para acessar os dados e gráficos correspondentes. 
        Em cada visão, utilize os filtros disponíveis na barra lateral para refinar sua análise conforme necessário.
        """
    )
    st.markdown("---")
    st.markdown("Desenvolvido por Patryck.")

if __name__ == "__main__":
    main()
