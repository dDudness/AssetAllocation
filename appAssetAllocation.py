# Importamos las librerias
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Importamos yahoo finance library para poder descargar los datos desde el codigo
import yfinance as yf

# Imprtamos warnings para ignorar advertencias y el codigo sea mas limpio
import warnings
warnings.filterwarnings('ignore')

# Importamos scipy.stats para poder hacer ajustes de distribiciones
from scipy.stats import kurtosis, skew, norm, t

# Importamos plotly para las gráficas (más bonitas que en matplot)
import plotly as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import date
import warnings
warnings.filterwarnings('ignore')


def dl(tickers: list, start_date='2010-01-01', end_date='2023-12-31'):

    # Usamos las fechas ya establecidas en la definicion de la funcion
    # Progress false para una mayor limpieza a la hora de la visualización

    data = yf.download(tickers, start=start_date, end=end_date, progress=False)['Close']
    return data



def calcular_estadisticas(daily_returns):
    
    # Estadísticas descriptivas básicas
    estadisticas_basicas = daily_returns.describe()  
    # Calcular curtosis y sesgo
    curtosis = kurtosis(daily_returns, fisher=True, axis=0)  
    sesgo = skew(daily_returns, axis=0)

    # Crear DataFrame para curtosis y sesgo
    adicionales = pd.DataFrame({
        'Curtosis': curtosis,
        'Sesgo': sesgo
    }, index=daily_returns.columns).T  

    estadisticas_completas = pd.concat([estadisticas_basicas, adicionales])

    return estadisticas_completas

def var_cvar(df):
    VaR = df.quantile(1 - 0.95)  # VaR al 5% (percentil 95% de las pérdidas)
    cvar = df[df <= VaR].mean()  # CVaR es el promedio de los valores por debajo de VaR
    data = pd.DataFrame({'VaR (5%)': [VaR], 'CVaR (5%)': [cvar]})
    return data

def sharpe(returns, risk_free_rate=0.02):

 
  # Calculamos el exceso de retorno sobre la tasa libre de riesgo
  excess_returns = returns - (risk_free_rate / 252)
  # Retorno ajustado por riesgo (Sharpe Ratio)
  return np.sqrt(252) * excess_returns.mean() / excess_returns.std()



def sortino(returns, risk_free_rate=0.02, target_return=0):


  excess_returns = returns - (risk_free_rate / 252)
  downside_returns = excess_returns[excess_returns < target_return]
  downside_deviation = np.sqrt(np.mean(downside_returns**2))

  # Retorno ajustado por riesgo (Sortino Ratio), con manejo de error cuando downside_deviation = 0
  return np.sqrt(252) * excess_returns.mean() / downside_deviation if downside_deviation != 0 else np.nan



def drawdawn(precios):
    # Calcular el high water mark (máximos históricos acumulados)
    high_water_mark = precios.expanding().max()
    
    # Calcular el drawdown como la proporción de caída desde el máximo histórico
    drawdown = (precios - high_water_mark) / high_water_mark
    
    return drawdown, high_water_mark


def crear_histograma_distribucion(returns, var_95, cvar_95, title):
    """
    Crea un histograma interactivo con la distribución de los retornos y resalta el VaR y CVaR al 95%.

    Parámetros:
    - returns: Serie de retornos diarios.
    - var_95: Valor en riesgo (VaR) al 95%.
    - cvar_95: Valor en riesgo condicional (CVaR) al 95%.
    - title: Título del gráfico.

    Retorna:
    - fig: Objeto de figura interactiva de Plotly.
    """
    import plotly.graph_objects as go
    import numpy as np

    # Calcular los bins y frecuencias para el histograma
    counts, bins = np.histogram(returns, bins=50)

    # Máscara para identificar valores por debajo del VaR
    mask_before_var = bins[:-1] <= var_95

    # Crear la figura base
    fig = go.Figure()

    # Añadir el histograma para retornos < VaR (rojo)
    fig.add_trace(go.Bar(
        x=bins[:-1][mask_before_var],  # Lados izquierdos de los bins
        y=counts[mask_before_var],    # Frecuencias correspondientes
        width=np.diff(bins)[mask_before_var],  # Ancho de los bins
        name='Retornos < VaR',
        marker_color='rgba(255, 65, 54, 0.6)'
    ))

    # Añadir el histograma para retornos >= VaR (azul)
    fig.add_trace(go.Bar(
        x=bins[:-1][~mask_before_var],
        y=counts[~mask_before_var],
        width=np.diff(bins)[~mask_before_var],
        name='Retornos ≥ VaR',
        marker_color='rgba(31, 119, 180, 0.6)'
    ))

    # Añadir una línea vertical para el VaR al 95%
    fig.add_trace(go.Scatter(
        x=[var_95, var_95],
        y=[0, max(counts)],
        mode='lines',
        name='VaR 95%',
        line=dict(color='green', width=2, dash='dash')
    ))

    # Añadir una línea vertical para el CVaR al 95%
    fig.add_trace(go.Scatter(
        x=[cvar_95, cvar_95],
        y=[0, max(counts)],
        mode='lines',
        name='CVaR 95%',
        line=dict(color='purple', width=2, dash='dot')
    ))

    # Configuración del diseño del gráfico
    fig.update_layout(
        title=title,
        xaxis_title='Retornos',
        yaxis_title='Frecuencia',
        legend=dict(title='Leyenda'),
        barmode='overlay',  # Superposición de barras
        bargap=0.0          # Sin espacio entre las barras
    )
    return fig



# Configuración de la página
st.set_page_config(page_title="Analizador de Portafolio", layout="wide")
st.sidebar.title("Analizador de Portafolio de Inversión")

# Entrada de símbolos y pesos
simbolos_input = st.sidebar.text_input("Ingrese los símbolos de las acciones separados por comas (por ejemplo: AAPL,GOOGL,MSFT):", "AAPL,GOOGL,MSFT,AMZN,NVDA")
pesos_input = st.sidebar.text_input("Ingrese los pesos correspondientes separados por comas (deben sumar 1):", "0.2,0.2,0.2,0.2,0.2")

simbolos = [s.strip() for s in simbolos_input.split(',')]
pesos = [float(w.strip()) for w in pesos_input.split(',')]

# Selección del benchmark
benchmark_options = {
    "S&P 500": "^GSPC",
    "Nasdaq": "^IXIC",
    "Dow Jones": "^DJI",
    "Russell 2000": "^RUT",
    "ACWI": "ACWI"
}
selected_benchmark = st.sidebar.selectbox("Seleccione el benchmark:", list(benchmark_options.keys()))
benchmark = benchmark_options[selected_benchmark]

# Selección de la ventana de tiempo
end_date = datetime.now()
start_date_options = {
    "1 mes": end_date - timedelta(days=30),
    "3 meses": end_date - timedelta(days=90),
    "6 meses": end_date - timedelta(days=180),
    "1 año": end_date - timedelta(days=365),
    "3 años": end_date - timedelta(days=3*365),
    "5 años": end_date - timedelta(days=5*365),
    "10 años": end_date - timedelta(days=10*365)
}
selected_window = st.sidebar.selectbox("Seleccione la ventana de tiempo para el análisis:", list(start_date_options.keys()))
start_date = start_date_options[selected_window]

if len(simbolos) != len(pesos) or abs(sum(pesos) - 1) > 1e-6:
    st.sidebar.error("El número de símbolos debe coincidir con el número de pesos, y los pesos deben sumar 1.")
else:
    # Obtener datos
    all_symbols = simbolos + [benchmark]
    df_stocks = dl(all_symbols, start_date, end_date)  # Esta función debe ser definida para obtener los datos
    returns, cumulative_returns, normalized_prices = calcular_metricas(df_stocks)  # Definir esta función según tu lógica
    
    # Rendimientos del portafolio
    portfolio_returns = calcular_rendimientos_portafolio(returns[simbolos], pesos)  # Función a definir
    portfolio_cumulative_returns = (1 + portfolio_returns).cumprod() - 1

    # Crear pestañas
    tab1, tab2 = st.tabs(["Análisis de Activos Individuales", "Análisis del Portafolio"])

    with tab1:
        st.header("Análisis de Activos Individuales")
        
        selected_asset = st.selectbox("Seleccione un activo para analizar:", simbolos)
        
        # Calcular VaR y CVaR para el activo seleccionado
        var_95, cvar_95 = calcular_var_cvar(returns[selected_asset])  # Función a definir
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Rendimiento Total", f"{cumulative_returns[selected_asset].iloc[-1]:.2%}")
        col2.metric("Sharpe Ratio", f"{calcular_sharpe_ratio(returns[selected_asset]):.2f}")  # Función a definir
        col3.metric("Sortino Ratio", f"{calcular_sortino_ratio(returns[selected_asset]):.2f}")  # Función a definir
        
        col4, col5 = st.columns(2)
        col4.metric("VaR 95%", f"{var_95:.2%}")
        col5.metric("CVaR 95%", f"{cvar_95:.2%}")
        
        # Gráfico de precio normalizado del activo seleccionado vs benchmark
        fig_asset = go.Figure()
        fig_asset.add_trace(go.Scatter(x=normalized_prices.index, y=normalized_prices[selected_asset], name=selected_asset))
        fig_asset.add_trace(go.Scatter(x=normalized_prices.index, y=normalized_prices[benchmark], name=selected_benchmark))
        fig_asset.update_layout(title=f'Precio Normalizado: {selected_asset} vs {selected_benchmark} (Base 100)', xaxis_title='Fecha', yaxis_title='Precio Normalizado')
        st.plotly_chart(fig_asset, use_container_width=True, key="price_normalized")
        
        # Beta del activo vs benchmark
        beta_asset = calcular_beta(returns[selected_asset], returns[benchmark])  # Función a definir
        st.metric(f"Beta vs {selected_benchmark}", f"{beta_asset:.2f}")
        
        st.subheader(f"Distribución de Retornos: {selected_asset} vs {selected_benchmark}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Histograma para el activo seleccionado
            var_asset, cvar_asset = calcular_var_cvar(returns[selected_asset])
            fig_hist_asset = crear_histograma_distribucion(
                returns[selected_asset],
                var_asset,
                cvar_asset,
                f'Distribución de Retornos - {selected_asset}'  # Esta es la función de visualización
            )
            st.plotly_chart(fig_hist_asset, use_container_width=True, key="hist_asset")
            
        with col2:
            # Histograma para el benchmark
            var_bench, cvar_bench = calcular_var_cvar(returns[benchmark])
            fig_hist_bench = crear_histograma_distribucion(
                returns[benchmark],
                var_bench,
                cvar_bench,
                f'Distribución de Retornos - {selected_benchmark}'
            )
            st.plotly_chart(fig_hist_bench, use_container_width=True, key="hist_bench_1")


    with tab2:
        st.header("Análisis del Portafolio")
        
        # Calcular VaR y CVaR para el portafolio
        portfolio_var_95, portfolio_cvar_95 = calcular_var_cvar(portfolio_returns)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Rendimiento Total del Portafolio", f"{portfolio_cumulative_returns.iloc[-1]:.2%}")
        col2.metric("Sharpe Ratio del Portafolio", f"{calcular_sharpe_ratio(portfolio_returns):.2f}")
        col3.metric("Sortino Ratio del Portafolio", f"{calcular_sortino_ratio(portfolio_returns):.2f}")

        col4, col5 = st.columns(2)
        col4.metric("VaR 95% del Portafolio", f"{portfolio_var_95:.2%}")
        col5.metric("CVaR 95% del Portafolio", f"{portfolio_cvar_95:.2%}")

        # Gráfico de rendimientos acumulados del portafolio vs benchmark
        fig_cumulative = go.Figure()
        fig_cumulative.add_trace(go.Scatter(x=portfolio_cumulative_returns.index, y=portfolio_cumulative_returns, name='Portafolio'))
        fig_cumulative.add_trace(go.Scatter(x=cumulative_returns.index, y=cumulative_returns[benchmark], name=selected_benchmark))
        fig_cumulative.update_layout(title=f'Rendimientos Acumulados: Portafolio vs {selected_benchmark}', xaxis_title='Fecha', yaxis_title='Rendimiento Acumulado')
        st.plotly_chart(fig_cumulative, use_container_width=True, key="cumulative_returns")


        # Beta del portafolio vs benchmark
        beta_portfolio = calcular_beta(portfolio_returns, returns[benchmark])  # Función a definir
        st.metric(f"Beta del Portafolio vs {selected_benchmark}", f"{beta_portfolio:.2f}")

        st.subheader("Distribución de Retornos del Portafolio vs Benchmark")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Histograma para el portafolio
            var_port, cvar_port = calcular_var_cvar(portfolio_returns)
            fig_hist_port = crear_histograma_distribucion(
                portfolio_returns,
                var_port,
                cvar_port,
                'Distribución de Retornos - Portafolio'  # Función de visualización
            )
            st.plotly_chart(fig_hist_port, use_container_width=True, key="hist_port")
            
        with col2:
            # Histograma para el benchmark
            var_bench, cvar_bench = calcular_var_cvar(returns[benchmark])
            fig_hist_bench = crear_histograma_distribucion(
                returns[benchmark],
                var_bench,
                cvar_bench,
                f'Distribución de Retornos - {selected_benchmark}'
            )
            st.plotly_chart(fig_hist_bench, use_container_width=True, key="hist_bench_2")

        # Rendimientos y métricas de riesgo en diferentes ventanas de tiempo
        st.sub
