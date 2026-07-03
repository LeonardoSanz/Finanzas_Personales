# Monte Carlo Retiro Fijo

App Streamlit para simular patrimonio en MM CLP con dos fases:

1. **Acumulación:** ahorro mensual positivo hasta la edad de inicio de retiro.
2. **Retiro:** ahorro mensual en cero y retiro fijo mensual desde la edad seleccionada.

## Estilo visual

La app incluye un estilo tipo dashboard Quant/CMF:

- Modo oscuro.
- Fondo azul profundo.
- Acentos morados corporativos.
- Detalles cyan.
- Cards para KPIs.
- Gráficos Plotly con plantilla oscura.
- Archivo `.streamlit/config.toml` incluido.

## Instalación

```bash
pip install -r requirements.txt
streamlit run app_montecarlo_retiro.py
```

## Archivos

- `app_montecarlo_retiro.py`: interfaz Streamlit y visualizaciones.
- `montecarlo_engine.py`: motor Monte Carlo.
- `requirements.txt`: dependencias.
- `.streamlit/config.toml`: tema visual de Streamlit.

## Unidad

Todos los montos están en **MM CLP**. Por ejemplo, `3.0` equivale a $3.000.000.
