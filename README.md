# Monte Carlo Retiro Fijo

App Streamlit para simular patrimonio personal en **CLP** con todos los ceros visibles en la interfaz.
El motor interno mantiene los cálculos en MM CLP para estabilidad numérica, pero la app muestra los resultados como pesos completos, por ejemplo `$1.000.000.000`.

## Qué hace

La simulación tiene dos fases:

1. **Acumulación:** capital inicial + ahorro mensual hasta la edad de inicio de retiro.
2. **Retiro:** desde la edad elegida el ahorro se vuelve cero y se descuenta un retiro fijo mensual.

La edad final está fija en **90 años**.

## Nuevas funcionalidades

- Inputs ya no están en la barra izquierda: todo queda ordenado en la pantalla principal.
- Montos ingresados y mostrados en CLP, no como `MM`.
- Edad final fija en 90 años.
- Tabla para flujos mensuales recurrentes:
  - jubilación,
  - arriendos,
  - dividendos,
  - gastos recurrentes,
  - cualquier ingreso o egreso mensual desde una edad específica.
- Tabla para flujos esporádicos:
  - bonos,
  - venta de activos,
  - herencias,
  - gastos únicos,
  - aportes extraordinarios.
- Gráficos con números anotados:
  - P50 al inicio del retiro,
  - P50 final,
  - P5 final,
  - P95 final,
  - mediana de agotamiento si existe.
- KPIs renombrados para que sean más claros.
- Explicación de las métricas dentro de la app.

## Instalación

```bash
pip install -r requirements.txt
streamlit run app_montecarlo_retiro.py
```

## Archivos

- `app_montecarlo_retiro.py`: interfaz Streamlit, inputs, KPIs y gráficos.
- `montecarlo_engine.py`: motor Monte Carlo.
- `requirements.txt`: dependencias.
- `.streamlit/config.toml`: tema visual oscuro.

## Nota

Esto es una herramienta de simulación y no una recomendación financiera.
