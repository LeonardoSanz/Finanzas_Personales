# Monte Carlo patrimonial: acumulación + retiro fijo

App Streamlit para simular patrimonio en MM CLP con dos fases:

1. **Acumulación:** ahorro mensual hasta una edad definida.
2. **Retiro:** ahorro mensual en cero y retiro mensual fijo desde la edad seleccionada.

## Archivos

- `app_montecarlo_retiro.py`: aplicación Streamlit completa.
- `requirements.txt`: dependencias mínimas.

## Instalación

Desde Anaconda Prompt o terminal:

```bash
cd ruta/de/la/carpeta/montecarlo_retiro_streamlit
pip install -r requirements.txt
streamlit run app_montecarlo_retiro.py
```

## Unidad de los montos

Todos los montos están en **MM CLP**.

Ejemplo:

- `3.0` = $3.000.000
- `1_000` = $1.000.000.000

## Métricas principales

- P50 al inicio del retiro.
- P50 final.
- Probabilidad de no agotar patrimonio.
- Probabilidad de terminar con más patrimonio que al inicio del retiro.
- Probabilidad de alcanzar el objetivo patrimonial.
- Edad mediana de agotamiento, si ocurre.

## Notas de modelo

- El modo **anual suavizado** replica la lógica original: se simula un retorno anual y se reparte en 12 retornos mensuales iguales.
- El modo **mensual IID** simula retornos mensuales directamente y suele capturar mejor el riesgo de secuencia en la fase de retiro.
