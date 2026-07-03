# Monte Carlo Retiro Fijo

App Streamlit para simular patrimonio personal en **CLP**, con todos los ceros visibles y separadores de miles en la interfaz.
El motor interno mantiene los cálculos en **MM CLP** para estabilidad numérica, pero la app muestra los resultados como pesos completos, por ejemplo `$1.000.000.000`.

## Qué hace

La simulación tiene dos fases:

1. **Acumulación:** capital inicial + ahorro mensual hasta la edad de inicio de retiro.
2. **Retiro:** desde la edad elegida el ahorro se vuelve cero y se descuenta un retiro fijo mensual.

El ahorro ya no tiene que ser constante durante toda la vida laboral. Ahora puede cambiar por tramos de edad, por ejemplo:

| Tramo | Ahorro mínimo | Ahorro más probable | Ahorro máximo |
|---|---:|---:|---:|
| 28 a 30 | $2.500.000 | $3.000.000 | $3.500.000 |
| 30 a 40 | $1.000.000 | $1.500.000 | $2.000.000 |
| 40 a 50 | $2.000.000 | $2.500.000 | $3.000.000 |

Desde la edad de inicio de retiro, el ahorro se corta automáticamente aunque exista un tramo cargado más largo.

La edad final está fija en **90 años**.

## Funcionalidades principales

- Inputs en la pantalla principal, sin barra lateral.
- Montos ingresados y mostrados en CLP, no como `MM`, con separadores de miles visibles: `1.000.000.000`.
- Edad final fija en 90 años.
- Inputs monetarios como texto para poder escribir `50.000.000`, `$50.000.000` o `50 MM`.
- Tabla para **rangos de ahorro por edad**:
  - de ahora a 30,
  - de 30 a 40,
  - de 40 a 50,
  - cualquier tramo personalizado con mínimo / más probable / máximo.
- Bloque para calcular **jubilación AFP estimada** usando saldo actual, ahorro mensual AFP, retorno real y tasa de retiro.
- Tabla para flujos mensuales recurrentes indexables:
  - arriendos,
  - otros ingresos o egresos,
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
- Gráfico de flujos rediseñado:
  - barras positivas para plata que entra,
  - barras negativas para plata que sale,
  - línea de flujo neto mensual antes del retorno,
  - diamantes para flujos esporádicos del año.
- Cortes P5/P50/P95 en la distribución final con etiquetas escalonadas para que no se tapen.
- Alertas interpretativas para retiro indexado, flujos indexados e IID mensual.

## Nueva lógica de indexación

El retiro fijo puede indexarse por inflación. Ahora los **ingresos recurrentes** también pueden indexarse, de modo que jubilación y arriendos no queden artificialmente planos mientras el retiro sube.

Para los flujos marcados como `Indexar inflación`, el monto se interpreta como **pesos de hoy**. El motor lo lleva a monto nominal de cada edad aplicando la inflación anual configurada.

Ejemplo:

- Arriendo actual: `$2.000.000`
- Inflación anual: `3%`
- Inicio del arriendo: edad 40

El gráfico mostrará el arriendo nominal estimado a los 40, 50, 65 y 90 años, no un monto plano de `$2.000.000`.

## Jubilación AFP estimada

La app incluye un bloque para calcular una jubilación AFP aproximada:

- saldo AFP actual,
- ahorro mensual AFP,
- edad de jubilación AFP,
- retorno real anual, por defecto `5%`,
- tasa de retiro anual, por defecto `3,2%`.

La fórmula trabaja en pesos reales/de hoy:

1. Capitaliza el saldo AFP y el ahorro mensual AFP con retorno real.
2. Al jubilar, calcula una pensión mensual como `saldo al jubilar × tasa de retiro anual / 12`.
3. Esa pensión se agrega como ingreso recurrente indexado por inflación desde hoy.

Esto evita el problema de comparar un retiro indexado con una jubilación/arriendo plano.


## Diferencia entre modelos de retorno

### Anual suavizado

Replica la lógica del script original: se simula un retorno anual y se reparte suavemente en 12 meses. Es más estable visualmente, pero subestima el riesgo de secuencia dentro del año.

### Mensual IID

Simula shocks mensuales independientes. Es más exigente para retiro porque permite malas secuencias de retornos justo cuando empiezas a retirar. Por eso puede mostrar mayor probabilidad de agotamiento o menor patrimonio final.

Esta versión también optimiza memoria: ya no guarda matrices gigantes de retornos y ahorros mensuales por path, por lo que el modo IID debería ser más estable en Streamlit Cloud.

## Formato de montos

Puedes ingresar montos de estas formas:

- `1.000.000.000`
- `$1.000.000.000`
- `1000000000`
- `1.000 MM`
- `3,5 MM`

La app los convierte internamente a CLP y muestra una equivalencia en millones para validar visualmente el número.

## Instalación

```bash
pip install -r requirements.txt
streamlit run app_montecarlo_retiro.py
```

## Archivos

- `app_montecarlo_retiro.py`: interfaz Streamlit, inputs, KPIs y gráficos.
- `montecarlo_engine.py`: motor Monte Carlo, incluyendo ahorro triangular por tramos de edad.
- `requirements.txt`: dependencias.
- `.streamlit/config.toml`: tema visual oscuro.

## Nota

Esto es una herramienta de simulación y no una recomendación financiera.

## Corrección rápida - 03 jul 2026

- Corregido error `KeyError: monto_inicio_nominal_mm` asociado a resultados/caché de flujos recurrentes.
- Eliminadas las vistas duplicadas con separadores bajo los editores de ahorro, flujos recurrentes y flujos esporádicos.
- Eliminadas las tablas auxiliares bajo el gráfico de flujos para dejar la pantalla más limpia. El detalle sigue disponible en el tab **Tablas** y en las descargas CSV.

## Mejora - distribución por edad

El tab de distribución ya no queda fijo solo a los 90 años. Ahora permite elegir la edad que se quiere revisar y muestra P5, P50 y P95 para esa edad específica. Esto sirve para comparar momentos clave como inicio de retiro, jubilación AFP, etapa con hijos o edad final.

## Lectura de flujos esporádicos

Los flujos esporádicos son eventos únicos. Un ingreso positivo aumenta el patrimonio en ese mes; un egreso negativo reduce el patrimonio en ese mes. Después, el patrimonio resultante sigue rentando según el modelo de retorno elegido. Los flujos recurrentes indexados, como AFP o arriendos, sí crecen mes a mes con inflación.

## Actualización anti-crash

- Corregido `Edad inicio retiro`: cuando la edad inicial sube sobre 40, el valor por defecto ahora se ajusta automáticamente a la edad inicial. Antes podía quedar `value=40` con `min_value>40`, lo que hacía caer Streamlit al cambiar parámetros.
- Se agregó un clamp extra al selector de distribución por edad para evitar índices fuera de rango si se cambia el horizonte/edad y queda estado anterior en sesión.

## Mejora estética y defaults personales

- Inputs reordenados en pestañas: Base, Ahorro por edad, AFP, Ingresos/gastos, Eventos únicos y Mercado.
- Se cargan por defecto los supuestos usados como punto de partida:
  - edad inicial 27,
  - retiro desde 42,
  - edad final 90,
  - capital inicial $35.000.000,
  - retiro mensual $5.000.000 indexado,
  - inflación 3%,
  - AFP actual $40.000.000,
  - ahorro mensual AFP $600.000,
  - jubilación AFP a los 60,
  - retorno real AFP 5%,
  - tasa retiro AFP 3,2%,
  - arriendo desde los 52 por $1.200.000 indexado,
  - evento único a los 31 por $40.000.000.
- Eliminadas las equivalencias visuales bajo cada input monetario para reducir ruido visual.
- Simplificado el hero superior y agregada una nota compacta del escenario base.

## Seguridad por clave

La app ahora tiene una pantalla de acceso antes de mostrar inputs o resultados.

Orden de lectura de la clave:

1. `APP_PASSWORD` en Streamlit Secrets.
2. Variable de entorno `APP_PASSWORD`.
3. Fallback local: `quant2026`.

Para Streamlit Cloud, agrega en **App settings → Secrets**:

```toml
APP_PASSWORD = "tu_clave_segura"
```

Recomendación: usar Secrets y cambiar la clave fallback si el repositorio queda público.

## Exportación CSV

En el tab **Tablas** se agregó una sección de exportación:

- ZIP completo con CSVs del escenario.
- Tabla por edad CSV.
- Flujos mensuales CSV.
- Distribución final por path CSV.
- Resumen CSV.
- Inputs del modelo CSV.

El ZIP incluye:

- metadata,
- inputs del modelo,
- resumen de percentiles,
- tabla por edad,
- calendario mensual de flujos,
- distribución final por path,
- tramos de ahorro cargados,
- flujos recurrentes cargados,
- flujos esporádicos cargados,
- cálculo AFP.

Existe una opción para incluir los paths completos en CLP. Puede generar un archivo pesado si se usan muchas simulaciones.

## Modo avanzado premium

Se agregó una capa visual superior con flujo de trabajo por módulos:

1. Base.
2. Ahorro.
3. AFP.
4. Flujos.
5. Eventos.
6. Mercado.

La app sigue siendo avanzada, pero queda más vendible visualmente y menos saturada al ingresar parámetros.

## Diagnóstico FIRE / Coast FIRE

Se agregó un nuevo tab **FIRE / Coast FIRE** con un cálculo aparte basado en los mismos supuestos del escenario simulado.

El usuario define una probabilidad mínima de éxito, por defecto `90%`, y la app prueba edades candidatas desde la edad inicial hasta la edad de retiro objetivo.

### FIRE anticipado

Pregunta:

> ¿Cuál es la edad más temprana desde la cual podría empezar a retirar el monto mensual deseado y mantener alta probabilidad de no agotar patrimonio hasta los 90?

La métrica principal es `prob_no_agotar_pct`.

### Coast FIRE

Pregunta:

> ¿Desde qué edad podría dejar de aportar capital nuevo y, aun así, jubilarme en la edad objetivo con buena probabilidad de éxito?

Se muestran dos lecturas:

- **Coast FIRE robusto:** primera edad donde se puede dejar de ahorrar y mantener probabilidad de no agotarse hasta los 90 mayor o igual al umbral.
- **Coast a la meta:** primera edad donde se puede dejar de ahorrar y llegar a la meta patrimonial en la edad objetivo con probabilidad mayor o igual al umbral.

El diagnóstico usa una cantidad menor de simulaciones que el escenario principal para no volver lenta la app. Por defecto usa hasta `8.000` simulaciones, editable en el tab.

El resultado puede descargarse como CSV individual y también queda incluido dentro del ZIP completo si ya fue calculado.
