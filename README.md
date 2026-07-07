# Monte Carlo Retiro Fijo

App Streamlit para simular patrimonio personal en **CLP**, con todos los ceros visibles y separadores de miles en la interfaz.
El motor interno mantiene los cálculos en **MM CLP** para estabilidad numérica, pero la app muestra los resultados como pesos completos, por ejemplo `$1.000.000.000`.

## Qué hace

La simulación tiene dos fases:

1. **Acumulación:** capital inicial + ahorro mensual hasta la edad de inicio de retiro.
2. **Retiro:** desde la edad elegida el ahorro se vuelve cero y se descuenta un retiro fijo mensual.

El ahorro patrimonial se ingresa por **tramos de edad**. En cada tramo escribes el ahorro mensual esperado en pesos de hoy; la app aplica una banda fija de **±$500.000** alrededor de ese monto usando distribución triangular:

| Input | Mínimo simulado | Más probable | Máximo simulado |
|---|---:|---:|---:|
| $3.000.000 | $2.500.000 | $3.000.000 | $3.500.000 |

Desde la edad de inicio de retiro, el ahorro se corta automáticamente. Si activas indexación, cada tramo de ahorro se interpreta como pesos de hoy y sube con inflación hasta el retiro.

La edad final está fija en **90 años**.

## Funcionalidades principales

- Inputs en la pantalla principal, sin barra lateral.
- Montos ingresados y mostrados en CLP, no como `MM`, con separadores de miles visibles: `1.000.000.000`.
- Edad final fija en 90 años.
- Inputs monetarios como texto para poder escribir `50.000.000`, `$50.000.000` o `50 MM`.
- Tabla de **ahorro mensual por edad**, con un monto esperado por tramo, banda automática fija de ±$500.000 e indexación opcional por inflación.
- Bloque para calcular **jubilación AFP estimada** usando saldo actual, ahorro mensual AFP, fondo AFP, retorno real promedio y desviación estándar según supuestos de la Superintendencia de Pensiones.
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

La app incluye un bloque para calcular una jubilación AFP aproximada por simulación:

- saldo AFP actual,
- ahorro mensual AFP,
- edad de jubilación AFP,
- fondo AFP o supuesto SP,
- escenario usado como pensión: P5, P25, P50 o P75,
- tasa de retiro anual, por defecto `3,2%`.

Los retornos reales anualizados usados para AFP vienen de la tabla indicada por el usuario como fuente Superintendencia de Pensiones:

| Fondo | Promedio real anual | Desv. est. |
|---|---:|---:|
| Fondo A | 4,49% | 10,99% |
| Fondo B | 4,02% | 8,53% |
| Fondo C | 3,38% | 6,19% |
| Fondo D | 2,81% | 4,52% |
| Fondo E | 2,17% | 4,12% |
| Renta Vitalicia | 3,11% | 0,65% |

La lógica trabaja en pesos reales/de hoy:

1. Simula el saldo AFP al jubilar usando promedio y desviación estándar del fondo elegido.
2. Selecciona un escenario de saldo, por defecto P50 mediano.
3. Calcula una pensión mensual como `saldo elegido al jubilar × tasa de retiro anual / 12`.
4. Esa pensión se agrega como ingreso recurrente indexado por inflación.

Esto evita comparar un retiro indexado con una jubilación/arriendo plano y además reconoce la incertidumbre del fondo AFP.


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
  - Fondo AFP C por defecto, usando promedio 3,38% y desviación estándar 6,19%,
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

## Matriz de capital requerido

La matriz vive dentro del tab **FIRE / Coast / Matriz**.

La matriz responde:

> Si comienzo a retirar a cierta edad, ¿cuánto patrimonio debo tener justo en esa fecha para llegar a los 90 años con una probabilidad de éxito objetivo?

Por defecto evalúa estas edades:

- 35
- 37
- 40
- 43
- 45
- 48
- 50
- 55
- 60
- 65

Y estas probabilidades de éxito:

- 70%
- 80%
- 90%
- 95%

La matriz usa los mismos supuestos del escenario activo:

- retiro mensual,
- indexación por inflación,
- arriendos,
- AFP,
- flujos recurrentes,
- eventos únicos,
- modelo de retorno,
- retorno esperado,
- volatilidad,
- truncamiento.

Importante: este cálculo no incluye ahorro antes de la edad analizada. La celda calcula el capital que ya deberías tener acumulado en esa edad para financiar el retiro desde ahí hasta los 90.

La matriz puede descargarse como CSV y también queda incluida en el ZIP completo del escenario si ya fue calculada antes de exportar.

## Corrección importante de indexación

El retiro mensual fijo se interpreta como **dinero de hoy**. Si ingresas `$5.000.000` e indexas a 3%, el primer retiro nominal en la edad de retiro ya se calcula ajustando inflación desde la edad inicial hasta esa edad. Luego continúa indexándose mes a mes hasta los 90.

## Lectura FIRE / Coast FIRE

La app ya no usa una tasa fija tipo 4% o 3,2% para calcular el FIRE patrimonial. El input central es el **retiro mensual deseado en pesos de hoy**. La app muestra:

- **Retiro real deseado:** monto mensual ingresado en dinero de hoy.
- **Primer retiro nominal:** monto mensual ajustado por inflación a la edad de retiro seleccionada.
- **Éxito simulado:** probabilidad de llegar a los 90 sin agotar patrimonio bajo el escenario activo.
- **Capital mediano al FIRE elegido:** patrimonio P50 justo al inicio del retiro elegido.

La tasa de retiro se mantiene solamente para la AFP, porque sirve para transformar el saldo AFP estimado en una pensión mensual aproximada.

## Matriz nominal de capital requerido

El tab **FIRE / Coast / Matriz** calcula cuánto patrimonio nominal necesitas tener exactamente a cada edad para llegar a los 90 con una probabilidad objetivo. Usa los mismos supuestos del escenario: retiro en pesos de hoy indexado, AFP, arriendos, eventos únicos, inflación y modelo de retornos.

La matriz principal queda en **capital nominal CLP a la edad de retiro**. Esto significa que el número puede subir con la edad por inflación acumulada, aunque en poder adquisitivo de hoy el capital económico requerido pueda ser menor.

## Actualización FIRE realista

La sección **FIRE / Coast / Matriz** ahora responde la pregunta operacional:

> ¿Cuál es la edad mínima a la que puedo retirarme con el monto mensual deseado y llegar a los 90 con X% de éxito?

El cálculo se hace por simulación para cada edad evaluada. Para cada edad, la app acumula patrimonio hasta esa edad usando los tramos de ahorro, retornos, flujos recurrentes, AFP y eventos únicos. Luego corta el ahorro, aplica el retiro mensual deseado —en pesos de hoy e indexado si corresponde— y mide si el patrimonio sobrevive hasta los 90.

La matriz ahora es **realista**: muestra el capital nominal requerido solo cuando el escenario completo logra el porcentaje de éxito de la fila. Si con los supuestos actuales no se llega a esa condición, la celda muestra **No alcanza**.

También se agregó **Coast FIRE / Cost FIRE** por simulación, pero responde una pregunta distinta al FIRE anticipado: usa la **edad de retiro elegida en el escenario base** y busca la primera edad en que podrías dejar de ahorrar, mantener el patrimonio invertido, y aun así jubilarte en esa edad elegida con el porcentaje de éxito objetivo.

Se mantienen solo cuatro cards principales del escenario:

- Patrimonio mediano al iniciar retiro.
- Patrimonio mediano a los 90.
- Probabilidad de no agotar patrimonio.
- Si falla, edad mediana de agotamiento.

## Ajustes recientes

- Se agregó **indexación del ahorro por inflación**. Los montos de ahorro en los tramos se interpretan como pesos de hoy y, si la casilla está activa, crecen con inflación hasta la edad de retiro. Esto permite modelar sueldos/ahorros reajustados por IPC.
- Se rediseñó el heatmap de **Matriz FIRE realista** para evitar etiquetas superpuestas. Las celdas viables muestran el capital nominal requerido en formato compacto; las celdas no viables se marcan con `—`.


## Último ajuste: ahorro por edad simplificado

- Se recuperó la tabla de ahorro por edad.
- Cada fila permite definir edad inicio, edad fin y ahorro mensual esperado.
- El mínimo y máximo ya no se ingresan manualmente: se calculan como ahorro esperado ±$500.000.
- Si la indexación está activa, los montos de ahorro de cada tramo se interpretan como pesos de hoy y crecen con inflación hasta el retiro.

## Reporte ejecutivo Excel para cliente

Se agregó un exportador **Excel ejecutivo** pensado para explicar la simulación a una persona no financiera. El archivo incluye hojas en este orden:

1. **Inputs**: supuestos principales, ahorro por edad, AFP, ingresos/gastos recurrentes y eventos únicos.
2. **Flujos**: tabla anual por edad con ahorro, retiros, flujos recurrentes, eventos únicos y flujo neto antes de retorno.
3. **FIRE / Coast**: edad FIRE anticipada, patrimonio para FIRE, edad Coast FIRE y detalle por edad.
4. **Matriz FIRE**: matriz realista de capital nominal requerido, con colores y celdas “No alcanza”.
5. **Percentiles**: evolución anual del patrimonio simulado.

También se eliminó de la interfaz la vista/caption auxiliar de banda triangular para no ensuciar la pantalla. La lógica se mantiene: cada tramo usa ahorro esperado ±$500.000 por detrás.

## Ajuste AFP UI

La selección de fondo AFP se dejó como botones horizontales en vez de dropdown/selectbox para evitar que el menú desplegable desordene visualmente la sección dentro de tabs/form en Streamlit Cloud. La lógica de cálculo no cambia: cada escenario AFP se simula con el promedio y desviación estándar real anual del fondo elegido.
