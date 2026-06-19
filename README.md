# DS_TPO

Pipeline de analisis para evaluar cambios de comportamiento alrededor de parciales usando datos personales de consumo digital y actividad fisica.

Documentacion completa del proyecto: `DOCUMENTACION.md`.

## Objetivo

Este proyecto cruza tres fuentes principales para responder la hipotesis de si los estudiantes cambian su rutina cuando tienen examenes:

1. Netflix (visualizaciones por dia)
2. Spotify (minutos escuchados por dia)
3. Fit/Salud (pasos y sueno por dia)

El enfoque trabaja con ventanas temporales relativas al examen y compara pre vs post parcial.

## Fuentes soportadas

El script detecta automaticamente fuentes dentro del workspace:

1. Netflix: archivos `NetflixViewingHistory nombre.csv`
2. Spotify: `Spotify Account Data nombre/StreamingHistory*.json`
3. Samsung Health: carpetas `Samsung Health nombre/...csv`
4. Apple Health: carpetas `apple_health_export nombre/export.xml` o `exportar.xml`
5. Google Fit Takeout: carpetas `Takeout nombre/Fit/Daily activity metrics/*.csv`

## Como correr

Desde la carpeta raiz del proyecto:

```powershell
python analisis_comportamiento_parciales.py
```

El script no requiere argumentos y escribe todos los resultados en `outputs/`.

## Logica de analisis

1. Lee el calendario de examenes desde `cronograma_cuatrimestre_2026.csv`.
2. Genera una ventana de trabajo por examen de `-14` a `+14` dias.
3. Filtra y agrega datos diarios por estudiante y fuente.
4. Calcula estadisticas pre/post:
   1. `pre_mean`, `post_mean`
   2. `delta_abs`, `delta_pct`
   3. `effect_size`
   4. `support_score`
5. Genera visualizaciones y rankings para priorizar hallazgos con mayor soporte.

## Procedimientos de ML y analisis

El proyecto combina analisis estadistico, feature engineering y un paso de clustering no supervisado. No es un pipeline de prediccion de nota, sino de deteccion de cambios de comportamiento alrededor de parciales.

### 1. Ingesta y normalizacion de datos

El script detecta automaticamente las carpetas y archivos disponibles y transforma cada fuente a un formato comun por `estudiante-dia`.

1. Netflix: cada visualizacion se convierte en un evento diario.
2. Spotify: cada sesion se convierte en minutos escuchados por dia.
3. Samsung Health: se agregan pasos, distancia y calorias por dia.
4. Apple Health: se parsean registros XML grandes en streaming para obtener pasos y sueno.
5. Google Fit Takeout: se suman los bloques diarios de actividad para reconstruir el total diario.

Esta etapa sirve como limpieza y unificacion de esquema, para poder comparar fuentes distintas con la misma grilla temporal.

### 2. Seleccion temporal por ventana relativa

Cada examen se usa como un evento central. A partir de eso se construye una ventana de analisis de 14 dias antes y 14 dias despues.

1. Se calcula la fecha relativa de cada observacion respecto del parcial.
2. Se descartan registros fuera de la ventana para reducir ruido y volumen.
3. Se conservan solo las fechas relevantes para mejorar rendimiento en carpetas grandes.

Esta estrategia es clave porque la hipotesis no busca tendencias generales, sino cambios alrededor de la situacion de examen.

### 3. Feature engineering

Cada fuente se transforma en variables que resumen el comportamiento diario:

1. Netflix: cantidad de visualizaciones por dia.
2. Spotify: minutos escuchados por dia.
3. Steps: pasos diarios.
4. Sleep: minutos de sueno por dia.

Sobre esas variables se derivan medidas adicionales:

1. `pre_mean`: promedio en los 7 dias previos al parcial.
2. `post_mean`: promedio en los 7 dias posteriores.
3. `baseline_mean`: promedio en la semana anterior a la ventana pre.
4. `delta_abs`: cambio absoluto entre post y pre.
5. `delta_pct`: cambio porcentual entre post y pre.
6. `pre_std` y `post_std`: dispersion de cada tramo.
7. `effect_size`: cambio estandarizado aproximado.
8. `support_score`: score heuristico que combina magnitud del cambio y cantidad de observaciones.

### 4. Agregacion estadistica pre/post

Esta es la parte central de la respuesta a la hipotesis.

1. Se agrupan los datos por estudiante y por fuente.
2. Se calculan medias separadas para tramos pre y post parcial.
3. Se evita depender de un unico dia puntual.
4. Se exige soporte minimo de datos para que un cambio entre al ranking.

Con esto se obtienen comparaciones mas robustas que una simple lectura visual de series sueltas.

### 5. Ranking de evidencia

Se generan tres rankings para priorizar hallazgos:

1. `ranking_delta_abs.csv`: ordena por cambio absoluto mas grande.
2. `ranking_delta_pct.csv`: ordena por cambio porcentual mas grande.
3. `ranking_support_score.csv`: prioriza cambios con mas magnitud y mas soporte de datos.

Este paso funciona como una seleccion automatica de evidencia, para identificar rapidamente que estudiantes y metricas muestran el mayor cambio cerca de parciales.

### 6. Clustering no supervisado

Se aplica clustering sobre las series por estudiante para agrupar patrones similares.

1. Cada estudiante se representa como un vector con valores diarios alrededor del parcial.
2. Se estandarizan las series para que la escala no domine el agrupamiento.
3. Se evalua K-Means para distintos valores de `k` con metodo del codo e indice de silueta.
4. Se prueba clustering jerarquico aglomerativo con enlace promedio sobre los mismos vectores.
5. Se usa el `k` con mejor silueta disponible y se deja la inercia para inspeccionar el codo manualmente.

Interpretacion posible de los clusters:

1. estudiantes con subida fuerte de uso digital antes del parcial.
2. estudiantes con caida de actividad fisica o cambio mixto.
3. perfiles estables o con cambios leves.

Esto no predice nada, pero ayuda a descubrir perfiles de comportamiento distintos dentro del grupo.

### 7. Visualizacion y validacion

Los resultados se inspeccionan con graficos complementarios:

1. Lineas por dia relativo para ver la tendencia promedio.
2. Heatmaps por estudiante para detectar heterogeneidad individual.
3. CSVs de ranking para auditar manualmente los casos mas fuertes.

La validacion no se apoya solo en una metrica; combina magnitud del cambio, dispersion y soporte de observaciones.

## Archivos de salida

### Resumen principal

1. `outputs/before_after_summary.csv`

Columnas clave:

1. `student`, `source`
2. `pre_mean`, `post_mean`
3. `pre_n`, `post_n` (cantidad de observaciones)
4. `delta_abs`, `delta_pct`
5. `effect_size`
6. `support_score`

### Rankings (opcion 3)

1. `outputs/ranking_delta_abs.csv`
2. `outputs/ranking_delta_pct.csv`
3. `outputs/ranking_support_score.csv`

Estos rankings filtran por cantidad minima de datos pre/post para evitar conclusiones fragiles.

### Visualizaciones

1. `outputs/netflix_relative_day.svg`
2. `outputs/spotify_relative_day.svg`
3. `outputs/steps_relative_day.svg`
4. `outputs/sleep_relative_day.svg` (si hay datos)
5. `outputs/*_heatmap.svg`
6. `outputs/top_changes_support.svg`
7. `outputs/delta_pct_matrix.svg`
8. `outputs/support_score_matrix.svg`
9. `outputs/interactive_dashboard.html`

### Clustering

1. `outputs/clusters_views.csv`
2. `outputs/clusters_minutes.csv`
3. `outputs/clusters_steps.csv`
4. `outputs/clusters_views_comparison.csv`
5. `outputs/clusters_minutes_comparison.csv`
6. `outputs/clusters_steps_comparison.csv`
7. `outputs/kmeans_evaluation_views.csv`
8. `outputs/kmeans_evaluation_minutes.csv`
9. `outputs/kmeans_evaluation_steps.csv`
10. `outputs/agglomerative_evaluation_views.csv`
11. `outputs/agglomerative_evaluation_minutes.csv`
12. `outputs/agglomerative_evaluation_steps.csv`
13. `outputs/clusters_views_agglomerative.csv`
14. `outputs/clusters_minutes_agglomerative.csv`
15. `outputs/clusters_steps_agglomerative.csv`

Si Windows bloquea la creacion de archivos nuevos dentro de `outputs/`, el script guarda los `kmeans_evaluation_*.csv`, `agglomerative_evaluation_*.csv` y `clusters_*_agglomerative.csv` en la raiz del proyecto y avisa por consola.

### Reevaluacion del clustering

HDBSCAN se saco del flujo porque, con esta muestra chica y vectores diarios estandarizados, devolvia todos los puntos como ruido (`-1`). Eso no aportaba una segmentacion interpretable y podia confundir la lectura del dashboard.

La validacion queda concentrada en K-Means:

1. `kmeans_evaluation_*.csv` muestra la inercia para revisar el codo.
2. La columna `silhouette` mide separacion y cohesion de los clusters para cada `k` valido.
3. La columna `selected` marca el `k` usado finalmente en los archivos `clusters_*`.
4. Si la silueta es baja o cambia poco entre valores de `k`, los clusters deben leerse como exploratorios y no como perfiles fuertes.

Tambien se prueba clustering jerarquico aglomerativo:

1. `agglomerative_evaluation_*.csv` usa los mismos vectores normalizados que K-Means.
2. El enlace usado es `average`, que compara clusters por distancia promedio entre sus puntos.
3. `clusters_*_comparison.csv` incluye `kmeans_cluster` y `agglomerative_cluster` para comparar metodo contra metodo.

## Interpretacion sugerida

1. Revisar primero `ranking_support_score.csv` para priorizar cambios con mayor intensidad y mayor volumen de datos.
2. Contrastar cada hallazgo con `*_relative_day.svg` y `*_heatmap.svg` para validar que no sea un outlier.
3. Usar `delta_pct` para comparar entre metricas de distinta escala (por ejemplo minutos vs pasos).
4. Revisar `effect_size` cuando quieras comparar cambios entre estudiantes con distinta cantidad de actividad.
5. Usar `delta_pct_matrix.svg` para ver de un vistazo qué estudiante se mueve más en cada fuente.
6. Usar `top_changes_support.svg` para presentar rápidamente los casos más fuertes en la defensa.
7. Abrir `interactive_dashboard.html` si querés filtrar por estudiante/fuente y rotar la vista 3D en vivo.
8. Revisar `kmeans_evaluation_*.csv` y `agglomerative_evaluation_*.csv` antes de interpretar clusters; la inercia ayuda a ver el codo y la silueta marca la separacion de los grupos.

## Limitaciones actuales

1. La calidad del resultado depende de la cobertura temporal de cada integrante cerca de los examenes.
2. Distintas plataformas de fit pueden medir de forma diferente.
3. Si faltan datos de sueno en la ventana de examenes, la salida de sueno puede quedar vacia.
4. El clustering es exploratorio y no debe leerse como clasificacion definitiva.
5. Con tan pocos estudiantes, la seleccion de `k` por silueta puede ser sensible a casos individuales.
6. El metodo del codo queda como apoyo visual/manual; no siempre hay un codo claro en muestras chicas.

## Conclusiones

A partir de los rankings y los graficos generados, la hipotesis queda apoyada de forma parcial: si aparecen cambios de comportamiento alrededor de los parciales, pero no son iguales en todas las fuentes ni en todos los estudiantes.

1. La senal mas clara aparece en actividad fisica, sobre todo en pasos.
2. En steps se observan cambios relevantes en varios integrantes, con casos de aumento y de disminucion segun la persona.
3. En Spotify la variacion existe, pero es mas heterogenea y de menor magnitud que en steps.
4. En Netflix el cambio es mas debil y menos consistente, por lo que sirve mas como complemento que como evidencia principal.
5. Los rankings de soporte muestran que no alcanza con un cambio grande: tambien importa cuantas observaciones respaldan ese cambio.
6. Apple Health, Samsung Health y Google Fit permiten ampliar la cobertura de la muestra de actividad fisica y hacer la respuesta mas solida.

Lectura practica para la defensa:

1. Si queres sostener la hipotesis, el argumento mas fuerte es que la rutina fisica cambia cerca de examenes.
2. Si queres matizarla, podes decir que el consumo digital tambien cambia, pero con patrones menos uniformes.
3. La mejor evidencia no es un unico grafico, sino la combinacion de ranking de soporte, series relativas al parcial y comparación pre/post.
