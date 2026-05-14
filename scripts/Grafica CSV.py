import os
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURACIÓN
# ============================================================

carpeta_resultados = r"C:\Users\Raiden AsusROG\OneDrive\Documentos\Archivo Académico\Politécnico Grancolombiano - Ing. de Sistemas\8 Semestre\Semillero HPC\HPC\results"

# ============================================================
# ARCHIVOS FIJOS (SIN FECHA)
# ============================================================

archivo_parquet = os.path.join(
    carpeta_resultados,
    "benchmark_parquet_duckdb.csv"
)

archivo_csv = os.path.join(
    carpeta_resultados,
    "benchmark_csv_python.csv"
)

archivo_errores_csv = os.path.join(
    carpeta_resultados,
    "errores_csv_python.csv"
)

# ============================================================
# CARPETA DE GRÁFICAS
# ============================================================

carpeta_graficas = os.path.join(
    carpeta_resultados,
    "graficas"
)

os.makedirs(carpeta_graficas, exist_ok=True)


# ============================================================
# VALIDAR EXISTENCIA DE ARCHIVOS
# ============================================================

if not os.path.exists(archivo_parquet):
    raise FileNotFoundError(
        f"No existe el archivo Parquet:\n{archivo_parquet}"
    )

if not os.path.exists(archivo_csv):
    raise FileNotFoundError(
        f"No existe el archivo CSV:\n{archivo_csv}"
    )


# ============================================================
# CARGA DE DATOS
# ============================================================

print("Cargando resultados...")

df_parquet = pd.read_csv(archivo_parquet)
df_csv = pd.read_csv(archivo_csv)

df_csv["tiempo_promedio_s"] = pd.to_numeric(
    df_csv["tiempo_promedio_s"],
    errors="coerce"
)
df = pd.concat([df_parquet, df_csv], ignore_index=True)

# ============================================================
# LIMPIEZA DE TIPOS
# ============================================================

columnas_numericas = [
    "tiempo_promedio_s",
    "speedup",
    "eficiencia_paralela",
    "threads"
]

for columna in columnas_numericas:
    if columna in df.columns:
        df[columna] = pd.to_numeric(
            df[columna],
            errors="coerce"
        )

# Solo resultados válidos
df = df[df["estado"] == "OK"].copy()

print("Datos cargados correctamente.")


# ============================================================
# FUNCIÓN AUXILIAR
# ============================================================

def guardar_grafica(nombre_archivo):
    ruta = os.path.join(
        carpeta_graficas,
        nombre_archivo
    )

    plt.tight_layout()
    plt.savefig(ruta, dpi=300)
    plt.close()

    print(f"Gráfica guardada: {ruta}")


# ============================================================
# FUNCIÓN GENERAL DE GRÁFICAS
# ============================================================

def graficar_lineas_por_consulta(
    columna_y,
    titulo,
    etiqueta_y,
    nombre_archivo
):
    consultas = df["consulta"].unique()

    for consulta in consultas:

        datos = df[df["consulta"] == consulta].copy()

        plt.figure(figsize=(10, 6))

        for formato in datos["formato"].unique():

            datos_formato = (
                datos[datos["formato"] == formato]
                .sort_values("threads")
            )

            plt.plot(
                datos_formato["threads"],
                datos_formato[columna_y],
                marker="o",
                linewidth=2,
                label=formato
            )

        plt.title(f"{titulo}\n{consulta}")
        plt.xlabel("Número de hilos / procesos")
        plt.ylabel(etiqueta_y)

        plt.xticks(
            sorted(
                datos["threads"]
                .dropna()
                .unique()
            )
        )

        plt.grid(
            True,
            linestyle="--",
            alpha=0.5
        )

        plt.legend()

        guardar_grafica(
            f"{nombre_archivo}_{consulta}.png"
        )


# ============================================================
# 1. TIEMPO PROMEDIO
# ============================================================

print("\nGenerando gráficas de tiempo promedio...")

graficar_lineas_por_consulta(
    columna_y="tiempo_promedio_s",
    titulo="Tiempo promedio de ejecución",
    etiqueta_y="Tiempo promedio (segundos)",
    nombre_archivo="tiempo_promedio"
)


# ============================================================
# 2. SPEEDUP
# ============================================================

print("\nGenerando gráficas de speedup...")

graficar_lineas_por_consulta(
    columna_y="speedup",
    titulo="Speedup",
    etiqueta_y="Speedup",
    nombre_archivo="speedup"
)


# ============================================================
# 3. EFICIENCIA PARALELA
# ============================================================

print("\nGenerando gráficas de eficiencia paralela...")

graficar_lineas_por_consulta(
    columna_y="eficiencia_paralela",
    titulo="Eficiencia paralela",
    etiqueta_y="Eficiencia paralela",
    nombre_archivo="eficiencia_paralela"
)


# ============================================================
# 4. COMPARACIÓN DE TIEMPOS A 16 HILOS
# ============================================================

print("\nGenerando comparación general de tiempos...")

df_16 = df[df["threads"] == 16].copy()

pivot_16 = df_16.pivot_table(
    index="consulta",
    columns="formato",
    values="tiempo_promedio_s",
    aggfunc="mean"
)

plt.figure(figsize=(12, 6))

pivot_16.plot(
    kind="bar",
    figsize=(12, 6)
)

plt.title(
    "Comparación de tiempo promedio a 16 hilos"
)

plt.xlabel("Consulta")
plt.ylabel("Tiempo promedio (segundos)")

plt.xticks(
    rotation=45,
    ha="right"
)

plt.grid(
    axis="y",
    linestyle="--",
    alpha=0.5
)

guardar_grafica(
    "comparacion_tiempo_16_hilos.png"
)


# ============================================================
# 5. COMPARACIÓN DE SPEEDUP
# ============================================================

print("\nGenerando comparación de speedup...")

pivot_speedup = df_16.pivot_table(
    index="consulta",
    columns="formato",
    values="speedup",
    aggfunc="mean"
)

plt.figure(figsize=(12, 6))

pivot_speedup.plot(
    kind="bar",
    figsize=(12, 6)
)

plt.title(
    "Comparación de speedup a 16 hilos"
)

plt.xlabel("Consulta")
plt.ylabel("Speedup")

plt.xticks(
    rotation=45,
    ha="right"
)

plt.grid(
    axis="y",
    linestyle="--",
    alpha=0.5
)

guardar_grafica(
    "comparacion_speedup_16_hilos.png"
)


# ============================================================
# 6. REGISTROS VÁLIDOS VS INVÁLIDOS
# ============================================================

print("\nGenerando gráfica de registros válidos e inválidos...")

if (
    "filas_validas" in df_csv.columns
    and "filas_invalidas" in df_csv.columns
):

    filas_validas = pd.to_numeric(
        df_csv["filas_validas"],
        errors="coerce"
    ).max()

    filas_invalidas = pd.to_numeric(
        df_csv["filas_invalidas"],
        errors="coerce"
    ).max()

    plt.figure(figsize=(7, 6))

    plt.bar(
        ["Filas válidas", "Filas inválidas"],
        [filas_validas, filas_invalidas]
    )

    plt.title(
        "Registros válidos e inválidos detectados"
    )

    plt.ylabel("Cantidad de registros")

    plt.grid(
        axis="y",
        linestyle="--",
        alpha=0.5
    )

    guardar_grafica(
        "registros_validos_invalidos_csv.png"
    )


# ============================================================
# 7. TIPOS DE ERRORES
# ============================================================

print("\nGenerando gráfica de tipos de errores...")

if os.path.exists(archivo_errores_csv):

    df_errores = pd.read_csv(
        archivo_errores_csv
    )

    if (
        not df_errores.empty
        and "tipo_error" in df_errores.columns
    ):

        conteo_errores = (
            df_errores["tipo_error"]
            .value_counts()
            .sort_values(ascending=True)
        )

        plt.figure(figsize=(10, 7))

        conteo_errores.plot(
            kind="barh"
        )

        plt.title(
            "Tipos de errores detectados en CSV"
        )

        plt.xlabel("Cantidad de errores")
        plt.ylabel("Tipo de error")

        plt.grid(
            axis="x",
            linestyle="--",
            alpha=0.5
        )

        guardar_grafica(
            "tipos_errores_csv.png"
        )


# ============================================================
# 8. OVERHEAD DE VALIDACIÓN
# ============================================================

print("\nGenerando gráfica de overhead de validación...")

if "tiempo_validacion_s" in df_csv.columns:

    tiempo_validacion = pd.to_numeric(
        df_csv["tiempo_validacion_s"],
        errors="coerce"
    ).max()

    tiempos_consultas_csv = (
        df_csv[df_csv["estado"] == "OK"]
        .groupby("consulta")["tiempo_promedio_s"]
        .mean()
    )

    plt.figure(figsize=(12, 6))

    tiempos_consultas_csv.plot(
        kind="bar"
    )

    plt.axhline(
        y=tiempo_validacion,
        linestyle="--",
        label=f"Tiempo validación: {tiempo_validacion:.4f} s"
    )

    plt.title(
        "Costo de validación frente al tiempo de consultas CSV"
    )

    plt.xlabel("Consulta")
    plt.ylabel("Tiempo promedio (segundos)")

    plt.xticks(
        rotation=45,
        ha="right"
    )

    plt.grid(
        axis="y",
        linestyle="--",
        alpha=0.5
    )

    plt.legend()

    guardar_grafica(
        "overhead_validacion_csv.png"
    )


print("\n=== TODAS LAS GRÁFICAS FUERON GENERADAS ===")
print(f"Carpeta de salida:\n{carpeta_graficas}")