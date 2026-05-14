import os
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# CONFIGURACIÓN
# ============================================================

archivo_csv = r"C:\Users\Raiden AsusROG\OneDrive\Documentos\Archivo Académico\Politécnico Grancolombiano - Ing. de Sistemas\8 Semestre\Semillero HPC\HPC\results\benchmark_parquet_duckdb_20260513_164114.csv"

carpeta_salida = r"C:\Users\Raiden AsusROG\OneDrive\Documentos\Archivo Académico\Politécnico Grancolombiano - Ing. de Sistemas\8 Semestre\Semillero HPC\HPC\results\graficos_parquet"

os.makedirs(carpeta_salida, exist_ok=True)

# ============================================================
# CARGA DE DATOS
# ============================================================

df = pd.read_csv(archivo_csv)

# Evitar duplicado si "maximo equipo" usa los mismos threads que "16 hilos"
df = df[df["escenario"] != "maximo equipo"]

# Convertir columnas numéricas
df["threads"] = pd.to_numeric(df["threads"], errors="coerce")
df["tiempo_promedio_s"] = pd.to_numeric(df["tiempo_promedio_s"], errors="coerce")
df["speedup"] = pd.to_numeric(df["speedup"], errors="coerce")
df["eficiencia_paralela"] = pd.to_numeric(df["eficiencia_paralela"], errors="coerce")

# Dejar solo resultados correctos
df = df[df["estado"] == "OK"]

# ============================================================
# FUNCIÓN PARA GRAFICAR
# ============================================================

def graficar_metrica(nombre_columna, titulo, eje_y, nombre_archivo):
    plt.figure(figsize=(10, 6))

    for consulta in df["consulta"].unique():
        datos = df[df["consulta"] == consulta].sort_values("threads")

        plt.plot(
            datos["threads"],
            datos[nombre_columna],
            marker="o",
            label=consulta
        )

    plt.title(titulo)
    plt.xlabel("Número de hilos")
    plt.ylabel(eje_y)
    plt.xticks(sorted(df["threads"].dropna().unique()))
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    ruta_salida = os.path.join(carpeta_salida, nombre_archivo)
    plt.savefig(ruta_salida, dpi=300)
    plt.close()

    print(f"Gráfico generado: {ruta_salida}")

# ============================================================
# GENERAR GRÁFICOS
# ============================================================

graficar_metrica(
    nombre_columna="tiempo_promedio_s",
    titulo="Tiempo promedio de ejecución por número de hilos - Parquet",
    eje_y="Tiempo promedio (segundos)",
    nombre_archivo="grafico_tiempo_promedio_parquet.png"
)

graficar_metrica(
    nombre_columna="speedup",
    titulo="Speedup por número de hilos - Parquet",
    eje_y="Speedup",
    nombre_archivo="grafico_speedup_parquet.png"
)

graficar_metrica(
    nombre_columna="eficiencia_paralela",
    titulo="Eficiencia paralela por número de hilos - Parquet",
    eje_y="Eficiencia paralela",
    nombre_archivo="grafico_eficiencia_paralela_parquet.png"
)

print("\n=== GRÁFICOS PARQUET GENERADOS CORRECTAMENTE ===")
print(f"Carpeta de salida: {carpeta_salida}")