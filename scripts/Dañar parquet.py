import os
import random
import time
import duckdb
import pandas as pd
import csv

# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

carpeta_entrada = r"C:\Users\Raiden AsusROG\OneDrive\Documentos\Archivo Académico\Politécnico Grancolombiano - Ing. de Sistemas\8 Semestre\Semillero HPC\HPC\data"

carpeta_salida = r"C:\Users\Raiden AsusROG\OneDrive\Documentos\Archivo Académico\Politécnico Grancolombiano - Ing. de Sistemas\8 Semestre\Semillero HPC\HPC\data_csv_sucios"

archivo_resumen = os.path.join(carpeta_salida, "resumen_conversion.csv")

porcentaje_error = 0.10  # 10% de filas con errores

random.seed(42)

hilos_por_defecto = 12

# Puedes cambiar los hilos por archivo aquí.
# Si el archivo no aparece en esta lista, usará hilos_por_defecto.
configuracion_archivos = {
    "yellow_tripdata_2024-04.parquet": 12,
    "yellow_tripdata_2024-05.parquet": 12,
    "yellow_tripdata_2024-06.parquet": 12,
    "yellow_tripdata_2024-07.parquet": 12,
    "yellow_tripdata_2024-08.parquet": 12,
    "yellow_tripdata_2024-09.parquet": 12,
    "yellow_tripdata_2024-10.parquet": 12,
    "yellow_tripdata_2018-11.parquet": 12,
    "yellow_tripdata_2024-12.parquet": 12,
}

columnas_objetivo = [
    "tpep_pickup_datetime",
    "passenger_count",
    "trip_distance",
    "fare_amount",
    "payment_type"
]

# ============================================================
# FUNCIONES
# ============================================================

def obtener_columnas_existentes(ruta_parquet):
    con = duckdb.connect()
    try:
        con.execute("SET threads = 1;")
        query = f"""
        SELECT *
        FROM read_parquet('{ruta_parquet}')
        LIMIT 1
        """
        df_muestra = con.execute(query).fetchdf()
        return list(df_muestra.columns)
    finally:
        con.close()


def inyectar_error_en_fila(fila):
    errores_posibles = []

    if "passenger_count" in fila.index:
        errores_posibles.extend([
            "passenger_texto",
            "passenger_negativo",
            "passenger_vacio",
            "passenger_outlier"
        ])

    if "trip_distance" in fila.index:
        errores_posibles.extend([
            "distance_texto",
            "distance_negativa",
            "distance_vacia",
            "distance_outlier"
        ])

    if "fare_amount" in fila.index:
        errores_posibles.extend([
            "fare_texto",
            "fare_negativo",
            "fare_vacio",
            "fare_outlier"
        ])

    if "tpep_pickup_datetime" in fila.index:
        errores_posibles.extend([
            "fecha_invalida",
            "fecha_vacia",
            "fecha_texto"
        ])

    if "payment_type" in fila.index:
        errores_posibles.extend([
            "payment_texto",
            "payment_vacio"
        ])

    if not errores_posibles:
        return fila, "sin_error_aplicable"

    tipo_error = random.choice(errores_posibles)

    if tipo_error == "passenger_texto":
        fila["passenger_count"] = "abc"

    elif tipo_error == "passenger_negativo":
        fila["passenger_count"] = -3

    elif tipo_error == "passenger_vacio":
        fila["passenger_count"] = ""

    elif tipo_error == "passenger_outlier":
        fila["passenger_count"] = 999

    elif tipo_error == "distance_texto":
        fila["trip_distance"] = "muy_lejos"

    elif tipo_error == "distance_negativa":
        fila["trip_distance"] = -25.7

    elif tipo_error == "distance_vacia":
        fila["trip_distance"] = ""

    elif tipo_error == "distance_outlier":
        fila["trip_distance"] = 99999.99

    elif tipo_error == "fare_texto":
        fila["fare_amount"] = "no_aplica"

    elif tipo_error == "fare_negativo":
        fila["fare_amount"] = -50.0

    elif tipo_error == "fare_vacio":
        fila["fare_amount"] = ""

    elif tipo_error == "fare_outlier":
        fila["fare_amount"] = 1000000.0

    elif tipo_error == "fecha_invalida":
        fila["tpep_pickup_datetime"] = "2025-99-99 99:99:99"

    elif tipo_error == "fecha_vacia":
        fila["tpep_pickup_datetime"] = ""

    elif tipo_error == "fecha_texto":
        fila["tpep_pickup_datetime"] = "ayer_en_la_tarde"

    elif tipo_error == "payment_texto":
        fila["payment_type"] = "desconocido_total"

    elif tipo_error == "payment_vacio":
        fila["payment_type"] = ""

    return fila, tipo_error


def leer_parquet_con_duckdb(ruta_parquet, hilos, columnas):
    con = duckdb.connect()

    try:
        con.execute(f"SET threads = {hilos};")

        columnas_sql = ", ".join(columnas)

        query = f"""
        SELECT {columnas_sql}
        FROM read_parquet('{ruta_parquet}')
        """

        df = con.execute(query).fetchdf()
        return df

    finally:
        con.close()


def procesar_parquet(ruta_parquet, hilos):
    nombre_archivo = os.path.basename(ruta_parquet)
    nombre_base = os.path.splitext(nombre_archivo)[0]

    ruta_csv_salida = os.path.join(
        carpeta_salida,
        f"{nombre_base}_sucio_{hilos}_hilos.csv"
    )

    print(f"\nProcesando: {nombre_archivo}")
    print(f"Hilos asignados: {hilos}")

    inicio = time.perf_counter()

    try:
        columnas_disponibles = obtener_columnas_existentes(ruta_parquet)

        columnas_validas = [
            col for col in columnas_objetivo
            if col in columnas_disponibles
        ]

        if not columnas_validas:
            print("No se encontraron columnas objetivo en este Parquet.")
            return None

        print(f"Columnas usadas: {columnas_validas}")

        df = leer_parquet_con_duckdb(
            ruta_parquet=ruta_parquet,
            hilos=hilos,
            columnas=columnas_validas
        )

    except Exception as e:
        print(f"Error leyendo Parquet: {e}")
        return None

    total_filas = len(df)

    if total_filas == 0:
        print("Archivo vacío.")
        return None

    cantidad_errores = int(total_filas * porcentaje_error)

    indices_con_error = set(
        random.sample(range(total_filas), cantidad_errores)
    )

    conteo_errores = {}
    filas_modificadas = []

    for idx, (_, fila) in enumerate(df.iterrows()):
        fila = fila.copy()

        if idx in indices_con_error:
            fila, tipo_error = inyectar_error_en_fila(fila)
            conteo_errores[tipo_error] = conteo_errores.get(tipo_error, 0) + 1

        filas_modificadas.append(fila)

    df_sucio = pd.DataFrame(filas_modificadas)

    try:
        df_sucio.to_csv(ruta_csv_salida, index=False, encoding="utf-8")
    except Exception as e:
        print(f"Error guardando CSV: {e}")
        return None

    fin = time.perf_counter()
    tiempo_total = fin - inicio

    print(f"CSV generado: {ruta_csv_salida}")
    print(f"Filas totales: {total_filas}")
    print(f"Filas con error: {cantidad_errores}")
    print(f"Tiempo total: {tiempo_total:.4f} s")
    print(f"Detalle errores: {conteo_errores}")

    return {
        "archivo_parquet": nombre_archivo,
        "hilos_usados": hilos,
        "filas_totales": total_filas,
        "filas_con_error": cantidad_errores,
        "porcentaje_error": porcentaje_error,
        "tiempo_segundos": round(tiempo_total, 4),
        "csv_generado": ruta_csv_salida,
        "detalle_errores": str(conteo_errores)
    }


def guardar_resumen(resultados):
    if not resultados:
        print("\nNo hay resultados para guardar.")
        return

    with open(archivo_resumen, mode="w", newline="", encoding="utf-8") as archivo:
        campos = [
            "archivo_parquet",
            "hilos_usados",
            "filas_totales",
            "filas_con_error",
            "porcentaje_error",
            "tiempo_segundos",
            "csv_generado",
            "detalle_errores"
        ]

        writer = csv.DictWriter(archivo, fieldnames=campos)
        writer.writeheader()
        writer.writerows(resultados)

    print(f"\nResumen guardado en: {archivo_resumen}")


# ============================================================
# PROGRAMA PRINCIPAL
# ============================================================

def main():
    os.makedirs(carpeta_salida, exist_ok=True)

    archivos_parquet = [
        archivo for archivo in os.listdir(carpeta_entrada)
        if archivo.lower().endswith(".parquet")
    ]

    if not archivos_parquet:
        print("No se encontraron archivos Parquet.")
        return

    print("=== CONVERSIÓN PARQUET A CSV CON ERRORES ===")
    print(f"Carpeta entrada: {carpeta_entrada}")
    print(f"Carpeta salida: {carpeta_salida}")
    print(f"Porcentaje de error: {porcentaje_error * 100}%")
    print(f"Hilos por defecto: {hilos_por_defecto}")

    resultados = []

    for archivo in archivos_parquet:
        ruta_parquet = os.path.join(carpeta_entrada, archivo)

        hilos = configuracion_archivos.get(
            archivo,
            hilos_por_defecto
        )

        resultado = procesar_parquet(
            ruta_parquet=ruta_parquet,
            hilos=hilos
        )

        if resultado:
            resultados.append(resultado)

    guardar_resumen(resultados)

    print("\n=== PROCESO FINALIZADO ===")
    print(f"Archivos procesados correctamente: {len(resultados)}")


if __name__ == "__main__":
    main()