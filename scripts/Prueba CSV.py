import os
import time
import logging
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
import psutil


# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

CARPETA_CSV = r"C:\Users\Raiden AsusROG\OneDrive\Documentos\Archivo Académico\Politécnico Grancolombiano - Ing. de Sistemas\8 Semestre\Semillero HPC\HPC\data_csv_sucios"

CARPETA_RESULTADOS = r"C:\Users\Raiden AsusROG\OneDrive\Documentos\Archivo Académico\Politécnico Grancolombiano - Ing. de Sistemas\8 Semestre\Semillero HPC\HPC\results"

CARPETA_LOGS = os.path.join(CARPETA_RESULTADOS, "logs")

ARCHIVO_RESULTADOS = os.path.join(CARPETA_RESULTADOS, "resultados_benchmark_csv_final.csv")
ARCHIVO_ERRORES = os.path.join(CARPETA_RESULTADOS, "errores_detectados_csv_final.csv")
ARCHIVO_RESUMEN = os.path.join(CARPETA_RESULTADOS, "resumen_csv_final.csv")
ARCHIVO_RESUMEN_TIPOS = os.path.join(CARPETA_RESULTADOS, "resumen_tipos_error_csv_final.csv")

ARCHIVO_LOG = os.path.join(CARPETA_LOGS, "benchmark_csv_final.log")

COLUMNAS_REQUERIDAS = [
    "tpep_pickup_datetime",
    "passenger_count",
    "trip_distance",
    "fare_amount",
    "payment_type"
]

MAX_PASSENGER_COUNT = 10
MAX_TRIP_DISTANCE = 500
MAX_FARE_AMOUNT = 5000

MAX_ERROR_RATE = 0.40

PROCESOS_PARALELOS = 4


# ============================================================
# LOGGING
# ============================================================

def configurar_logging():
    os.makedirs(CARPETA_LOGS, exist_ok=True)

    logging.basicConfig(
        filename=ARCHIVO_LOG,
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        encoding="utf-8"
    )


# ============================================================
# UTILIDADES
# ============================================================

def crear_error(archivo, fila, columna, valor, tipo_error, severidad, descripcion):
    return {
        "archivo": archivo,
        "fila": fila,
        "columna": columna,
        "valor_detectado": valor,
        "tipo_error": tipo_error,
        "severidad": severidad,
        "descripcion": descripcion
    }


def leer_csv_seguro(ruta_csv):
    try:
        df = pd.read_csv(
            ruta_csv,
            dtype=str,
            low_memory=False,
            on_bad_lines="skip",
            encoding="utf-8"
        )
        return df, None

    except UnicodeDecodeError:
        try:
            df = pd.read_csv(
                ruta_csv,
                dtype=str,
                low_memory=False,
                on_bad_lines="skip",
                encoding="latin-1"
            )
            return df, None
        except Exception as e:
            return None, f"Error de codificación: {e}"

    except FileNotFoundError:
        return None, "Archivo no encontrado"

    except pd.errors.EmptyDataError:
        return None, "Archivo CSV vacío"

    except pd.errors.ParserError as e:
        return None, f"Error de estructura CSV: {e}"

    except Exception as e:
        return None, f"Error inesperado leyendo CSV: {type(e).__name__}: {e}"


def obtener_metricas_sistema():
    proceso = psutil.Process(os.getpid())

    return {
        "memoria_mb": round(proceso.memory_info().rss / (1024 * 1024), 2),
        "cpu_percent": proceso.cpu_percent(interval=None)
    }


# ============================================================
# VALIDACIÓN VECTORIAL
# ============================================================

def detectar_errores_columna_numerica(
    df,
    archivo,
    columna,
    serie_convertida,
    maximo,
    prefijo
):
    errores = []

    original = df[columna].copy()
    texto_limpio = original.fillna("").astype(str).str.strip()

    mask_vacio = texto_limpio.eq("") | texto_limpio.str.lower().isin(["nan", "none", "null"])
    mask_texto = serie_convertida.isna() & ~mask_vacio
    mask_negativo = serie_convertida < 0
    mask_outlier = serie_convertida > maximo

    for idx in df.index[mask_vacio]:
        errores.append(
            crear_error(
                archivo,
                idx,
                columna,
                original.loc[idx],
                f"{prefijo}_vacio",
                "WARNING",
                f"Valor vacío en {columna}"
            )
        )

    for idx in df.index[mask_texto]:
        errores.append(
            crear_error(
                archivo,
                idx,
                columna,
                original.loc[idx],
                f"{prefijo}_texto",
                "ERROR",
                f"Valor no numérico en {columna}"
            )
        )

    for idx in df.index[mask_negativo]:
        errores.append(
            crear_error(
                archivo,
                idx,
                columna,
                original.loc[idx],
                f"{prefijo}_negativo",
                "ERROR",
                f"Valor negativo en {columna}"
            )
        )

    for idx in df.index[mask_outlier]:
        errores.append(
            crear_error(
                archivo,
                idx,
                columna,
                original.loc[idx],
                f"{prefijo}_outlier",
                "ERROR",
                f"Valor fuera de rango en {columna}"
            )
        )

    mascara_valida = (
        serie_convertida.notna()
        & (serie_convertida >= 0)
        & (serie_convertida <= maximo)
    )

    return errores, mascara_valida


def detectar_errores_fecha(df, archivo):
    errores = []

    columna = "tpep_pickup_datetime"
    original = df[columna].copy()
    texto_limpio = original.fillna("").astype(str).str.strip()

    fechas_convertidas = pd.to_datetime(
        original,
        errors="coerce",
        format=None
    )

    mask_vacia = texto_limpio.eq("") | texto_limpio.str.lower().isin(["nan", "none", "null"])
    mask_invalida = fechas_convertidas.isna() & ~mask_vacia
    mask_texto = mask_invalida & texto_limpio.str.lower().str.contains("ayer|tarde", regex=True)

    for idx in df.index[mask_vacia]:
        errores.append(
            crear_error(
                archivo,
                idx,
                columna,
                original.loc[idx],
                "fecha_vacia",
                "WARNING",
                "Fecha vacía"
            )
        )

    for idx in df.index[mask_texto]:
        errores.append(
            crear_error(
                archivo,
                idx,
                columna,
                original.loc[idx],
                "fecha_texto",
                "ERROR",
                "Fecha escrita como texto no procesable"
            )
        )

    for idx in df.index[mask_invalida & ~mask_texto]:
        errores.append(
            crear_error(
                archivo,
                idx,
                columna,
                original.loc[idx],
                "fecha_invalida",
                "ERROR",
                "Fecha inválida o imposible"
            )
        )

    mascara_valida = fechas_convertidas.notna()

    return errores, mascara_valida, fechas_convertidas


def detectar_errores_payment_type(df, archivo):
    errores = []

    columna = "payment_type"
    original = df[columna].copy()
    texto_limpio = original.fillna("").astype(str).str.strip()

    convertido = pd.to_numeric(original, errors="coerce")

    mask_vacio = texto_limpio.eq("") | texto_limpio.str.lower().isin(["nan", "none", "null"])
    mask_texto = convertido.isna() & ~mask_vacio

    for idx in df.index[mask_vacio]:
        errores.append(
            crear_error(
                archivo,
                idx,
                columna,
                original.loc[idx],
                "payment_vacio",
                "WARNING",
                "Tipo de pago vacío"
            )
        )

    for idx in df.index[mask_texto]:
        errores.append(
            crear_error(
                archivo,
                idx,
                columna,
                original.loc[idx],
                "payment_texto",
                "ERROR",
                "Tipo de pago no numérico"
            )
        )

    mascara_valida = convertido.notna()

    return errores, mascara_valida, convertido


def limpiar_y_validar(df, archivo):
    errores = []

    columnas_presentes = set(df.columns)
    columnas_faltantes = [
        col for col in COLUMNAS_REQUERIDAS
        if col not in columnas_presentes
    ]

    for columna in columnas_faltantes:
        errores.append(
            crear_error(
                archivo,
                None,
                columna,
                None,
                "columna_faltante",
                "CRITICAL",
                f"No existe la columna requerida: {columna}"
            )
        )

    if columnas_faltantes:
        return pd.DataFrame(), errores

    df = df[COLUMNAS_REQUERIDAS].copy()

    mascara_global = pd.Series(True, index=df.index)

    # passenger_count
    passenger_num = pd.to_numeric(df["passenger_count"], errors="coerce")
    err, mask = detectar_errores_columna_numerica(
        df,
        archivo,
        "passenger_count",
        passenger_num,
        MAX_PASSENGER_COUNT,
        "passenger"
    )
    errores.extend(err)
    mascara_global &= mask
    df["passenger_count"] = passenger_num

    # trip_distance
    distance_num = pd.to_numeric(df["trip_distance"], errors="coerce")
    err, mask = detectar_errores_columna_numerica(
        df,
        archivo,
        "trip_distance",
        distance_num,
        MAX_TRIP_DISTANCE,
        "distance"
    )
    errores.extend(err)
    mascara_global &= mask
    df["trip_distance"] = distance_num

    # fare_amount
    fare_num = pd.to_numeric(df["fare_amount"], errors="coerce")
    err, mask = detectar_errores_columna_numerica(
        df,
        archivo,
        "fare_amount",
        fare_num,
        MAX_FARE_AMOUNT,
        "fare"
    )
    errores.extend(err)
    mascara_global &= mask
    df["fare_amount"] = fare_num

    # fecha
    err, mask, fechas = detectar_errores_fecha(df, archivo)
    errores.extend(err)
    mascara_global &= mask
    df["tpep_pickup_datetime"] = fechas

    # payment_type
    err, mask, payment_num = detectar_errores_payment_type(df, archivo)
    errores.extend(err)
    mascara_global &= mask
    df["payment_type"] = payment_num

    df_limpio = df[mascara_global].copy()

    return df_limpio, errores


# ============================================================
# BENCHMARK
# ============================================================

def ejecutar_consulta(df, archivo, nombre, funcion):
    inicio = time.perf_counter()

    try:
        resultado = funcion(df)
        fin = time.perf_counter()

        return {
            "archivo": archivo,
            "consulta": nombre,
            "resultado": str(resultado),
            "tiempo_segundos": round(fin - inicio, 6),
            "estado": "OK",
            "error": ""
        }

    except Exception as e:
        fin = time.perf_counter()

        return {
            "archivo": archivo,
            "consulta": nombre,
            "resultado": "",
            "tiempo_segundos": round(fin - inicio, 6),
            "estado": "ERROR",
            "error": f"{type(e).__name__}: {e}"
        }


def ejecutar_benchmark(df, archivo):
    consultas = [
        (
            "COUNT(*)",
            lambda df: len(df)
        ),
        (
            "SUM(passenger_count)",
            lambda df: df["passenger_count"].sum()
        ),
        (
            "AVG(trip_distance)",
            lambda df: df["trip_distance"].mean()
        ),
        (
            "GROUP BY passenger_count",
            lambda df: df.groupby("passenger_count").size().to_dict()
        ),
        (
            "WHERE trip_distance > 5 GROUP BY passenger_count",
            lambda df: df[df["trip_distance"] > 5].groupby("passenger_count").size().to_dict()
        ),
        (
            "AVG(fare_amount)",
            lambda df: df["fare_amount"].mean()
        ),
        (
            "GROUP BY payment_type",
            lambda df: df.groupby("payment_type").size().to_dict()
        )
    ]

    resultados = []

    for nombre, funcion in consultas:
        resultados.append(
            ejecutar_consulta(df, archivo, nombre, funcion)
        )

    return resultados


# ============================================================
# PROCESAMIENTO POR ARCHIVO
# ============================================================

def procesar_archivo_csv(ruta_csv):
    archivo = os.path.basename(ruta_csv)

    inicio_total = time.perf_counter()

    errores = []
    resultados = []

    metricas_inicio = obtener_metricas_sistema()

    try:
        df, error_lectura = leer_csv_seguro(ruta_csv)

        if error_lectura:
            errores.append(
                crear_error(
                    archivo,
                    None,
                    None,
                    None,
                    "error_lectura_csv",
                    "CRITICAL",
                    error_lectura
                )
            )

            resumen = {
                "archivo": archivo,
                "estado": "ERROR_LECTURA",
                "filas_originales": 0,
                "filas_validas": 0,
                "filas_descartadas": 0,
                "errores_detectados": len(errores),
                "error_rate": 1,
                "tiempo_total_segundos": 0,
                "throughput_filas_segundo": 0,
                "memoria_inicio_mb": metricas_inicio["memoria_mb"],
                "memoria_fin_mb": metricas_inicio["memoria_mb"],
                "cpu_inicio_percent": metricas_inicio["cpu_percent"],
                "cpu_fin_percent": metricas_inicio["cpu_percent"]
            }

            return resultados, errores, resumen

        filas_originales = len(df)

        df_limpio, errores_validacion = limpiar_y_validar(df, archivo)
        errores.extend(errores_validacion)

        filas_validas = len(df_limpio)
        filas_descartadas = filas_originales - filas_validas

        error_rate = filas_descartadas / filas_originales if filas_originales > 0 else 1

        if error_rate > MAX_ERROR_RATE:
            errores.append(
                crear_error(
                    archivo,
                    None,
                    None,
                    error_rate,
                    "tasa_error_excedida",
                    "CRITICAL",
                    f"La tasa de error supera el máximo permitido: {MAX_ERROR_RATE}"
                )
            )

            estado = "ABORTADO_POR_EXCESO_ERRORES"
        else:
            resultados = ejecutar_benchmark(df_limpio, archivo)
            estado = "OK"

        fin_total = time.perf_counter()
        tiempo_total = fin_total - inicio_total

        metricas_fin = obtener_metricas_sistema()

        resumen = {
            "archivo": archivo,
            "estado": estado,
            "filas_originales": filas_originales,
            "filas_validas": filas_validas,
            "filas_descartadas": filas_descartadas,
            "errores_detectados": len(errores),
            "error_rate": round(error_rate, 6),
            "tiempo_total_segundos": round(tiempo_total, 6),
            "throughput_filas_segundo": round(filas_originales / tiempo_total, 2) if tiempo_total > 0 else 0,
            "memoria_inicio_mb": metricas_inicio["memoria_mb"],
            "memoria_fin_mb": metricas_fin["memoria_mb"],
            "cpu_inicio_percent": metricas_inicio["cpu_percent"],
            "cpu_fin_percent": metricas_fin["cpu_percent"]
        }

        return resultados, errores, resumen

    except Exception as e:
        fin_total = time.perf_counter()
        tiempo_total = fin_total - inicio_total

        errores.append(
            crear_error(
                archivo,
                None,
                None,
                None,
                "error_general_procesamiento",
                "CRITICAL",
                f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
            )
        )

        metricas_fin = obtener_metricas_sistema()

        resumen = {
            "archivo": archivo,
            "estado": "ERROR_GENERAL",
            "filas_originales": 0,
            "filas_validas": 0,
            "filas_descartadas": 0,
            "errores_detectados": len(errores),
            "error_rate": 1,
            "tiempo_total_segundos": round(tiempo_total, 6),
            "throughput_filas_segundo": 0,
            "memoria_inicio_mb": metricas_inicio["memoria_mb"],
            "memoria_fin_mb": metricas_fin["memoria_mb"],
            "cpu_inicio_percent": metricas_inicio["cpu_percent"],
            "cpu_fin_percent": metricas_fin["cpu_percent"]
        }

        return resultados, errores, resumen


# ============================================================
# PROGRAMA PRINCIPAL
# ============================================================

def main():
    os.makedirs(CARPETA_RESULTADOS, exist_ok=True)
    configurar_logging()

    logging.info("Inicio del benchmark CSV final")

    archivos_csv = [
        os.path.join(CARPETA_CSV, archivo)
        for archivo in os.listdir(CARPETA_CSV)
        if archivo.lower().endswith(".csv")
    ]

    if not archivos_csv:
        print("No se encontraron archivos CSV.")
        logging.warning("No se encontraron archivos CSV.")
        return

    print("=== BENCHMARK CSV FINAL CON MANEJO ROBUSTO DE EXCEPCIONES ===")
    print(f"Archivos encontrados: {len(archivos_csv)}")
    print(f"Procesos paralelos: {PROCESOS_PARALELOS}")
    print(f"Máxima tasa de error permitida: {MAX_ERROR_RATE * 100}%")

    todos_resultados = []
    todos_errores = []
    todos_resumenes = []

    inicio_global = time.perf_counter()

    with ProcessPoolExecutor(max_workers=PROCESOS_PARALELOS) as executor:
        futuros = {
            executor.submit(procesar_archivo_csv, ruta): ruta
            for ruta in archivos_csv
        }

        for futuro in as_completed(futuros):
            ruta = futuros[futuro]
            archivo = os.path.basename(ruta)

            try:
                resultados, errores, resumen = futuro.result()

                todos_resultados.extend(resultados)
                todos_errores.extend(errores)
                todos_resumenes.append(resumen)

                print(f"\nArchivo procesado: {archivo}")
                print(f"Estado: {resumen['estado']}")
                print(f"Filas originales: {resumen['filas_originales']}")
                print(f"Filas válidas: {resumen['filas_validas']}")
                print(f"Filas descartadas: {resumen['filas_descartadas']}")
                print(f"Errores detectados: {resumen['errores_detectados']}")
                print(f"Tiempo: {resumen['tiempo_total_segundos']} s")

                logging.info(f"Procesado {archivo} | Estado: {resumen['estado']}")

            except Exception as e:
                logging.error(f"Error procesando {archivo}: {e}")

                todos_errores.append(
                    crear_error(
                        archivo,
                        None,
                        None,
                        None,
                        "error_futuro_paralelo",
                        "CRITICAL",
                        f"{type(e).__name__}: {e}"
                    )
                )

    fin_global = time.perf_counter()
    tiempo_global = fin_global - inicio_global

    df_resultados = pd.DataFrame(todos_resultados)
    df_errores = pd.DataFrame(todos_errores)
    df_resumen = pd.DataFrame(todos_resumenes)

    df_resultados.to_csv(ARCHIVO_RESULTADOS, index=False, encoding="utf-8")
    df_errores.to_csv(ARCHIVO_ERRORES, index=False, encoding="utf-8")
    df_resumen.to_csv(ARCHIVO_RESUMEN, index=False, encoding="utf-8")

    if not df_errores.empty:
        df_resumen_tipos = (
            df_errores
            .groupby(["tipo_error", "severidad"])
            .size()
            .reset_index(name="cantidad")
            .sort_values(by="cantidad", ascending=False)
        )

        df_resumen_tipos.to_csv(
            ARCHIVO_RESUMEN_TIPOS,
            index=False,
            encoding="utf-8"
        )
    else:
        pd.DataFrame(columns=["tipo_error", "severidad", "cantidad"]).to_csv(
            ARCHIVO_RESUMEN_TIPOS,
            index=False,
            encoding="utf-8"
        )

    print("\n=== PROCESO FINALIZADO ===")
    print(f"Tiempo global: {round(tiempo_global, 4)} s")
    print(f"Resultados benchmark: {ARCHIVO_RESULTADOS}")
    print(f"Errores detectados: {ARCHIVO_ERRORES}")
    print(f"Resumen por archivo: {ARCHIVO_RESUMEN}")
    print(f"Resumen por tipo de error: {ARCHIVO_RESUMEN_TIPOS}")
    print(f"Log del proceso: {ARCHIVO_LOG}")

    logging.info(f"Proceso finalizado en {round(tiempo_global, 4)} segundos")


if __name__ == "__main__":
    main()