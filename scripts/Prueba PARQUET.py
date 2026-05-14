import os
import csv
import time
import traceback
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd


# ============================================================
# CONFIGURACIÓN DEL EXPERIMENTO CSV
# ============================================================

carpeta_csv = r"C:\Users\Raiden AsusROG\OneDrive\Documentos\Archivo Académico\Politécnico Grancolombiano - Ing. de Sistemas\8 Semestre\Semillero HPC\HPC\data_csv_sucios"

carpeta_resultados = r"C:\Users\Raiden AsusROG\OneDrive\Documentos\Archivo Académico\Politécnico Grancolombiano - Ing. de Sistemas\8 Semestre\Semillero HPC\HPC\results"

os.makedirs(carpeta_resultados, exist_ok=True)

fecha_ejecucion = datetime.now().strftime("%Y%m%d_%H%M%S")

archivo_resultados = os.path.join(
    carpeta_resultados,
    f"benchmark_csv_python_{fecha_ejecucion}.csv"
)

archivo_errores = os.path.join(
    carpeta_resultados,
    f"errores_csv_python_{fecha_ejecucion}.csv"
)

archivo_resumen_validacion = os.path.join(
    carpeta_resultados,
    f"resumen_validacion_csv_{fecha_ejecucion}.csv"
)

hilos_logicos = os.cpu_count()
nucleos_estimados = hilos_logicos // 2 if hilos_logicos else 1

repeticiones = 3
hacer_warmup = True

escenarios = [
    {"nombre": "1 hilo", "threads": 1},
    {"nombre": "2 hilos", "threads": 2},
    {"nombre": "4 hilos", "threads": 4},
    {"nombre": "8 hilos", "threads": 8},
    {"nombre": "16 hilos", "threads": 16},
    {"nombre": "maximo equipo", "threads": hilos_logicos},
]

columnas_requeridas = [
    "tpep_pickup_datetime",
    "passenger_count",
    "trip_distance",
    "fare_amount",
    "payment_type"
]

MAX_PASSENGER_COUNT = 10
MAX_TRIP_DISTANCE = 500
MAX_FARE_AMOUNT = 5000


# ============================================================
# CONSULTAS ANALÍTICAS EQUIVALENTES AL BENCHMARK PARQUET
# ============================================================

consultas = [
    {
        "nombre": "Q1_count_total",
        "descripcion": "Conteo total de registros válidos",
        "funcion": lambda df: len(df)
    },
    {
        "nombre": "Q2_suma_pasajeros",
        "descripcion": "Suma total de pasajeros",
        "funcion": lambda df: df["passenger_count"].sum()
    },
    {
        "nombre": "Q3_promedio_distancia",
        "descripcion": "Promedio de distancia recorrida",
        "funcion": lambda df: df["trip_distance"].mean()
    },
    {
        "nombre": "Q4_group_by_pasajeros",
        "descripcion": "Agrupación por cantidad de pasajeros",
        "funcion": lambda df: (
            df.groupby("passenger_count")
            .agg(
                total_viajes=("passenger_count", "count"),
                distancia_promedio=("trip_distance", "mean"),
                total_recaudado=("fare_amount", "sum")
            )
            .reset_index()
        )
    },
    {
        "nombre": "Q5_filtro_group_by",
        "descripcion": "Filtro por distancia mayor a 5 y agrupación por pasajeros",
        "funcion": lambda df: (
            df[df["trip_distance"] > 5]
            .groupby("passenger_count")
            .agg(
                total_viajes=("passenger_count", "count"),
                distancia_promedio=("trip_distance", "mean"),
                total_recaudado=("fare_amount", "sum")
            )
            .reset_index()
        )
    }
]


# ============================================================
# MANEJO DE ERRORES
# ============================================================

def crear_error(archivo, fila, columna, valor, tipo_error, descripcion):
    return {
        "fecha_ejecucion": fecha_ejecucion,
        "archivo": archivo,
        "fila": fila,
        "columna": columna,
        "valor_detectado": valor,
        "tipo_error": tipo_error,
        "descripcion": descripcion
    }


def detectar_errores_numericos(df, archivo, columna, maximo, prefijo):
    errores = []

    original = df[columna].copy()
    texto = original.fillna("").astype(str).str.strip()
    convertido = pd.to_numeric(original, errors="coerce")

    mask_vacio = texto.eq("") | texto.str.lower().isin(["nan", "none", "null"])
    mask_texto = convertido.isna() & ~mask_vacio
    mask_negativo = convertido < 0
    mask_outlier = convertido > maximo

    for idx in df.index[mask_vacio]:
        errores.append(
            crear_error(
                archivo,
                idx,
                columna,
                original.loc[idx],
                f"{prefijo}_vacio",
                f"Valor vacío en la columna {columna}"
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
                f"Valor no numérico en la columna {columna}"
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
                f"Valor negativo en la columna {columna}"
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
                f"Valor fuera de rango en la columna {columna}"
            )
        )

    mascara_valida = (
        convertido.notna()
        & (convertido >= 0)
        & (convertido <= maximo)
    )

    return convertido, mascara_valida, errores


def detectar_errores_fecha(df, archivo):
    errores = []

    columna = "tpep_pickup_datetime"
    original = df[columna].copy()
    texto = original.fillna("").astype(str).str.strip()

    convertido = pd.to_datetime(original, errors="coerce")

    mask_vacio = texto.eq("") | texto.str.lower().isin(["nan", "none", "null"])
    mask_invalido = convertido.isna() & ~mask_vacio
    mask_texto = mask_invalido & texto.str.lower().str.contains("ayer|tarde", regex=True)

    for idx in df.index[mask_vacio]:
        errores.append(
            crear_error(
                archivo,
                idx,
                columna,
                original.loc[idx],
                "fecha_vacia",
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
                "Fecha escrita como texto no procesable"
            )
        )

    for idx in df.index[mask_invalido & ~mask_texto]:
        errores.append(
            crear_error(
                archivo,
                idx,
                columna,
                original.loc[idx],
                "fecha_invalida",
                "Fecha inválida o imposible"
            )
        )

    mascara_valida = convertido.notna()

    return convertido, mascara_valida, errores


def detectar_errores_payment(df, archivo):
    errores = []

    columna = "payment_type"
    original = df[columna].copy()
    texto = original.fillna("").astype(str).str.strip()

    convertido = pd.to_numeric(original, errors="coerce")

    mask_vacio = texto.eq("") | texto.str.lower().isin(["nan", "none", "null"])
    mask_texto = convertido.isna() & ~mask_vacio

    for idx in df.index[mask_vacio]:
        errores.append(
            crear_error(
                archivo,
                idx,
                columna,
                original.loc[idx],
                "payment_vacio",
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
                "Tipo de pago no numérico"
            )
        )

    mascara_valida = convertido.notna()

    return convertido, mascara_valida, errores


# ============================================================
# LECTURA Y VALIDACIÓN DEL CSV
# ============================================================

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


def validar_y_limpiar_csv(ruta_csv):
    archivo = os.path.basename(ruta_csv)

    inicio_validacion = time.perf_counter()

    errores = []

    df, error_lectura = leer_csv_seguro(ruta_csv)

    if error_lectura:
        errores.append(
            crear_error(
                archivo,
                None,
                None,
                None,
                "error_lectura_csv",
                error_lectura
            )
        )

        fin_validacion = time.perf_counter()

        resumen = {
            "archivo": archivo,
            "filas_originales": 0,
            "filas_validas": 0,
            "filas_invalidas": 0,
            "errores_detectados": len(errores),
            "tiempo_validacion_s": round(fin_validacion - inicio_validacion, 6),
            "estado_validacion": "ERROR_LECTURA"
        }

        return pd.DataFrame(), errores, resumen

    filas_originales = len(df)

    columnas_presentes = set(df.columns)
    columnas_faltantes = [
        col for col in columnas_requeridas
        if col not in columnas_presentes
    ]

    if columnas_faltantes:
        for columna in columnas_faltantes:
            errores.append(
                crear_error(
                    archivo,
                    None,
                    columna,
                    None,
                    "columna_faltante",
                    f"No existe la columna requerida: {columna}"
                )
            )

        fin_validacion = time.perf_counter()

        resumen = {
            "archivo": archivo,
            "filas_originales": filas_originales,
            "filas_validas": 0,
            "filas_invalidas": filas_originales,
            "errores_detectados": len(errores),
            "tiempo_validacion_s": round(fin_validacion - inicio_validacion, 6),
            "estado_validacion": "COLUMNAS_FALTANTES"
        }

        return pd.DataFrame(), errores, resumen

    df = df[columnas_requeridas].copy()

    mascara_global = pd.Series(True, index=df.index)

    # passenger_count
    convertido, mascara, err = detectar_errores_numericos(
        df,
        archivo,
        "passenger_count",
        MAX_PASSENGER_COUNT,
        "passenger"
    )
    df["passenger_count"] = convertido
    mascara_global &= mascara
    errores.extend(err)

    # trip_distance
    convertido, mascara, err = detectar_errores_numericos(
        df,
        archivo,
        "trip_distance",
        MAX_TRIP_DISTANCE,
        "distance"
    )
    df["trip_distance"] = convertido
    mascara_global &= mascara
    errores.extend(err)

    # fare_amount
    convertido, mascara, err = detectar_errores_numericos(
        df,
        archivo,
        "fare_amount",
        MAX_FARE_AMOUNT,
        "fare"
    )
    df["fare_amount"] = convertido
    mascara_global &= mascara
    errores.extend(err)

    # fecha
    convertido, mascara, err = detectar_errores_fecha(df, archivo)
    df["tpep_pickup_datetime"] = convertido
    mascara_global &= mascara
    errores.extend(err)

    # payment_type
    convertido, mascara, err = detectar_errores_payment(df, archivo)
    df["payment_type"] = convertido
    mascara_global &= mascara
    errores.extend(err)

    df_limpio = df[mascara_global].copy()

    filas_validas = len(df_limpio)
    filas_invalidas = filas_originales - filas_validas

    fin_validacion = time.perf_counter()

    resumen = {
        "archivo": archivo,
        "filas_originales": filas_originales,
        "filas_validas": filas_validas,
        "filas_invalidas": filas_invalidas,
        "errores_detectados": len(errores),
        "tiempo_validacion_s": round(fin_validacion - inicio_validacion, 6),
        "estado_validacion": "OK"
    }

    return df_limpio, errores, resumen


# ============================================================
# PROCESAMIENTO PARALELO POR PARTICIONES
# ============================================================

def dividir_dataframe(df, partes):
    if len(df) == 0:
        return []

    partes = max(1, partes)
    tamano = len(df) // partes

    particiones = []

    for i in range(partes):
        inicio = i * tamano
        fin = len(df) if i == partes - 1 else (i + 1) * tamano
        particiones.append(df.iloc[inicio:fin].copy())

    return particiones


def ejecutar_consulta_particion(nombre_consulta, df_particion):
    if nombre_consulta == "Q1_count_total":
        return len(df_particion)

    if nombre_consulta == "Q2_suma_pasajeros":
        return df_particion["passenger_count"].sum()

    if nombre_consulta == "Q3_promedio_distancia":
        return {
            "suma": df_particion["trip_distance"].sum(),
            "conteo": df_particion["trip_distance"].count()
        }

    if nombre_consulta == "Q4_group_by_pasajeros":
        return (
            df_particion.groupby("passenger_count")
            .agg(
                total_viajes=("passenger_count", "count"),
                suma_distancia=("trip_distance", "sum"),
                conteo_distancia=("trip_distance", "count"),
                total_recaudado=("fare_amount", "sum")
            )
            .reset_index()
        )

    if nombre_consulta == "Q5_filtro_group_by":
        filtrado = df_particion[df_particion["trip_distance"] > 5]

        return (
            filtrado.groupby("passenger_count")
            .agg(
                total_viajes=("passenger_count", "count"),
                suma_distancia=("trip_distance", "sum"),
                conteo_distancia=("trip_distance", "count"),
                total_recaudado=("fare_amount", "sum")
            )
            .reset_index()
        )

    raise ValueError(f"Consulta no reconocida: {nombre_consulta}")


def combinar_resultados(nombre_consulta, resultados_particiones):
    if nombre_consulta == "Q1_count_total":
        return sum(resultados_particiones), 1

    if nombre_consulta == "Q2_suma_pasajeros":
        return sum(resultados_particiones), 1

    if nombre_consulta == "Q3_promedio_distancia":
        suma_total = sum(r["suma"] for r in resultados_particiones)
        conteo_total = sum(r["conteo"] for r in resultados_particiones)

        promedio = suma_total / conteo_total if conteo_total else 0
        return promedio, 1

    if nombre_consulta in ["Q4_group_by_pasajeros", "Q5_filtro_group_by"]:
        dfs = [r for r in resultados_particiones if isinstance(r, pd.DataFrame)]

        if not dfs:
            return pd.DataFrame(), 0

        combinado = pd.concat(dfs, ignore_index=True)

        final = (
            combinado.groupby("passenger_count")
            .agg(
                total_viajes=("total_viajes", "sum"),
                suma_distancia=("suma_distancia", "sum"),
                conteo_distancia=("conteo_distancia", "sum"),
                total_recaudado=("total_recaudado", "sum")
            )
            .reset_index()
        )

        final["distancia_promedio"] = (
            final["suma_distancia"] / final["conteo_distancia"]
        )

        final = final[
            [
                "passenger_count",
                "total_viajes",
                "distancia_promedio",
                "total_recaudado"
            ]
        ]

        return final, len(final)

    raise ValueError(f"Consulta no reconocida: {nombre_consulta}")


def ejecutar_consulta_paralela(df, consulta, threads):
    nombre_consulta = consulta["nombre"]

    inicio = time.perf_counter()

    try:
        particiones = dividir_dataframe(df, threads)

        if not particiones:
            return None, 0, time.perf_counter() - inicio, "Dataset vacío"

        with ProcessPoolExecutor(max_workers=threads) as executor:
            futuros = [
                executor.submit(
                    ejecutar_consulta_particion,
                    nombre_consulta,
                    particion
                )
                for particion in particiones
            ]

            resultados_particiones = []

            for futuro in as_completed(futuros):
                resultados_particiones.append(futuro.result())

        resultado_final, filas_resultado = combinar_resultados(
            nombre_consulta,
            resultados_particiones
        )

        fin = time.perf_counter()

        return resultado_final, filas_resultado, fin - inicio, None

    except Exception as e:
        fin = time.perf_counter()

        error = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"

        return None, 0, fin - inicio, error


# ============================================================
# MÉTRICAS
# ============================================================

def calcular_metricas(tiempos):
    tiempos_validos = [t for t in tiempos if t is not None]

    if not tiempos_validos:
        return None, None, None

    promedio = sum(tiempos_validos) / len(tiempos_validos)
    minimo = min(tiempos_validos)
    maximo = max(tiempos_validos)

    return promedio, minimo, maximo


def resultado_a_texto(resultado):
    if isinstance(resultado, pd.DataFrame):
        return resultado.to_dict(orient="records")

    return resultado


# ============================================================
# GUARDADO DE RESULTADOS
# ============================================================

def guardar_csv(ruta, campos, filas):
    with open(ruta, mode="w", newline="", encoding="utf-8") as archivo:
        writer = csv.DictWriter(archivo, fieldnames=campos)
        writer.writeheader()
        writer.writerows(filas)


# ============================================================
# EJECUCIÓN PRINCIPAL
# ============================================================

def main():
    archivos_csv = [
        os.path.join(carpeta_csv, archivo)
        for archivo in os.listdir(carpeta_csv)
        if archivo.lower().endswith(".csv")
    ]

    if not archivos_csv:
        print("No se encontraron archivos CSV.")
        return

    print("=== BENCHMARK FINAL CSV CON ERRORES Y MANEJO DE EXCEPCIONES ===\n")
    print(f"Carpeta CSV: {carpeta_csv}")
    print(f"Hilos lógicos detectados: {hilos_logicos}")
    print(f"Núcleos físicos estimados: {nucleos_estimados}")
    print(f"Repeticiones por prueba: {repeticiones}")
    print(f"Warmup activado: {hacer_warmup}")
    print(f"Archivos CSV encontrados: {len(archivos_csv)}")
    print(f"Archivo de resultados: {archivo_resultados}\n")

    resultados_benchmark = []
    errores_globales = []
    resumenes_validacion = []

    # Validar todos los CSV una sola vez
    print("=== FASE 1: VALIDACIÓN Y LIMPIEZA DE CSV ===")

    dataframes_limpios = []

    for ruta_csv in archivos_csv:
        archivo = os.path.basename(ruta_csv)

        print(f"\nValidando archivo: {archivo}")

        df_limpio, errores, resumen = validar_y_limpiar_csv(ruta_csv)

        errores_globales.extend(errores)
        resumenes_validacion.append(resumen)

        if not df_limpio.empty:
            dataframes_limpios.append(df_limpio)

        print(f"Filas originales: {resumen['filas_originales']}")
        print(f"Filas válidas: {resumen['filas_validas']}")
        print(f"Filas inválidas: {resumen['filas_invalidas']}")
        print(f"Errores detectados: {resumen['errores_detectados']}")
        print(f"Tiempo validación: {resumen['tiempo_validacion_s']} s")
        print(f"Estado validación: {resumen['estado_validacion']}")

    if not dataframes_limpios:
        print("\nNo hay datos válidos para ejecutar benchmark.")
        return

    df_dataset = pd.concat(dataframes_limpios, ignore_index=True)

    total_filas_originales = sum(r["filas_originales"] for r in resumenes_validacion)
    total_filas_validas = sum(r["filas_validas"] for r in resumenes_validacion)
    total_filas_invalidas = sum(r["filas_invalidas"] for r in resumenes_validacion)
    total_errores = len(errores_globales)
    tiempo_total_validacion = sum(r["tiempo_validacion_s"] for r in resumenes_validacion)

    print("\n=== RESUMEN GLOBAL DE VALIDACIÓN ===")
    print(f"Filas originales: {total_filas_originales}")
    print(f"Filas válidas: {total_filas_validas}")
    print(f"Filas inválidas: {total_filas_invalidas}")
    print(f"Errores detectados: {total_errores}")
    print(f"Tiempo total validación: {tiempo_total_validacion:.6f} s")

    # Benchmark
    print("\n=== FASE 2: BENCHMARK ANALÍTICO CSV ===")

    for consulta in consultas:
        nombre_consulta = consulta["nombre"]
        descripcion = consulta["descripcion"]

        print("\n==========================================")
        print(f"Consulta: {nombre_consulta}")
        print(f"Descripción: {descripcion}")
        print("==========================================")

        tiempos_por_threads = {}

        for escenario in escenarios:
            nombre_escenario = escenario["nombre"]
            threads = escenario["threads"]

            if threads is None:
                continue

            if hilos_logicos and threads > hilos_logicos:
                print(f"Saltando {nombre_escenario}: excede capacidad detectada.")
                continue

            print(f"\nEscenario: {nombre_escenario} | Procesos: {threads}")

            if hacer_warmup:
                _, _, tiempo_warmup, error_warmup = ejecutar_consulta_paralela(
                    df_dataset,
                    consulta,
                    threads
                )

                if error_warmup:
                    print(f"  Warmup: ERROR -> {error_warmup}")
                else:
                    print(f"  Warmup: {tiempo_warmup:.4f} s")

            tiempos = []
            filas_resultado = 0
            resultado_final = ""
            error_final = ""
            estado = "OK"

            for intento in range(1, repeticiones + 1):
                resultado, filas, tiempo, error = ejecutar_consulta_paralela(
                    df_dataset,
                    consulta,
                    threads
                )

                if error:
                    estado = "ERROR"
                    error_final = error
                    tiempos.append(None)
                    print(f"  Intento {intento}: ERROR -> {error}")
                else:
                    tiempos.append(tiempo)
                    filas_resultado = filas
                    resultado_final = resultado_a_texto(resultado)
                    print(f"  Intento {intento}: {tiempo:.4f} s")

            promedio, minimo, maximo = calcular_metricas(tiempos)

            if promedio is not None:
                tiempos_por_threads[threads] = promedio
                print(
                    f"  Promedio: {promedio:.4f} s | "
                    f"Mín: {minimo:.4f} s | "
                    f"Máx: {maximo:.4f} s"
                )
            else:
                print("  No se pudieron calcular métricas.")

            resultados_benchmark.append({
                "fecha_ejecucion": fecha_ejecucion,
                "formato": "CSV",
                "consulta": nombre_consulta,
                "descripcion": descripcion,
                "escenario": nombre_escenario,
                "threads": threads,
                "hilos_logicos_detectados": hilos_logicos,
                "nucleos_estimados": nucleos_estimados,
                "repeticiones": repeticiones,
                "warmup": hacer_warmup,
                "tiempo_promedio_s": round(promedio, 6) if promedio is not None else "",
                "tiempo_minimo_s": round(minimo, 6) if minimo is not None else "",
                "tiempo_maximo_s": round(maximo, 6) if maximo is not None else "",
                "speedup": "",
                "eficiencia_paralela": "",
                "filas_resultado": filas_resultado,
                "estado": estado,
                "error": error_final,
                "filas_originales": total_filas_originales,
                "filas_validas": total_filas_validas,
                "filas_invalidas": total_filas_invalidas,
                "errores_detectados": total_errores,
                "porcentaje_error_detectado": round(
                    total_filas_invalidas / total_filas_originales,
                    6
                ) if total_filas_originales else "",
                "tiempo_validacion_s": round(tiempo_total_validacion, 6),
                "resultado": str(resultado_final)[:1000]
            })

        tiempo_base_1_hilo = tiempos_por_threads.get(1)

        if tiempo_base_1_hilo:
            for fila in resultados_benchmark:
                if fila["consulta"] == nombre_consulta and fila["estado"] == "OK":
                    threads = fila["threads"]
                    tiempo_actual = fila["tiempo_promedio_s"]

                    if tiempo_actual:
                        speedup = tiempo_base_1_hilo / tiempo_actual
                        eficiencia = speedup / threads

                        fila["speedup"] = round(speedup, 6)
                        fila["eficiencia_paralela"] = round(eficiencia, 6)

    # Guardar resultados
    campos_resultados = [
        "fecha_ejecucion",
        "formato",
        "consulta",
        "descripcion",
        "escenario",
        "threads",
        "hilos_logicos_detectados",
        "nucleos_estimados",
        "repeticiones",
        "warmup",
        "tiempo_promedio_s",
        "tiempo_minimo_s",
        "tiempo_maximo_s",
        "speedup",
        "eficiencia_paralela",
        "filas_resultado",
        "estado",
        "error",
        "filas_originales",
        "filas_validas",
        "filas_invalidas",
        "errores_detectados",
        "porcentaje_error_detectado",
        "tiempo_validacion_s",
        "resultado"
    ]

    campos_errores = [
        "fecha_ejecucion",
        "archivo",
        "fila",
        "columna",
        "valor_detectado",
        "tipo_error",
        "descripcion"
    ]

    campos_resumen = [
        "archivo",
        "filas_originales",
        "filas_validas",
        "filas_invalidas",
        "errores_detectados",
        "tiempo_validacion_s",
        "estado_validacion"
    ]

    guardar_csv(archivo_resultados, campos_resultados, resultados_benchmark)
    guardar_csv(archivo_errores, campos_errores, errores_globales)
    guardar_csv(archivo_resumen_validacion, campos_resumen, resumenes_validacion)

    print("\n=== BENCHMARK CSV FINALIZADO ===")
    print(f"Resultados guardados en:")
    print(archivo_resultados)
    print(f"\nErrores detectados guardados en:")
    print(archivo_errores)
    print(f"\nResumen de validación guardado en:")
    print(archivo_resumen_validacion)


if __name__ == "__main__":
    main()