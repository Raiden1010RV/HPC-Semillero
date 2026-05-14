import duckdb
import pandas as pd

# Ruta a los Parquet
parquet_files = r"C:\Users\Raiden AsusROG\OneDrive\Documentos\Archivo Académico\Politécnico Grancolombiano - Ing. de Sistemas\8 Semestre\Semillero HPC\HPC\data\*.parquet"

# Conexión
con = duckdb.connect()

print("\n=== COLUMNAS DEL DATASET ===\n")

# Obtener columnas como DataFrame
df_columns = con.execute(f"""
DESCRIBE SELECT * 
FROM read_parquet('{parquet_files}')
""").df()

print(df_columns)

print("\n=== PRIMEROS 10 REGISTROS ===\n")

# Obtener datos como DataFrame
df = con.execute(f"""
SELECT *
FROM read_parquet('{parquet_files}')
LIMIT 10
""").df()

print(df)

df.to_csv("preview.csv", index=False)

con.close()