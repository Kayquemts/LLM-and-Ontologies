import os
import pandas as pd
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")


# ================================================================
# CONFIGURAÇÕES
# ================================================================
BASE = r"CACHET-CADB"
SIG_DIR = os.path.join(BASE, "annotations")
CSV = r"generated-csv"

FS = 1024  # taxa de amostragem do ECG (para converter Start/End → segundos)

# ================================================================
# LER O CSV QUE VOCÊ GEROU ANTES
# ================================================================
def carregar_pacientes_com_arritmia():
    path = os.path.join(CSV, "pacientes_com_arritmias.csv")
    df = pd.read_csv(path)
    print(f"📌 Carregado pacientes_com_arritmias.csv: {len(df)} linhas")
    return df

# ================================================================
# LER O CONTEXT.XLSX DE CADA SESSÃO
# ================================================================
def carregar_context(patient, session, segment):
    context_path = os.path.join(SIG_DIR, patient, session, segment, "context.xlsx")
    
    if not os.path.exists(context_path):
        print(f"⚠ context.xlsx não encontrado: {context_path}")
        return None
    
    try:
        df = pd.read_excel(context_path)
        return df
    except Exception as e:
        print(f"❌ Erro ao ler {context_path}: {e}")
        return None

# ================================================================
# FILTRAR OS INTERVALOS CORRESPONDENTES AO START–END DO ECG
# ================================================================
def extrair_intervalo_context(df_context, start, end):
    # converter start/end (amostras) → segundos
    start_s = start / FS
    end_s = end / FS

    # a coluna Time rel [s] contém o tempo relativo
    df_filtered = df_context[
        (df_context["Time rel [s]"] >= start_s) & 
        (df_context["Time rel [s]"] < end_s)
    ]
    
    return df_filtered

# ================================================================
# PROCESSAR TODAS AS ARRTIMIAS E RESGATAR CONTEXTO
# ================================================================
def gerar_contexto_para_arritmias(df_arritmias):
    registros = []

    for _, row in df_arritmias.iterrows():
        patient = row["patient"]
        session = row["session"]
        segment = row["segment"]
        start = row["Start"]
        end = row["End"]
        gende = row["gender"]
        weight = row["weight"]
        age = row["age"]
        height = row["height"]
        classe = row["Class"]

        print(f"🔍 Processando {patient} | {session} | {segment} | {start}-{end} | Class {classe}")

        # Carregar context.xlsx correspondente
        df_context = carregar_context(patient, session, segment)
        if df_context is None:
            continue

        # Extrair apenas o intervalo correto
        df_intervalo = extrair_intervalo_context(df_context, start, end)
        if df_intervalo.empty:
            continue

        # Colocar colunas extras (paciente, classe etc.)
        df_intervalo = df_intervalo.copy()
        df_intervalo["patient"] = patient
        df_intervalo["session"] = session
        df_intervalo["segment"] = segment
        df_intervalo["Start"] = start
        df_intervalo["End"] = end
        df_intervalo["gender"] = gende
        df_intervalo["weight"] = weight 
        df_intervalo["age"] = age
        df_intervalo["height"] = height
        df_intervalo["Class"] = classe

        registros.append(df_intervalo)

    # Concatenar tudo
    if registros:
        df_final = pd.concat(registros, ignore_index=True)
        out_path = os.path.join(CSV, "contexto_das_arritmias.csv")
        df_final.to_csv(out_path, index=False)
        print(f"\n✅ CSV salvo: {out_path}")
        print(df_final.head())
        return df_final
    else:
        print("❌ Nenhum intervalo encontrado.")
        return None

# ================================================================
# EXECUÇÃO PRINCIPAL
# ================================================================
if __name__ == "__main__":
    df_arr = carregar_pacientes_com_arritmia()
    gerar_contexto_para_arritmias(df_arr)
