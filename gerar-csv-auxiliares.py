import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import detrend
import xml.etree.ElementTree as ET

# ================================================================
# CONFIGURAÇÕES DO DATASET
# ================================================================

BASE = r"CACHET-CADB"
CSV = r"generated-csv"
ANN_DIR = os.path.join(BASE, "annotations")
SIG_DIR = os.path.join(BASE, "signal")

FS = 1024              # taxa de amostragem
SCALE = 1/1000         # conversão para mV

CLASSES = {
    1: "AF (Atrial Fibrillation)",
    2: "NSR (Normal Sinus Rhythm)",
    3: "Noise",
    4: "Others"
}

# ================================================================
# 1) PARSE DO UNISENS.XML → EXTRAIR INFORMAÇÕES DO PACIENTE
# ================================================================

def ler_unisens(unisens_path):
    if not os.path.exists(unisens_path):
        return None

    tree = ET.parse(unisens_path)
    root = tree.getroot()

    # Detectar namespace
    if root.tag.startswith("{"):
        uri = root.tag.split("}")[0][1:]
        ns = {"u": uri}
        find = lambda path: root.find(path, ns)
        findall = lambda path: root.findall(path, ns)
    else:
        ns = {}
        find = lambda path: root.find(path)
        findall = lambda path: root.findall(path)

    info = {}

    # --- customAttributes ---
    custom_attrs = find("u:customAttributes" if ns else "customAttributes")
    if custom_attrs is not None:
        for attr in custom_attrs.findall("u:customAttribute" if ns else "customAttribute"):
            info[attr.attrib.get("key")] = attr.attrib.get("value")

    # --- signals ---
    sinais = []
    for signal in findall("u:signalEntry" if ns else "signalEntry"):
        id_arq = signal.attrib.get("id")
        canais = [c.attrib.get("name") for c in signal.findall("u:channel" if ns else "channel")]
        sinais.append(f"{id_arq} ({', '.join(canais)})")

    info["sinais"] = "; ".join(sinais)
    info["unisens_path"] = unisens_path
    return info

def gerar_csv_unisens():
    print("\n📌 Lendo estrutura unisens.xml…")
    dados = []

    for paciente in sorted(p for p in os.listdir(SIG_DIR) if p.startswith("P")):
        pac_dir = os.path.join(SIG_DIR, paciente)

        for dispositivo in sorted(os.listdir(pac_dir)):
            disp_dir = os.path.join(pac_dir, dispositivo)

            for sessao in sorted(os.listdir(disp_dir)):
                unisens_path = os.path.join(disp_dir, sessao, "unisens.xml")
                info = ler_unisens(unisens_path)

                if info:
                    info["patient"] = paciente
                    info["session"] = dispositivo
                    info["segment"] = sessao
                    dados.append(info)

    df = pd.DataFrame(dados)
    out_path = os.path.join(CSV, "informacoes_pacientes.csv")
    df.to_csv(out_path, index=False)
    print(f"✅ CSV gerado: {out_path}\n")
    print(df.head(1))

    return df

# ================================================================
# 2) LER TODAS AS ANOTAÇÕES DO DATASET
# ================================================================

def carregar_anotacoes():
    print("🔍 Procurando annotation.csv…")
    linhas = []

    for dirpath, _, filenames in os.walk(ANN_DIR):
        if "annotation.csv" in filenames:
            csv_path = os.path.join(dirpath, "annotation.csv")

            parts = csv_path.split(os.sep)
            patient = parts[-4]
            session = parts[-3]
            segment = parts[-2]

            try:
                df = pd.read_csv(csv_path)
                required = {"Start", "End", "Class"}
                
                if not required.issubset(df.columns):
                    print(f"⚠ Ignorado (colunas faltando): {csv_path}")
                    continue

                for _, row in df.iterrows():
                    linhas.append({
                        "patient": patient,
                        "session": session,
                        "segment": segment,
                        "Start": int(row["Start"]),
                        "End": int(row["End"]),
                        "Class": int(row["Class"]),
                    })

            except Exception as e:
                continue

    df_final = pd.DataFrame(linhas)
    print(f"📌 Total de anotações carregadas: {len(df_final)}\n")

    out_csv = os.path.join(CSV, "todas_anotacoes.csv")
    df_final.to_csv(out_csv, index=False)
    print(f"📁 CSV salvo: {out_csv}")
    print(df_final.head(1))

    return df_final

# ================================================================
# 3) CARREGAR UM ECG.BIN
# ================================================================

def carregar_sinal(patient, session, segment):
    ecg_path = os.path.join(SIG_DIR, patient, session, segment, "ecg.bin")

    if not os.path.exists(ecg_path):
        print(f"❌ Sinal não encontrado: {ecg_path}")
        return None
    
    ecg = np.fromfile(ecg_path, dtype=np.int16) * SCALE
    return ecg

# ================================================================
# 4) PLOTAR AS QUATRO CLASSES (Figura estilo artigo)
# ================================================================

def plotar_amostras(df):
    fig, axes = plt.subplots(4, 1, figsize=(16, 10), sharex=True)
    fig.suptitle("Amostras de ECG — 4 Classes (CACHET-CADB)", fontsize=15)

    for idx, (class_id, class_name) in enumerate(CLASSES.items()):
        ax = axes[idx]
        subset = df[df["Class"] == class_id]

        if subset.empty:
            ax.text(0.1, 0.5, f"{class_name}\n(não encontrado)", fontsize=12)
            ax.set_axis_off()
            continue

        row = subset.iloc[0]
        ecg = carregar_sinal(row["patient"], row["session"], row["segment"])

        if ecg is None:
            ax.text(0.1, 0.5, f"{class_name}\n(erro ao carregar)", fontsize=12)
            ax.set_axis_off()
            continue

        segment = detrend(ecg[row["Start"]:row["End"]])
        t = np.arange(len(segment)) / FS

        ax.plot(t, segment, linewidth=1)
        ax.set_ylabel("mV")
        ax.set_title(class_name, loc="left")
        ax.grid(True, linestyle="--", alpha=0.5)

    axes[-1].set_xlabel("Tempo [s]")
    plt.tight_layout()
    plt.show()

def juntar_info_paciente_e_arritmia(df_arritmia, df_paciente):

    # As chaves que identificam um registro único no dataset
    keys = ["patient", "session", "segment"]

    # Merge 1→N: um paciente pode ter várias arritmias
    df_merged = df_arritmia.merge(df_paciente, on=keys, how="left")

    # Remover colunas que nunca são úteis no CSV final
    drop_cols = [
        "sensorLocation", 
        "sensorVersion",
        "sensorType",
        "personId",
        "sinais",
        "unisens_path"
    ]
    df_merged = df_merged.drop(columns=[c for c in drop_cols if c in df_merged.columns])

    # Tratar dados vazios importantes
    df_merged = df_merged.dropna(subset=["gender", "age", "weight", "height"], how="any")

    out_path = os.path.join(CSV, "pacientes_com_arritmias.csv")
    df_merged.to_csv(out_path, index=False)

    print(f"📁 Base final criada: {out_path}")
    print(df_merged.head(3))

    return df_merged


# ================================================================
# EXECUÇÃO PRINCIPAL
# ================================================================

if __name__ == "__main__":
    print("\n==================== INÍCIO DO PROCESSAMENTO ====================\n")

    df_info_patient = gerar_csv_unisens()
    df_info_arrhythmia = carregar_anotacoes()
    df_base_final = juntar_info_paciente_e_arritmia(df_info_arrhythmia, df_info_patient)
    plotar_amostras(df_info_arrhythmia)

    print("\n==================== PROCESSO CONCLUÍDO ====================\n")
