import streamlit as st
import os
import pandas as pd
from google import genai
from google.genai import types
import ast 

# ==============================================================================
# 0. CONFIGURAÇÃO E INICIALIZAÇÃO DA API
# ==============================================================================
try:
    # A API_KEY é lida de forma segura a partir do Streamlit Secrets
    API_KEY = st.secrets["API_KEY"] 
    client = genai.Client(api_key=API_KEY)
    MODEL = "gemini-2.5-flash" 
    # st.toast("✅ Configuração da API bem-sucedida.") # Opcional
except Exception as e:
    # Mostra o erro de forma mais amigável, sem expor a chave
    st.error(f"❌ Erro ao configurar a API. Por favor, verifique se a chave está configurada corretamente nos Secrets do Streamlit Cloud.")
    st.info("Certifique-se de que a chave está no formato: API_KEY = \"SUA_CHAVE_AQUI\"")
    client = None
    st.stop() # Interrompe a execução para evitar erros adicionais

# Configurações globais
LARGURA = 80 # Largura não é mais crucial, mas mantemos o conceito.
NUM_QUESTOES = 10 # Mantendo as 10 questões, conforme solicitado.

# Inicialização do Streamlit Session State (Estado da Aplicação)
if 'placar' not in st.session_state:
    st.session_state.placar = []
if 'questoes_geradas' not in st.session_state:
    st.session_state.questoes_geradas = []
if 'indice_questao' not in st.session_state:
    st.session_state.indice_questao = 0
if 'prova_iniciada' not in st.session_state:
    st.session_state.prova_iniciada = False

# ==============================================================================
# 1. PROMPTS DINÂMICOS
# ==============================================================================

def construir_prompt_avaliador(rigor_nivel):
    deducao_por_erro = 0.05 + (rigor_nivel / 10) * 0.15
    rigor_conteudo_desc = "um critério de correção focado na ideia principal e menos rigoroso no conteúdo." if rigor_nivel <= 5 else "um critério de correção rigoroso, exigindo precisão total no conteúdo."

    return f"""
Você é o Avaliador Crítico de Prova. Sua única função é receber uma resposta digitada e, com base em critérios de precisão, profundidade e coerência com o material de estudo:
1. Fazer uma crítica breve e objetiva (máximo 3 frases) sobre a resposta.
2. Corrigir erros de Português e ortografia na resposta digitada. Para cada erro encontrado, retire **{deducao_por_erro:.2f} ponto** da nota final.
3. Atribuir uma nota final estrita de 0 a 10, considerando a profundidade do conteúdo E a dedução dos erros de escrita. Utilize {rigor_conteudo_desc}.
4. Gerar uma resposta sucinta, mas completa, que seria a resposta esperada para a pergunta.
5. Formatar sua saída APENAS da seguinte maneira:
   CRITICA: [Sua crítica aqui, incluindo menção explícita aos erros de escrita e à dedução.]
   NOTA: [A nota numérica final atribuída após todas as deduções]
   RESPOSTA_ESPERADA: [A resposta completa e sucinta]
"""

def construir_prompt_gerador(dificuldade_nivel):
    dificuldade_desc = ""
    if dificuldade_nivel <= 3:
        dificuldade_desc = "perguntas FÁCEIS e diretas."
    elif dificuldade_nivel <= 7:
        dificuldade_desc = "perguntas de dificuldade MODERADA e específicas."
    else:
        dificuldade_desc = "perguntas AVANÇADAS, específicas e que exijam análise crítica."
        
    return f"""
Você é um gerador de questões de prova. Sua função é ler o conteúdo de estudo fornecido e criar **EXATAMENTE {NUM_QUESTOES} questões abertas** baseadas no material. Crie {dificuldade_desc}
É obrigatório que sua saída seja APENAS e ESTREITAMENTE uma lista Python (list of strings). NÃO inclua nenhum texto introdutório, cabeçalho, explicação, ou formatação de código Markdown.
Formato Exigido: ["Questão 1 aqui.", "Questão 2 aqui.", ..., "Questão {NUM_QUESTOES} aqui."]
"""

# ==============================================================================
# 2. FUNÇÕES DE FLUXO (ADAPTADAS PARA STREAMLIT)
# ==============================================================================

# Função cacheada: garante que as questões só sejam geradas uma vez
@st.cache_data(show_spinner="⏳ Gerando questões com o Gemini...")
def gerar_questoes_do_material(_uploaded_files, dificuldade_nivel, file_names):
    if not client: return []
    
    gemini_files = []
    
    try:
        # 1. Envia TODOS os arquivos para a API do Gemini
        for uploaded_file in _uploaded_files:
            # Salva o arquivo temporariamente (necessário para o genai.Client.files.upload)
            with open(uploaded_file.name, "wb") as f:
                f.write(uploaded_file.getbuffer())
                
            gemini_file = client.files.upload(file=uploaded_file.name)
            gemini_files.append(gemini_file)

        if not gemini_files:
            return []
            
        # 2. Chama o modelo para gerar as questões
        contents = [
            f"Com base no conteúdo de todos estes arquivos ({', '.join(file_names)}), gere **exatamente {NUM_QUESTOES} questões abertas**, estritamente como uma lista Python. Use apenas o conteúdo dos anexos."
        ]
        contents.extend(gemini_files) 
        
        response = client.models.generate_content(
            model=MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=construir_prompt_gerador(dificuldade_nivel)
            ),
        )
        
        # 3. Processamento de Resposta (Mantendo a lógica robusta de correção)
        questoes_raw = response.text.strip()
        questoes_list = []

        if questoes_raw.startswith("```"):
            questoes_raw = questoes_raw.strip('`').replace('python\n', '', 1).strip()
        
        try:
            questoes_list = ast.literal_eval(questoes_raw)
        except Exception:
            # Fallback para extração
            lines = [line.strip() for line in questoes_raw.split('\n') if line.strip()]
            for line in lines:
                if line.startswith(('[', '"', "'")): continue
                questoes_list.append(line)
            questoes_list = questoes_list[:NUM_QUESTOES]
            
        if not isinstance(questoes_list, list) or len(questoes_list) != NUM_QUESTOES:
             st.warning(f"O modelo gerou um número incorreto de questões ({len(questoes_list)}). Usando o que foi gerado.")
             
        return questoes_list[:NUM_QUESTOES]
        
    except Exception as e:
        st.error(f"❌ Erro durante a geração de questões: {e}")
        return []
        
    finally:
        # 4. Limpeza (deleta os arquivos da API)
        for gem_file in gemini_files:
            try:
                client.files.delete(name=gem_file.name)
            except:
                pass 
        # Limpa arquivos temporários
        for uploaded_file in _uploaded_files:
            if os.path.exists(uploaded_file.name):
                os.remove(uploaded_file.name)


def avaliar_resposta(questao, resposta_digitada, rigor_nivel):
    """Chama a API do Gemini para avaliar e pontuar a resposta."""
    prompt = f"""
    Questão: "{questao}"
    Resposta Digitada: "{resposta_digitada}"
    
    Avalie a resposta digitada para a questão e gere a resposta esperada.
    """
    
    critica = "Erro na API/Formatação durante a avaliação."
    nota = 0.0
    resposta_esperada = "Não foi possível gerar a resposta esperada devido a um erro na API."

    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=construir_prompt_avaliador(rigor_nivel)
            ),
        )
        
        avaliacao_raw = response.text.strip()
        
        # Extração da nota, crítica e resposta esperada
        critica = avaliacao_raw.split("CRITICA:")[1].split("NOTA:")[0].strip()
        nota = float(avaliacao_raw.split("NOTA:")[1].split("RESPOSTA_ESPERADA:")[0].strip())
        resposta_esperada = avaliacao_raw.split("RESPOSTA_ESPERADA:")[1].strip()
        
        return critica, nota, resposta_esperada

    except Exception as e:
        st.error(f"Erro na avaliação: {e}")
        return critica, nota, resposta_esperada

# ==============================================================================
# 3. INTERFACE STREAMLIT
# ==============================================================================

st.set_page_config(layout="wide", page_title="📝 Prova Dinâmica Gemini")

# --- CABEÇALHO COM LOGO NO DOBRO DO TAMANHO E AUTORIA EM DESTAQUE ---
col1, col2 = st.columns([1, 4])
with col1:
    # AJUSTE: Aumentamos a largura da imagem para 300 (dobro de 150)
    try:
        st.image("zumtec_logo.png", width=300) 
    except FileNotFoundError:
        st.warning("Logo 'zumtec_logo.png' não encontrado no repositório.")
with col2:
    st.title("📝 Gerador e Avaliador de Provas (Gemini)")
    st.caption("Centralize o controle da dificuldade, rigor e aplicação de provas para seus alunos.")
    st.markdown("Criado por **Dr. Gustavo Feitosa** (Zumtec Digital Health Solutions)")
st.markdown("---") # Separador para o cabeçalho

# --- BARRA LATERAL PARA CONFIGURAÇÃO ---
with st.sidebar:
    st.header("⚙️ Configurações da Prova")
    
    # Sliders substituem o input() do Colab
    dificuldade = st.slider(
        "Nível de DIFICULDADE das perguntas", 
        min_value=0, max_value=10, value=5, 
        help="0=Fácil (Direto do texto), 10=Difícil (Exige análise crítica)."
    )
    rigor = st.slider(
        "Nível de RIGOR de correção", 
        min_value=0, max_value=10, value=5, 
        help="0=Flexível (Foca na ideia principal), 10=Rigoroso (Exige precisão total e pune erros gramaticais)."
    )

    st.subheader("📚 Upload do Material")
    uploaded_files = st.file_uploader(
        "Selecione um ou mais arquivos de estudo (PDF, TXT, DOCX, etc.)",
        type=['pdf', 'txt', 'docx', 'pptx', 'jpg', 'jpeg', 'png'], 
        accept_multiple_files=True
    )
    
    # Botão para gerar as questões
    if st.button("▶️ Gerar Questões"):
        if uploaded_files:
            file_names = [f.name for f in uploaded_files]
            
            # Chama a função de geração (cached)
            questoes = gerar_questoes_do_material(uploaded_files, dificuldade, file_names)
            
            if questoes:
                st.session_state.questoes_geradas = questoes
                st.session_state.indice_questao = 0
                st.session_state.placar = [] # Limpa resultados anteriores
                st.session_state.prova_iniciada = True
                st.success(f"✅ {NUM_QUESTOES} Questões geradas com sucesso!")
            else:
                st.error("Falha ao gerar questões. Verifique o conteúdo dos arquivos.")
        else:
            st.warning("Por favor, faça o upload dos materiais de estudo.")

# --- LÓGICA DE CORREÇÃO E AVANÇO ---
def corrigir_e_avancar():
    indice = st.session_state.indice_questao
    questao_atual = st.session_state.questoes_geradas[indice]
    
    # Pega a resposta do text_area usando a chave
    resposta_digitada = st.session_state[f"resposta_q_{indice}"] 
    
    if not resposta_digitada.strip():
        st.error("Sua resposta está vazia.")
        return # Não avança se a resposta for vazia

    with st.spinner("🔎 Avaliando a resposta..."):
        # Avalia a resposta usando o nível de rigor da sidebar
        critica, nota, resposta_esperada = avaliar_resposta(
            questao_atual, resposta_digitada, rigor
        )
    
    # Armazena o resultado no placar
    st.session_state.placar.append({
        "Questão": f"Q{indice + 1}",
        "Conteúdo": questao_atual,
        "Resposta_Aluno": resposta_digitada,
        "Critica_Avaliador": critica,
        "Resposta_Esperada": resposta_esperada,
        "Nota": nota
    })
    
    # --- Exibe o Feedback Imediato ---
    st.subheader(f"Feedback da Questão {indice + 1}")
    
    if nota >= 7.0:
        st.balloons()
        st.success(f"✨ NOTA FINAL: {nota:.1f}/10 - Ótimo trabalho!")
    elif nota >= 5.0:
        st.warning(f"🟡 NOTA FINAL: {nota:.1f}/10 - Você está quase lá, revise a crítica abaixo.")
    else:
        st.error(f"🔴 NOTA FINAL: {nota:.1f}/10 - Revise o conteúdo.")

    with st.expander("Ver Crítica e Resposta Esperada"):
        st.markdown(f"**Crítica:** \n\n {critica}")
        if nota < 7.0:
            st.markdown(f"**Oportunidade de Aprendizado (Resposta Esperada):** \n\n {resposta_esperada}")
            
    # Avança para a próxima questão
    st.session_state.indice_questao += 1
    # st.experimental_rerun() # Não é mais necessário aqui


# --- ÁREA PRINCIPAL DA PROVA ---

if st.session_state.prova_iniciada and st.session_state.indice_questao < NUM_QUESTOES:
    
    # Variáveis da questão atual
    indice = st.session_state.indice_questao
    questao_atual = st.session_state.questoes_geradas[indice]
    
    st.markdown(f"---")
    st.header(f"➡️ QUESTÃO {indice + 1} de {NUM_QUESTOES}")
    st.markdown(f"## **{questao_atual}**")
    st.markdown(f"---")

    # Área de resposta
    # Adicionamos uma chave única e o on_change para acionar a correção
    st.text_area(
        "✍️ DIGITE SUA RESPOSTA AQUI:", 
        height=200, 
        key=f"resposta_q_{indice}" 
    )
    
    st.button("Corrigir e Próxima Questão", on_click=corrigir_e_avancar)


# --- RELATÓRIO FINAL ---
elif st.session_state.prova_iniciada and st.session_state.indice_questao >= NUM_QUESTOES:
    st.header("🏁 Prova Finalizada!")
    
    df_placar = pd.DataFrame(st.session_state.placar)
    nota_media = df_placar['Nota'].mean()
    
    st.markdown(f"### Média Final: **{nota_media:.2f}/10**")
    
    st.subheader("📋 Relatório Detalhado")
    df_summary = df_placar[['Questão', 'Nota', 'Resposta_Aluno', 'Critica_Avaliador']]
    st.dataframe(df_summary, use_container_width=True, hide_index=True)
    
    st.markdown("---")

    # Função para gerar o arquivo Excel em memória (Buffer)
    def to_excel(df):
        # Usamos BytesIO para criar o arquivo em memória
        import io
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Resultados Completos')
        output.seek(0) # Volta para o início do buffer
        return output

    # Botão de download
    excel_buffer = to_excel(df_placar)
    st.download_button(
        label="⬇️ Baixar Relatório Completo (Excel)",
        data=excel_buffer,
        file_name='Relatorio_Prova_Gemini.xlsx',
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        help="Baixe todas as perguntas, respostas, notas e críticas."
    )

# --- FLUXO INICIAL ---
else:
    st.info("⬅️ Por favor, use a barra lateral para configurar o nível de dificuldade, rigor e fazer o upload dos materiais de estudo para iniciar a prova.")
