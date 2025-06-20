# AG2 – Análise de Risco de Crédito
*Engenharias de Computação e Software - INATEL*

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![MySQL](https://img.shields.io/badge/MySQL-8.0+-orange.svg)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3+-green.svg)

Este projeto implementa um sistema de análise de risco de crédito utilizando Machine Learning, com interface web interativa e conexão com banco de dados MySQL. Desenvolvido para a disciplina AG02 do INATEL.

## 📁 Estrutura do Projeto

```
ag02/
├── src/
│   ├── app/
│   │   └── app.py              # Interface web Streamlit
│   ├── models/
│   │   └── model.py            # Modelo de Machine Learning
│   └── utils/
│       └── translations.py     # Traduções e mapeamentos
├── scripts/
│   └── run_app.sh             # Script de execução
├── .env.example               # Exemplo de configuração
├── requirements.txt           # Dependências Python
└── README.md                  # Este arquivo
```

## 🚀 Início Rápido

### Pré-requisitos
- Python 3.8 ou superior
- MySQL Server 8.0+
- Git

### 1. Clonagem e Configuração Inicial

```bash
# Clonar o repositório
git clone <url-do-repositorio>
cd ag02

# Criar ambiente virtual (recomendado)
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# ou
.venv\Scripts\activate     # Windows

# Instalar dependências
pip install -r requirements.txt
```

### 2. Configuração do Banco de Dados

```bash
# Copiar arquivo de configuração
cp .env.example .env

# Editar arquivo .env com suas credenciais
nano .env  # ou vim .env
```

### 3. Executar a Aplicação

#### Opção 1: Interface Web (Recomendado) 🌐
```bash
# Usar script auxiliar
chmod +x scripts/run_app.sh
./scripts/run_app.sh

# Ou executar diretamente
streamlit run src/app/app.py
```

#### Opção 2: Apenas o Modelo (Terminal)
```bash
python src/models/model.py
```

## 🌐 Interface Web

### 📊 **Páginas Principais**

#### 🏠 **Dashboard**
- Métricas de performance do modelo em tempo real
- Gráficos de análise de dados
- Estatísticas gerais do dataset
- Indicadores de saúde do sistema

#### 🔮 **Nova Predição**
- **Casos de Teste**: Perfis pré-definidos para teste rápido
- **Análise Personalizada**: Formulário completo para novos clientes
- **Resultados Visuais**: Predições com explicabilidade
- **Validação de Entrada**: Verificação automática dos dados

#### 📈 **Análise do Modelo**
- Matriz de confusão interativa
- Gráfico de importância das features
- Métricas detalhadas (Accuracy, Precision, Recall, F1-Score)
- Curvas ROC e análise de performance

## ⚙️ Tecnologias Utilizadas

### **Backend & Machine Learning**
- **Python 3.8+**: Linguagem principal
- **Scikit-learn**: Algoritmos de Machine Learning
- **Pandas**: Manipulação e análise de dados
- **MySQL Connector**: Conexão com banco de dados
- **SQLAlchemy**: ORM para Python
- **Python-dotenv**: Gerenciamento de variáveis de ambiente

### **Frontend & Visualização**
- **Streamlit**: Framework para aplicações web
- **Plotly**: Gráficos interativos
- **CSS Customizado**: Estilização da interface

### **Banco de Dados**
- **MySQL 8.0+**: Sistema de gerenciamento de banco
- **German Credit Dataset**: Dataset para análise de crédito


## 🎯 Guia de Uso

1. Configure o ambiente virtual Python
2. Instale as dependências do `requirements.txt`
3. Configure as credenciais do MySQL no `.env`
4. Execute a aplicação com `streamlit run src/app/app.py` ou `./scripts/run_app.sh`

## 🛠️ Desenvolvimento

### **Estrutura de Arquivos**
- `src/app/app.py`: Interface principal Streamlit
- `src/models/model.py`: Lógica do modelo ML
- `src/utils/translations.py`: Sistema de traduções
- `scripts/run_app.sh`: Script de execução automatizada

## 📊 Dataset German Credit

O projeto utiliza o German Credit Dataset que contém:
- **1000 registros** de clientes
- **20 atributos** (características financeiras e pessoais)
- **Variável target**: Risco de crédito (Bom/Ruim)
- **Aplicação real**: Análise de concessão de crédito bancário

---

**Desenvolvido por:** João Victor de Oliveira Goulart Costa
