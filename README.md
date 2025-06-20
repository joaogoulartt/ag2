# AG2 – Engenharias de Computação e Software

Este projeto implementa um modelo de Machine Learning para análise de risco de crédito usando dados do banco MySQL para a matéria de AG02 do Inatel.

## 🚀 Início Rápido

### 1. Interface Web (Recomendado) 🌐
A aplicação possui uma interface web moderna e intuitiva desenvolvida com Streamlit:

```bash
# Instalar dependências
pip install -r requirements.txt

# Executar a aplicação web
streamlit run app.py
# ou usar o script helper
./run_app.sh
```

**Funcionalidades da Interface Web:**
- 🏠 **Dashboard**: Visualização de métricas e performance do modelo
- 🔮 **Nova Predição**: Interface para análise de novos clientes
- 📊 **Dados**: Exploração e visualização dos dados
- ⚙️ **Configurações**: Gerenciamento do sistema

### 2. Executar apenas o Modelo (Terminal)
```bash
python main.py
```

## 📋 Configuração Manual

### 1. Instalar Dependências

```bash
pip install -r requirements.txt
```

### 2. Configurar Credenciais

Copie o arquivo `.env.example` para `.env` e configure suas credenciais:

```bash
cp .env.example .env
```

Edite o arquivo `.env` com suas credenciais reais

## 🖥️ Interface Web

A aplicação Streamlit possui uma interface moderna e intuitiva com:

### 🧪 **Casos de Teste**
- Perfis pré-definidos de clientes
- Testes rápidos do modelo
- Visualização imediata dos resultados

### 👤 **Análise Personalizada**
- Formulário completo para novos clientes
- Entrada de dados com validação
- Predição em tempo real

### 📈 **Análise do Modelo**
- Métricas de performance
- Gráfico de importância das features
- Matriz de confusão interativa

### 📋 **Documentação**
- Informações sobre o German Credit Dataset
- Descrição detalhada das variáveis
- Contexto do problema de negócio

## 📈 Funcionalidades

- ✅ Interface web moderna com Streamlit
- ✅ Carregamento automático de dados do MySQL
- ✅ Treinamento de modelo de Árvore de Decisão
- ✅ Avaliação de performance do modelo
- ✅ Análise visual com gráficos Plotly
- ✅ Casos de teste pré-configurados
- ✅ Formulário interativo para novos clientes
- ✅ Sistema de logging e monitoramento
- ✅ Configuração segura via variáveis de ambiente

## 🎯 Como Usar

1. **Configure o banco MySQL** com suas credenciais no arquivo `.env`
2. **Execute a aplicação**: `streamlit run app.py`
3. **Acesse** `http://localhost:8501` no seu navegador
4. **Teste** os casos pré-definidos ou crie análises personalizadas

## 📱 Screenshots

A interface possui:
- **Design moderno** com gradientes e cores visuais
- **Métricas em tempo real** do modelo
- **Gráficos interativos** para análise
- **Formulários intuitivos** para entrada de dados
- **Resultados visuais** das predições
