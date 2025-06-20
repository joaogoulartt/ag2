#!/bin/bash

# Script para executar a aplicação Streamlit
echo "🚀 Iniciando aplicação AG2 - Análise de Risco de Crédito..."
echo "📝 Certifique-se de que:"
echo "   1. O MySQL está rodando"
echo "   2. As credenciais estão configuradas no arquivo .env"
echo "   3. A tabela 'germancredit' existe no banco"
echo ""
echo "🌐 A aplicação será aberta no navegador em http://localhost:8501"
echo ""

# Navegar para o diretório raiz do projeto
cd "$(dirname "$0")/.."

# Adicionar o diretório src ao PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Executar a aplicação Streamlit
streamlit run src/app/app.py
