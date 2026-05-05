#!/bin/bash

# Script para testear todos los componentes del Ejercicio 1 (Perceptrón)

# Colores para la salida
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Activar entorno virtual si existe
if [ -d "venv" ]; then
    source venv/bin/activate
fi

echo -e "${BLUE}=== Iniciando Tests del Ejercicio 1 ===${NC}"

# 1. Test de compuertas lógicas (AND / OR)
echo -e "\n${BLUE}[1/4] Testeando Perceptrón Simple (AND)${NC}"
python3 algorithmEjercicio1/perceptron_and.py
if [ $? -eq 0 ]; then
    echo -e "${GREEN}AND Perceptron: OK${NC}"
else
    echo -e "${RED}AND Perceptron: FALLÓ${NC}"
fi

echo -e "\n${BLUE}[2/4] Testeando Perceptrón Simple (OR)${NC}"
python3 algorithmEjercicio1/perceptron_or.py
if [ $? -eq 0 ]; then
    echo -e "${GREEN}OR Perceptron: OK${NC}"
else
    echo -e "${RED}OR Perceptron: FALLÓ${NC}"
fi

# 2. Test de Fraude (Knowledge Distillation)
echo -e "\n${BLUE}[3/4] Ejecutando Perceptrón para Detección de Fraude (Ajuste Inicial)${NC}"
python3 algorithmEjercicio1/perceptron_fraud.py
if [ $? -eq 0 ]; then
    echo -e "${GREEN}Perceptron Fraud: OK${NC}"
    echo -e "Gráficos generados: plot_mse_epochs.png"
else
    echo -e "${RED}Perceptron Fraud: FALLÓ${NC}"
fi

# 3. Test de Generalización y Análisis Económico
echo -e "\n${BLUE}[4/4] Ejecutando Estudio de Generalización y Análisis Económico${NC}"
python3 algorithmEjercicio1/generalizacion_fraud.py
if [ $? -eq 0 ]; then
    echo -e "${GREEN}Generalización y Economía: OK${NC}"
    echo -e "Gráficos generados: plot_umbral.png, plot_costo_economico.png"
else
    echo -e "${RED}Generalización y Economía: FALLÓ${NC}"
fi

echo -e "\n${BLUE}=== Tests Finalizados ===${NC}"
