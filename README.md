\
# Leitor de Gabarito — OMR (Cartão-Resposta)

Este projeto lê fotos do cartão-resposta (modelo fixo) e gera uma planilha `.xlsx` com as respostas.

## Como rodar (dev)

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/mac:
source .venv/bin/activate

pip install -r requirements.txt
python app.py
```

## Como gerar executável (Windows)

### Opção A — PyInstaller (rápida)
```bash
pip install pyinstaller
pyinstaller --onefile --windowed --name LeitorGabarito app.py
```

> Obs.: `--onefile` empacota em um único `.exe`, mas ele precisa “descompactar” ao executar, então pode ser mais lento
na primeira abertura. (Isso é esperado no modo onefile.)

### Opção B — pyside6-deploy (multiplataforma)
O `pyside6-deploy` (Qt for Python) usa o Nuitka por baixo e gera `.exe` (Windows), `.bin` (Linux) ou `.app` (macOS).

```bash
pip install pyside6
pyside6-deploy
```

Depois a gente ajusta o arquivo de configuração do deploy para ficar 100% “um clique e pronto”.

## Ajustes finos
Se aparecer muito `blank` ou `MULTI` em fotos reais, a gente ajusta:
- os parâmetros de recorte (onde começa a área das 90 questões),
- `blank_delta` e `ambiguous_delta`,
- e, se necessário, um “filtro” para ignorar círculos fora da região das bolhas.
