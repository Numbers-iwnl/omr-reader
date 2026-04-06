# Leitor de Gabarito (Cartão-Resposta) — v3

Lê automaticamente fotos de cartões-resposta com 90 questões (6 painéis × 15 questões × 5 opções A-E) e exporta um `.xlsx` com as respostas.

## Como usar

```
pip install -r requirements.txt
python app.py
```

Selecione a foto do cartão. O `.xlsx` é salvo na mesma pasta da imagem.

## O que mudou na v3 (vs v2)

| Problema (v2) | Solução (v3) |
|---|---|
| Warp falhava em 9/10 fotos (buscava contorno maior = fundo) | Multi-estratégia: 3 passes Canny com parâmetros diferentes; fallback para resize sem warp |
| Template fixo de coordenadas → qualquer erro de warp quebrava tudo | **Detecção dos 6 painéis como âncoras independentes**: cada caixa de painel é detectada separadamente; as coordenadas das bolhas são normalizadas dentro da caixa |
| `rr=4px` de scoring → pouca separação entre marcado/vazio | `rr=6px` (fill_frac 0.40→0.60): muito mais sinal nas bolhas marcadas |
| Threshold fixo (blank_min_score=3.0) | **Threshold adaptativo** por painel: percentil 65 dos scores reais |
| Ratio 2º/1º → falsos "ambiguous" por letras impressas | **Baseline por questão + ratio 3º/1º**: neutraliza o "ruído" das letras A-E impressas dentro de cada bolha |
| Local search ±2px | Local search ±6px: cobre distorções residuais reais |

## Dicas de foto

- Enquadre a folha inteira (4 bordas visíveis)
- Evite sombra forte sobre os painéis de resposta
- Iluminação uniforme dá melhores resultados
- Funciona mesmo sem perspectiva perfeita graças à detecção de painéis

## Estrutura dos arquivos

| Arquivo | Função |
|---|---|
| `omr_reader.py` | **Motor principal** — warp, detecção de painéis, scoring |
| `app.py` | Interface gráfica (PySide6) |
| `exporter.py` | Exporta respostas para `.xlsx` |
| `build_template.py` | Gera `template.json` a partir de uma imagem de referência (manutenção) |
| `template.json` | Posições de referência dos painéis (não usado diretamente na v3) |

## Status de resposta no .xlsx

| Status | Significado |
|---|---|
| `ok` | Resposta clara e única |
| `blank` | Nenhuma bolha marcada com confiança |
| `ambiguous` | Duas bolhas muito próximas em score — verificar manualmente |
