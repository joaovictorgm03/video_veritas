# Código de terceiros

`swift/qwen_vl_utils/` foi copiado **sem alterações** do repositório oficial do VideoVeritas:

- origem: https://github.com/EricTan7/VideoVeritas
- commit: `82c5fb84d327ccd3118ed0ebffb3aae45aa83694` (30/07/2026)
- licença: Apache License 2.0 (ver `LICENSE` nesta pasta)

É o pré-processamento de vídeo do modelo (leitura dos frames, amostragem e redimensionamento,
controlado pelas variáveis `FPS`, `DURATION` e `VIDEO_MAX_TOKEN_NUM`). Os scripts deste repositório
carregam só esse arquivo, sem instalar o pacote `swift` inteiro.
