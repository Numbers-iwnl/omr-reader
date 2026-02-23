\
from __future__ import annotations

import os
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QLabel,
    QFileDialog, QMessageBox, QCheckBox
)

from omr_reader import read_omr_answers, OMRConfig
from exporter import export_answers_to_xlsx


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Leitor de Gabarito (Cartão-Resposta)")

        layout = QVBoxLayout()
        layout.setSpacing(12)

        self.title = QLabel("Selecione a foto do cartão-resposta e eu salvo um .xlsx com as respostas.")
        self.title.setWordWrap(True)

        self.btn = QPushButton("Selecionar imagem…")
        self.btn.clicked.connect(self.select_image)

        self.debug = QCheckBox("Gerar arquivos de debug (warp/overlay) ao lado do .xlsx")
        self.debug.setChecked(False)

        self.hint = QLabel("Dica: pegue as 4 bordas da folha na foto e evite sombra forte.")
        self.hint.setStyleSheet("color: #666;")
        self.hint.setWordWrap(True)

        layout.addWidget(self.title)
        layout.addWidget(self.btn)
        layout.addWidget(self.debug)
        layout.addWidget(self.hint)

        self.setLayout(layout)
        self.setMinimumWidth(520)

    def select_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Escolher imagem do cartão-resposta",
            "",
            "Imagens (*.jpg *.jpeg *.png *.bmp *.webp);;Todos os arquivos (*.*)"
        )
        if not file_path:
            return

        try:
            cfg = OMRConfig()
            answers = read_omr_answers(file_path, cfg, debug_dir=None)

            # salva ao lado da imagem
            img = Path(file_path)
            out_path = img.with_name(img.stem + "_respostas.xlsx")
            export_answers_to_xlsx(answers, file_path, str(out_path))

            if self.debug.isChecked():
                debug_dir = img.with_name(img.stem + "_debug")
                # reprocessa só pra salvar overlays (sem mexer no fluxo do usuário)
                read_omr_answers(file_path, cfg, debug_dir=str(debug_dir))

            QMessageBox.information(
                self,
                "Pronto!",
                f"Planilha salva em:\n\n{out_path}\n\n"
                "Se aparecer muito 'blank' ou 'MULTI', tente uma foto mais nítida/centralizada."
            )

        except Exception as e:
            QMessageBox.critical(
                self,
                "Erro ao ler o gabarito",
                f"{type(e).__name__}: {e}\n\n"
                "Sugestões:\n"
                "• Garanta que a folha inteira (4 bordas) apareça na foto\n"
                "• Evite sombras e reflexos\n"
                "• Aumente um pouco a distância e mantenha a câmera mais paralela à folha"
            )


def main():
    app = QApplication([])
    w = MainWindow()
    w.show()
    app.exec()


if __name__ == "__main__":
    main()
