from __future__ import annotations

from pathlib import Path
from typing import List

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QMessageBox,
    QCheckBox, QButtonGroup, QRadioButton, QGroupBox,
    QProgressBar, QFrame, QScrollArea
)
from PySide6.QtGui import QFont

from omr_reader import read_omr_answers, OMRConfig
from exporter import export_answers_to_xlsx


# ---------------------------------------------------------------------------
# Worker thread – processa imagens sem travar a UI
# ---------------------------------------------------------------------------
class WorkerThread(QThread):
    progress    = Signal(int, int, str)   # (atual, total, mensagem)
    finished    = Signal(list)            # lista de (img_path, out_path, n_answers, n_blank, n_ambig)
    error       = Signal(str, str)        # (img_path, mensagem de erro)

    def __init__(self, files: List[str], q_start: int, debug: bool):
        super().__init__()
        self.files   = files
        self.q_start = q_start
        self.debug   = debug

    def run(self):
        results = []
        for i, fp in enumerate(self.files):
            self.progress.emit(i + 1, len(self.files), Path(fp).name)
            try:
                debug_dir = None
                if self.debug:
                    p = Path(fp)
                    debug_dir = str(p.with_name(p.stem + "_debug"))

                answers = read_omr_answers(
                    fp,
                    config=OMRConfig(),
                    debug_dir=debug_dir,
                    q_start=self.q_start,
                )

                out_path = Path(fp).with_name(Path(fp).stem + "_respostas.xlsx")
                export_answers_to_xlsx(answers, fp, str(out_path))

                n_blank = sum(1 for a in answers if a.status == "blank")
                n_ambig = sum(1 for a in answers if a.status == "ambiguous")
                results.append((fp, str(out_path), len(answers), n_blank, n_ambig))

            except Exception as e:
                self.error.emit(fp, f"{type(e).__name__}: {e}")

        self.finished.emit(results)


# ---------------------------------------------------------------------------
# UI principal
# ---------------------------------------------------------------------------
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Leitor de Gabarito — Amo Medicina")
        self.setMinimumWidth(560)
        self._worker = None

        root = QVBoxLayout()
        root.setSpacing(14)
        root.setContentsMargins(18, 18, 18, 18)

        # ── Título ──────────────────────────────────────────────────────────
        title = QLabel("Leitor de Cartão-Resposta")
        f = QFont(); f.setPointSize(14); f.setBold(True)
        title.setFont(f)
        root.addWidget(title)

        sub = QLabel("Converte fotos de gabaritos em planilha .xlsx automaticamente.")
        sub.setStyleSheet("color: #555;")
        root.addWidget(sub)

        root.addWidget(self._hline())

        # ── Seleção de Dia ───────────────────────────────────────────────────
        dia_box = QGroupBox("Dia do simulado")
        dia_lay = QHBoxLayout()
        dia_lay.setSpacing(20)

        self._dia_group = QButtonGroup(self)
        self._r_dia1 = QRadioButton("Dia 1  (questões 1 – 90)")
        self._r_dia2 = QRadioButton("Dia 2  (questões 91 – 180)")
        self._r_dia1.setChecked(True)
        self._dia_group.addButton(self._r_dia1, 1)
        self._dia_group.addButton(self._r_dia2, 2)

        dia_lay.addWidget(self._r_dia1)
        dia_lay.addWidget(self._r_dia2)
        dia_lay.addStretch()
        dia_box.setLayout(dia_lay)
        root.addWidget(dia_box)

        # ── Opções ───────────────────────────────────────────────────────────
        self._chk_debug = QCheckBox(
            "Salvar imagens de debug (overlay + caixas detectadas) ao lado do .xlsx"
        )
        root.addWidget(self._chk_debug)

        # ── Botão de seleção ─────────────────────────────────────────────────
        self._btn = QPushButton("📂  Selecionar imagem(ns)…")
        self._btn.setMinimumHeight(40)
        self._btn.clicked.connect(self._pick_files)
        root.addWidget(self._btn)

        # ── Barra de progresso (oculta até usar) ────────────────────────────
        self._prog_label = QLabel("")
        self._prog_label.setStyleSheet("color: #444; font-size: 11px;")
        self._prog_label.hide()
        root.addWidget(self._prog_label)

        self._prog_bar = QProgressBar()
        self._prog_bar.setTextVisible(False)
        self._prog_bar.hide()
        root.addWidget(self._prog_bar)

        root.addWidget(self._hline())

        # ── Área de resultados ───────────────────────────────────────────────
        self._result_label = QLabel("Resultados aparecerão aqui.")
        self._result_label.setStyleSheet("color: #666; font-size: 11px;")
        self._result_label.setWordWrap(True)
        self._result_label.setAlignment(Qt.AlignTop)
        root.addWidget(self._result_label)

        # ── Dica ─────────────────────────────────────────────────────────────
        hint = QLabel(
            "💡 Dica: enquadre a folha inteira na foto (4 bordas visíveis) "
            "e evite sombras sobre os painéis de resposta."
        )
        hint.setStyleSheet("color: #888; font-size: 10px;")
        hint.setWordWrap(True)
        root.addWidget(hint)

        self.setLayout(root)

    # ── helpers ─────────────────────────────────────────────────────────────
    @staticmethod
    def _hline() -> QFrame:
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet("color: #ddd;")
        return line

    def _q_start(self) -> int:
        return 91 if self._dia_group.checkedId() == 2 else 1

    # ── slots ────────────────────────────────────────────────────────────────
    def _pick_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Escolher imagem(ns) do cartão-resposta",
            "",
            "Imagens (*.jpg *.jpeg *.png *.bmp *.webp);;Todos os arquivos (*.*)",
        )
        if not files:
            return
        self._process(files)

    def _process(self, files: List[str]):
        self._btn.setEnabled(False)
        self._prog_bar.setMaximum(len(files))
        self._prog_bar.setValue(0)
        self._prog_bar.show()
        self._prog_label.show()
        self._result_label.setText("Processando…")

        self._worker = WorkerThread(
            files,
            q_start=self._q_start(),
            debug=self._chk_debug.isChecked(),
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_progress(self, current: int, total: int, name: str):
        self._prog_bar.setValue(current)
        self._prog_label.setText(f"Processando {current}/{total}: {name}")

    def _on_finished(self, results):
        self._btn.setEnabled(True)
        self._prog_bar.hide()
        self._prog_label.hide()

        if not results:
            self._result_label.setText("Nenhum arquivo processado com sucesso.")
            return

        lines = []
        for fp, out_path, n_total, n_blank, n_ambig in results:
            name  = Path(fp).name
            flags = []
            if n_blank:
                flags.append(f"{n_blank} em branco")
            if n_ambig:
                flags.append(f"{n_ambig} ambíguo(s) — verificar manualmente")
            flag_str = f"  ⚠️ {', '.join(flags)}" if flags else "  ✅ tudo certo"
            lines.append(f"✔ {name}\n   → {Path(out_path).name}{flag_str}")

        self._result_label.setText("\n\n".join(lines))

        # Popup de confirmação apenas quando há poucos arquivos
        if len(results) <= 3:
            msg_lines = []
            for fp, out_path, n_total, n_blank, n_ambig in results:
                msg_lines.append(f"• {Path(out_path).name}")
                if n_blank or n_ambig:
                    msg_lines.append(
                        f"  ({n_blank} em branco, {n_ambig} ambíguo(s))"
                    )
            QMessageBox.information(
                self,
                "Pronto!",
                "Planilha(s) salva(s):\n\n" + "\n".join(msg_lines),
            )

    def _on_error(self, fp: str, msg: str):
        name = Path(fp).name
        self._result_label.setText(
            self._result_label.text().replace(
                "Processando…", f"❌ Erro em {name}:\n{msg}"
            )
        )
        QMessageBox.warning(
            self,
            f"Erro — {name}",
            f"{msg}\n\n"
            "Sugestões:\n"
            "• Verifique se selecionou o Dia correto (1 ou 2)\n"
            "• Garanta que as 4 bordas da folha apareçam na foto\n"
            "• Evite sombras e reflexos sobre os painéis de resposta\n"
            "• Mantenha a câmera paralela à folha",
        )


def main():
    app = QApplication([])
    w = MainWindow()
    w.show()
    app.exec()


if __name__ == "__main__":
    main()
