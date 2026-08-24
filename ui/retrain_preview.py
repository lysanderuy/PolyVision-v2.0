# RetrainPreview.py
"""
Visual-only preview gallery for retraining UI design work.

This module intentionally avoids importing Retrain.py, Detectron2, database
helpers, or training code. All data shown here is static mock data so the
preview can be used safely while iterating on UI changes.
"""

import sys

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QIcon
from PyQt5.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QProgressBar,
    QScrollArea,
    QSizePolicy,
    QStackedWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

try:
    from app_paths import resource_path
except Exception:
    resource_path = None


class RetrainPreviewGallery(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Retraining UI Preview Gallery")
        self.setMinimumSize(1024, 700)
        self.resize(1320, 860)
        if resource_path:
            self.setWindowIcon(QIcon(resource_path("res", "PolyVisionLogo.png")))

        self.setStyleSheet(PREVIEW_STYLESHEET)

        root = QHBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(14)

        self.nav = QListWidget()
        self.nav.setObjectName("previewNav")
        self.nav.setFixedWidth(260)
        self.nav.setSpacing(3)
        root.addWidget(self.nav)

        right = QVBoxLayout()
        right.setSpacing(10)
        root.addLayout(right, stretch=1)

        header = QLabel("Preview-only retraining screens")
        header.setObjectName("previewHeader")
        right.addWidget(header)

        note = QLabel(
            "Static mock widgets only. Buttons are decorative and do not import data, "
            "reset databases, launch repair scripts, or start training."
        )
        note.setObjectName("previewNote")
        note.setWordWrap(True)
        right.addWidget(note)

        self.stack = QStackedWidget()
        right.addWidget(self.stack, stretch=1)

        self.add_screen("Main Retraining - Idle", self.build_main_retraining(training=False))
        self.add_screen(
            "Main Progress - Final Status Panel",
            self.build_main_retraining(training=True, progress_variant="status_panel"),
        )
        self.add_screen("Progress Bar Concepts", self.build_progress_bar_concepts())
        self.add_screen("Progress Bar Designs", self.build_progress_bar_designs())
        self.add_screen("Import COCO Dataset", self.build_import_dialog_preview())
        self.add_screen("GPU Repair Prompt", self.build_gpu_repair_prompt())
        self.add_screen("Low Data Warning", self.build_low_data_prompt())
        self.add_screen("Confirm Retraining", self.build_confirm_retraining_prompt())
        self.add_screen("Retraining Failed", self.build_failure_prompt())
        self.add_screen("Summary & Deployment", self.build_comparison_preview())
        self.add_screen("Confirm Reset", self.build_reset_prompt())
        self.add_screen("Close During Training", self.build_close_training_prompt())

        self.nav.currentRowChanged.connect(self.stack.setCurrentIndex)
        self.nav.setCurrentRow(0)

    def add_screen(self, title, widget):
        item = QListWidgetItem(title)
        self.nav.addItem(item)
        self.stack.addWidget(self.wrap_preview(widget))

    def wrap_preview(self, widget):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setObjectName("previewScroll")

        holder = QWidget()
        layout = QVBoxLayout(holder)
        layout.setContentsMargins(18, 18, 18, 18)
        if widget.objectName() == "retrainSurface":
            layout.addWidget(widget, stretch=1)
        else:
            layout.addWidget(widget, alignment=Qt.AlignTop | Qt.AlignHCenter)
        layout.addStretch()

        scroll.setWidget(holder)
        return scroll

    def build_main_retraining(self, training=False, progress_variant="default"):
        panel = QFrame()
        panel.setObjectName("retrainSurface")
        panel.setMinimumSize(920, 680)
        panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        layout = QVBoxLayout(panel)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        model_group = QGroupBox("Model Selection")
        model_group.setObjectName("modelSelectionGroup")
        model_layout = QVBoxLayout(model_group)
        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("Select model to retrain:"))
        model_row.addStretch()

        binary_button = QPushButton("Binary")
        multiclass_button = QPushButton("Multiclass")
        for button in (binary_button, multiclass_button):
            button.setObjectName("modelChoiceButton")
            button.setCheckable(True)
            button.setEnabled(not training)
            button.setMinimumWidth(120)
        model_row.addWidget(binary_button)
        model_row.addWidget(multiclass_button)
        model_layout.addLayout(model_row)

        model_accent = QFrame()
        model_accent.setObjectName("modelAccentLine")
        model_accent.setFixedHeight(3)
        model_layout.addWidget(model_accent)
        layout.addWidget(model_group)

        summary_group = QGroupBox("Retraining Data Summary")
        summary_group.setObjectName("dashboardGroup")
        summary_layout = QHBoxLayout(summary_group)
        summary_layout.setSpacing(10)

        context = QLabel("537")
        context.setObjectName("infoText")

        if training:
            dataset_card = self.dashboard_card(
                "Dataset",
                [
                    ("Usable", "537"),
                    ("Positive", "531"),
                    ("Negative", "6"),
                ],
            )
            runtime_card = self.dashboard_card(
                "Runtime",
                [
                    ("Device", "CUDA GPU"),
                    ("Iteration", "742 / 1358"),
                    ("ETA", "18m 42s"),
                    ("Elapsed", "11m 03s"),
                ],
            )
            status_card = self.dashboard_card(
                "Training",
                [
                    ("Phase", "Stage 3/4"),
                    ("Current Loss", "0.382"),
                    ("Best Loss", "0.351"),
                    ("First Loss", "1.284"),
                    ("LR", "0.000015"),
                ],
            )
        else:
            dataset_card = self.dashboard_card(
                "Dataset",
                [
                    ("Positive", "531"),
                    ("Negative", "6"),
                    ("Total", "537"),
                    ("Usable", context),
                ],
            )
            runtime_card = self.dashboard_card(
                "Run Plan",
                [
                    ("Model", "Binary"),
                    ("Device", "GPU Ready"),
                    ("Iterations", "1358"),
                ],
            )
            status_card = self.dashboard_card(
                "Status",
                [
                    ("Phase", "Ready"),
                    ("ETA", "-"),
                    ("Readiness", "Ready"),
                ],
            )

        summary_layout.addWidget(dataset_card)
        summary_layout.addWidget(runtime_card)
        summary_layout.addWidget(status_card)
        layout.addWidget(summary_group)

        progress_group = QGroupBox("Retraining Progress")
        progress_layout = QVBoxLayout(progress_group)
        progress_layout.setSpacing(10)

        if training:
            progress_header = self.main_progress_header(progress_variant)
            progress_body = self.progress_highlights(
                current_step="Stage 3/4: Training new Challenger model",
                summary="Binary model on CUDA GPU · Iteration 742 / 1358 · ETA 18m 42s",
                completed=[
                    "Checked GPU and training environment",
                    "Stage 1/4: Prepared training data",
                    "Stage 2/4: Identified champion model",
                    "Loaded config and training arguments",
                ],
                active=[
                    "Phase: Challenger training",
                    "Current loss: 0.382",
                    "Best loss so far: 0.351",
                    "First logged loss: 1.284",
                    "Learning rate: 0.000015",
                ],
                upcoming=[
                    "Verify model_final.pth was saved",
                    "Stage 4/4: Benchmark champion vs challenger",
                    "Ask whether to deploy the new model",
                ],
                technical_log=(
                    "GPU: CUDA support available\n"
                    "Stage 1/4: Preparing training data...\n"
                    "--- Starting Data Preparation (Bridge Strategy) ---\n"
                    "Found 60 new data points.\n"
                    "Selecting 60 RANDOM old images...\n"
                    "Total unique anchor images selected: 90\n"
                    "Stage 2/4: Identifying current Champion model...\n"
                    "Champion model found: Retrained Model\n"
                    "Stage 3/4: Training new Challenger model...\n"
                    "--- Training device selected: cuda ---\n"
                    "iter: 1 total_loss: 1.284 lr: 0.000001\n"
                    "iter: 340 total_loss: 0.514 lr: 0.000015\n"
                    "iter: 611 total_loss: 0.351 lr: 0.000015\n"
                    "iter: 742 total_loss: 0.382 lr: 0.000015\n"
                    "Training in progress..."
                ),
            )
        else:
            progress_header = self.main_idle_status_panel()
            progress_body = self.progress_highlights(
                current_step="Ready to start retraining",
                summary="Review the Binary run plan, then start when ready.",
                completed=[
                    "Found 537 binary-relevant samples",
                    "Loaded retraining configuration",
                    "GPU support is available",
                ],
                active=[
                    "Waiting for user confirmation",
                ],
                upcoming=[
                    "Prepare retraining dataset",
                    "Train updated Binary model",
                    "Benchmark and compare results",
                ],
                technical_log="Training logs will appear here after retraining starts.",
            )

        progress_layout.addWidget(progress_header)
        progress_layout.addWidget(progress_body)
        layout.addWidget(progress_group, stretch=2)

        button_layout = QHBoxLayout()
        button_layout.setSpacing(8)
        import_button = QPushButton("Import Dataset")
        reset_button = QPushButton("Reset Data")
        start_button = QPushButton("Start Retraining")
        cancel_button = QPushButton("Cancel")
        close_button = QPushButton("Close")

        import_button.setProperty("buttonRole", "secondary")
        reset_button.setProperty("buttonRole", "danger")
        start_button.setProperty("buttonRole", "primary")
        cancel_button.setProperty("buttonRole", "warning")
        close_button.setProperty("buttonRole", "secondary")

        if training:
            import_button.setEnabled(False)
            reset_button.setEnabled(False)
            start_button.setEnabled(False)
            cancel_button.setEnabled(True)
        else:
            cancel_button.setEnabled(False)

        for button in (import_button, reset_button, start_button, cancel_button, close_button):
            button.setObjectName("footerButton")
            button.setProperty("previewOnly", True)
            button.setMinimumHeight(32)
            button.setMinimumWidth(96)

        start_button.setMinimumWidth(132)

        def refresh_model_choice(model_type):
            is_binary = model_type == "Binary"
            meta = self.model_preview_meta(model_type)
            color = meta["color"]
            context.setText(meta["usable_samples"])
            context.setStyleSheet(f"color: {color}; font-weight: bold;")
            if not training:
                self.set_dashboard_value(runtime_card, "Model", model_type)
                self.update_idle_status_panel(progress_header, model_type)
                self.update_idle_progress_copy(progress_body, model_type)
                start_button.setStyleSheet(self.primary_button_style(meta))
            model_group.setStyleSheet(
                "QGroupBox#modelSelectionGroup { "
                f"border: 1px solid {color}; "
                "border-radius: 5px; margin-top: 10px; } "
                "QGroupBox#modelSelectionGroup::title { "
                "subcontrol-origin: margin; subcontrol-position: top left; padding: 0 5px; }"
            )
            model_accent.setStyleSheet(f"background: {color};")
            binary_button.setChecked(is_binary)
            multiclass_button.setChecked(not is_binary)
            binary_button.setStyleSheet(self.model_button_style("#0f5f9e", is_binary))
            multiclass_button.setStyleSheet(self.model_button_style("#0f7f45", not is_binary))

        binary_button.clicked.connect(lambda: refresh_model_choice("Binary"))
        multiclass_button.clicked.connect(lambda: refresh_model_choice("Multiclass"))
        refresh_model_choice("Binary")

        button_layout.addWidget(import_button, alignment=Qt.AlignLeft)
        button_layout.addWidget(reset_button, alignment=Qt.AlignLeft)
        button_layout.addStretch()
        button_layout.addWidget(start_button)
        button_layout.addWidget(cancel_button)
        button_layout.addWidget(close_button)
        layout.addLayout(button_layout)

        return panel

    def dashboard_card(self, title, rows):
        card = QFrame()
        card.setObjectName("dashboardCard")
        card.setMinimumWidth(210)
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(12, 10, 12, 10)
        card_layout.setSpacing(6)

        title_label = QLabel(title)
        title_label.setObjectName("dashboardTitle")
        card_layout.addWidget(title_label)

        for label, value in rows:
            row = QHBoxLayout()
            row.setSpacing(8)
            label_widget = QLabel(label)
            label_widget.setObjectName("dashboardLabel")

            if isinstance(value, QLabel):
                value_widget = value
            else:
                value_widget = QLabel(value)
            value_widget.setObjectName("dashboardValue")
            value_widget.setProperty("metricName", label)
            value_widget.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

            row.addWidget(label_widget)
            row.addStretch()
            row.addWidget(value_widget)
            card_layout.addLayout(row)

        card_layout.addStretch()
        return card

    def set_dashboard_value(self, card, metric_name, value):
        labels = card.findChildren(QLabel)
        for label in labels:
            if label.property("metricName") == metric_name:
                label.setText(value)
                return

    def model_preview_meta(self, model_type):
        if model_type == "Multiclass":
            return {
                "model_type": "Multiclass",
                "color": "#0f7f45",
                "hover": "#0b6f3b",
                "pressed": "#095f33",
                "usable_samples": "531",
                "sample_phrase": "531 multiclass-ready samples ready",
                "completed_sample": "Found 531 multiclass-ready samples",
                "summary": "Review the Multiclass run plan, then start when ready.",
                "upcoming_train": "Train updated Multiclass model",
                "technical_log": (
                    "Multiclass model selected.\n"
                    "531 positive annotated samples are ready for multiclass retraining.\n"
                    "GPU support is available.\n"
                    "Training logs will appear here after retraining starts."
                ),
            }
        return {
            "model_type": "Binary",
            "color": "#0f5f9e",
            "hover": "#0c5288",
            "pressed": "#0a4775",
            "usable_samples": "537",
            "sample_phrase": "537 binary-relevant samples ready",
            "completed_sample": "Found 537 binary-relevant samples",
            "summary": "Review the Binary run plan, then start when ready.",
            "upcoming_train": "Train updated Binary model",
            "technical_log": (
                "Binary model selected.\n"
                "All 537 collected samples are available for binary retraining.\n"
                "GPU support is available.\n"
                "Training logs will appear here after retraining starts."
            ),
        }

    def main_idle_status_panel(self):
        capsule = QFrame()
        capsule.setObjectName("mainStatusPanel")
        capsule_layout = QVBoxLayout(capsule)
        capsule_layout.setContentsMargins(14, 10, 14, 10)
        capsule_layout.setSpacing(8)

        top = QHBoxLayout()
        state = QLabel("Ready")
        state.setObjectName("mainStatusPercent")
        state.setProperty("idleStatusRole", "state")
        model = QLabel("Model: Binary")
        model.setObjectName("mainStatusEta")
        model.setProperty("idleStatusRole", "model")
        health = QLabel("GPU Ready")
        health.setObjectName("mainStatusHealth")
        top.addWidget(state)
        top.addSpacing(12)
        top.addWidget(model)
        top.addStretch()
        top.addWidget(health)
        capsule_layout.addLayout(top)

        progress = QProgressBar()
        progress.setObjectName("mainStatusProgress")
        progress.setMaximum(100)
        progress.setValue(0)
        progress.setTextVisible(False)
        capsule_layout.addWidget(progress)

        bottom = QLabel("")
        bottom.setObjectName("mainStatusBottom")
        bottom.setProperty("idleStatusRole", "bottom")
        capsule_layout.addWidget(bottom)
        self.update_idle_status_panel(capsule, "Binary")
        return capsule

    def update_idle_status_panel(self, panel, model_type):
        meta = self.model_preview_meta(model_type)
        panel.setStyleSheet(
            "QFrame#mainStatusPanel { "
            "background: #f8fbfd; "
            "border: 1px solid #bfd0dc; "
            f"border-left: 5px solid {meta['color']}; "
            "border-radius: 7px; }"
        )
        for label in panel.findChildren(QLabel):
            role = label.property("idleStatusRole")
            if role == "state":
                label.setText("Ready")
                label.setStyleSheet(
                    f"color: {meta['color']}; font-size: 17pt; font-weight: 800;"
                )
            elif role == "model":
                label.setText(f"Model: {model_type}")
            elif role == "bottom":
                label.setText(f"Waiting for confirmation · {meta['sample_phrase']}")

        progress = panel.findChild(QProgressBar, "mainStatusProgress")
        if progress:
            progress.setStyleSheet(
                "QProgressBar#mainStatusProgress { "
                "height: 18px; border: 1px solid #bfd0dc; "
                "border-radius: 9px; background: #edf3f7; } "
                "QProgressBar#mainStatusProgress::chunk { "
                f"border-radius: 8px; background: {meta['color']}; }}"
            )

    def update_idle_progress_copy(self, panel, model_type):
        meta = self.model_preview_meta(model_type)
        for label in panel.findChildren(QLabel):
            role = label.property("progressRole")
            if role == "current":
                label.setText(f"Ready to start {model_type} retraining")
            elif role == "summary":
                label.setText(meta["summary"])

        self.set_highlight_items(
            panel,
            "done",
            [
                meta["completed_sample"],
                "Loaded retraining configuration",
                "GPU support is available",
            ],
        )
        self.set_highlight_items(
            panel,
            "active",
            [
                "Waiting for user confirmation",
            ],
        )
        self.set_highlight_items(
            panel,
            "next",
            [
                "Prepare retraining dataset",
                meta["upcoming_train"],
                "Benchmark and compare results",
            ],
        )

        for editor in panel.findChildren(QTextEdit):
            if editor.property("progressRole") == "technicalLog":
                editor.setPlainText(meta["technical_log"])

    def set_highlight_items(self, panel, status, items):
        marker = {"done": "✓", "active": "•", "next": "→"}.get(status, "•")
        labels = [
            label for label in panel.findChildren(QLabel)
            if label.property("highlightStatus") == status
        ]
        labels.sort(key=lambda label: label.property("highlightIndex") or 0)
        for index, label in enumerate(labels):
            if index < len(items):
                label.setText(f"{marker} {items[index]}")
                label.setVisible(True)
            else:
                label.setVisible(False)

    def primary_button_style(self, meta):
        return (
            "QPushButton#footerButton { "
            "color: #ffffff; "
            f"background: {meta['color']}; "
            f"border: 1px solid {meta['color']}; "
            "border-radius: 4px; "
            "padding: 6px 14px; "
            "font-weight: 700; } "
            "QPushButton#footerButton:hover { "
            f"background: {meta['hover']}; border-color: {meta['hover']}; }} "
            "QPushButton#footerButton:pressed { "
            f"background: {meta['pressed']}; border-color: {meta['pressed']}; }} "
            "QPushButton#footerButton:disabled { "
            "color: #9aa5b1; background: #eef1f4; border-color: #d5dbe1; }"
        )

    def main_progress_header(self, variant):
        if variant == "status_panel":
            capsule = QFrame()
            capsule.setObjectName("mainStatusPanel")
            capsule_layout = QVBoxLayout(capsule)
            capsule_layout.setContentsMargins(14, 10, 14, 10)
            capsule_layout.setSpacing(8)

            top = QHBoxLayout()
            percent = QLabel("54%")
            percent.setObjectName("mainStatusPercent")
            eta = QLabel("ETA 18m 42s")
            eta.setObjectName("mainStatusEta")
            health = QLabel("Running normally")
            health.setObjectName("mainStatusHealth")
            top.addWidget(percent)
            top.addSpacing(12)
            top.addWidget(eta)
            top.addStretch()
            top.addWidget(health)
            capsule_layout.addLayout(top)

            progress = QProgressBar()
            progress.setObjectName("mainStatusProgress")
            progress.setMaximum(100)
            progress.setValue(54)
            progress.setTextVisible(False)
            capsule_layout.addWidget(progress)

            bottom = QLabel("Stage 3/4 · Training challenger model · Iteration 742 / 1358")
            bottom.setObjectName("mainStatusBottom")
            capsule_layout.addWidget(bottom)
            return capsule

        progress = QProgressBar()
        progress.setObjectName("mainRoundedProgress")
        progress.setMaximum(100)
        progress.setValue(54)
        progress.setFormat("%p%")
        progress.setTextVisible(True)
        return progress

    def build_progress_bar_concepts(self):
        panel = QFrame()
        panel.setObjectName("conceptSurface")
        panel.setMinimumSize(920, 680)
        panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        layout = QVBoxLayout(panel)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(14)

        title = QLabel("Progress Bar Concepts")
        title.setObjectName("windowTitle")
        layout.addWidget(title)

        intro = QLabel(
            "Four visual directions for the retraining progress indicator. "
            "All examples use the same mock state: Stage 3/4, iteration 742 / 1358, 54% complete."
        )
        intro.setObjectName("previewNote")
        intro.setWordWrap(True)
        layout.addWidget(intro)

        grid = QGridLayout()
        grid.setSpacing(14)
        grid.addWidget(
            self.progress_concept_percent_only(),
            0,
            0,
        )
        grid.addWidget(
            self.progress_concept_stage_label(),
            0,
            1,
        )
        grid.addWidget(
            self.progress_concept_milestones(),
            1,
            0,
        )
        grid.addWidget(
            self.progress_concept_eta(),
            1,
            1,
        )
        layout.addLayout(grid)
        layout.addStretch()
        return panel

    def concept_card(self, title, description):
        card = QFrame()
        card.setObjectName("conceptCard")
        card.setMinimumSize(390, 220)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(9)

        title_label = QLabel(title)
        title_label.setObjectName("conceptTitle")
        layout.addWidget(title_label)

        description_label = QLabel(description)
        description_label.setObjectName("conceptDescription")
        description_label.setWordWrap(True)
        layout.addWidget(description_label)
        return card, layout

    def progress_concept_percent_only(self):
        card, layout = self.concept_card(
            "1. Percent Only",
            "Cleanest option. The bar only says the percent, while details stay in the dashboard/highlights.",
        )
        progress = QProgressBar()
        progress.setObjectName("percentOnlyProgress")
        progress.setMaximum(100)
        progress.setValue(54)
        progress.setFormat("%p%")
        layout.addWidget(progress)
        detail = QLabel("Best when the surrounding UI already shows iteration, ETA, and phase.")
        detail.setObjectName("conceptCaption")
        detail.setWordWrap(True)
        layout.addWidget(detail)
        layout.addStretch()
        return card

    def progress_concept_stage_label(self):
        card, layout = self.concept_card(
            "2. Stage-Aware Bar",
            "Keeps the bar simple, but adds current pipeline stage directly above it.",
        )
        stage = QLabel("Stage 3/4 · Training Challenger Model")
        stage.setObjectName("progressStageLabel")
        layout.addWidget(stage)
        progress = QProgressBar()
        progress.setObjectName("stageAwareProgress")
        progress.setMaximum(100)
        progress.setValue(54)
        progress.setFormat("%p%")
        layout.addWidget(progress)
        detail = QLabel("Iteration 742 / 1358")
        detail.setObjectName("conceptCaption")
        layout.addWidget(detail)
        layout.addStretch()
        return card

    def progress_concept_milestones(self):
        card, layout = self.concept_card(
            "3. Pipeline Milestones",
            "Shows where the user is in the four major retraining stages.",
        )
        steps = QHBoxLayout()
        steps.setSpacing(8)
        step_data = [
            ("1", "Data", "done"),
            ("2", "Champion", "done"),
            ("3", "Training", "active"),
            ("4", "Evaluate", "next"),
        ]
        for number, label, state in step_data:
            steps.addWidget(self.pipeline_step(number, label, state))
        layout.addLayout(steps)
        progress = QProgressBar()
        progress.setObjectName("milestoneProgress")
        progress.setMaximum(100)
        progress.setValue(54)
        progress.setFormat("%p%")
        layout.addWidget(progress)
        layout.addStretch()
        return card

    def progress_concept_eta(self):
        card, layout = self.concept_card(
            "4. ETA + Confidence",
            "Best for long runs. Puts time remaining and health signal near the bar.",
        )
        top = QHBoxLayout()
        eta = QLabel("ETA 18m 42s")
        eta.setObjectName("etaLabel")
        health = QLabel("Running normally")
        health.setObjectName("healthyLabel")
        top.addWidget(eta)
        top.addStretch()
        top.addWidget(health)
        layout.addLayout(top)
        progress = QProgressBar()
        progress.setObjectName("etaProgress")
        progress.setMaximum(100)
        progress.setValue(54)
        progress.setFormat("%p%")
        layout.addWidget(progress)
        detail = QLabel("Elapsed 11m 03s · Current loss 0.382 · Best loss 0.351")
        detail.setObjectName("conceptCaption")
        detail.setWordWrap(True)
        layout.addWidget(detail)
        layout.addStretch()
        return card

    def pipeline_step(self, number, label, state):
        step = QFrame()
        step.setObjectName(f"pipelineStep_{state}")
        step.setMinimumWidth(82)
        layout = QVBoxLayout(step)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)

        number_label = QLabel(number)
        number_label.setObjectName("pipelineStepNumber")
        number_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(number_label)

        text = QLabel(label)
        text.setObjectName("pipelineStepLabel")
        text.setAlignment(Qt.AlignCenter)
        text.setWordWrap(True)
        layout.addWidget(text)
        return step

    def build_progress_bar_designs(self):
        panel = QFrame()
        panel.setObjectName("conceptSurface")
        panel.setMinimumSize(920, 680)
        panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        layout = QVBoxLayout(panel)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(14)

        title = QLabel("Progress Bar Design Concepts")
        title.setObjectName("windowTitle")
        layout.addWidget(title)

        intro = QLabel(
            "These all use the preferred direction: pipeline milestones plus health signal. "
            "The difference is only the progress bar's visual treatment."
        )
        intro.setObjectName("previewNote")
        intro.setWordWrap(True)
        layout.addWidget(intro)

        grid = QGridLayout()
        grid.setSpacing(14)
        grid.addWidget(self.progress_design_rounded_rail(), 0, 0)
        grid.addWidget(self.progress_design_segmented(), 0, 1)
        grid.addWidget(self.progress_design_timeline(), 1, 0)
        grid.addWidget(self.progress_design_status_capsule(), 1, 1)
        layout.addLayout(grid)
        layout.addStretch()
        return panel

    def design_card(self, title, description):
        card = QFrame()
        card.setObjectName("conceptCard")
        card.setMinimumSize(390, 250)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(10)

        title_label = QLabel(title)
        title_label.setObjectName("conceptTitle")
        layout.addWidget(title_label)

        description_label = QLabel(description)
        description_label.setObjectName("conceptDescription")
        description_label.setWordWrap(True)
        layout.addWidget(description_label)

        layout.addLayout(self.design_status_row())
        return card, layout

    def design_status_row(self):
        row = QHBoxLayout()
        phase = QLabel("Stage 3/4 · Training")
        phase.setObjectName("progressStageLabel")
        health = QLabel("Running normally")
        health.setObjectName("healthyLabel")
        row.addWidget(phase)
        row.addStretch()
        row.addWidget(health)
        return row

    def progress_design_rounded_rail(self):
        card, layout = self.design_card(
            "1. Rounded Rail",
            "Simple upgrade from the current bar: softer shape, cleaner fill, percent-only text.",
        )
        progress = QProgressBar()
        progress.setObjectName("roundedRailProgress")
        progress.setMaximum(100)
        progress.setValue(54)
        progress.setFormat("%p%")
        layout.addWidget(progress)
        layout.addLayout(self.compact_stage_row())
        layout.addWidget(self.design_caption("Best if we want a conservative improvement without changing much layout."))
        layout.addStretch()
        return card

    def progress_design_segmented(self):
        card, layout = self.design_card(
            "2. Segmented Stage Bar",
            "Progress is divided into the four retraining phases so users understand what 54% means.",
        )
        segments = QHBoxLayout()
        segments.setSpacing(4)
        for label, state in [
            ("Data", "done"),
            ("Champion", "done"),
            ("Training", "active"),
            ("Evaluate", "next"),
        ]:
            segments.addWidget(self.segment_piece(label, state))
        layout.addLayout(segments)
        layout.addLayout(self.compact_metric_row("54%", "742 / 1358", "ETA 18m 42s"))
        layout.addWidget(self.design_caption("Best match for the pipeline concept. It makes stage progress explicit."))
        layout.addStretch()
        return card

    def progress_design_timeline(self):
        card, layout = self.design_card(
            "3. Timeline Nodes",
            "A more guided visual: users see completed stages as checkpoints along one progress path.",
        )
        timeline = QFrame()
        timeline.setObjectName("timelineRail")
        timeline_layout = QHBoxLayout(timeline)
        timeline_layout.setContentsMargins(10, 8, 10, 8)
        timeline_layout.setSpacing(0)
        nodes = [
            ("1", "Data", "done"),
            ("2", "Champion", "done"),
            ("3", "Training", "active"),
            ("4", "Evaluate", "next"),
        ]
        for index, (number, label, state) in enumerate(nodes):
            timeline_layout.addWidget(self.timeline_node(number, label, state))
            if index < len(nodes) - 1:
                connector = QFrame()
                connector.setObjectName("timelineConnector")
                connector.setFixedHeight(3)
                timeline_layout.addWidget(connector, stretch=1)
        layout.addWidget(timeline)
        layout.addLayout(self.compact_metric_row("54%", "Stage 3/4", "Running normally"))
        layout.addWidget(self.design_caption("Best if we want the progress area to feel more guided and less technical."))
        layout.addStretch()
        return card

    def progress_design_status_capsule(self):
        card, layout = self.design_card(
            "4. Integrated Status Panel",
            "Keeps the capsule's rich status idea, but brings it back into the light retraining UI style.",
        )
        capsule = QFrame()
        capsule.setObjectName("statusCapsule")
        capsule_layout = QVBoxLayout(capsule)
        capsule_layout.setContentsMargins(14, 12, 14, 12)
        capsule_layout.setSpacing(9)
        top = QHBoxLayout()
        percent = QLabel("54%")
        percent.setObjectName("capsulePercent")
        eta = QLabel("ETA 18m 42s")
        eta.setObjectName("capsuleEta")
        health = QLabel("Running normally")
        health.setObjectName("capsuleHealth")
        top.addWidget(percent)
        top.addSpacing(12)
        top.addWidget(eta)
        top.addStretch()
        top.addWidget(health)
        capsule_layout.addLayout(top)
        progress = QProgressBar()
        progress.setObjectName("capsuleProgress")
        progress.setMaximum(100)
        progress.setValue(54)
        progress.setTextVisible(False)
        capsule_layout.addWidget(progress)
        bottom = QLabel("Stage 3/4 · Training challenger model")
        bottom.setObjectName("capsuleBottom")
        capsule_layout.addWidget(bottom)
        layout.addWidget(capsule)
        layout.addWidget(self.design_caption("Potential final direction if we want more presence than the rounded rail."))
        layout.addStretch()
        return card

    def compact_stage_row(self):
        row = QHBoxLayout()
        for label, state in [
            ("Data", "done"),
            ("Champion", "done"),
            ("Training", "active"),
            ("Evaluate", "next"),
        ]:
            tag = QLabel(label)
            tag.setObjectName(f"stageTag_{state}")
            tag.setAlignment(Qt.AlignCenter)
            row.addWidget(tag)
        return row

    def compact_metric_row(self, left, center, right):
        row = QHBoxLayout()
        for value in (left, center, right):
            label = QLabel(value)
            label.setObjectName("conceptMetric")
            row.addWidget(label)
        row.addStretch()
        return row

    def segment_piece(self, label, state):
        piece = QLabel(label)
        piece.setObjectName(f"segmentPiece_{state}")
        piece.setAlignment(Qt.AlignCenter)
        piece.setMinimumHeight(34)
        return piece

    def timeline_node(self, number, label, state):
        node = QFrame()
        node.setObjectName("timelineNode")
        node_layout = QVBoxLayout(node)
        node_layout.setContentsMargins(0, 0, 0, 0)
        node_layout.setSpacing(4)
        dot = QLabel(number)
        dot.setObjectName(f"timelineDot_{state}")
        dot.setAlignment(Qt.AlignCenter)
        dot.setFixedSize(32, 32)
        text = QLabel(label)
        text.setObjectName("timelineLabel")
        text.setAlignment(Qt.AlignCenter)
        node_layout.addWidget(dot, alignment=Qt.AlignHCenter)
        node_layout.addWidget(text)
        return node

    def design_caption(self, text):
        caption = QLabel(text)
        caption.setObjectName("conceptCaption")
        caption.setWordWrap(True)
        return caption

    def progress_highlights(
        self,
        current_step,
        summary,
        completed,
        active,
        upcoming,
        technical_log,
    ):
        panel = QFrame()
        panel.setObjectName("highlightsPanel")
        panel.setMinimumHeight(330)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(9)

        current = QLabel(current_step)
        current.setObjectName("currentStep")
        current.setProperty("progressRole", "current")
        current.setWordWrap(True)
        layout.addWidget(current)

        summary_label = QLabel(summary)
        summary_label.setObjectName("stepSummary")
        summary_label.setProperty("progressRole", "summary")
        layout.addWidget(summary_label)

        columns = QHBoxLayout()
        columns.setSpacing(10)
        columns.addWidget(self.highlight_column("Completed", completed, "done"))
        columns.addWidget(self.highlight_column("In Progress", active, "active"))
        columns.addWidget(self.highlight_column("Next", upcoming, "next"))
        layout.addLayout(columns)

        technical_button = QPushButton("Show Technical Log")
        technical_button.setCheckable(True)
        technical_button.setObjectName("technicalLogToggle")

        technical_view = QTextEdit()
        technical_view.setReadOnly(True)
        technical_view.setProperty("progressRole", "technicalLog")
        technical_view.setFont(QFont("Courier", 9))
        technical_view.setPlainText(technical_log)
        technical_view.setVisible(False)
        technical_view.setFixedHeight(170)

        def toggle_log(checked):
            technical_view.setVisible(checked)
            technical_button.setText("Hide Technical Log" if checked else "Show Technical Log")

        technical_button.toggled.connect(toggle_log)
        layout.addWidget(technical_button, alignment=Qt.AlignRight)
        layout.addWidget(technical_view)
        return panel

    def highlight_column(self, title, items, status):
        card = QFrame()
        card.setObjectName("highlightCard")
        card.setMinimumWidth(240)
        card.setMinimumHeight(150)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(7)

        title_label = QLabel(title)
        title_label.setObjectName("highlightTitle")
        layout.addWidget(title_label)

        marker = {"done": "✓", "active": "•", "next": "→"}.get(status, "•")
        for index, item in enumerate(items):
            label = QLabel(f"{marker} {item}")
            label.setObjectName(f"highlightItem_{status}")
            label.setProperty("highlightStatus", status)
            label.setProperty("highlightIndex", index)
            label.setWordWrap(True)
            layout.addWidget(label)

        layout.addStretch()
        return card

    def model_button_style(self, color, selected):
        if selected:
            return (
                f"QPushButton#modelChoiceButton {{ color: white; background: {color}; "
                f"border: 1px solid {color}; border-radius: 3px; padding: 5px 14px; }}"
            )
        return (
            "QPushButton#modelChoiceButton { color: #1f2933; background: #f5f5f5; "
            f"border: 1px solid {color}; border-radius: 3px; padding: 5px 14px; }} "
            "QPushButton#modelChoiceButton:hover { background: #eef6f7; }"
        )

    def build_import_dialog_preview(self):
        panel = self.dialog_panel("Import COCO Dataset", width=560)
        form = QGridLayout()
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(12)

        form.addWidget(QLabel("Image Directory:"), 0, 0)
        image_label = QLabel("<i>mock_dataset/train/images</i>")
        image_label.setObjectName("mutedValue")
        form.addWidget(image_label, 0, 1)
        form.addWidget(QPushButton("Select Image Folder..."), 1, 1)

        form.addWidget(QLabel("Annotation File:"), 2, 0)
        json_label = QLabel("<i>mock_dataset/train/_annotations.coco.json</i>")
        json_label.setObjectName("mutedValue")
        form.addWidget(json_label, 2, 1)
        form.addWidget(QPushButton("Select COCO .json File..."), 3, 1)

        panel.layout().addLayout(form)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("Import")
        panel.layout().addWidget(buttons)
        return panel

    def build_gpu_repair_prompt(self):
        return self.message_panel(
            "GPU Support Repair Available",
            "This machine appears to have NVIDIA GPU hardware, but PolyVision cannot use CUDA in this environment.",
            "GPU: NVIDIA GeForce RTX Mock\nPyTorch CUDA: unavailable\nDetectron2 CUDA: unavailable\n\n"
            "You can repair GPU support now, continue this retraining run on CPU, or cancel.",
            ["Repair GPU Support", "Continue on CPU", "Cancel"],
            tone="warning",
        )

    def build_low_data_prompt(self):
        return self.message_panel(
            "Low Data Warning",
            "You have only collected 18 new data samples.",
            "It is recommended to have at least 50 new samples to see a significant improvement "
            "in model performance.\n\nDo you want to continue with the retraining anyway?",
            ["Yes", "No"],
            tone="warning",
        )

    def build_confirm_retraining_prompt(self):
        return self.message_panel(
            "Confirm Retraining",
            "This will start the model retraining process.",
            "Retraining can take a long time and consume significant computer resources. "
            "We recommend charging your device before proceeding.\n\nAre you sure you want to continue?",
            ["Yes", "No"],
            tone="info",
        )

    def build_failure_prompt(self):
        return self.message_panel(
            "Retraining Failed",
            "The retraining pipeline could not complete.",
            "FATAL: Test set not found at the configured path. Cannot benchmark.\n\n"
            "No model was deployed. Existing model files were retained.",
            ["OK"],
            tone="error",
        )

    def build_reset_prompt(self):
        return self.message_panel(
            "Confirm Reset",
            "This will permanently delete all collected retraining images and their labels.",
            "This action cannot be undone.\n\nAre you sure you want to continue?",
            ["Yes", "No"],
            tone="error",
        )

    def build_close_training_prompt(self):
        return self.message_panel(
            "Retraining in Progress",
            "Retraining is currently in progress.",
            "Are you sure you want to close and cancel it?",
            ["Yes", "No"],
            tone="warning",
        )

    def build_comparison_preview(self):
        panel = self.dialog_panel("Retraining Summary & Deployment", width=720)

        main_label = QLabel("Retraining and evaluation complete.")
        main_label.setObjectName("sectionTitle")
        panel.layout().addWidget(main_label)

        comparison_layout = QHBoxLayout()
        comparison_layout.setSpacing(12)
        comparison_layout.addWidget(
            self.metrics_group(
                "Current Model (Champion)",
                ap=48.72,
                ap50=72.14,
                ap75=51.08,
            )
        )
        comparison_layout.addWidget(
            self.metrics_group(
                "New Model (Challenger)",
                ap=52.31,
                ap50=76.42,
                ap75=55.19,
                delta=3.59,
            )
        )
        panel.layout().addLayout(comparison_layout)

        deploy_label = QLabel("The new model shows the performance above. Do you want to deploy it?")
        deploy_label.setWordWrap(True)
        panel.layout().addWidget(deploy_label)

        buttons = QDialogButtonBox(QDialogButtonBox.Yes | QDialogButtonBox.No)
        panel.layout().addWidget(buttons)
        return panel

    def metrics_group(self, title, ap, ap50, ap75, delta=None):
        group = QGroupBox(title)
        layout = QVBoxLayout(group)
        layout.addWidget(QLabel(f"<b>Overall AP:</b> {ap:.2f}"))
        layout.addWidget(QLabel(f"AP50 (IoU > 0.5): {ap50:.2f}"))
        layout.addWidget(QLabel(f"AP75 (IoU > 0.75): {ap75:.2f}"))
        if delta is not None:
            delta_label = QLabel(f"Change: {delta:+.2f} AP")
            delta_label.setObjectName("positiveDelta" if delta >= 0 else "negativeDelta")
            layout.addSpacing(8)
            layout.addWidget(delta_label)
        layout.addStretch()
        return group

    def dialog_panel(self, title, width=620):
        panel = QFrame()
        panel.setObjectName("dialogSurface")
        panel.setMinimumWidth(width)
        panel.setMaximumWidth(width)
        panel.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Maximum)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(14)

        title_label = QLabel(title)
        title_label.setObjectName("windowTitle")
        layout.addWidget(title_label)
        return panel

    def message_panel(self, title, text, detail, buttons, tone="info"):
        panel = self.dialog_panel(title, width=620)
        body = QHBoxLayout()
        body.setSpacing(12)

        icon = QLabel(self.icon_text_for_tone(tone))
        icon.setObjectName(f"{tone}Icon")
        icon.setAlignment(Qt.AlignTop | Qt.AlignHCenter)
        icon.setFixedWidth(42)
        body.addWidget(icon)

        text_layout = QVBoxLayout()
        headline = QLabel(text)
        headline.setObjectName("messageHeadline")
        headline.setWordWrap(True)
        detail_label = QLabel(detail)
        detail_label.setObjectName("messageDetail")
        detail_label.setWordWrap(True)
        text_layout.addWidget(headline)
        text_layout.addWidget(detail_label)
        body.addLayout(text_layout)

        panel.layout().addLayout(body)

        button_row = QHBoxLayout()
        button_row.addStretch()
        for label in buttons:
            button = QPushButton(label)
            button.setProperty("previewOnly", True)
            button_row.addWidget(button)
        panel.layout().addLayout(button_row)
        return panel

    def icon_text_for_tone(self, tone):
        if tone == "warning":
            return "!"
        if tone == "error":
            return "x"
        return "i"


PREVIEW_STYLESHEET = """
QDialog {
    background: #eef1f4;
    font-family: Segoe UI, Arial, sans-serif;
}
#previewHeader {
    font-size: 16pt;
    font-weight: 700;
    color: #17212b;
}
#previewNote {
    color: #52616f;
}
#previewNav {
    background: #ffffff;
    border: 1px solid #ccd4dc;
    border-radius: 6px;
    padding: 8px;
    outline: 0;
}
#previewNav::item {
    padding: 10px 9px;
    border-radius: 5px;
}
#previewNav::item:selected {
    background: #0f7c80;
    color: #ffffff;
}
#previewNav::item:hover:!selected {
    background: #e5edef;
}
#previewScroll {
    border: 1px solid #ccd4dc;
    border-radius: 6px;
    background: #f8fafb;
}
#dialogSurface {
    background: #ffffff;
    border: 1px solid #cbd5df;
    border-radius: 6px;
}
#retrainSurface {
    background: #f0f0f0;
    border: 1px solid #c0c0c0;
}
#conceptSurface {
    background: #f0f0f0;
    border: 1px solid #c0c0c0;
}
#windowTitle {
    font-size: 13pt;
    font-weight: 700;
    color: #17212b;
}
#sectionTitle {
    font-size: 14pt;
    font-weight: 700;
}
QGroupBox {
    background: transparent;
    border: 1px solid #c0c0c0;
    border-radius: 5px;
    margin-top: 10px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 5px;
}
QPushButton {
    color: #1f2933;
    background: #ffffff;
    border: 1px solid #b8c2cc;
    border-radius: 4px;
    padding: 6px 12px;
    font-weight: 600;
}
QPushButton:hover {
    background: #f1f7fb;
    border-color: #0f5f9e;
}
QPushButton:pressed {
    background: #e3edf4;
}
QPushButton:disabled {
    color: #9aa5b1;
    background: #eef1f4;
    border-color: #d5dbe1;
}
QPushButton#footerButton {
    padding: 6px 14px;
}
QPushButton#footerButton[buttonRole="primary"] {
    color: #ffffff;
    background: #0f5f9e;
    border-color: #0f5f9e;
    font-weight: 700;
}
QPushButton#footerButton[buttonRole="primary"]:hover {
    background: #0c5288;
    border-color: #0c5288;
}
QPushButton#footerButton[buttonRole="primary"]:pressed {
    background: #0a4775;
    border-color: #0a4775;
}
QPushButton#footerButton[buttonRole="danger"] {
    color: #b42318;
    background: #fff8f7;
    border-color: #efb0aa;
}
QPushButton#footerButton[buttonRole="danger"]:hover {
    background: #fff0ee;
    border-color: #d92d20;
}
QPushButton#footerButton[buttonRole="danger"]:pressed {
    background: #fee4e2;
    border-color: #b42318;
}
QPushButton#footerButton[buttonRole="warning"] {
    color: #8a4b00;
    background: #fff8e8;
    border-color: #dfb76b;
}
QPushButton#footerButton[buttonRole="warning"]:hover {
    background: #fff0c2;
    border-color: #b7791f;
}
QPushButton#footerButton[buttonRole="warning"]:pressed {
    background: #fce3a8;
    border-color: #8a4b00;
}
QPushButton#footerButton:disabled {
    color: #9aa5b1;
    background: #eef1f4;
    border-color: #d5dbe1;
}
QProgressBar {
    border: 1px solid #c0c0c0;
    border-radius: 0px;
    height: 24px;
    text-align: center;
    background: #e5e5e5;
}
QProgressBar::chunk {
    background: #0078d7;
}
QTextEdit {
    border: 1px solid #c0c0c0;
    border-radius: 0px;
    background: #ffffff;
}
QComboBox {
    padding: 2px 6px;
    border: 1px solid #c0c0c0;
    border-radius: 0px;
    background: #ffffff;
}
#infoText {
    color: blue;
    font-weight: 700;
}
#dashboardCard {
    background: #ffffff;
    border: 1px solid #d2d8de;
    border-radius: 4px;
}
#dashboardTitle {
    color: #2b3640;
    font-weight: 700;
}
#dashboardLabel {
    color: #52616f;
}
#dashboardValue {
    color: #1f2933;
    font-weight: 700;
}
#highlightsPanel {
    background: #ffffff;
    border: 1px solid #c0c0c0;
    border-radius: 4px;
}
#currentStep {
    color: #17212b;
    font-size: 12pt;
    font-weight: 700;
}
#stepSummary {
    color: #52616f;
}
#highlightCard {
    background: #f8fafb;
    border: 1px solid #d2d8de;
    border-radius: 4px;
}
#highlightTitle {
    color: #2b3640;
    font-weight: 700;
}
#highlightItem_done {
    color: #107c41;
}
#highlightItem_active {
    color: #0f5f9e;
    font-weight: 600;
}
#highlightItem_next {
    color: #52616f;
}
#technicalLogToggle {
    color: #405160;
}
#mainRoundedProgress {
    height: 30px;
    border: 1px solid #bfd0dc;
    border-radius: 15px;
    background: #edf3f7;
    font-weight: 800;
}
#mainRoundedProgress::chunk {
    border-radius: 14px;
    background: #0f5f9e;
}
#mainStatusPanel {
    background: #f8fbfd;
    border: 1px solid #bfd0dc;
    border-left: 5px solid #0f5f9e;
    border-radius: 7px;
}
#mainStatusPercent {
    color: #0f5f9e;
    font-size: 17pt;
    font-weight: 800;
}
#mainStatusEta {
    color: #405160;
    font-weight: 700;
}
#mainStatusHealth {
    color: #107c41;
    background: #e1f4e9;
    border: 1px solid #a8d8b8;
    border-radius: 10px;
    padding: 2px 9px;
    font-weight: 700;
}
#mainStatusProgress {
    height: 18px;
    border: 1px solid #bfd0dc;
    border-radius: 9px;
    background: #edf3f7;
}
#mainStatusProgress::chunk {
    border-radius: 8px;
    background: #0f5f9e;
}
#mainStatusBottom {
    color: #52616f;
    font-weight: 600;
}
#conceptCard {
    background: #ffffff;
    border: 1px solid #c8d0d8;
    border-radius: 5px;
}
#conceptTitle {
    color: #17212b;
    font-size: 12pt;
    font-weight: 700;
}
#conceptDescription,
#conceptCaption {
    color: #52616f;
}
#percentOnlyProgress,
#stageAwareProgress,
#milestoneProgress,
#etaProgress {
    height: 30px;
    font-weight: 700;
}
#percentOnlyProgress::chunk {
    background: #0f5f9e;
}
#stageAwareProgress::chunk {
    background: #0f7f45;
}
#milestoneProgress::chunk {
    background: #7057c8;
}
#etaProgress::chunk {
    background: #0f7c80;
}
#progressStageLabel {
    color: #0f7f45;
    font-weight: 700;
}
#etaLabel {
    color: #17212b;
    font-weight: 700;
}
#healthyLabel {
    color: #107c41;
    font-weight: 700;
}
#pipelineStep_done,
#pipelineStep_active,
#pipelineStep_next {
    border: 1px solid #c8d0d8;
    border-radius: 5px;
    background: #f8fafb;
}
#pipelineStep_done {
    border-color: #107c41;
}
#pipelineStep_active {
    border-color: #7057c8;
    background: #f2efff;
}
#pipelineStep_next {
    color: #52616f;
}
#pipelineStepNumber {
    font-weight: 800;
    font-size: 13pt;
}
#pipelineStepLabel {
    font-weight: 600;
}
#roundedRailProgress {
    height: 32px;
    border: 1px solid #bfd0dc;
    border-radius: 16px;
    background: #edf3f7;
    font-weight: 800;
}
#roundedRailProgress::chunk {
    border-radius: 15px;
    background: #0f5f9e;
}
#stageTag_done,
#stageTag_active,
#stageTag_next {
    border: 1px solid #c8d0d8;
    border-radius: 10px;
    padding: 3px 8px;
    font-size: 9pt;
}
#stageTag_done {
    color: #107c41;
    border-color: #107c41;
}
#stageTag_active {
    color: #ffffff;
    background: #0f5f9e;
    border-color: #0f5f9e;
}
#stageTag_next {
    color: #52616f;
}
#segmentPiece_done,
#segmentPiece_active,
#segmentPiece_next {
    border-radius: 4px;
    font-weight: 700;
}
#segmentPiece_done {
    background: #dff4e7;
    color: #107c41;
}
#segmentPiece_active {
    background: #0f5f9e;
    color: #ffffff;
}
#segmentPiece_next {
    background: #e8edf2;
    color: #52616f;
}
#conceptMetric {
    color: #405160;
    font-weight: 700;
    padding-right: 10px;
}
#timelineRail {
    background: #f8fafb;
    border: 1px solid #d2d8de;
    border-radius: 5px;
}
#timelineConnector {
    background: #c8d0d8;
}
#timelineDot_done,
#timelineDot_active,
#timelineDot_next {
    border-radius: 16px;
    font-weight: 800;
}
#timelineDot_done {
    background: #107c41;
    color: #ffffff;
}
#timelineDot_active {
    background: #0f5f9e;
    color: #ffffff;
}
#timelineDot_next {
    background: #e8edf2;
    color: #52616f;
}
#timelineLabel {
    color: #405160;
    font-weight: 600;
    font-size: 9pt;
}
#statusCapsule {
    background: #f8fbfd;
    border: 1px solid #bfd0dc;
    border-left: 5px solid #0f5f9e;
    border-radius: 7px;
}
#capsulePercent {
    color: #0f5f9e;
    font-size: 17pt;
    font-weight: 800;
}
#capsuleEta {
    color: #405160;
    font-weight: 700;
}
#capsuleHealth {
    color: #107c41;
    background: #e1f4e9;
    border: 1px solid #a8d8b8;
    border-radius: 10px;
    padding: 2px 9px;
    font-weight: 700;
}
#capsuleProgress {
    height: 20px;
    border: 1px solid #bfd0dc;
    border-radius: 10px;
    background: #edf3f7;
}
#capsuleProgress::chunk {
    border-radius: 9px;
    background: #0f5f9e;
}
#capsuleBottom {
    color: #52616f;
    font-weight: 600;
}
#mutedValue {
    color: #52616f;
}
#messageHeadline {
    font-weight: 700;
    font-size: 11pt;
}
#messageDetail {
    color: #405160;
}
#warningIcon,
#errorIcon,
#infoIcon {
    border-radius: 21px;
    min-height: 42px;
    max-height: 42px;
    font-weight: 800;
    font-size: 17pt;
}
#warningIcon {
    background: #fff4d6;
    color: #9a6700;
}
#errorIcon {
    background: #fee4e2;
    color: #b42318;
}
#infoIcon {
    background: #d9f0ff;
    color: #0f5f9e;
}
#positiveDelta {
    color: #107c41;
    font-weight: 700;
}
#negativeDelta {
    color: #b42318;
    font-weight: 700;
}
"""


def main():
    app = QApplication(sys.argv)
    gallery = RetrainPreviewGallery()
    gallery.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
