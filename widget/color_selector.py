from PyQt5.QtWidgets import (QLabel, QWidget, QPushButton, QDialog, QVBoxLayout, QHBoxLayout, QComboBox, QColorDialog)

from PyQt5.QtGui import QColor


class ColorSelectorDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Color")
        self.color = QColor(255, 255, 255)

        # Create a combo box to select between solid color and dimension-based color
        self.color_mode_combo = QComboBox(self)
        self.color_mode_combo.addItem("Solid Color")
        self.color_mode_combo.addItem("Dimension-based Color")
        self.color_mode_combo.currentIndexChanged.connect(self.update_color_mode)

        # Dimension-based color selection widgets
        self.dimension_combo = QComboBox(self)
        self.dimension_combo.addItems(["X", "Y", "Z", "Intensity"])
        self.dimension_combo.currentIndexChanged.connect(self.update_dimension_color)

        # Create containers for dimension-based color widgets
        self.dimension_color_widget = QWidget(self)
        dimension_color_layout = QVBoxLayout(self.dimension_color_widget)
        dimension_color_layout.addWidget(QLabel("Select Dimension:"))
        dimension_color_layout.addWidget(self.dimension_combo)

        # Layout setup
        layout = QVBoxLayout()
        layout.addWidget(QLabel("Color Mode:"))
        layout.addWidget(self.color_mode_combo)
        layout.addWidget(self.dimension_color_widget)

        # Button layout
        button_layout = QHBoxLayout()
        ok_button = QPushButton("OK", self)
        ok_button.clicked.connect(self.accept)
        cancel_button = QPushButton("Cancel", self)
        cancel_button.clicked.connect(self.reject)

        button_layout.addWidget(ok_button)
        button_layout.addWidget(cancel_button)
        layout.addLayout(button_layout)

        self.setLayout(layout)
        self.update_color_mode()

    def update_color_mode(self):
        if self.color_mode_combo.currentText() == "Solid Color":
            # Open QColorDialog for solid color selection
            self.dimension_color_widget.setVisible(False)
            self.color = QColorDialog.getColor(self.color, self, "Select Solid Color")
        else:
            self.dimension_color_widget.setVisible(True)

    def update_dimension_color(self):
        # This method will be called when the dimension is changed
        # You can implement the logic to update the color based on the selected dimension
        pass

    def get_color_mode(self):
        return self.color_mode_combo.currentText()

    def get_selected_dimension(self):
        return self.dimension_combo.currentText()


