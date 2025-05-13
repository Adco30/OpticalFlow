"""
Custom widgets for the OpticalFlow visualization interface.
"""
from PyQt5.QtWidgets import (
    QFrame, QLabel, QVBoxLayout, QHBoxLayout, QWidget, QSlider,
    QComboBox, QCheckBox, QScrollArea, QGroupBox, QTextEdit, QPushButton
)
from PyQt5.QtCore import Qt, pyqtSignal, QSize
from PyQt5.QtGui import QPixmap, QImage, QFont
import numpy as np
from typing import Optional, List, Tuple

from utils.config import FlowConfiguration


class FlowDisplayCell(QFrame):
    """Widget for displaying images with titles and info text."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setFrameStyle(QFrame.Box | QFrame.Plain)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(10, 10, 10, 10)

        self.title_label = QLabel()
        self.title_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.layout.addWidget(self.title_label)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.image_label)

        self.info_label = QLabel()
        self.info_label.setAlignment(Qt.AlignRight | Qt.AlignBottom)
        self.layout.addWidget(self.info_label)

        # Set size policy to allow horizontal expanding
        self.setSizePolicy(self.sizePolicy().Expanding, self.sizePolicy().Fixed)
        self.stored_image = None
        self.stored_pixmap = None

    def set_display_content(self, title: str, image: np.ndarray, info: str = ""):
        """Set the content of this display cell."""
        self.title_label.setText(title)
        self.info_label.setText(info)
        self.stored_image = image

        if image is None:
            return

        # Convert image to display format
        if image.dtype != np.uint8:
            img_display = (np.clip(image, 0, 1) * 255).astype(np.uint8)
        else:
            img_display = image

        h, w = img_display.shape[:2]

        # Create QImage
        if len(img_display.shape) == 2:
            qimg = QImage(img_display.data, w, h, w, QImage.Format_Grayscale8)
        else:
            qimg = QImage(img_display.data, w, h, w * 3, QImage.Format_RGB888)

        pixmap = QPixmap.fromImage(qimg)
        self.stored_pixmap = pixmap
        self.update_display_size()

    def update_display_size(self):
        """Scale the displayed image to fit available width."""
        if not self.stored_pixmap:
            return

        available_width = self.width() - 20
        if available_width <= 0:
            return

        scaled_pixmap = self.stored_pixmap.scaledToWidth(
            available_width,
            Qt.SmoothTransformation
        )

        self.image_label.setPixmap(scaled_pixmap)
        self.updateGeometry()

    def resizeEvent(self, event):
        """Handle widget resizing."""
        super().resizeEvent(event)
        self.update_display_size()

    def sizeHint(self):
        """Calculate appropriate size hint."""
        if not self.stored_pixmap:
            return QSize(200, 200)

        available_width = self.width() - 20
        if available_width <= 0:
            available_width = 200

        aspect_ratio = self.stored_pixmap.height() / self.stored_pixmap.width()
        image_height = int(available_width * aspect_ratio)

        # Add space for labels
        total_height = (
            image_height +
            self.title_label.sizeHint().height() +
            self.info_label.sizeHint().height() + 40
        )

        return QSize(self.width(), total_height)


class SelectableFlowCell(FlowDisplayCell):
    """Flow display cell that can be selected by clicking."""
    clicked = pyqtSignal(int)

    def __init__(self, row_index: int, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.row_index = row_index
        self.setCursor(Qt.PointingHandCursor)

    def mousePressEvent(self, event):
        self.clicked.emit(self.row_index)
        super().mousePressEvent(event)


class ParameterSliderControl(QWidget):
    """Slider widget for adjusting floating-point parameters."""
    valueChanged = pyqtSignal(str, float)

    def __init__(self, name: str, value_range: Tuple[float, float], default_value: float, parent: Optional[QWidget] = None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        header = QHBoxLayout()
        label = QLabel(name.replace('_', ' ').title())
        self.value_label = QLabel(f"{default_value:.3f}")
        header.addWidget(label)
        header.addWidget(self.value_label)
        layout.addLayout(header)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 100)

        # Set initial position
        initial_pos = int((default_value - value_range[0]) * 100 / (value_range[1] - value_range[0]))
        self.slider.setValue(initial_pos)
        layout.addWidget(self.slider)

        self.parameter_name = name
        self.range_min = value_range[0]
        self.range_max = value_range[1]

        self.slider.valueChanged.connect(self._handle_value_change)

    def _handle_value_change(self, position: int):
        """Convert slider position to float value and emit signal."""
        value = self.range_min + (position / 100.0) * (self.range_max - self.range_min)
        self.value_label.setText(f"{value:.3f}")
        self.valueChanged.emit(self.parameter_name, value)

    def set_value(self, value: float):
        """Set slider to display specific value."""
        value = max(self.range_min, min(self.range_max, value))
        position = int((value - self.range_min) * 100 / (self.range_max - self.range_min))
        self.slider.setValue(position)
        self.value_label.setText(f"{value:.3f}")


class ParameterDropdownControl(QWidget):
    """Dropdown widget for discrete parameter selection."""
    valueChanged = pyqtSignal(str, object)

    def __init__(self, name: str, options: List[Tuple], default_value, parent: Optional[QWidget] = None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        label = QLabel(name.replace('_', ' ').title())
        layout.addWidget(label)

        self.combo = QComboBox()
        for value, display_text in options:
            self.combo.addItem(display_text, value)

        # Set default value
        for i in range(self.combo.count()):
            if self.combo.itemData(i) == default_value:
                self.combo.setCurrentIndex(i)
                break

        layout.addWidget(self.combo)
        self.parameter_name = name
        self.combo.currentIndexChanged.connect(self._handle_selection_change)

    def _handle_selection_change(self, index: int):
        """Emit selected value when dropdown changes."""
        value = self.combo.itemData(index)
        self.valueChanged.emit(self.parameter_name, value)

    def set_value(self, value):
        """Set dropdown to specific value."""
        for i in range(self.combo.count()):
            if self.combo.itemData(i) == value:
                self.combo.setCurrentIndex(i)
                break


class ParameterControlPanel(QScrollArea):
    """Scrollable panel containing all flow parameter controls."""
    parametersChanged = pyqtSignal(FlowConfiguration, bool)  # Added bool for update button

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setFrameStyle(QFrame.Box | QFrame.Plain)
        self.setFixedWidth(300)

        content = QWidget()
        self.setWidget(content)

        self.flow_config = FlowConfiguration()
        layout = QVBoxLayout(content)
        layout.setSpacing(15)

        # Status label
        self.status_label = QLabel("Optimization running...")
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)

        # Update button
        self.update_button = QPushButton("Update Flow")
        self.update_button.setEnabled(False)
        self.update_button.clicked.connect(self._on_update_clicked)
        layout.addWidget(self.update_button)

        # Parameter sections
        sections = [
            ("Preprocessing", ['gamma', 'gaussian_sigma', 'gradient_scale', 'block_detection_threshold']),
            ("LK Algorithm", ['window_size', 'temporal_kernel_size', 'det_threshold', 'magnitude_threshold']),
            ("Hierarchical", ['pyramid_levels']),
            ("Bilateral Filter", ['bilateral_d', 'bilateral_sigma_color', 'bilateral_sigma_space']),
            ("Visualization", ['grid_spacing', 'vector_scale', 'motion_threshold', 'downsampling_factor'])
        ]

        # Create sliders
        ranges = FlowConfiguration.get_parameter_ranges()
        self.parameter_sliders = {}

        for section_name, param_names in sections:
            group = QGroupBox(section_name)
            group_layout = QVBoxLayout()

            for param_name in param_names:
                range_min, range_max = ranges[param_name]
                default = getattr(self.flow_config, param_name)
                slider = ParameterSliderControl(param_name, (range_min, range_max), default)
                slider.valueChanged.connect(self._on_slider_value_changed)
                group_layout.addWidget(slider)
                self.parameter_sliders[param_name] = slider

            group.setLayout(group_layout)
            layout.addWidget(group)

        # Dropdown controls
        self.parameter_dropdowns = {}
        dropdown_group = QGroupBox("Method Options")
        dropdown_layout = QVBoxLayout()

        # Kernel type
        kernel_dropdown = ParameterDropdownControl(
            'kernel_type',
            FlowConfiguration.get_kernel_types(),
            self.flow_config.kernel_type
        )
        kernel_dropdown.valueChanged.connect(self._on_dropdown_value_changed)
        dropdown_layout.addWidget(kernel_dropdown)
        self.parameter_dropdowns['kernel_type'] = kernel_dropdown

        # Interpolation method
        interp_dropdown = ParameterDropdownControl(
            'interpolation_method',
            FlowConfiguration.get_interpolation_methods(),
            self.flow_config.interpolation_method
        )
        interp_dropdown.valueChanged.connect(self._on_dropdown_value_changed)
        dropdown_layout.addWidget(interp_dropdown)
        self.parameter_dropdowns['interpolation_method'] = interp_dropdown

        # Border mode
        border_dropdown = ParameterDropdownControl(
            'border_mode',
            FlowConfiguration.get_border_modes(),
            self.flow_config.border_mode
        )
        border_dropdown.valueChanged.connect(self._on_dropdown_value_changed)
        dropdown_layout.addWidget(border_dropdown)
        self.parameter_dropdowns['border_mode'] = border_dropdown

        dropdown_group.setLayout(dropdown_layout)
        layout.addWidget(dropdown_group)

        # Options group
        options_group = QGroupBox("Options")
        options_layout = QVBoxLayout()

        self.bilateral_filter_toggle = QCheckBox("Use Bilateral Filter")
        self.bilateral_filter_toggle.stateChanged.connect(self._on_filter_toggle)
        options_layout.addWidget(self.bilateral_filter_toggle)

        self.flow_overlay_check = QCheckBox("Overlay Flow on Difference")
        self.flow_overlay_check.stateChanged.connect(self._on_overlay_changed)
        options_layout.addWidget(self.flow_overlay_check)

        self.difference_matrices_check = QCheckBox("Show Difference Matrices")
        self.difference_matrices_check.stateChanged.connect(self._on_overlay_changed)
        options_layout.addWidget(self.difference_matrices_check)

        options_group.setLayout(options_layout)
        layout.addWidget(options_group)

        # Vector display containers
        self.vector_display_widgets = {}

        # Disable controls until optimization completes
        self.set_panel_enabled(False)
        self.pending_update = False

    def set_panel_enabled(self, enabled: bool):
        """Enable or disable all parameter controls."""
        for slider in self.parameter_sliders.values():
            slider.setEnabled(enabled)
        for dropdown in self.parameter_dropdowns.values():
            dropdown.setEnabled(enabled)
        self.bilateral_filter_toggle.setEnabled(enabled)
        self.flow_overlay_check.setEnabled(enabled)
        self.difference_matrices_check.setEnabled(enabled)

        if enabled and not self.pending_update:
            self.update_button.setEnabled(False)

    def _enable_update_button(self):
        """Enable the update button when parameters change."""
        self.update_button.setEnabled(True)
        self.pending_update = True

    def _on_update_clicked(self):
        """Handle update button click."""
        self.update_button.setEnabled(False)
        self.pending_update = False
        self.parametersChanged.emit(self.flow_config, True)

    def _on_slider_value_changed(self, param_name: str, value: float):
        """Handle slider value changes."""
        self.flow_config.update_parameter(param_name, value)
        self._enable_update_button()

    def _on_dropdown_value_changed(self, param_name: str, value):
        """Handle dropdown value changes."""
        self.flow_config.update_parameter(param_name, value)
        self._enable_update_button()

    def _on_filter_toggle(self, state: int):
        """Handle bilateral filter toggle."""
        self.flow_config.use_bilateral = (state == Qt.Checked)
        self._enable_update_button()

    def _on_overlay_changed(self):
        """Handle overlay/difference matrix toggles."""
        self._enable_update_button()

    def is_overlay_enabled(self) -> bool:
        """Check if flow overlay is enabled."""
        return self.flow_overlay_check.isChecked()

    def show_differences_enabled(self) -> bool:
        """Check if difference matrices should be shown."""
        return self.difference_matrices_check.isChecked()

    def load_configuration(self, config: FlowConfiguration):
        """Update controls from configuration object."""
        for param_name, slider in self.parameter_sliders.items():
            if hasattr(config, param_name):
                slider.set_value(getattr(config, param_name))

        for param_name, dropdown in self.parameter_dropdowns.items():
            if hasattr(config, param_name):
                dropdown.set_value(getattr(config, param_name))

        if hasattr(config, 'use_bilateral'):
            self.bilateral_filter_toggle.setChecked(config.use_bilateral)

        # Update internal config
        for param_name in self.parameter_sliders.keys():
            if hasattr(config, param_name):
                self.flow_config.update_parameter(param_name, getattr(config, param_name))

        for param_name in self.parameter_dropdowns.keys():
            if hasattr(config, param_name):
                self.flow_config.update_parameter(param_name, getattr(config, param_name))

        self.status_label.setText("Optimization complete! Parameters loaded.")
        self.set_panel_enabled(True)
        self.update_button.setEnabled(False)
        self.pending_update = False

    def create_vector_text_display(self, U: np.ndarray, V: np.ndarray, img_shape: Tuple[int, int],
                                 config: FlowConfiguration, region_rect: Optional[Tuple[int, int, int, int]] = None) -> QTextEdit:
        """Create text widget displaying flow vector data."""
        text_area = QTextEdit()
        text_area.setReadOnly(True)
        text_area.setFixedHeight(150)
        text_area.setFont(QFont("Courier New", 9))

        if U is None or V is None:
            text_area.setText("No flow vectors available")
            return text_area

        # Get sampling factor
        sample_factor = int(config.downsampling_factor)
        h, w = img_shape

        # Build text representation
        text_lines = []
        text_lines.append(f"Vector field ({w}x{h}) sampled at factor {sample_factor}")
        if region_rect:
            x, y, bw, bh = region_rect
            text_lines.append(f"Motion region at ({x},{y}) size {bw}x{bh}")
        text_lines.append("Format: (x,y)→(u,v)|magnitude")
        text_lines.append("-" * 30)

        # Calculate motion threshold
        magnitudes = np.sqrt(U**2 + V**2)
        avg_magnitude = np.mean(magnitudes)
        threshold = avg_magnitude * config.motion_threshold

        # Sample vector field
        rows_to_sample = min(15, h)
        row_indices = np.linspace(0, h-1, rows_to_sample, dtype=int)

        for y in row_indices:
            line = ""
            cols_to_sample = min(20, w)
            col_indices = np.linspace(0, w-1, cols_to_sample, dtype=int)

            for x in col_indices:
                u = U[y, x]
                v = V[y, x]
                mag = np.sqrt(u**2 + v**2)

                # Check if point is in motion region
                in_region = False
                if region_rect:
                    bx, by, bw, bh = region_rect
                    in_region = (bx <= x < bx+bw) and (by <= y < by+bh)

                if mag < threshold:
                    if in_region:
                        line += "[-0-] "
                    else:
                        line += "  .   "
                else:
                    marker = "*" if in_region else " "
                    line += f"{marker}({u:.1f},{v:.1f}){marker} "

            text_lines.append(line)

        text_area.setText("\n".join(text_lines))
        return text_area

    def create_difference_text_display(self, diff: np.ndarray) -> QTextEdit:
        """Create text widget displaying difference matrix data."""
        text_area = QTextEdit()
        text_area.setReadOnly(True)
        text_area.setFixedHeight(120)
        text_area.setFont(QFont("Courier New", 9))

        if diff is None:
            text_area.setText("No difference data available")
            return text_area

        h, w = diff.shape[:2]

        text_lines = []
        text_lines.append(f"Difference matrix ({w}x{h}) sampled")
        text_lines.append("-" * 30)

        # Sample the difference matrix
        rows_to_sample = min(8, h)
        row_indices = np.linspace(0, h-1, rows_to_sample, dtype=int)

        for y in row_indices:
            line = ""
            cols_to_sample = min(30, w)
            col_indices = np.linspace(0, w-1, cols_to_sample, dtype=int)

            for x in col_indices:
                # Get pixel value
                if len(diff.shape) == 3:
                    val = np.mean(diff[y, x])
                else:
                    val = diff[y, x]

                # Convert to single digit
                digit = min(9, int(val * 10 / 255))
                line += f"{digit}"

            text_lines.append(line)

        text_area.setText("\n".join(text_lines))
        return text_area

    def update_vector_displays(self, results: List):
        """Update text displays for vector information."""
        # Clean up old displays
        for display in self.vector_display_widgets.values():
            display.setParent(None)
            display.deleteLater()
        self.vector_display_widgets.clear()

        layout = self.widget().layout()

        for filename, img, preprocessed_img, diff, U, V, region_rect in results:
            group = QGroupBox(f"Vectors: {filename}")
            group_layout = QVBoxLayout()

            flow_display = self.create_vector_text_display(U, V, img.shape, self.flow_config, region_rect)
            group_layout.addWidget(flow_display)

            if self.show_differences_enabled():
                diff_display = self.create_difference_text_display(diff)
                group_layout.addWidget(diff_display)

            group.setLayout(group_layout)
            layout.addWidget(group)
            self.vector_display_widgets[filename] = group