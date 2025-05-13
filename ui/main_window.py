import os
import sys
import numpy as np
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QGridLayout,
    QScrollArea, QSplitter, QProgressBar, QLabel
)
from PyQt5.QtCore import Qt, QTimer, QThread, QSettings
from PyQt5.QtGui import QPixmap, QImage
import cv2
from PyQt5.QtCore import pyqtSignal

from .widgets import (
    SelectableFlowCell, ParameterControlPanel, 
    FlowDisplayCell
)
from core.image_operations import load_test_images, preprocess_image, compute_image_difference
from core.flow_algorithms import basic_lucas_kanade_flow, hierarchical_lucas_kanade_flow
from optimization.parameter_tuner import ParameterOptimizer, compute_flow_quality_score
from analysis.movement_analyzer import MotionAnalyzer
from utils.config import FlowConfiguration
from utils.video_loader import FlowVisualizationGenerator, FlowProcessingWorker


class RowSelectionManager:
    """Manages row selection and parameter state for different rows."""
    
    def __init__(self):
        self.row_configurations = {}
        self.selected_row = -1
    
    def store_row_config(self, row_index: int, config: FlowConfiguration):
        """Store configuration for a specific row."""
        self.row_configurations[row_index] = config
    
    def get_row_config(self, row_index: int) -> FlowConfiguration:
        """Retrieve configuration for a specific row."""
        return self.row_configurations.get(row_index)
    
    def select_row(self, row_index: int) -> FlowConfiguration:
        """Select a row and return its configuration."""
        self.selected_row = row_index
        return self.get_row_config(row_index)


class AllRowsProcessingWorker(QThread):
    """Worker for processing all images with updated parameters."""
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(list)
    
    def __init__(self, base_image: np.ndarray, test_images: list, 
                 config: FlowConfiguration, motion_regions: dict):
        super().__init__()
        self.base_image = base_image
        self.test_images = test_images
        self.config = config
        self.abort_flag = False
        self.motion_regions = motion_regions
    
    def abort_processing(self):
        """Request processing abort."""
        self.abort_flag = True
    
    def run(self):
        """Process all images with current configuration."""
        if self.abort_flag:
            return
        
        results = []
        total = len(self.test_images)
        
        for i, (filename, img) in enumerate(self.test_images):
            if self.abort_flag:
                return
            
            self.progress.emit(int(i/total * 100), f"Processing {filename}...")
            
            # Preprocess images
            preprocessed_base = preprocess_image(self.base_image, self.config)
            preprocessed_img = preprocess_image(img, self.config)
            
            # Determine algorithm
            use_hierarchical = 'R5' in filename or 'R10' in filename or 'R20' in filename or 'R40' in filename
            
            # Compute optical flow
            if use_hierarchical:
                U, V = hierarchical_lucas_kanade_flow(
                    preprocessed_base, 
                    preprocessed_img,
                    self.config.pyramid_levels,
                    int(self.config.window_size),
                    self.config.kernel_type,
                    self.config.gaussian_sigma,
                    self.config.interpolation_method,
                    self.config.border_mode
                )
            else:
                U, V = basic_lucas_kanade_flow(
                    preprocessed_base, 
                    preprocessed_img, 
                    int(self.config.window_size), 
                    self.config.kernel_type, 
                    self.config.gaussian_sigma
                )
            
            # Compute difference
            diff = compute_image_difference(preprocessed_base, preprocessed_img)
            region_rect = self.motion_regions.get(filename)
            
            # Calculate quality score
            score = compute_flow_quality_score(U, V, region_rect)
            
            result = (filename, img, preprocessed_img, diff, U, V, region_rect)
            results.append(result)
            
            self.progress.emit(int((i + 1)/total * 100), f"Completed {filename} (quality: {score:.4f})")
        
        self.finished.emit(results)


class SingleRowProcessingWorker(QThread):
    """Worker for processing a single image row with specific parameters."""
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(int, tuple)
    
    def __init__(self, base_image: np.ndarray, filename: str, image: np.ndarray, 
                 config: FlowConfiguration, motion_regions: dict, row_index: int):
        super().__init__()
        self.base_image = base_image
        self.filename = filename
        self.image = image
        self.config = config
        self.abort_flag = False
        self.motion_regions = motion_regions
        self.row_index = row_index
    
    def abort_processing(self):
        """Request processing abort."""
        self.abort_flag = True
    
    def run(self):
        """Process the image with current configuration."""
        if self.abort_flag:
            return
        
        self.progress.emit(0, f"Processing {self.filename}...")
        
        # Preprocess images
        preprocessed_base = preprocess_image(self.base_image, self.config)
        preprocessed_img = preprocess_image(self.image, self.config)
        
        # Determine algorithm
        use_hierarchical = 'R5' in self.filename or 'R10' in self.filename or 'R20' in self.filename or 'R40' in self.filename
        
        # Compute optical flow
        if use_hierarchical:
            U, V = hierarchical_lucas_kanade_flow(
                preprocessed_base, 
                preprocessed_img,
                self.config.pyramid_levels,
                int(self.config.window_size),
                self.config.kernel_type,
                self.config.gaussian_sigma,
                self.config.interpolation_method,
                self.config.border_mode
            )
        else:
            U, V = basic_lucas_kanade_flow(
                preprocessed_base, 
                preprocessed_img, 
                int(self.config.window_size), 
                self.config.kernel_type, 
                self.config.gaussian_sigma
            )
        
        # Compute difference
        diff = compute_image_difference(preprocessed_base, preprocessed_img)
        region_rect = self.motion_regions.get(self.filename)
        
        # Calculate quality score
        score = compute_flow_quality_score(U, V, region_rect)
        
        self.progress.emit(100, f"Completed {self.filename} (quality: {score:.4f})")
        
        result = (self.filename, self.image, preprocessed_img, diff, U, V, region_rect)
        self.finished.emit(self.row_index, result)


class FlowVisualizationWindow(QMainWindow):
    """Main window for optical flow visualization and parameter optimization."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Optical Flow Analysis and Optimization")
        self.setMinimumSize(1200, 800)
        
        # Settings for persistence
        self.settings = QSettings("OpticalFlow", "MainWindow")
        
        # Initialize selection manager
        self.row_manager = RowSelectionManager()
        
        # Create main interface
        self.init_ui()
        
        # Initialize data
        self.base_image = None
        self.test_images = None
        self.motion_regions = {}
        self.current_results = None
        self.optimized_config = None
        
        # Thread management
        self.worker_thread = None
        self.current_worker = None
        
        # Load initial data
        self.load_image_data()
        
        # Set up resize timer
        self.resize_timer = QTimer()
        self.resize_timer.setSingleShot(True)
        self.resize_timer.timeout.connect(self.update_grid_layout)
    
    def init_ui(self):
        """Initialize the user interface."""
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Create splitter
        self.splitter = QSplitter(Qt.Horizontal)
        
        # Create scrollable area for flow visualization grid
        self.scroll_area = QScrollArea()
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll_area.setWidgetResizable(True)
        
        # Grid widget for image displays
        self.grid_widget = QWidget()
        self.grid_layout = QGridLayout(self.grid_widget)
        self.grid_layout.setSpacing(10)
        self.grid_layout.setContentsMargins(10, 10, 10, 10)
        self.scroll_area.setWidget(self.grid_widget)
        
        # Parameter control panel
        self.control_panel = ParameterControlPanel()
        self.control_panel.parametersChanged.connect(self.on_parameters_changed)
        
        # Add widgets to splitter
        self.splitter.addWidget(self.scroll_area)
        self.splitter.addWidget(self.control_panel)
        
        # Restore splitter state or set default
        if self.settings.contains("splitter_state"):
            self.splitter.restoreState(self.settings.value("splitter_state"))
        else:
            self.splitter.setSizes([self.width() - 300, 300])
        
        main_layout.addWidget(self.splitter)
        
        # Status bar with progress indicator
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumHeight(15)
        self.statusBar().addPermanentWidget(self.progress_bar, 1)
        self.progress_label = QLabel()
        self.statusBar().addWidget(self.progress_label)
        self.progress_bar.hide()
    
    def closeEvent(self, event):
        """Handle application exit."""
        self.settings.setValue("splitter_state", self.splitter.saveState())
        self.cleanup_workers()
        super().closeEvent(event)
    
    def resizeEvent(self, event):
        """Handle window resize."""
        super().resizeEvent(event)
        self.resize_timer.start(100)
    
    def cleanup_workers(self):
        """Clean up any running worker threads."""
        if self.current_worker:
            if hasattr(self.current_worker, 'abort_processing'):
                self.current_worker.abort_processing()
        if self.worker_thread and self.worker_thread.isRunning():
            self.worker_thread.quit()
            self.worker_thread.wait(1000)
    
    def load_image_data(self):
        """Load base image and test images, detect motion regions."""
        img_path = os.path.join('input_images', 'TestSeq')
        self.base_image, self.test_images = load_test_images(img_path)
        
        # Detect motion regions
        motion_analyzer = MotionAnalyzer()
        for filename, img in self.test_images:
            region_rect = motion_analyzer.detect_moving_region(
                self.base_image, img, 0.1, 20
            )
            if region_rect:
                self.motion_regions[filename] = region_rect
        
        self.start_parameter_optimization()
    
    def start_parameter_optimization(self):
        """Begin Bayesian optimization for flow parameters."""
        self.cleanup_workers()
        self.progress_bar.show()
        self.progress_bar.setValue(0)
        self.progress_label.setText("Starting optimization...")
        
        self.worker_thread = QThread()
        self.current_worker = ParameterOptimizer(self.base_image, self.test_images)
        self.current_worker.motion_regions = self.motion_regions
        self.current_worker.moveToThread(self.worker_thread)
        
        self.worker_thread.started.connect(self.current_worker.run_optimization)
        self.current_worker.progress.connect(self.update_progress)
        self.current_worker.optimized_params.connect(self.on_parameters_optimized)
        self.current_worker.finished.connect(self.on_processing_complete)
        
        self.worker_thread.start()
    
    def update_progress(self, value: int, message: str):
        """Update progress indicators."""
        self.progress_bar.setValue(value)
        self.progress_label.setText(message)
    
    def on_parameters_optimized(self, config: FlowConfiguration):
        """Handle optimized parameters received."""
        self.optimized_config = config
        self.control_panel.load_configuration(config)
    
    def on_processing_complete(self, results: list):
        """Handle completion of processing."""
        self.progress_bar.hide()
        self.progress_label.setText("")
        
        if results:
            self.current_results = results
            self.update_visualization_display()
            self.control_panel.update_vector_displays(results)
    
    def copy_configuration(self, source_config: FlowConfiguration) -> FlowConfiguration:
        """Create deep copy of configuration."""
        config_copy = FlowConfiguration()
        
        # Copy all attributes
        for attr in vars(source_config):
            setattr(config_copy, attr, getattr(source_config, attr))
        
        return config_copy
    
    def on_parameters_changed(self, config: FlowConfiguration, is_update_button: bool):
        """Handle parameter changes from control panel."""
        if not is_update_button:
            # Just a parameter change, not from update button
            return
            
        # Reprocess all images with new parameters
        self.cleanup_workers()
        self.progress_bar.show()
        self.progress_bar.setValue(0)
        self.progress_label.setText("Reprocessing all images...")
        
        # Create deep copy of configuration for worker thread
        config_copy = self.copy_configuration(config)
        
        # Create worker for all rows
        self.worker_thread = QThread()
        self.current_worker = AllRowsProcessingWorker(
            self.base_image, 
            self.test_images, 
            config_copy,
            self.motion_regions
        )
        self.current_worker.moveToThread(self.worker_thread)
        
        self.worker_thread.started.connect(self.current_worker.run)
        self.current_worker.progress.connect(self.update_progress)
        self.current_worker.finished.connect(self.on_all_rows_processed)
        
        self.worker_thread.start()
    
    def on_all_rows_processed(self, results: list):
        """Handle completion of all rows processing."""
        self.current_results = results
        self.update_visualization_display()
        self.control_panel.update_vector_displays(results)
        self.progress_bar.hide()
        self.progress_label.setText(f"Processed {len(results)} images successfully.")
        
        # Store configuration for current displayed rows
        config_copy = self.copy_configuration(self.control_panel.flow_config)
        for i in range(len(results)):
            self.row_manager.store_row_config(i, config_copy)
    
    def update_visualization_display(self):
        """Update the main visualization grid."""
        if not self.current_results:
            return
        
        # Clear existing grid
        while self.grid_layout.count():
            item = self.grid_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        # Calculate column width
        scrollbar_width = self.scroll_area.verticalScrollBar().width() if self.scroll_area.verticalScrollBar().isVisible() else 0
        available_width = self.scroll_area.width() - scrollbar_width - 20
        
        if available_width <= 0:
            available_width = 800
        
        col_count = 3
        spacing_width = self.grid_layout.horizontalSpacing() * (col_count - 1)
        cell_width = (available_width - spacing_width) // col_count
        
        overlay_enabled = self.control_panel.is_overlay_enabled()
        
        # Create grid cells for each result
        for row, (filename, original_img, preprocessed_img, diff, U, V, region_rect) in enumerate(self.current_results):
            avg_u = np.mean(np.abs(U))
            avg_v = np.mean(np.abs(V))
            
            # Original image cell
            img_cell = SelectableFlowCell(row)
            img_cell.setFixedWidth(cell_width)
            img_cell.set_display_content(filename, original_img)
            img_cell.clicked.connect(self.on_row_selection_changed)
            
            # Flow visualization
            flow_bitmap = FlowVisualizationGenerator.create_flow_visualization(
                U, V, original_img.shape, self.control_panel.flow_config, region_rect
            )
            
            # Highlight selected row
            if row == self.row_manager.selected_row:
                img_cell.setStyleSheet("border: 2px solid #3498db;")
            
            # Configure display based on overlay setting
            if overlay_enabled:
                # Show overlay of flow on difference
                overlay_img = FlowVisualizationGenerator.overlay_flow_on_difference(flow_bitmap, diff)
                diff_cell = SelectableFlowCell(row)
                diff_cell.setFixedWidth(cell_width)
                diff_cell.set_display_content("Difference + Flow", overlay_img)
                diff_cell.clicked.connect(self.on_row_selection_changed)
                
                if row == self.row_manager.selected_row:
                    diff_cell.setStyleSheet("border: 2px solid #3498db;")
                
                self.grid_layout.addWidget(img_cell, row, 0)
                self.grid_layout.addWidget(diff_cell, row, 1)
                
                # Preprocessed image
                prep_cell = SelectableFlowCell(row)
                prep_cell.setFixedWidth(cell_width)
                prep_cell.set_display_content("Preprocessed", preprocessed_img)
                prep_cell.clicked.connect(self.on_row_selection_changed)
                
                if row == self.row_manager.selected_row:
                    prep_cell.setStyleSheet("border: 2px solid #3498db;")
                
                self.grid_layout.addWidget(prep_cell, row, 2)
            else:
                # Show separate difference and flow
                diff_cell = SelectableFlowCell(row)
                diff_cell.setFixedWidth(cell_width)
                diff_cell.set_display_content("Difference", diff)
                diff_cell.clicked.connect(self.on_row_selection_changed)
                
                flow_cell = SelectableFlowCell(row)
                flow_cell.setFixedWidth(cell_width)
                
                # Add region info if available
                region_info = ""
                if region_rect:
                    x, y, w, h = region_rect
                    region_info = f"\nMotion region: ({x},{y}) {w}x{h}"
                
                flow_cell.set_display_content(
                    "Flow Visualization",
                    flow_bitmap,
                    f"Avg. displacement: ({avg_u:.2f}, {avg_v:.2f}){region_info}"
                )
                flow_cell.clicked.connect(self.on_row_selection_changed)
                
                # Apply selection highlighting
                if row == self.row_manager.selected_row:
                    img_cell.setStyleSheet("border: 2px solid #3498db;")
                    diff_cell.setStyleSheet("border: 2px solid #3498db;")
                    flow_cell.setStyleSheet("border: 2px solid #3498db;")
                
                self.grid_layout.addWidget(img_cell, row, 0)
                self.grid_layout.addWidget(diff_cell, row, 1)
                self.grid_layout.addWidget(flow_cell, row, 2)
    
    def on_row_selection_changed(self, row_index: int):
        """Handle row selection change."""
        if row_index != self.row_manager.selected_row:
            self.row_manager.selected_row = row_index
            stored_config = self.row_manager.select_row(row_index)
            
            # Update control panel with stored configuration
            if stored_config:
                self.control_panel.load_configuration(stored_config)
            
            # Update visual highlighting
            self.update_selection_highlighting()
            
            self.statusBar().showMessage(f"Selected row {row_index+1}")
    
    def update_selection_highlighting(self):
        """Update visual highlighting for selected row."""
        for i in range(self.grid_layout.count()):
            item = self.grid_layout.itemAt(i)
            if item and item.widget():
                widget = item.widget()
                if isinstance(widget, SelectableFlowCell):
                    if widget.row_index == self.row_manager.selected_row:
                        widget.setStyleSheet("border: 2px solid #3498db;")
                    else:
                        widget.setStyleSheet("")
    
    def update_grid_layout(self):
        """Recalculate grid cell widths on window resize."""
        scrollbar_width = self.scroll_area.verticalScrollBar().width() if self.scroll_area.verticalScrollBar().isVisible() else 0
        available_width = self.scroll_area.width() - scrollbar_width - 20
        
        if available_width <= 0:
            return
        
        col_count = 3
        spacing_width = self.grid_layout.horizontalSpacing() * (col_count - 1)
        cell_width = (available_width - spacing_width) // col_count
        
        # Update all cell widths
        for i in range(self.grid_layout.count()):
            item = self.grid_layout.itemAt(i)
            if item and item.widget():
                item.widget().setFixedWidth(cell_width)
