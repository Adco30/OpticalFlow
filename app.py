"""
Main application entry point for the OpticalFlow Visualization system.
"""
import sys
from PyQt5.QtWidgets import QApplication
from ui.main_window import FlowVisualizationWindow


def main():
    """Entry point for the optical flow visualization application."""
    app = QApplication(sys.argv)
    window = FlowVisualizationWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
