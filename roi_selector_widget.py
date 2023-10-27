from PyQt5.QtWidgets import QWidget, QLabel, QStackedWidget, QVBoxLayout
from .qt_widgets import NDarray_to_QPixmap


class ROISelectorWidget(QWidget):

    def __init__(self, image = None, *args, **kwargs):
    
        super().__init__(*args, **kwargs)
        
        self.assignment = None
        self.image = image 
        self.declare_components()
        self.layout_components()
    
    def declare_components(self):
        self.image_label = QLabel(self)
        self.image_label.setPixmap(NDarray_to_QPixmap(self.image))

    def layout_components(self):
        pass