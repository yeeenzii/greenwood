# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'AW.ui'
#
# Created by: PyQt5 UI code generator 5.15.7
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from time import strftime
from urllib.request import urlopen
from datetime import date
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5 import QtCore, QtGui, QtWidgets

class Downloader(QThread):
    #found the download stuff on the internet, lol. I'll reference in later.

    # Signal for the window to establish the maximum value
    # of the progress bar.
    setTotalProgress = pyqtSignal(int)
    # Signal to increase the progress.
    setCurrentProgress = pyqtSignal(int)
    # Signal to be emitted when the file has been downloaded successfully.
    succeeded = pyqtSignal()

    def __init__(self, url, filename):
        super().__init__()
        self._url = url
        self._filename = filename

    def run(self):
        url = "https://www.python.org/ftp/python/3.7.2/python-3.7.2.exe"
        filename = "python-3.7.2.exe"
        readBytes = 0
        chunkSize = 1024
        # Open the URL address.
        with urlopen(url) as r:
            # Tell the window the amount of bytes to be downloaded.
            self.setTotalProgress.emit(int(r.info()["Content-Length"]))
            with open(filename, "ab") as f:
                while True:
                    # Read a piece of the file we are downloading.
                    chunk = r.read(chunkSize)
                    # If the result is `None`, that means data is not
                    # downloaded yet. Just keep waiting.
                    if chunk is None:
                        continue
                    # If the result is an empty `bytes` instance, then
                    # the file is complete.
                    elif chunk == b"":
                        break
                    # Write into the local file the downloaded chunk.
                    f.write(chunk)
                    readBytes += chunkSize
                    # Tell the window how many bytes we have received.
                    self.setCurrentProgress.emit(readBytes)
        # If this line is reached then no exception has ocurred in
        # the previous lines.
        self.succeeded.emit()


class Ui_MainWindow(object):

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.progressBar = QtWidgets.QProgressBar(self.centralwidget)
        self.progressBar.setGeometry(QtCore.QRect(30, 80, 731, 31))
        self.progressBar.setProperty("value", 0)
        self.progressBar.setObjectName("progressBar")
        self.DownloadReport = QtWidgets.QPushButton(self.centralwidget)
        self.DownloadReport.setGeometry(QtCore.QRect(632, 121, 131, 31))
        self.DownloadReport.setObjectName("DownloadReport")
        self.DownloadReport.pressed.connect(self.initDownload)
        self.Share = QtWidgets.QRadioButton(self.centralwidget)
        self.Share.setGeometry(QtCore.QRect(640, 150, 111, 20))
        self.Share.setObjectName("Share")
        self.widget = QtWidgets.QWidget(self.centralwidget)
        self.widget.setGeometry(QtCore.QRect(40, 180, 591, 191))
        self.widget.setAutoFillBackground(False)
        self.widget.setStyleSheet("background-color: rgb(120, 120, 120)")
        self.widget.setObjectName("widget")
        self.LocationLabel = QtWidgets.QLabel(self.centralwidget)
        self.LocationLabel.setGeometry(QtCore.QRect(40, 110, 60, 16))
        self.LocationLabel.setObjectName("LocationLabel")
        self.DateLabel = QtWidgets.QLabel(self.centralwidget)
        self.DateLabel.setGeometry(QtCore.QRect(250, 110, 60, 16))
        self.DateLabel.setObjectName("DateLabel")
        self.LocationLabel2 = QtWidgets.QLabel(self.centralwidget)
        self.LocationLabel2.setGeometry(QtCore.QRect(100, 110, 60, 16))
        palette = QtGui.QPalette()
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(126, 126, 126))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(205, 205, 205))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Light, brush)
        brush = QtGui.QBrush(QtGui.QColor(170, 170, 170))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Midlight, brush)
        brush = QtGui.QBrush(QtGui.QColor(68, 68, 68))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Dark, brush)
        brush = QtGui.QBrush(QtGui.QColor(91, 91, 91))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Mid, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Text, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.BrightText, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.ButtonText, brush)
        brush = QtGui.QBrush(QtGui.QColor(126, 126, 126))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(126, 126, 126))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Window, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Shadow, brush)
        brush = QtGui.QBrush(QtGui.QColor(195, 195, 195))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.AlternateBase, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 220))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.ToolTipBase, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.ToolTipText, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(126, 126, 126))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(205, 205, 205))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Light, brush)
        brush = QtGui.QBrush(QtGui.QColor(170, 170, 170))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Midlight, brush)
        brush = QtGui.QBrush(QtGui.QColor(68, 68, 68))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Dark, brush)
        brush = QtGui.QBrush(QtGui.QColor(91, 91, 91))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Mid, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Text, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.BrightText, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.ButtonText, brush)
        brush = QtGui.QBrush(QtGui.QColor(126, 126, 126))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(126, 126, 126))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Window, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Shadow, brush)
        brush = QtGui.QBrush(QtGui.QColor(195, 195, 195))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.AlternateBase, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 220))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.ToolTipBase, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.ToolTipText, brush)
        brush = QtGui.QBrush(QtGui.QColor(68, 68, 68))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(126, 126, 126))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(205, 205, 205))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Light, brush)
        brush = QtGui.QBrush(QtGui.QColor(170, 170, 170))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Midlight, brush)
        brush = QtGui.QBrush(QtGui.QColor(68, 68, 68))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Dark, brush)
        brush = QtGui.QBrush(QtGui.QColor(91, 91, 91))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Mid, brush)
        brush = QtGui.QBrush(QtGui.QColor(68, 68, 68))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Text, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.BrightText, brush)
        brush = QtGui.QBrush(QtGui.QColor(68, 68, 68))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.ButtonText, brush)
        brush = QtGui.QBrush(QtGui.QColor(126, 126, 126))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(126, 126, 126))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Window, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Shadow, brush)
        brush = QtGui.QBrush(QtGui.QColor(136, 136, 136))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.AlternateBase, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 220))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.ToolTipBase, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.ToolTipText, brush)
        self.LocationLabel2.setPalette(palette)
        self.LocationLabel2.setStyleSheet("background-color: rgb(126, 126, 126)")
        self.LocationLabel2.setText("")
        self.LocationLabel2.setObjectName("LocationLabel2")
        self.DateLabel2 = QtWidgets.QLabel(self.centralwidget)
        self.DateLabel2.setGeometry(QtCore.QRect(290, 110, 60, 16))
        self.DateLabel2.setStyleSheet("background-color: rgb(126, 126, 126)")
        self.DateLabel2.setText("")
        self.DateLabel2.setObjectName("DateLabel2")
        self.ElephantsLabel = QtWidgets.QLabel(self.centralwidget)
        self.ElephantsLabel.setGeometry(QtCore.QRect(60, 400, 191, 21))
        self.ElephantsLabel.setObjectName("ElephantsLabel")
        self.MotorcycleLabe = QtWidgets.QLabel(self.centralwidget)
        self.MotorcycleLabe.setGeometry(QtCore.QRect(60, 440, 201, 21))
        self.MotorcycleLabe.setObjectName("MotorcycleLabe")
        self.GunshotLabel = QtWidgets.QLabel(self.centralwidget)
        self.GunshotLabel.setGeometry(QtCore.QRect(60, 480, 191, 21))
        self.GunshotLabel.setObjectName("GunshotLabel")
        self.HumanLabel = QtWidgets.QLabel(self.centralwidget)
        self.HumanLabel.setGeometry(QtCore.QRect(440, 400, 191, 21))
        self.HumanLabel.setObjectName("HumanLabel")
        self.LoggingLabel = QtWidgets.QLabel(self.centralwidget)
        self.LoggingLabel.setGeometry(QtCore.QRect(440, 440, 191, 21))
        self.LoggingLabel.setObjectName("LoggingLabel")
        self.ElephantLable2 = QtWidgets.QLabel(self.centralwidget)
        self.ElephantLable2.setGeometry(QtCore.QRect(280, 400, 60, 16))
        self.ElephantLable2.setText("")
        self.ElephantLable2.setObjectName("ElephantLable2")
        self.HumanLabel2 = QtWidgets.QLabel(self.centralwidget)
        self.HumanLabel2.setGeometry(QtCore.QRect(640, 400, 60, 16))
        self.HumanLabel2.setText("")
        self.HumanLabel2.setObjectName("HumanLabel2")
        self.LoggingLabel2 = QtWidgets.QLabel(self.centralwidget)
        self.LoggingLabel2.setGeometry(QtCore.QRect(640, 440, 60, 16))
        self.LoggingLabel2.setText("")
        self.LoggingLabel2.setObjectName("LoggingLabel2")
        self.GunshotLabel2 = QtWidgets.QLabel(self.centralwidget)
        self.GunshotLabel2.setGeometry(QtCore.QRect(280, 480, 60, 16))
        self.GunshotLabel2.setText("")
        self.GunshotLabel2.setObjectName("GunshotLabel2")
        self.CycleLabel2 = QtWidgets.QLabel(self.centralwidget)
        self.CycleLabel2.setGeometry(QtCore.QRect(280, 440, 60, 16))
        self.CycleLabel2.setText("")
        self.CycleLabel2.setObjectName("CycleLabel2")
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.DownloadReport.setText(_translate("MainWindow", "Download Report"))
        self.Share.setText(_translate("MainWindow", "Share to Cloud"))
        self.LocationLabel.setText(_translate("MainWindow", "Location"))
        self.DateLabel.setText(_translate("MainWindow", "Date"))
        self.ElephantsLabel.setText(_translate("MainWindow", "Frequency of Elephant Sounds"))
        self.MotorcycleLabe.setText(_translate("MainWindow", "Frequency of Motorcycle Sounds"))
        self.GunshotLabel.setText(_translate("MainWindow", "Frequency of Gunshot Sounds"))
        self.HumanLabel.setText(_translate("MainWindow", "Frequency of Human Sounds"))
        self.LoggingLabel.setText(_translate("MainWindow", "Frequency of Loggin Sounds"))

    def initDownload(self):

        # Disable the button while the file is downloading.
        self.DownloadReport.setEnabled(False)
        # Run the download in a new thread.
        self.downloader = Downloader(
            #random link
            "https://www.python.org/ftp/python/3.7.2/python-3.7.2.exe",
            "python-3.7.2.exe"
        )
        # Connect the signals which send information about the download
        # progress with the proper methods of the progress bar.
        self.downloader.setTotalProgress.connect(self.progressBar.setMaximum)
        self.downloader.setCurrentProgress.connect(self.progressBar.setValue)
        # Qt will invoke the `succeeded()` method when the file has been
        # downloaded successfully and `downloadFinished()` when the
        # child thread finishes.
        self.downloader.succeeded.connect(self.downloadSucceeded)
        self.downloader.finished.connect(self.downloadFinished)
        self.downloader.start()

    def downloadSucceeded(self):
        # Set the progress at 100%.
        self.progressBar.setValue(self.progressBar.maximum())
        self.DownloadReport.setText("Downloaded!")

    def downloadFinished(self):
        # Restore the button.
        self.DownloadReport.setEnabled(True)
        # Delete the thread when no longer needed.
        del self.downloader

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
