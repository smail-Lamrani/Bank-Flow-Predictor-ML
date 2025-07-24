# -*- mode: python -*-
from PyInstaller.utils.hooks import collect_data_files
from PyInstaller.utils.hooks import collect_dynamic_libs
import xgboost

block_cipher = None

# Collecte des ressources PyQt5 (plugins, translations…)
datas = collect_data_files('PyQt5', subdir='Qt/plugins')
datas += collect_data_files('PyQt5', subdir='Qt/plugins/platforms')
datas.append((os.path.join(os.path.dirname(xgboost.__file__), "VERSION"), "xgboost"))
datas.append(("ressources/main_window.ui", "ressources"))
datas += [
    ("ressources/global_style.css", "ressources"),
    ("ressources/py_frames/home/home_style.css", "ressources/py_frames/home"),
    ("ressources/py_frames/data_preparation/data_exploration_style.css", "ressources/py_frames/data_preparation"),
    ("ressources/py_frames/forcasting/clf/clf_style.css", "ressources/py_frames/forcasting/clf"),
    ("ressources/py_frames/about/about_style.css", "ressources/py_frames/about"),
    ("ressources/py_frames/data_preparation/visualisation/vis_style.css", "ressources/py_frames/data_preparation/visualisation"),
]



# Spécification pour build ONE-FILE
# Ne pas inclure la section COLLECT : tout est packagé dans EXE

a = Analysis(
    ['main.py'],          # Script principal
    pathex=[],            # Chemins supplémentaires si besoin
    binaries=collect_dynamic_libs("xgboost"),          # Pas de binaries externes (PyInstaller gère la DLL Python automatiquement)
    datas=datas,          # Ressources PyQt5
    hiddenimports=[       # Modules PyQt5 dynamiques
        'PyQt5.QtCore',
        'PyQt5.QtGui',
        'PyQt5.QtWidgets',
        'PyQt5.uic',
        'numpy.core._multiarray_umath',
         'numpy.core.multiarray',
    ],
    excludes=[            # Paquets à exclure
        'tensorflow',
        'keras',
        'torch',
        'pytest',
        'tkinter',
    ],
    hookspath=[],
    runtime_hooks=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
)

pyz = PYZ(
    a.pure,
    a.zipped_data,
    cipher=block_cipher,
)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    exclude_binaries=False,  # FALSE => inclusion des binaires dans l'exe => one-file
    name='mon_app',
    debug=False,             # passez à True pour mode debug du bootloader
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
)
