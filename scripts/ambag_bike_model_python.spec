# -*- mode: python -*-
a = Analysis(['ambag_bike_model_python.py'],
             pathex=['C:\\Projects\\AMBAG\\PythonCode_crf_edits'],
             hiddenimports=[],
             hookspath=None)
pyz = PYZ(a.pure)
exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          name=os.path.join('dist', 'ambag_bike_model_python.exe'),
          debug=False,
          strip=None,
          upx=True,
          console=True )
