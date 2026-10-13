# PyInstaller recipe for the ProTrack3D app. Build from the repository root:
#     pyinstaller --noconfirm --distpath packaging/dist --workpath packaging/build packaging/ProTrack3D.spec
# The result is packaging/dist/ProTrack3D.app on macOS and packaging/dist/ProTrack3D/ on Windows and Linux.
import os
import sys

root = os.path.abspath(os.path.join(SPECPATH, '..'))
# The tracking, family-tree and analysis modules sit at the repository root and are imported by name.
science = ['myTrackStep1', 'myTrackStep2', 'correlation20220708', 'functions', 'create_database', 'createEventIntensityPlots',
           'allAnalysis', 'allAnalysisApp', 'sizeQuartileAnalysis', 'FamilyTreeWithSplitAndMerge', 'FamilyTreeComplete',
           'segment.pruneAndFilter', 'segment.saveTif', 'segment.batch_mask_dia_from_armseg',
           'segment.batch_quantify_arm_from_armseg', 'czifile']
# Matplotlib picks its output code by name at run time: the PNG and PDF writers and the 3D axes must be listed.
plotting = ['matplotlib.backends.backend_agg', 'matplotlib.backends.backend_pdf', 'matplotlib.backends.backend_svg',
            'mpl_toolkits.mplot3d']
icon = os.path.join(SPECPATH, 'ProTrack3D.icns' if sys.platform == 'darwin' else 'ProTrack3D.ico')

a = Analysis([os.path.join(SPECPATH, 'protrack3d_app.py')], pathex=[root], hiddenimports=science + plotting,
             datas=[(os.path.join(root, 'protrack3d', 'model_catalog.json'), 'protrack3d')],
             excludes=['tkinter', 'tensorflow', 'PyQt5', 'PyQt6', 'IPython', 'jupyter', 'notebook', 'pytest'], noarchive=False)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name='ProTrack3D', console=False,
          icon=icon if os.path.isfile(icon) else None)
coll = COLLECT(exe, a.binaries, a.datas, name='ProTrack3D')
if sys.platform == 'darwin':
    app = BUNDLE(coll, name='ProTrack3D.app', icon=icon if os.path.isfile(icon) else None, bundle_identifier='edu.unlv.protrack3d',
                 info_plist={'CFBundleShortVersionString': '2.0', 'NSHighResolutionCapable': True,
                             'LSMinimumSystemVersion': '11.0'})
