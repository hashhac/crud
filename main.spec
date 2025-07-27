# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[('static', 'static'), ('templates', 'templates'), ('C:/Users/GGPC/.cache/huggingface/hub/models--openai--whisper-small', 'models/models--openai--whisper-small'), ('C:/Users/GGPC/.cache/huggingface/hub/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0', 'models/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0'), ('C:/Users/GGPC/.cache/huggingface/hub/models--microsoft--speecht5_tts', 'models/models--microsoft--speecht5_tts'), ('C:/Users/GGPC/.cache/huggingface/hub/models--microsoft--speecht5_hifigan', 'models/models--microsoft--speecht5_hifigan')],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='main',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='main',
)
