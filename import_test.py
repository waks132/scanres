import importlib
modules = ['network_svd_analyzer','optimized_offensive_engine','main_application']
for m in modules:
    try:
        importlib.import_module(m)
        print(f"{m}: OK")
    except Exception as e:
        print(f"{m}: FAILED: {type(e).__name__}: {e}")
